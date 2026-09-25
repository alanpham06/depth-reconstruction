"""export.py — the completion as a .pte for ExecuTorch, Vulkan backend, GPU only.

    python -m export.export --checkpoint checkpoints/<run>_version_N_best.ckpt --size 256

Lowering is torch.export, then
to_edge_transform_and_lower(partitioner=[VulkanPartitioner()]), then to_executorch.
The shape is baked in, because the Vulkan partitioner wants static ones, so one
export serves one --size.

The graph is the whole completion rather than the bare UNet. It takes sparse depth
and its mask as one (1, 2, H, W) tensor, which is what a sensor gives, and returns
metric depth (1, 1, H, W): assemble_input's centring on the mean sparse depth and
the add-back afterwards are inside it, so a caller needs nothing from this repo.

GPU only. Vulkan is the one backend, and an op the partitioner leaves outside the
delegate is an error that writes nothing. Such an op runs on the portable CPU
fallback, and fails at load time in an app built with
EXECUTORCH_BUILD_PORTABLE_OPS=OFF, so a .pte holding one only looks exported.
Measured 2026-09-24 on a base-32 UNet at 256x256: one Vulkan subgraph, nothing left
over, and the exported graph matching eager exactly.

Only the network is imported, never the training wrapper, so this runs in the
executorch conda env, which has neither Lightning nor matplotlib.
"""

from __future__ import annotations

import argparse
import operator
import os
import shutil
from pathlib import Path

import torch

from datasets.sparse_depth import assemble_input
from models import predict
from utils.checkpoint import load_network
from utils.paths import default_output, run_name

# Parity below this is fusion and float-order noise rather than a different network
PARITY_TOLERANCE = 1e-3


class Completion(torch.nn.Module):
    """Sparse depth and its mask in, metric depth out: the whole of what ships."""

    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, sparse: torch.Tensor) -> torch.Tensor:
        inputs, ref = assemble_input(sparse[:, 0], sparse[:, 1])
        return predict(self.network, inputs, ref)


def use_bundled_flatc() -> Path | None:
    """Put ExecuTorch's own flatc in front of whatever else is on PATH.

    Serialization shells out to `flatc` by bare name, so PATH decides which one
    runs. A conda base environment with the executorch wheel installed leaves a
    console script called flatc in its bin directory, and that shim is a Python
    entry point pointing at *its* interpreter -- so from any other environment it
    fails to import and the export dies at the last step, after the whole graph
    has already lowered successfully. The binary that ships beside the executorch
    package is the one that matches this installation, so it goes first.
    """
    try:
        import executorch
    except ImportError:
        return None
    # A namespace package, so __file__ is None and the search goes through __path__
    for root in getattr(executorch, "__path__", []):
        bundled = Path(root) / "data" / "bin" / "flatc"
        if bundled.is_file():
            os.environ["PATH"] = (
                f"{bundled.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            )
            return bundled
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export the completion as a Vulkan .pte")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--size",
        type=int,
        default=256,
        help="image width and height; the graph is exported for this one shape",
    )
    p.add_argument("--output", default=None, help="defaults to output/export/<run>")
    return p.parse_args(argv)


def example_input(size: int) -> tuple[torch.Tensor]:
    """One (1, 2, size, size) input: depth at 5% of pixels, then the mask of which.

    Seeded, so a parity number is reproducible, and realistic rather than zeros, so
    it exercises the centring on the mean sparse depth.
    """
    generator = torch.Generator().manual_seed(0)
    mask = (torch.rand(1, 1, size, size, generator=generator) < 0.05).float()
    depth = (2.0 + 3.0 * torch.rand(1, 1, size, size, generator=generator)) * mask
    return (torch.cat([depth, mask], dim=1),)


def lower(module, example):
    """The exported graph, and the Vulkan program it lowers to."""
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    if (use_bundled_flatc() or shutil.which("flatc")) is None:
        raise SystemExit("no flatc on PATH and none bundled with executorch")
    exported = torch.export.export(module, example)
    lowered = to_edge_transform_and_lower(exported, partitioner=[VulkanPartitioner()])
    return exported, lowered.to_executorch()


def undelegated_ops(program) -> list[str]:
    """The ops the backend left for the CPU, read off the lowered graph.

    Read rather than matched against a list: a hardcoded list of "ops Vulkan cannot
    take" goes stale silently in both directions, and the graph already says which
    nodes ended up outside a delegate call. Every call_function node counts except
    the delegate call itself and operator.getitem, which only unpacks the
    delegate's own output tuple. Anything else listed here runs on the portable
    fallback, and fails at LOAD time in an app built with
    EXECUTORCH_BUILD_PORTABLE_OPS=OFF -- so a clean export here is not proof the
    .pte runs on the device.
    """
    graph = program.exported_program().graph_module.graph
    ops = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if "executorch_call_delegate" in str(node.target):
            continue
        if node.target is operator.getitem:
            continue
        ops.add(str(node.target))
    return sorted(ops)


def parity(network, exported, example) -> float:
    """Largest absolute difference between the eager model and the exported graph.

    It compares the exported graph rather than the .pte, because executing the
    .pte needs the runtime built for this host; what it catches is export-time
    divergence -- a traced constant, a branch folded the wrong way, a dtype
    promoted.
    """
    with torch.no_grad():
        reference = network(*example)
        replayed = exported.module()(*example)
    return float((reference - replayed).abs().max())


def main(argv=None):
    args = parse_args(argv)
    network = load_network(args.checkpoint)
    completion = Completion(network).eval()
    example = example_input(args.size)
    exported, program = lower(completion, example)
    drift = parity(completion, exported, example)
    stragglers = undelegated_ops(program)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Shape:      {tuple(example[0].shape)} -> (1, 1, {args.size}, {args.size})")
    print(f"Parity:     {drift:.2e} against eager (tolerance {PARITY_TOLERANCE:g})")
    for op in stragglers:
        print(f"  NOT DELEGATED  {op}", flush=True)
    if stragglers:
        raise SystemExit(
            f"{len(stragglers)} op(s) would run on the CPU fallback; exports are "
            "GPU only, so no .pte was written"
        )
    if drift > PARITY_TOLERANCE:
        raise SystemExit(
            f"export diverges from the eager model by {drift:.2e}, over the "
            f"{PARITY_TOLERANCE:g} tolerance; no .pte was written"
        )
    output = (
        Path(args.output) if args.output else default_output("export", args.checkpoint)
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{run_name(args.checkpoint)}_vulkan.pte"
    path.write_bytes(program.buffer)
    print(f"Wrote:      {path.resolve()}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
