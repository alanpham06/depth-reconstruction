"""eval.py — score a checkpoint and both baselines on held-out captures.

    python -m eval.eval --checkpoint checkpoints/<run>_version_N_best.ckpt --val data/val

Every number is pooled over the valid pixels of what it names -- one capture file,
which is one shape, or all of them -- exactly as training's val/ tags are, so the
"all" row of a best.ckpt reproduces the val/mae it was chosen on. The constant and
nearest rows score the same views with no network at all: read the model against
nearest, never on its own.

Writes metrics.json and panel.png to output/eval/<run>_<source>/, where <source> is
the directory the captures came from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.sparse_depth import SparseDepthDataset, capture_files, evenly_spaced_batch
from models import predict
from models.baselines import BASELINES
from utils.checkpoint import load_network
from utils.metrics import METRICS, PooledMetrics
from utils.panel import PANEL_COLUMNS, depth_panel
from utils.paths import CAPTURE_SUFFIX, default_output, source_label


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--val",
        type=Path,
        nargs="+",
        required=True,
        help="*_gt.pt files, or directories of them",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="defaults to output/eval/<run>_<source>",
    )
    p.add_argument(
        "--num-panels", type=int, default=6, help="views in panel.png; 0 writes none"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


@torch.no_grad()
def score(network, dataset, batch_size: int, device) -> dict[str, dict[str, float]]:
    """The model and every baseline, pooled over each valid pixel of `dataset`."""
    pooled = PooledMetrics()
    for batch in DataLoader(dataset, batch_size=batch_size):
        batch = {key: value.to(device) for key, value in batch.items()}
        sparse = batch["input"][:, 1:2]
        pred = predict(network, batch["input"], batch["ref"])
        pooled.add("model", pred, batch["target"], batch["valid"], sparse)
        for name, fill in BASELINES.items():
            fill_pred = fill(batch["input"], batch["ref"])
            pooled.add(name, fill_pred, batch["target"], batch["valid"], sparse)
    return pooled.summarize()


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    network = load_network(args.checkpoint, device)
    files = capture_files(args.val)
    names = [file.name.removesuffix(CAPTURE_SUFFIX) for file in files]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise SystemExit(
            f"--val names two captures called {', '.join(duplicates)}; "
            "score each directory on its own"
        )
    output = args.output or default_output(
        "eval", args.checkpoint, source_label(args.val)
    )
    output.mkdir(parents=True, exist_ok=True)

    results = {
        file.name.removesuffix(CAPTURE_SUFFIX): score(
            network, SparseDepthDataset([file]), args.batch_size, device
        )
        for file in files
    }
    results["all"] = score(network, SparseDepthDataset(files), args.batch_size, device)
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'capture':<12} {'predictor':<9} " + " ".join(f"{m:>9}" for m in METRICS))
    for capture, scores in results.items():
        for predictor, metrics in scores.items():
            values = " ".join(f"{metrics[m]:>9.4f}" for m in METRICS)
            print(f"{capture:<12} {predictor:<9} {values}")

    if args.num_panels > 0:
        batch = evenly_spaced_batch(SparseDepthDataset(files), args.num_panels)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            pred = predict(network, batch["input"], batch["ref"])
        save_image(
            depth_panel(batch, pred), output / "panel.png", nrow=len(PANEL_COLUMNS)
        )
    print(f"Wrote: {output.resolve()}")


if __name__ == "__main__":
    main()
