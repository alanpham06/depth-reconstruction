"""The export is the whole completion, matches eager, and leaves nothing on the CPU."""

import operator
from types import SimpleNamespace

import pytest
import torch

import export.export as export
from datasets.sparse_depth import assemble_input
from models import make_model, predict

SIZE = 32


def _network():
    torch.manual_seed(0)
    return make_model(base_channels=8).eval()


def test_the_graph_takes_raw_sparse_depth_and_returns_metric_depth():
    network = _network()
    (sparse,) = export.example_input(SIZE)
    inputs, ref = assemble_input(sparse[:, 0], sparse[:, 1])
    with torch.no_grad():
        torch.testing.assert_close(
            export.Completion(network)(sparse),
            predict(network, inputs, ref),
            rtol=0,
            atol=0,
        )


def test_the_exported_graph_matches_eager():
    completion = export.Completion(_network()).eval()
    example = export.example_input(SIZE)
    exported = torch.export.export(completion, example)
    assert export.parity(completion, exported, example) <= 1e-6


def test_there_is_no_cpu_backend_to_choose():
    with pytest.raises(SystemExit):
        export.parse_args(["--checkpoint", "x.ckpt", "--backend", "portable"])


def _fake_program(node_specs):
    """A stand-in for the lowered program, just deep enough for undelegated_ops:
    it only ever reads graph_module.graph.nodes off the exported program."""
    nodes = [SimpleNamespace(op=op, target=target) for op, target in node_specs]
    graph_module = SimpleNamespace(graph=SimpleNamespace(nodes=nodes))
    return SimpleNamespace(
        exported_program=lambda: SimpleNamespace(graph_module=graph_module)
    )


def test_a_non_aten_op_outside_the_delegate_still_counts_as_left_on_the_cpu():
    """Checking only for "aten" in the target let a non-aten op outside the
    delegate pass as if the whole graph had been lowered."""
    program = _fake_program(
        [
            ("call_function", "executorch_call_delegate"),
            ("call_function", operator.getitem),
            ("call_function", "my_custom_lib.frobnicate.default"),
            ("placeholder", "x"),
        ]
    )
    assert export.undelegated_ops(program) == ["my_custom_lib.frobnicate.default"]


def test_an_op_left_on_the_cpu_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "load_network", lambda path: _network())
    monkeypatch.setattr(
        export,
        "lower",
        lambda module, example: (
            torch.export.export(module, example),
            SimpleNamespace(buffer=b"pte"),
        ),
    )
    monkeypatch.setattr(export, "undelegated_ops", lambda program: ["aten.log1p.out"])
    with pytest.raises(SystemExit, match="GPU only"):
        export.main(
            [
                "--checkpoint",
                "checkpoints/tiny_version_0_best.ckpt",
                "--size",
                str(SIZE),
                "--output",
                str(tmp_path),
            ]
        )
    assert list(tmp_path.iterdir()) == []


def test_a_parity_drift_over_tolerance_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "load_network", lambda path: _network())
    monkeypatch.setattr(
        export,
        "lower",
        lambda module, example: (
            torch.export.export(module, example),
            SimpleNamespace(buffer=b"pte"),
        ),
    )
    monkeypatch.setattr(export, "undelegated_ops", lambda program: [])
    monkeypatch.setattr(export, "parity", lambda network, exported, example: 1.0)
    with pytest.raises(SystemExit, match="diverges"):
        export.main(
            [
                "--checkpoint",
                "checkpoints/tiny_version_0_best.ckpt",
                "--size",
                str(SIZE),
                "--output",
                str(tmp_path),
            ]
        )
    assert list(tmp_path.iterdir()) == []


def test_the_whole_graph_lowers_to_vulkan():
    pytest.importorskip("executorch")
    completion = export.Completion(_network()).eval()
    _, program = export.lower(completion, export.example_input(SIZE))
    assert export.undelegated_ops(program) == []
