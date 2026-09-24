"""A whole run, end to end, on a capture small enough to train in seconds.

The real render.capture and train.main on 32x32 views, then the checks a person
would otherwise make by opening TensorBoard and the run directory.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import render.capture as capture
import train
from eval import eval as eval_cli
from utils.metrics import METRICS

SIZE, PANEL_IMAGES = 32, 2
LOSS_TERMS = ("loss", "l1", "grad")
EXPECTED_SCALARS = {
    *(f"batch/{term}" for term in (*LOSS_TERMS, "grad_norm")),
    *(f"train/{term}" for term in LOSS_TERMS),
    *(f"val/{term}" for term in LOSS_TERMS),
    *(f"val/{metric}" for metric in METRICS),
    *(f"val/{m}_{b}" for m in METRICS for b in ("nearest", "constant")),
    "lr/recon",
    "epoch",
}
PANEL_TAGS = {"train/sparse_pred_gt_error", "val/sparse_pred_gt_error"}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    data = root / "data"
    with pytest.MonkeyPatch.context() as patch:
        for split, views, seed in (("train", 4, 0), ("val", 2, 1)):
            patch.setattr(
                sys,
                "argv",
                [
                    "capture",
                    "--views",
                    str(views),
                    "--size",
                    str(SIZE),
                    "--points",
                    "300",
                    "--seed",
                    str(seed),
                    "--out",
                    str(data / split),
                ],
            )
            capture.main()
        patch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--train",
                str(data / "train"),
                "--val",
                str(data / "val"),
                "--output",
                str(root / "runs"),
                "--name",
                "tiny",
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--base-channels",
                "8",
                "--num-workers",
                "0",
                "--precision",
                "32-true",
                "--panel-every",
                "1",
                "--panel-images",
                str(PANEL_IMAGES),
                "--log-every",
                "1",
                "--quiet",
            ],
        )
        train.main()
    return SimpleNamespace(
        root=root,
        dir=root / "runs" / "tiny" / "version_0",
        checkpoints=root / "checkpoints",
        val=data / "val",
    )


def _events(run_dir: Path) -> EventAccumulator:
    (path,) = run_dir.glob("events.out.tfevents.*")
    events = EventAccumulator(str(path), size_guidance={"scalars": 0, "images": 0})
    events.Reload()
    return events


def test_the_run_logs_exactly_the_documented_scalars(run):
    assert set(_events(run.dir).Tags()["scalars"]) == EXPECTED_SCALARS


def test_each_panel_has_one_row_per_view_and_four_columns(run):
    events = _events(run.dir)
    assert set(events.Tags()["images"]) == PANEL_TAGS
    pad = 2  # make_grid's default
    for tag in PANEL_TAGS:
        image = events.Images(tag)[-1]
        assert (image.width, image.height) == (
            4 * (SIZE + pad) + pad,
            PANEL_IMAGES * (SIZE + pad) + pad,
        )


def test_the_models_are_named_after_the_run_and_live_beside_runs(run):
    assert sorted(p.name for p in run.checkpoints.iterdir()) == [
        "tiny_version_0_best.ckpt",
        "tiny_version_0_last.ckpt",
    ]
    assert not list(run.dir.rglob("*.ckpt"))


def test_the_run_directory_holds_its_record(run):
    assert {"config.json", "summary.json", "hparams.yaml", "samples"} <= {
        p.name for p in run.dir.iterdir()
    }
    assert sorted(p.name for p in (run.dir / "samples").iterdir()) == [
        "epoch_0001.png",
        "epoch_0002.png",
    ]
    config = json.loads((run.dir / "config.json").read_text())
    assert config["name"] == "tiny" and config["epochs"] == 2
    summary = json.loads((run.dir / "summary.json").read_text())
    assert (summary["train_views"], summary["val_views"]) == (12, 6)
    assert summary["hyperparameters"]["base_channels"] == 8
    val_mae = [event.value for event in _events(run.dir).Scalars("val/mae")]
    assert summary["best_val_mae"] == pytest.approx(min(val_mae))


def test_the_schedule_steps_once_per_batch_over_the_whole_run(run):
    payload = torch.load(
        run.checkpoints / "tiny_version_0_last.ckpt", weights_only=False
    )
    (schedule,) = payload["lr_schedulers"]
    assert schedule["total_steps"] == 2 * 3  # two epochs, 12 views in batches of 4
    assert schedule["_step_count"] == 2 * 3 + 1
    (group,) = payload["optimizer_states"][0]["param_groups"]
    assert group["weight_decay"] == pytest.approx(1e-4)


def test_a_resume_with_a_different_epoch_count_is_refused(run):
    last = run.checkpoints / "tiny_version_0_last.ckpt"
    with pytest.raises(SystemExit, match="--epochs must match the resumed run"):
        train.check_resume(SimpleNamespace(resume=last, epochs=3))
    train.check_resume(SimpleNamespace(resume=last, epochs=2))


def _evaluate(run, output, *extra):
    eval_cli.main(
        [
            "--checkpoint",
            str(run.checkpoints / "tiny_version_0_best.ckpt"),
            "--val",
            str(run.val),
            "--output",
            str(output),
            *extra,
        ]
    )
    return json.loads((output / "metrics.json").read_text())


def test_eval_reproduces_the_number_best_ckpt_was_chosen_on(run):
    results = _evaluate(run, run.root / "eval")
    summary = json.loads((run.dir / "summary.json").read_text())
    assert results["all"]["model"]["mae"] == pytest.approx(
        summary["best_val_mae"], rel=1e-4
    )
    assert set(results) == {"cube", "sphere_ico", "sphere_uv", "all"}
    for scores in results.values():
        assert set(scores) == {"model", "constant", "nearest"}
        assert all(tuple(metrics) == METRICS for metrics in scores.values())
    assert (run.root / "eval" / "panel.png").is_file()


def test_eval_without_panels_writes_only_the_numbers(run):
    output = run.root / "eval_no_panels"
    _evaluate(run, output, "--num-panels", "0")
    assert sorted(p.name for p in output.iterdir()) == ["metrics.json"]
