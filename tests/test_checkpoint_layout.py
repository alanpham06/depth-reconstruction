"""Where a trained model is written, and what still finds it there.

They guard a defect that ships when nothing checks it: a checkpoint named in any
other shape must come back as itself, never as a name built from the checkout's
parent directories.
"""

from pathlib import Path

import pytest

from utils.paths import CHECKPOINTS, checkpoint_dir, run_name, run_stem

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_checkpoints_live_beside_runs_not_inside_them(tmp_path):
    """runs/ is telemetry and gets cleared; models must not go with it."""
    assert (
        checkpoint_dir(tmp_path / "runs" / "demo" / "version_0")
        == tmp_path / CHECKPOINTS
    )


def test_checkpoint_dir_follows_the_run_not_the_working_directory(tmp_path):
    """A bare Path("checkpoints") followed the cwd, and a test given a tmp_path
    wrote into the real repo instead of its sandbox."""
    got = checkpoint_dir(tmp_path / "runs" / "demo" / "version_0")
    assert tmp_path in got.parents, f"{got} escaped the sandbox"
    assert REPO_ROOT not in got.parents


@pytest.mark.parametrize(
    "path,expected",
    [
        ("checkpoints/demo_version_0_best.ckpt", "demo_version_0"),
        ("checkpoints/demo_version_0_last.ckpt", "demo_version_0"),
    ],
)
def test_a_checkpoint_names_its_run(path, expected):
    assert run_name(path) == expected


@pytest.mark.parametrize("name", ["last.ckpt", "demo_version_0_best-v1.ckpt"])
def test_an_unparsable_checkpoint_name_does_not_become_a_directory_name(name):
    """Lightning really writes both of these; neither ends in _best or _last, so
    each must come back as its own stem."""
    got = run_name(f"checkpoints/{name}")
    assert got in {"last", "demo_version_0_best-v1"}, got


def test_two_runs_of_the_same_name_under_different_parents_do_not_collide(tmp_path):
    """--output takes a directory runs live in, so these are different runs. Keyed
    on the last path component alone they produced one filename and the second
    silently overwrote the first."""
    a = tmp_path / "runs" / "ablation_a" / "myrun" / "version_0"
    b = tmp_path / "runs" / "ablation_b" / "myrun" / "version_0"
    assert run_stem(a) != run_stem(b), f"both runs write {run_stem(a)}"
    assert checkpoint_dir(a) == checkpoint_dir(b) == tmp_path / CHECKPOINTS


def test_a_nested_runs_directory_still_puts_models_outside_runs(tmp_path):
    """Taking the nearest `runs` ancestor instead of the outermost put them in
    runs/a/checkpoints -- inside the directory this exists to make safe to clear."""
    got = checkpoint_dir(tmp_path / "runs" / "a" / "runs" / "b" / "version_0")
    assert "runs" not in got.relative_to(tmp_path).parts, got


def test_the_ordinary_run_shape_keeps_the_name_it_already_had(tmp_path):
    """For the ordinary runs/<name>/version_N shape, the stem is <name>_version_N."""
    assert run_stem(tmp_path / "runs" / "demo" / "version_0") == "demo_version_0"


def test_last_ckpt_is_written_after_every_epoch(tmp_path, monkeypatch):
    """save_last only writes last.ckpt when training ends -- a run that dies
    mid-run would have nothing current on disk."""
    import sys

    import lightning as L
    import torch
    from torch.utils.data import DataLoader

    import render.capture as capture
    import train
    from datasets.sparse_depth import build_datasets
    from models.reconstruction_module import SparseDepthModule

    data = tmp_path / "data"
    for split, views, seed in (("train", 4, 0), ("val", 2, 1)):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "capture",
                "--views",
                str(views),
                "--size",
                "32",
                "--points",
                "300",
                "--seed",
                str(seed),
                "--out",
                str(data / split),
            ],
        )
        capture.main()
    train_set, val_set = build_datasets(
        [data / "train"], [data / "val"], 8, (0.3, 1.0), 0.0
    )

    run = tmp_path / "runs" / "demo" / "version_0"
    stem, directory = run_stem(run), checkpoint_dir(run)
    directory.mkdir(parents=True)
    last_path = directory / f"{stem}_last.ckpt"
    checkpoints = train.checkpoint_callbacks(run)
    best = checkpoints[0]  # best before last; see checkpoint_callbacks

    seen = []

    class _RecordLastEpoch(L.Callback):
        def on_train_epoch_start(self, trainer, module):
            if not last_path.is_file():
                seen.append(None)
                return
            payload = torch.load(last_path, map_location="cpu", weights_only=False)
            seen.append(payload["epoch"])
            # last.ckpt carries every callback's state, including best's -- if
            # last's hook ran before best's for the epoch just finished, this
            # would still read the previous epoch's score
            saved = payload["callbacks"][best.state_key]["best_model_score"]
            assert saved == pytest.approx(float(best.best_model_score)), (
                "last.ckpt's copy of best_model_score is stale"
            )

    trainer = L.Trainer(
        max_epochs=4,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=[*checkpoints, _RecordLastEpoch()],
    )
    trainer.fit(
        SparseDepthModule(base_channels=8, epochs=4),
        DataLoader(train_set, batch_size=4),
        DataLoader(val_set, batch_size=4),
    )

    assert seen == [None, 0, 1, 2]
    assert sorted(p.name for p in directory.iterdir()) == [
        f"{stem}_best.ckpt",
        f"{stem}_last.ckpt",
    ]
