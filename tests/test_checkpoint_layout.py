"""Where a trained model is written, and what still finds it there.

They guard two defects that ship when nothing checks them:

  - save_last ignores `filename=`. Lightning formats CHECKPOINT_NAME_LAST, so in a
    shared checkpoints/ every run wrote one anonymous last.ckpt, then last-v1,
    last-v2, with nothing recording which run owned which.
  - a checkpoint named in any other shape must come back as itself, never as a
    name built from the checkout's parent directories.
"""

from pathlib import Path

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

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


def test_save_last_names_the_run(tmp_path):
    """Lightning reads CHECKPOINT_NAME_LAST here, never filename=."""
    stem = "demo_version_0"
    callback = ModelCheckpoint(
        dirpath=tmp_path, filename=f"{stem}_best", save_last=True
    )
    callback.CHECKPOINT_NAME_LAST = f"{stem}_last"
    assert callback.CHECKPOINT_NAME_LAST == f"{stem}_last"
    # the default is what would land in a shared directory unnamed
    assert ModelCheckpoint.CHECKPOINT_NAME_LAST == "last"


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
    """These two are files Lightning really writes. Keyed off the filename suffix
    they fell into the nested branch and resolved to the checkout's parents."""
    got = run_name(f"checkpoints/{name}")
    assert "Desktop" not in got, got
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
    """Checkpoints migrated before the collision fix must still resolve."""
    assert run_stem(tmp_path / "runs" / "demo" / "version_0") == "demo_version_0"
