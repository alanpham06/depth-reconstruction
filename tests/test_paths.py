"""Where a tool's output lands, named after the run that produced it."""

from pathlib import Path

from utils.paths import default_output, run_name, source_label

CHECKPOINT = "checkpoints/unet_b32_version_0_best.ckpt"


def test_a_checkpoint_names_its_run_and_version():
    assert run_name(CHECKPOINT) == "unet_b32_version_0"


def test_no_source_is_the_run_on_its_own():
    assert default_output("export", CHECKPOINT) == Path(
        "output/export/unet_b32_version_0"
    )


def test_the_source_is_in_the_name():
    assert default_output("eval", CHECKPOINT, "val") == Path(
        "output/eval/unet_b32_version_0_val"
    )


def test_a_split_is_named_by_its_directory_however_it_was_given(tmp_path):
    val = tmp_path / "val"
    val.mkdir()
    (val / "cube_gt.pt").touch()
    assert source_label([val]) == source_label([val / "cube_gt.pt"]) == "val"
