"""--train and --val name captures by file or by directory, and the split follows."""

import pytest
import torch

from datasets.sparse_depth import build_datasets, capture_files, evenly_spaced_batch

H = W = 8
KEEP = (0.3, 1.0)


def _capture(path, views: int):
    """The four tensors the dataset reads, for `views` views of an 8x8 image."""
    depth = torch.rand(views, H, W) + 1.0
    sparse_mask = torch.rand(views, H, W) < 0.3
    torch.save(
        {
            "depth": depth,
            "mask": torch.ones(views, H, W, dtype=torch.bool),
            "sparse_depth": depth * sparse_mask,
            "sparse_mask": sparse_mask,
        },
        path,
    )
    return path


def test_a_directory_means_every_capture_in_it(tmp_path):
    cube = _capture(tmp_path / "cube_gt.pt", 2)
    sphere = _capture(tmp_path / "sphere_uv_gt.pt", 2)
    (tmp_path / "cube_gt.png").touch()  # the preview the capture writes beside it
    assert capture_files([tmp_path]) == [cube, sphere]


def test_a_file_is_taken_as_given(tmp_path):
    cube = _capture(tmp_path / "cube_gt.pt", 2)
    assert capture_files([cube]) == [cube]


def test_a_missing_path_is_an_error_that_names_it(tmp_path):
    missing = tmp_path / "val" / "*_gt.pt"  # a shell glob that matched nothing
    with pytest.raises(FileNotFoundError, match=r"\*_gt\.pt does not exist"):
        capture_files([missing])


def test_a_directory_without_captures_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="holds no"):
        capture_files([tmp_path])


def test_without_val_every_nth_view_is_held_out(tmp_path):
    _capture(tmp_path / "cube_gt.pt", 12)
    train, val = build_datasets([tmp_path], None, 8, KEEP, 0.0)
    assert (len(train), len(val)) == (10, 2)  # views 0 and 8
    assert train.augment and not val.augment


def test_with_val_the_splits_come_from_different_captures(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    _capture(tmp_path / "train" / "cube_gt.pt", 6)
    _capture(tmp_path / "val" / "cube_gt.pt", 3)
    train, val = build_datasets([tmp_path / "train"], [tmp_path / "val"], 8, KEEP, 0.0)
    assert (len(train), len(val)) == (6, 3)


def test_a_panel_batch_spans_the_split(tmp_path):
    _capture(tmp_path / "cube_gt.pt", 10)
    _, val = build_datasets([tmp_path], [tmp_path], 8, KEEP, 0.0)
    batch = evenly_spaced_batch(val, 4)
    assert batch["input"].shape == (4, 2, H, W)
    torch.testing.assert_close(batch["target"][0], val[0]["target"])
    torch.testing.assert_close(batch["target"][-1], val[9]["target"])
