"""The sample panel shows what save_preview showed, on the same colour scales."""

import torch
from matplotlib import colormaps

from utils.panel import PANEL_COLUMNS, colorize, depth_panel

H = W = 8


def _view(valid):
    depth = torch.linspace(2.0, 3.0, H * W).reshape(1, 1, H, W)
    points = torch.zeros(1, 1, H, W, dtype=torch.bool)
    points[..., ::3, ::3] = True
    ref = torch.full((1, 1, 1, 1), 2.5)
    batch = {
        "input": torch.cat([(depth - ref) * points, points.float()], dim=1),
        "target": depth,
        "valid": valid,
        "ref": ref,
    }
    return batch, depth


ALL = torch.ones(1, 1, H, W, dtype=torch.bool)


def test_four_tiles_per_view_in_column_order():
    batch, depth = _view(ALL)
    tiles = depth_panel(batch, depth + 0.1)
    assert PANEL_COLUMNS == ("sparse", "pred", "gt", "error")
    assert tiles.shape == (4, 3, H, W)
    # ground truth: viridis over the view's own range, as save_preview drew it
    expected = colorize(depth[0, 0], ALL[0, 0], 2.0, 3.0, "viridis")
    torch.testing.assert_close(tiles[2], expected)


def test_a_pixel_a_column_does_not_cover_is_black():
    batch, depth = _view(ALL)
    sparse = depth_panel(batch, depth)[0]
    no_point = batch["input"][0, 1] == 0
    assert (sparse[:, no_point] == 0).all()
    assert (sparse[:, ~no_point].sum(dim=0) > 0).all()


def test_the_error_scale_tops_out_at_five_percent_of_the_range_plus_one():
    batch, depth = _view(ALL)
    vmax = 0.05 * (3.0 - 2.0 + 1.0)  # save_preview's vmax
    error = depth_panel(batch, depth + vmax)[3]
    top = torch.tensor(colormaps["magma"](1.0)[:3], dtype=torch.float32)
    torch.testing.assert_close(error[:, 0, 0], top)


def test_a_view_the_object_misses_still_draws():
    """No valid pixel means no range to scale by; that must not end the run."""
    batch, depth = _view(torch.zeros(1, 1, H, W, dtype=torch.bool))
    tiles = depth_panel(batch, depth)
    assert (tiles[1:] == 0).all()  # prediction, target and error cover valid only
