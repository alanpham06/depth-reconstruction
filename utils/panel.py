"""The sample panel: sparse input | prediction | ground truth | |error|, per view.

The colour scales are the ones save_preview used on main. Depth is viridis over the
view's own ground-truth range, so the three depth columns read against each other,
and error is magma up to 5% of that range plus one. A pixel a column does not
cover -- no sparse point there, or a ray that missed -- is black.

Here rather than in the LightningModule because eval.py draws the same panel from
a bare network, without Lightning.
"""

from __future__ import annotations

import torch
from matplotlib import colormaps

# depth_panel's four tiles per view, in the order it appends them
PANEL_COLUMNS = ("sparse", "pred", "gt", "error")


def colorize(values, shown, low: float, high: float, cmap: str) -> torch.Tensor:
    """(H, W) -> (3, H, W) in [0, 1]: `cmap` over [low, high], black where not shown."""
    scaled = ((values.float() - low) / max(high - low, 1e-12)).clamp(0.0, 1.0)
    rgb = colormaps[cmap](scaled.cpu().numpy())[..., :3]
    return torch.from_numpy(rgb).permute(2, 0, 1).float() * shown.cpu().float()


@torch.no_grad()
def depth_panel(batch: dict[str, torch.Tensor], pred: torch.Tensor) -> torch.Tensor:
    """(4N, 3, H, W): the four tiles of each view in turn, for make_grid(nrow=4)."""
    tiles = []
    for i in range(pred.shape[0]):
        gt, valid = batch["target"][i, 0], batch["valid"][i, 0].bool()
        points = batch["input"][i, 1] > 0
        sparse = batch["input"][i, 0] + batch["ref"][i].reshape(())
        low, high = (
            (gt[valid].min().item(), gt[valid].max().item())
            if valid.any()
            else (0.0, 1.0)
        )
        tiles += [
            colorize(sparse, points, low, high, "viridis"),
            colorize(pred[i, 0], valid, low, high, "viridis"),
            colorize(gt, valid, low, high, "viridis"),
            colorize(
                (pred[i, 0] - gt).abs(), valid, 0.0, 0.05 * (high - low + 1.0), "magma"
            ),
        ]
    return torch.stack(tiles)
