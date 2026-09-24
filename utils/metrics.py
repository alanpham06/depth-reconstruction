"""Depth completion metrics, pooled over a whole split.

depth_metrics returns per-batch sums rather than means, so a split's metric is its
total error over its total pixel count: every valid pixel weighs the same, whichever
batch it arrived in.

  mae, rmse   depth error over the pixels that hit the surface
  abs_rel     |pred - gt| / gt over the same pixels
  delta1      the share of them with max(pred / gt, gt / pred) < 1.25
  hole_mae    mae over valid pixels that had no sparse point -- the ones the
              network actually had to fill in
"""

import torch

# The names summarize_metrics returns, in its order: the val/ tags and eval's columns
METRICS = ("mae", "rmse", "abs_rel", "delta1", "hole_mae")


@torch.no_grad()
def depth_metrics(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, sparse_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-batch sums (not means) so they can be accumulated over a whole split.

    `hole_*` counts only valid pixels that had no sparse input, i.e. the pixels
    the network actually had to fill in.
    """
    valid = valid.bool()
    holes = valid & ~sparse_mask.bool()
    err = (pred - gt).abs()
    gt_safe = gt.clamp(min=1e-6)
    ratio = torch.maximum(
        pred.clamp(min=1e-6) / gt_safe, gt_safe / pred.clamp(min=1e-6)
    )
    return {
        "n": valid.sum().double(),
        "abs": err[valid].double().sum(),
        "sq": err[valid].double().square().sum(),
        "rel": (err / gt_safe)[valid].double().sum(),
        "delta1": (ratio[valid] < 1.25).double().sum(),
        "hole_n": holes.sum().double(),
        "hole_abs": err[holes].double().sum(),
    }


def summarize_metrics(sums: dict[str, torch.Tensor]) -> dict[str, float]:
    n, hole_n = sums["n"].clamp(min=1), sums["hole_n"].clamp(min=1)
    return {
        "mae": (sums["abs"] / n).item(),
        "rmse": (sums["sq"] / n).sqrt().item(),
        "abs_rel": (sums["rel"] / n).item(),
        "delta1": (sums["delta1"] / n).item(),
        "hole_mae": (sums["hole_abs"] / hole_n).item(),
    }


class PooledMetrics:
    """depth_metrics sums for several predictors over one split, divided once at the end."""

    def __init__(self) -> None:
        self.sums: dict[str, dict[str, torch.Tensor]] = {}

    def add(self, name: str, pred, gt, valid, sparse_mask) -> None:
        sums = self.sums.setdefault(name, {})
        for key, value in depth_metrics(pred, gt, valid, sparse_mask).items():
            sums[key] = sums.get(key, 0) + value

    def summarize(self, reduce=None) -> dict[str, dict[str, float]]:
        """Metrics per predictor. `reduce`, if given, sums each total across processes."""
        results = {}
        for name, sums in self.sums.items():
            if reduce is not None:
                sums = {key: reduce(value) for key, value in sums.items()}
            results[name] = summarize_metrics(sums)
        return results
