"""Depth completion losses and evaluation metrics.

All terms are restricted to `valid` pixels (rays that hit the surface): the
background has no depth, so it is neither supervised nor scored.

  depth_l1       mean |pred - gt| over valid pixels
  gradient_loss  multi-scale L1 between the finite-difference gradients of
                 pred and gt. A difference is only used when both of its pixels
                 are valid, so the jump from the object to the empty background
                 is never supervised; creases inside the object (cube edges)
                 are, which is what keeps them sharp instead of rounded off.
"""

import torch
import torch.nn.functional as F


def depth_l1(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
	valid = valid.to(pred.dtype)
	return ((pred - gt).abs() * valid).sum() / valid.sum().clamp(min=1.0)


def _gradient_l1(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
	total, count = pred.new_zeros(()), pred.new_zeros(())
	for dim in (-1, -2):
		n = pred.shape[dim]
		first, rest = (lambda x: x.narrow(dim, 0, n - 1)), (lambda x: x.narrow(dim, 1, n - 1))
		both = (first(valid) & rest(valid)).to(pred.dtype)
		grad_pred = rest(pred) - first(pred)
		grad_gt = rest(gt) - first(gt)
		total = total + ((grad_pred - grad_gt).abs() * both).sum()
		count = count + both.sum()
	return total / count.clamp(min=1.0)


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, scales: int = 4) -> torch.Tensor:
	"""Gradient matching averaged over `scales` resolutions (full, 1/2, 1/4, ...).

	Coarser scales compare depth differences across wider pixel gaps, which
	constrains the overall surface slope and not just local texture. At each
	scale pred and gt are subsampled (not blurred) and a coarse pixel is valid
	only if its whole block was.
	"""
	losses = []
	for scale in range(scales):
		step = 2**scale
		if min(pred.shape[-2:]) < 2 * step:
			break
		block_valid = -F.max_pool2d(-valid.to(pred.dtype), step, ceil_mode=True) > 0.5 if step > 1 else valid.bool()
		losses.append(_gradient_l1(pred[..., ::step, ::step], gt[..., ::step, ::step], block_valid))
	return torch.stack(losses).mean()


def completion_loss(
	pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, grad_weight: float = 0.5
) -> tuple[torch.Tensor, dict[str, float]]:
	l1 = depth_l1(pred, gt, valid)
	grad = gradient_loss(pred, gt, valid)
	total = l1 + grad_weight * grad
	return total, {"loss": total.item(), "l1": l1.item(), "grad": grad.item()}


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
	ratio = torch.maximum(pred.clamp(min=1e-6) / gt_safe, gt_safe / pred.clamp(min=1e-6))
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
