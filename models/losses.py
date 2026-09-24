"""Depth completion losses.

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


def _gradient_l1(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    total, count = pred.new_zeros(()), pred.new_zeros(())
    for dim in (-1, -2):
        n = pred.shape[dim]
        first, rest = (
            (lambda x: x.narrow(dim, 0, n - 1)),
            (lambda x: x.narrow(dim, 1, n - 1)),
        )
        both = (first(valid) & rest(valid)).to(pred.dtype)
        grad_pred = rest(pred) - first(pred)
        grad_gt = rest(gt) - first(gt)
        total = total + ((grad_pred - grad_gt).abs() * both).sum()
        count = count + both.sum()
    return total / count.clamp(min=1.0)


def gradient_loss(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, scales: int = 4
) -> torch.Tensor:
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
        block_valid = (
            -F.max_pool2d(-valid.to(pred.dtype), step, ceil_mode=True) > 0.5
            if step > 1
            else valid.bool()
        )
        losses.append(
            _gradient_l1(
                pred[..., ::step, ::step], gt[..., ::step, ::step], block_valid
            )
        )
    return torch.stack(losses).mean()


def completion_loss(
    pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, grad_weight: float = 0.5
) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = depth_l1(pred, gt, valid)
    grad = gradient_loss(pred, gt, valid)
    total = l1 + grad_weight * grad
    return total, {"loss": total.item(), "l1": l1.item(), "grad": grad.item()}
