"""Checks for the sparse-to-dense baseline: input assembly, losses, and the UNet."""

import torch

from baselines import constant_fill, nearest_fill
from dataset import assemble_input
from losses import (
    completion_loss,
    depth_l1,
    depth_metrics,
    gradient_loss,
    summarize_metrics,
)
from unet import UNet


def _plane(
    height: int = 32, width: int = 32, slope: float = 0.01, offset: float = 3.0
) -> torch.Tensor:
    u = torch.arange(width, dtype=torch.float32)
    return (offset + slope * u).expand(height, width).clone()


def test_assemble_input_centers_on_sparse_mean():
    depth = torch.zeros(8, 8)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    depth[1, 2], depth[5, 6] = 3.0, 5.0
    mask[1, 2] = mask[5, 6] = True
    inputs, ref = assemble_input(depth, mask)
    assert inputs.shape == (2, 8, 8) and ref.shape == (1, 1, 1)
    assert ref.item() == 4.0
    assert inputs[0, 1, 2] == -1.0 and inputs[0, 5, 6] == 1.0
    assert inputs[0][~mask].abs().max() == 0  # empty pixels stay exactly zero
    assert torch.equal(inputs[1], mask.float())
    # Batched call gives one reference per map.
    batch, refs = assemble_input(
        torch.stack([depth, 2 * depth]), torch.stack([mask, mask])
    )
    assert batch.shape == (2, 2, 8, 8) and refs.flatten().tolist() == [4.0, 8.0]


def test_losses_vanish_on_perfect_prediction_and_ignore_background():
    gt = _plane()[None, None]
    valid = torch.zeros_like(gt, dtype=torch.bool)
    valid[..., 8:24, 8:24] = True
    pred = gt.clone()
    pred[~valid] = 100.0  # garbage off the surface must not count
    loss, parts = completion_loss(pred, gt, valid)
    assert loss.item() < 1e-6 and parts["l1"] < 1e-6 and parts["grad"] < 1e-6


def test_constant_offset_costs_l1_but_not_gradient():
    gt = _plane()[None, None]
    valid = torch.ones_like(gt, dtype=torch.bool)
    pred = gt + 0.2
    assert abs(depth_l1(pred, gt, valid).item() - 0.2) < 1e-5
    assert gradient_loss(pred, gt, valid).item() < 1e-6


def test_gradient_loss_detects_wrong_slope_and_rounded_crease():
    valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    gt = _plane(slope=0.02)[None, None]
    assert gradient_loss(_plane(slope=0.0)[None, None], gt, valid) > 0.01
    # A crease (two planes meeting) versus a smoothed version of it.
    u = torch.arange(32, dtype=torch.float32)
    crease = (3.0 + 0.05 * (u - 16).abs()).expand(1, 1, 32, 32)
    smooth = (3.0 + 0.05 * ((u - 16).square() + 16).sqrt()).expand(1, 1, 32, 32)
    assert (
        gradient_loss(smooth, crease, valid)
        > gradient_loss(crease, crease, valid) + 1e-3
    )


def test_metrics_on_known_error():
    gt = torch.full((1, 1, 4, 4), 2.0)
    valid = torch.ones_like(gt, dtype=torch.bool)
    sparse = torch.zeros_like(gt)
    sparse[..., 0, 0] = 1.0
    pred = gt + 0.4
    pred[..., 0, 0] = 2.0  # the sparse pixel is exact
    metrics = summarize_metrics(depth_metrics(pred, gt, valid, sparse))
    assert abs(metrics["mae"] - 0.4 * 15 / 16) < 1e-6
    assert abs(metrics["hole_mae"] - 0.4) < 1e-6
    assert metrics["delta1"] == 1.0  # 2.4 / 2 < 1.25


def test_unet_handles_arbitrary_sizes():
    model = UNet(base=8).eval()
    for size in [(64, 64), (50, 70)]:
        assert model(torch.zeros(2, 2, *size)).shape == (2, 1, *size)


def test_unet_can_fit_one_sample():
    torch.manual_seed(0)
    gt = _plane(slope=0.02)[None, None]
    valid = torch.ones_like(gt, dtype=torch.bool)
    sparse_mask = torch.rand(32, 32) < 0.1
    inputs, ref = assemble_input(gt[0, 0] * sparse_mask, sparse_mask)
    model = UNet(base=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss, _ = completion_loss(ref[None] + model(inputs[None]), gt, valid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.2 * losses[0]


def test_baselines_fill_from_sparse_points():
    depth = torch.zeros(6, 6)
    mask = torch.zeros(6, 6, dtype=torch.bool)
    depth[0, 0], depth[5, 5] = 2.0, 4.0
    mask[0, 0] = mask[5, 5] = True
    inputs, ref = assemble_input(depth, mask)
    constant = constant_fill(inputs[None], ref[None])
    assert constant.shape == (1, 1, 6, 6) and (constant == 3.0).all()
    nearest = nearest_fill(inputs[None], ref[None])[0, 0]
    assert nearest[1, 0] == 2.0 and nearest[4, 5] == 4.0 and nearest[5, 5] == 4.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
