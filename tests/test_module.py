"""The LightningModule predicts metric depth and records what built it."""

import torch

from models import predict
from models.reconstruction_module import SparseDepthModule, panel_tag
from utils.panel import PANEL_COLUMNS


def test_forward_is_the_metric_depth_prediction():
    torch.manual_seed(0)
    module = SparseDepthModule(base_channels=8).eval()
    inputs, ref = torch.randn(2, 2, 32, 32), torch.full((2, 1, 1, 1), 4.5)
    with torch.no_grad():
        torch.testing.assert_close(
            module(inputs, ref), predict(module.model, inputs, ref), rtol=0, atol=0
        )


def test_the_hyperparameters_are_recorded():
    module = SparseDepthModule(
        base_channels=8, lr=2e-3, weight_decay=1e-5, grad_weight=0.25, epochs=7
    )
    recorded = {
        k: module.hparams[k]
        for k in ("base_channels", "lr", "weight_decay", "grad_weight", "epochs")
    }
    assert recorded == {
        "base_channels": 8,
        "lr": 2e-3,
        "weight_decay": 1e-5,
        "grad_weight": 0.25,
        "epochs": 7,
    }


def test_the_panel_draws_the_first_views_of_its_batch():
    module = SparseDepthModule(base_channels=8, panel_images=2).eval()
    mask = (torch.rand(3, 32, 32) < 0.1).float()
    batch = {
        "input": torch.stack([torch.randn(3, 32, 32) * mask, mask], dim=1),
        "target": torch.rand(3, 1, 32, 32) + 2.0,
        "valid": torch.ones(3, 1, 32, 32, dtype=torch.bool),
        "ref": torch.full((3, 1, 1, 1), 2.5),
    }
    grid, ncols, columns = module.panel(batch)
    assert grid.shape == (2 * 4, 3, 32, 32)
    assert (ncols, columns) == (4, PANEL_COLUMNS)
    assert panel_tag(columns) == "sparse_pred_gt_error"
