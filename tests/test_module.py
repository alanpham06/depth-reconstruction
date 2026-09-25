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


def test_the_baseline_fills_run_only_through_the_first_validation(
    tmp_path, monkeypatch
):
    """constant_fill and nearest_fill use no network, so their numbers never
    change; the sanity check and the first validation must be the only
    validations that run them, and every later one just reads the numbers back.
    """
    import sys

    import lightning as L
    from torch.utils.data import DataLoader

    import models.reconstruction_module as reconstruction_module
    import render.capture as capture
    from datasets.sparse_depth import build_datasets
    from models.baselines import BASELINES

    data = tmp_path / "data"
    for split, views, seed in (("train", 4, 0), ("val", 2, 1)):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "capture",
                "--views",
                str(views),
                "--size",
                "32",
                "--points",
                "300",
                "--seed",
                str(seed),
                "--out",
                str(data / split),
            ],
        )
        capture.main()
    train_set, val_set = build_datasets(
        [data / "train"], [data / "val"], 8, (0.3, 1.0), 0.0
    )

    module = SparseDepthModule(base_channels=8, epochs=3)
    rounds = []

    def _counted(fill):
        def wrapped(inputs, ref):
            sanity = module.trainer.sanity_checking
            rounds.append("sanity" if sanity else module.trainer.current_epoch)
            return fill(inputs, ref)

        return wrapped

    monkeypatch.setattr(
        reconstruction_module,
        "BASELINES",
        {name: _counted(fill) for name, fill in BASELINES.items()},
    )

    first_nearest = {}

    class _RecordFirstNearest(L.Callback):
        # on_validation_end, not on_validation_epoch_end: callbacks see that one
        # before the module has logged its pooled metrics for the epoch
        def on_validation_end(self, trainer, _):
            if not trainer.sanity_checking and "mae" not in first_nearest:
                first_nearest["mae"] = trainer.callback_metrics["val/mae_nearest"]

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[_RecordFirstNearest()],
    )
    trainer.fit(
        module,
        DataLoader(train_set, batch_size=4),
        DataLoader(val_set, batch_size=4),
    )

    assert set(rounds) == {"sanity", 0}, rounds
    assert trainer.callback_metrics["val/mae_nearest"] == first_nearest["mae"]
