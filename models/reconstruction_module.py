"""reconstruction_module.py — the LightningModule and the callbacks behind a run.

The model, the loss and the schedule in one place, so train.py is argument parsing
and a Trainer rather than a hand-written epoch loop. The recipe is main's: AdamW,
and a OneCycle schedule stepped every batch over the whole run.

What is logged
--------------
    batch/*                loss, l1 and grad per step, and the gradient norm
    train/*, val/*         the same three terms as epoch curves
    val/<metric>           mae, rmse, abs_rel, delta1 and hole_mae, pooled over
                           every valid pixel of the split
    val/<metric>_nearest   the same five for nearest_fill and for constant_fill,
    val/<metric>_constant  which score the same views with no network at all
    lr/recon               the OneCycle rate at the start of each epoch

**Read val/mae against val/mae_nearest, never on its own.** Copying each pixel's
nearest sparse point already fills a smooth shape well; the gap between the two
lines is what the network is worth.

**The val metrics are pooled, not averaged per batch.** validation_step adds each
batch's sums, and on_validation_epoch_end divides once, exactly as evaluate() did
on main. Lightning's own epoch mean would weight a batch of small silhouettes the
same as a batch of large ones. val/loss, val/l1 and val/grad are Lightning's
batch-size-weighted means; they are not what picks best.ckpt.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.utils import make_grid, save_image

from models import make_model, predict
from models.baselines import BASELINES
from models.losses import completion_loss
from utils.metrics import PooledMetrics
from utils.panel import PANEL_COLUMNS, depth_panel

LOSS_TERMS = ("loss", "l1", "grad")


def panel_tag(columns: tuple[str, ...]) -> str:
    return "_".join(columns)


class SparseDepthModule(L.LightningModule):
    def __init__(
        self,
        base_channels: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_weight: float = 0.5,
        epochs: int = 100,
        panel_images: int = 6,
    ):
        super().__init__()
        # epochs is recorded so a resume can refuse a different one: the OneCycle
        # schedule is built from the run's length
        self.save_hyperparameters()
        self.model = make_model(base_channels)
        self.pooled = PooledMetrics()

    def forward(self, inputs: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return predict(self.model, inputs, ref)

    def _loss(self, batch):
        pred = self(batch["input"], batch["ref"])
        loss, parts = completion_loss(
            pred, batch["target"], batch["valid"], self.hparams.grad_weight
        )
        return pred, loss, parts

    def training_step(self, batch, _):
        _, loss, parts = self._loss(batch)
        size = batch["input"].shape[0]
        # train/ and batch/ under separate prefixes, so Lightning never splits a
        # tag into a _step / _epoch pair
        for term in LOSS_TERMS:
            self.log(
                f"train/{term}",
                parts[term],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=size,
                prog_bar=term == "loss",
            )
            self.log(f"batch/{term}", parts[term], on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.pooled = PooledMetrics()

    def validation_step(self, batch, _):
        pred, _, parts = self._loss(batch)
        size = batch["input"].shape[0]
        for term in LOSS_TERMS:
            self.log(
                f"val/{term}",
                parts[term],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=size,
            )
        sparse = batch["input"][:, 1:2]
        self.pooled.add("model", pred, batch["target"], batch["valid"], sparse)
        for name, fill in BASELINES.items():
            fill_pred = fill(batch["input"], batch["ref"])
            self.pooled.add(name, fill_pred, batch["target"], batch["valid"], sparse)

    def on_validation_epoch_end(self) -> None:
        def total(value):
            return self.trainer.strategy.reduce(value, reduce_op="sum")

        for name, metrics in self.pooled.summarize(reduce=total).items():
            suffix = "" if name == "model" else f"_{name}"
            for metric, value in metrics.items():
                # Already summed across processes; sync_dist only averages the
                # identical copies, which keeps Lightning from warning
                self.log(
                    f"val/{metric}{suffix}",
                    value,
                    sync_dist=True,
                    prog_bar=not suffix and metric == "mae",
                )

    def on_before_optimizer_step(self, optimizer):
        # A prediction that has started to diverge shows up here an epoch before
        # it shows up as a flat loss curve
        total = sum(
            float(p.grad.detach().norm(2)) ** 2
            for p in self.model.parameters()
            if p.grad is not None
        )
        self.log("batch/grad_norm", total**0.5, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # Over every step of the run, as on main: epochs times the batches an
        # epoch actually trains, which --limit-train-batches caps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_epoch_start(self) -> None:
        # Logged here rather than by LearningRateMonitor, which would name the tag
        # after the optimizer class rather than what it schedules
        self.log(
            "lr/recon",
            self.optimizers().param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
        )

    @torch.no_grad()
    def panel(self, batch):
        """The first panel_images views of `batch` as panel tiles, with the layout."""
        n = self.hparams.panel_images
        batch = {key: value[:n].to(self.device) for key, value in batch.items()}
        pred = self(batch["input"], batch["ref"])
        return depth_panel(batch, pred), len(PANEL_COLUMNS), PANEL_COLUMNS


class RunScopedCheckpoint(ModelCheckpoint):
    """A ModelCheckpoint that only ever deletes its own run's files.

    Every run writes into one shared checkpoints/ now, and that quietly disarmed
    both of the guards Lightning relies on to not delete somebody else's model:

      - `load_state_dict` reloads `best_k_models` only when its dirpath equals the
        one recorded in the checkpoint. Per-version dirpaths never matched, so a
        resume began with empty bookkeeping. One shared directory always matches,
        so a resumed run now starts out holding the *previous* version's best path.
      - `_should_remove_checkpoint` then permits a delete anywhere under dirpath,
        which used to be this run's own directory and is now every run's.

    With both gone, resuming version_0 into version_1 deleted
    checkpoints/<run>_version_0_best.ckpt on the first improvement -- exactly the
    loss that moving the models out of runs/ was meant to make impossible.

    The rule this restores is narrow and is only ever a refusal: a file whose name
    does not begin with this run's stem is not this run's to remove. Lightning's
    own refusals still apply on top.
    """

    def __init__(self, run_stem: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._run_stem = run_stem

    def _should_remove_checkpoint(self, trainer, previous: str, current: str) -> bool:
        if not Path(previous).name.startswith(self._run_stem):
            return False
        return super()._should_remove_checkpoint(trainer, previous, current)


class PanelWriter(L.Callback):
    """Writes the sample panels to TensorBoard, and the val one to samples/.

    Every `every` epochs and after the last, main's preview cadence, for both
    splits. The train panel is the views being fitted, so a gap opening between the
    two panels is the first sign of overfitting. Both use views fixed at startup, so
    what changes between panels is the model rather than the sample.
    """

    def __init__(self, batches: dict, every: int, samples_dir: Path | None = None):
        self.batches, self.every, self.samples_dir = batches, every, samples_dir

    def _write(self, trainer, module, stage: str) -> None:
        epoch = trainer.current_epoch + 1
        due = epoch % self.every == 0 or epoch == trainer.max_epochs
        batch = self.batches.get(stage)
        if batch is None or not due or not trainer.is_global_zero:
            return
        was_training = module.training
        module.eval()
        grid, ncols, columns = module.panel(batch)
        if was_training:
            module.train()
        if trainer.logger is not None:
            trainer.logger.experiment.add_image(
                f"{stage}/{panel_tag(columns)}", make_grid(grid, nrow=ncols), epoch
            )
        if stage == "val" and self.samples_dir is not None:
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            save_image(grid, self.samples_dir / f"epoch_{epoch:04d}.png", nrow=ncols)

    def on_train_epoch_end(self, trainer, module):
        self._write(trainer, module, "train")

    def on_validation_epoch_end(self, trainer, module):
        if not trainer.sanity_checking:
            self._write(trainer, module, "val")
