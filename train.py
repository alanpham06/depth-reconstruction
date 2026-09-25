"""train.py — sparse depth in, dense depth out, on the procedural shapes.

The network sees (sparse depth, valid-point mask) and predicts dense depth relative
to the mean sparse depth; the loss is L1 depth plus a multi-scale gradient-matching
term, both only on pixels that hit the surface.

The loop is Lightning's; models/reconstruction_module.py holds the model, the loss
and the schedule. A run is a config in configs/ plus a run name, and any flag beats
the config.

A run writes runs/<name>/version_N/, and its models to
checkpoints/<name>_version_N_{best,last}.ckpt. Re-running a name gets version_1
rather than overwriting version_0, so a repeat is a new directory and never a lost
result.

Recipe
------
  python -m render.capture --views 256 --seed 0 --out data/train
  python -m render.capture --views 24 --seed 1 --out data/val
  python train.py --config unet_b32 --train data/train --val data/val --name baseline

Different seeds and view counts give val its own camera poses and its own sparse
samples. Without --val, every --val-every'th view of the training captures is held
out instead.

--resume continues from a last.ckpt with the same --epochs, e.g. after a preempted
job: the OneCycle schedule is built from the run's length.

Smoke (two epochs, tiny):
  python train.py --train data/train --val data/val --epochs 2 --base-channels 8 \\
      --limit-train-batches 4 --name smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from arguments.config import apply_config, given_options, load_config
from datasets.sparse_depth import build_datasets, evenly_spaced_batch
from models import count_parameters
from models.reconstruction_module import (
    PanelWriter,
    RunScopedCheckpoint,
    SparseDepthModule,
)
from utils.checkpoint import load_payload
from utils.paths import RESOLVED_CONFIG, SAMPLES, SUMMARY, checkpoint_dir, run_stem

PRECISIONS = ("bf16-mixed", "16-mixed", "32-true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--train",
        type=Path,
        nargs="+",
        required=True,
        help="*_gt.pt files, or directories of them, to train on",
    )
    p.add_argument(
        "--val",
        type=Path,
        nargs="*",
        default=None,
        help="held-out *_gt.pt files, or directories of them",
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=8,
        help="without --val: hold out every n-th view of the training captures",
    )
    p.add_argument(
        "--config",
        default=None,
        help="a name in configs/ or a yaml path; a flag given here beats it",
    )
    p.add_argument(
        "--output",
        default="runs",
        help="where runs live; a run is <output>/<name>/version_N",
    )
    p.add_argument(
        "--name",
        default=None,
        help="run name under --output; defaults to the config name. Re-running a "
        "name gets version_1 rather than overwriting version_0",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--grad-weight", type=float, default=0.5, help="weight of the gradient loss"
    )
    p.add_argument(
        "--base-channels", type=int, default=32, help="UNet width at full resolution"
    )
    p.add_argument(
        "--keep-range",
        type=float,
        nargs=2,
        default=(0.3, 1.0),
        metavar=("LOW", "HIGH"),
        help="each training view keeps a fraction of its sparse points from here",
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="sparse depth noise during training, in scene units",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--precision",
        default="bf16-mixed",
        choices=PRECISIONS,
        help="Lightning precision; 32-true without a CUDA device",
    )
    p.add_argument(
        "--panel-every", type=int, default=10, help="epochs between sample panels"
    )
    p.add_argument(
        "--panel-images", type=int, default=6, help="rows in the sample panels"
    )
    p.add_argument(
        "--log-every", type=int, default=50, help="steps between batch/ writes"
    )
    p.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="stop each epoch after n batches, for smoke tests; the schedule follows",
    )
    p.add_argument(
        "--resume", type=Path, default=None, help="a last.ckpt to continue from"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="no progress bar")
    return p


def check_panel_settings(args) -> None:
    """A panel needs at least one row, and at least one epoch between panels."""
    for flag, value in (
        ("--panel-images", args.panel_images),
        ("--panel-every", args.panel_every),
    ):
        if value < 1:
            raise SystemExit(f"{flag} must be at least 1, got {value}")


def parse_args():
    """The command line, with a named config filling in what it did not set"""
    parser = build_parser()
    args = parser.parse_args()
    given = given_options(build_parser)
    if args.config:
        args.applied_config = apply_config(
            args, load_config(args.config), given, parser
        )
    else:
        args.applied_config = []
    if args.name is None:
        args.name = args.config or "run"
    check_panel_settings(args)
    return args


def resolve_precision(precision: str, cuda: bool) -> str:
    """Mixed precision wants a CUDA device; without one, full precision, and say so.

    --amp did the same on main: autocast on the CPU is slower than float32 rather
    than faster.
    """
    if cuda or precision == "32-true":
        return precision
    print(f"No CUDA device: --precision {precision} -> 32-true", flush=True)
    return "32-true"


def check_resume(args) -> None:
    """--epochs must match the run being resumed: the OneCycle schedule is built from it."""
    if args.resume is None:
        return
    recorded = load_payload(args.resume).get("hyper_parameters", {}).get("epochs")
    if recorded is not None and recorded != args.epochs:
        raise SystemExit(
            f"--epochs must match the resumed run ({recorded}): "
            "the LR schedule depends on it"
        )


def check_output_inside_runs(run: Path) -> None:
    """--output must put a run inside a directory named runs.

    checkpoint_dir falls back to writing beside the run itself when there is no
    runs ancestor, which would silently break the checkpoints-beside-runs layout.
    """
    if "runs" not in run.resolve().parts:
        raise SystemExit(
            f"{run} has no runs/ ancestor: checkpoints/ sits beside runs/, not "
            "inside it -- point --output at (or inside) a directory named runs"
        )


def check_no_existing_checkpoints(run: Path) -> None:
    """Refuses a --name whose checkpoints already exist.

    Hit after runs/ was cleared and Lightning's version numbering restarted from
    version_0; a resume always logs a new version, so a new stem, and is
    unaffected.
    """
    stem = run_stem(run)
    existing = sorted(checkpoint_dir(run).glob(f"{stem}_*.ckpt"))
    if existing:
        raise SystemExit(
            f"{existing[0]} already exists: delete it or choose another --name"
        )


def resolve_log_every(log_every: int, train_batches: int, limit: int | None) -> int:
    """batch/ writes are capped by the batches an epoch actually runs.

    A few hundred views is a shorter epoch than the default logging interval, and
    a smoke run's --limit-train-batches shortens it further; either way Lightning
    must write the batch/ curves at least once an epoch rather than never.
    """
    batches = min(train_batches, limit or train_batches)
    return min(log_every, max(1, batches))


def checkpoint_callbacks(run: Path) -> list[RunScopedCheckpoint]:
    """best and last, in the order that keeps last.ckpt's copy of best current.

    Lightning runs same-hook callbacks in list order, and a checkpoint captures
    every callback's state -- so best must process each epoch's validation
    before last saves, or every last.ckpt carries the previous epoch's best
    score. Best first does that.
    """
    stem = run_stem(run)
    directory = checkpoint_dir(run)
    best = RunScopedCheckpoint(
        stem,
        dirpath=directory,
        filename=f"{stem}_best",
        monitor="val/mae",
        mode="min",
    )
    # last.ckpt after every epoch, so a preempted job resumes where it stopped: no
    # monitor and save_top_k=1 keep the one newest checkpoint under this name.
    # save_last would write it only when training ends.
    last = RunScopedCheckpoint(
        stem,
        dirpath=directory,
        filename=f"{stem}_last",
        save_top_k=1,
        every_n_epochs=1,
    )
    return [best, last]


def main():
    args = parse_args()
    check_resume(args)
    args.precision = resolve_precision(args.precision, torch.cuda.is_available())
    L.seed_everything(args.seed, workers=True)
    # Tensor cores, on a matmul that does not need the last bits of precision.
    # nearest_fill does, and computes its distances without a matmul for that reason
    torch.set_float32_matmul_precision("high")

    print("=== Training run ===", flush=True)
    print("Run:", f"{args.output}/{args.name}", flush=True)
    if args.config:
        filled = ", ".join(args.applied_config) or "nothing the command line left unset"
        print(f"Config {args.config}: filled {filled}", flush=True)

    train_set, val_set = build_datasets(
        args.train, args.val, args.val_every, args.keep_range, args.noise_std
    )
    print(f"Views: {len(train_set)} train, {len(val_set)} val", flush=True)

    loader = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        # A partial last batch only when it is the only batch, as on main
        drop_last=len(train_set) > args.batch_size,
        **loader,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, **loader)

    module = SparseDepthModule(
        base_channels=args.base_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_weight=args.grad_weight,
        epochs=args.epochs,
        panel_images=args.panel_images,
    )
    print(
        f"Model: params={count_parameters(module.model):,} "
        f"(base_channels={args.base_channels})",
        flush=True,
    )
    print(f"Loss: l1 + {args.grad_weight} * grad", flush=True)

    # default_hp_metric would add an empty hp_metric tag to every run
    logger = TensorBoardLogger(
        save_dir=args.output, name=args.name, default_hp_metric=False
    )
    run = Path(logger.log_dir)
    check_output_inside_runs(run)
    check_no_existing_checkpoints(run)
    run.mkdir(parents=True, exist_ok=True)
    # So a result is reproducible from the run directory, not the shell history
    (run / RESOLVED_CONFIG).write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    log_every = resolve_log_every(
        args.log_every, len(train_loader), args.limit_train_batches
    )
    best, last = checkpoint_callbacks(run)
    panels = PanelWriter(
        {
            "train": evenly_spaced_batch(train_set, args.panel_images),
            "val": evenly_spaced_batch(val_set, args.panel_images),
        },
        every=args.panel_every,
        samples_dir=run / SAMPLES,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        logger=logger,
        default_root_dir=args.output,
        log_every_n_steps=log_every,
        limit_train_batches=args.limit_train_batches,
        callbacks=[best, last, panels],
        enable_progress_bar=not args.quiet,
    )
    print(f"TensorBoard events -> {run.resolve()}", flush=True)
    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    best_score = best.best_model_score
    (run / SUMMARY).write_text(
        json.dumps(
            {
                "run": args.name,
                "version": run.name,
                "config": args.config,
                "epochs": args.epochs,
                "train_views": len(train_set),
                "val_views": len(val_set),
                "parameters": count_parameters(module.model),
                "hyperparameters": dict(module.hparams),
                "best_checkpoint": best.best_model_path,
                "best_val_mae": None if best_score is None else float(best_score),
                "metrics": metrics,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    mae, nearest = metrics.get("val/mae"), metrics.get("val/mae_nearest")
    if mae is not None and nearest is not None:
        print(f"val/mae {mae:.4f} against nearest {nearest:.4f}", flush=True)
    print(f"Done -> {run.resolve()}")


if __name__ == "__main__":
    main()
