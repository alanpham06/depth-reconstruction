"""Train the sparse-to-dense UNet baseline on rendered ground truth.

The network sees (sparse depth, valid-point mask) and predicts dense depth
relative to the mean sparse depth; the loss is L1 depth plus a multi-scale
gradient-matching term, both only on pixels that hit the surface.

  python train.py --train data/train/*_gt.pt --val data/val/*_gt.pt --out runs/baseline

Without --val, every --val-every'th view of the training files is held out.
Each run directory gets config.json, baselines.json (non-learned fills scored
on the same validation split), metrics.csv, last.pt, best.pt (lowest
validation MAE) and preview_<epoch>.png. --resume continues from a last.pt
with the same --epochs (e.g. after a preempted job).
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from baselines import evaluate_baselines
from dataset import SparseDepthDataset, load_views
from losses import completion_loss, depth_metrics, summarize_metrics
from unet import UNet


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--train", type=Path, nargs="+", required=True, help="*_gt.pt files to train on")
	parser.add_argument("--val", type=Path, nargs="*", default=None, help="held-out *_gt.pt files")
	parser.add_argument("--val-every", type=int, default=8, help="without --val: hold out every n-th view")
	parser.add_argument("--out", type=Path, default=Path("runs") / time.strftime("%Y%m%d-%H%M%S"))
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--grad-weight", type=float, default=0.5, help="weight of the gradient loss")
	parser.add_argument("--base", type=int, default=32, help="UNet width at full resolution")
	parser.add_argument("--keep-range", type=float, nargs=2, default=(0.3, 1.0), metavar=("LOW", "HIGH"))
	parser.add_argument("--noise-std", type=float, default=0.0, help="sparse depth noise during training")
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
	parser.add_argument("--preview-every", type=int, default=10, help="epochs between preview images")
	parser.add_argument("--max-steps", type=int, default=None, help="stop each epoch after n batches (smoke tests)")
	parser.add_argument("--resume", type=Path, default=None, help="last.pt to continue from")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	return parser.parse_args()


def build_datasets(args: argparse.Namespace) -> tuple[SparseDepthDataset, SparseDepthDataset]:
	augment = dict(keep_range=tuple(args.keep_range), noise_std=args.noise_std)
	if args.val:
		return (
			SparseDepthDataset(args.train, augment=True, **augment),
			SparseDepthDataset(args.val),
		)
	n_views = load_views(args.train)["depth"].shape[0]
	held_out = torch.arange(n_views) % args.val_every == 0
	return (
		SparseDepthDataset(args.train, views=(~held_out).nonzero().squeeze(1), augment=True, **augment),
		SparseDepthDataset(args.train, views=held_out.nonzero().squeeze(1)),
	)


def predict(model: UNet, batch: dict[str, torch.Tensor], amp: bool) -> torch.Tensor:
	with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
		residual = model(batch["input"])
	return batch["ref"] + residual.float()


@torch.no_grad()
def evaluate(model: UNet, loader: DataLoader, args: argparse.Namespace) -> dict[str, float]:
	model.eval()
	sums, loss_sum, batches = {}, 0.0, 0
	for batch in loader:
		batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
		pred = predict(model, batch, args.amp)
		loss, _ = completion_loss(pred, batch["target"], batch["valid"], args.grad_weight)
		loss_sum, batches = loss_sum + loss.item(), batches + 1
		for key, value in depth_metrics(pred, batch["target"], batch["valid"], batch["input"][:, 1:2]).items():
			sums[key] = sums.get(key, 0) + value
	return {"loss": loss_sum / max(batches, 1), **summarize_metrics(sums)}


@torch.no_grad()
def save_preview(model: UNet, dataset: SparseDepthDataset, args: argparse.Namespace, path: Path, n: int = 6) -> None:
	model.eval()
	indices = torch.linspace(0, len(dataset) - 1, min(n, len(dataset))).round().long().tolist()
	figure, axes = plt.subplots(4, len(indices), figsize=(2.2 * len(indices), 9), squeeze=False)
	for column, index in enumerate(indices):
		sample = {key: value[None].to(args.device) for key, value in dataset[index].items()}
		pred = predict(model, sample, args.amp)[0, 0].cpu()
		gt, valid = sample["target"][0, 0].cpu(), sample["valid"][0, 0].cpu()
		sparse_mask = sample["input"][0, 1].cpu() > 0
		sparse = (sample["input"][0, 0].cpu() + sample["ref"][0, 0].cpu()).masked_fill(~sparse_mask, math.nan)
		low, high = gt[valid].min().item(), gt[valid].max().item()
		panels = [
			(sparse, dict(cmap="viridis", vmin=low, vmax=high)),
			(pred.masked_fill(~valid, math.nan), dict(cmap="viridis", vmin=low, vmax=high)),
			(gt.masked_fill(~valid, math.nan), dict(cmap="viridis", vmin=low, vmax=high)),
			((pred - gt).abs().masked_fill(~valid, math.nan), dict(cmap="magma", vmin=0, vmax=0.05 * (high - low + 1))),
		]
		for row, (image, style) in enumerate(panels):
			axes[row, column].imshow(image.numpy(), **style)
			axes[row, column].set_axis_off()
		axes[0, column].set_title(f"val {index}", fontsize=8)
	for row, label in enumerate(["sparse input", "prediction", "ground truth", "|error|"]):
		axes[row, 0].text(-0.08, 0.5, label, transform=axes[row, 0].transAxes, rotation=90, va="center", ha="right")
	figure.tight_layout()
	figure.savefig(path, dpi=110)
	plt.close(figure)


def main() -> None:
	args = parse_args()
	torch.manual_seed(args.seed)
	args.out.mkdir(parents=True, exist_ok=True)
	args.amp = args.amp and args.device.startswith("cuda")

	train_set, val_set = build_datasets(args)
	loader_options = dict(num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
	train_loader = DataLoader(
		train_set, batch_size=args.batch_size, shuffle=True, drop_last=len(train_set) > args.batch_size,
		persistent_workers=args.workers > 0, **loader_options,
	)
	val_loader = DataLoader(val_set, batch_size=args.batch_size, **loader_options)

	model = UNet(in_channels=2, out_channels=1, base=args.base).to(args.device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	steps_per_epoch = min(len(train_loader), args.max_steps or len(train_loader))
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05
	)
	start_epoch, best_mae = 0, math.inf
	if args.resume:
		state = torch.load(args.resume, map_location=args.device)
		if state["config"]["epochs"] != args.epochs:
			raise SystemExit(f"--epochs must match the resumed run ({state['config']['epochs']}): the LR schedule depends on it")
		model.load_state_dict(state["model"])
		optimizer.load_state_dict(state["optimizer"])
		scheduler.load_state_dict(state["scheduler"])
		start_epoch, best_mae = state["epoch"] + 1, state["best_mae"]

	config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
	config["train"], config["val"] = [str(p) for p in args.train], [str(p) for p in args.val or []]
	(args.out / "config.json").write_text(json.dumps(config, indent=2))
	n_params = sum(p.numel() for p in model.parameters())
	print(f"train={len(train_set)} val={len(val_set)} params={n_params / 1e6:.2f}M device={args.device} -> {args.out}")
	baselines = evaluate_baselines(val_loader)
	(args.out / "baselines.json").write_text(json.dumps(baselines, indent=2))
	for name, metrics in baselines.items():
		print(f"baseline {name:8s} val mae {metrics['mae']:.4f} rmse {metrics['rmse']:.4f} hole mae {metrics['hole_mae']:.4f} d1 {metrics['delta1']:.3f}")

	log_path = args.out / "metrics.csv"
	fields = ["epoch", "lr", "train_loss", "train_l1", "train_grad", "val_loss", "val_mae", "val_rmse",
		"val_abs_rel", "val_delta1", "val_hole_mae", "seconds"]
	if not log_path.exists():
		with log_path.open("w", newline="") as f:
			csv.writer(f).writerow(fields)

	for epoch in range(start_epoch, args.epochs):
		model.train()
		started = time.time()
		totals, steps = {"loss": 0.0, "l1": 0.0, "grad": 0.0}, 0
		for step, batch in enumerate(train_loader):
			if step == steps_per_epoch:
				break
			batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
			pred = predict(model, batch, args.amp)
			loss, parts = completion_loss(pred, batch["target"], batch["valid"], args.grad_weight)
			optimizer.zero_grad(set_to_none=True)
			lr = optimizer.param_groups[0]["lr"]
			loss.backward()
			optimizer.step()
			scheduler.step()
			totals = {key: totals[key] + parts[key] for key in totals}
			steps += 1
		train = {key: value / max(steps, 1) for key, value in totals.items()}
		val = evaluate(model, val_loader, args)
		seconds = time.time() - started

		row = [epoch, lr, train["loss"], train["l1"], train["grad"], val["loss"],
			val["mae"], val["rmse"], val["abs_rel"], val["delta1"], val["hole_mae"], seconds]
		with log_path.open("a", newline="") as f:
			csv.writer(f).writerow([f"{x:.6g}" if isinstance(x, float) else x for x in row])
		print(
			f"epoch {epoch:3d}  train loss {train['loss']:.4f} (l1 {train['l1']:.4f}, grad {train['grad']:.4f})  "
			f"val mae {val['mae']:.4f} rmse {val['rmse']:.4f} hole mae {val['hole_mae']:.4f} "
			f"d1 {val['delta1']:.3f}  {seconds:.0f}s"
		)

		state = {
			"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
			"epoch": epoch, "best_mae": min(best_mae, val["mae"]), "config": config, "val": val,
		}
		torch.save(state, args.out / "last.pt")
		if val["mae"] < best_mae:
			best_mae = val["mae"]
			torch.save(state, args.out / "best.pt")
		if (epoch + 1) % args.preview_every == 0 or epoch + 1 == args.epochs:
			save_preview(model, val_set, args, args.out / f"preview_{epoch + 1:03d}.png")


if __name__ == "__main__":
	main()
