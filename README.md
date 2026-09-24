# depth-reconstruction — sparse depth in, dense depth out

Completes a depth map from a sparse set of points on the surface. The ground truth
is procedural shapes rendered by exact ray casting, so the target is exact rather
than a sensor's estimate.

```
sparse depth (1, H, W) ─┐
                        ├─> UNet ─> depth - ref ─> + ref ─> dense depth (1, H, W)
point mask   (1, H, W) ─┘                 ref = the mean sparse depth
```

Each view of a cube, a UV sphere and an icosphere is ray-cast for exact z-depth and
normals. Its sparse input is surface points sampled by area, with extra density at
sharp edges and corners, kept only where visible. The network predicts depth
relative to the mean of those points, so it learns shape rather than distance. It
is always scored against two fills that use no network at all.

The deployment target is a Quest 3, through ExecuTorch's Vulkan backend.

## Setup

```bash
conda env create -f environment.yaml
conda activate dr
python -m pytest tests
```

## Code checks

The Python style is four spaces, double quotes and an 88-column formatter target.
Ruff is pinned in `environment.yaml`.
Run these from the repository root in its conda environment:

```bash
ruff check .
ruff format --check .
python -m pytest tests
```

Use `ruff check --fix .` and `ruff format .` to apply automatic cleanup.

## Layout

| | |
|---|---|
| `train.py` | the one entry point that trains |
| `arguments/`, `configs/` | the flag surface, and the yaml a `--config` name resolves to |
| `datasets/` | capture files, splits, augmentation |
| `models/` | `unet.py`, `losses.py`, `baselines.py`, `reconstruction_module.py` |
| `render/` | capture: the procedural meshes, the ray caster, the surface sampling |
| `eval/`, `export/` | scoring a checkpoint, and deployment |
| `plotting/`, `utils/` | the point-cloud figures, and the shared plumbing |
| `scripts/` | the cluster job |
| `tests/` | `python -m pytest tests` |

`runs/`, `checkpoints/`, `output/`, `logs/` and `data/` are gitignored.

## Capturing ground truth

```bash
python -m render.capture --views 256 --seed 0 --out data/train
python -m render.capture --views 24  --seed 1 --out data/val
```

About two minutes on the CPU. Each primitive gets one `<shape>_gt.pt`, with every
view stacked: the mesh, `K`, `world_to_cam`, the dense `depth`, `normals` and
`mask`, and the `sparse_depth`/`sparse_mask` input. Beside it goes a
`<shape>_gt.png` preview. Different seeds and view counts give val its own camera
poses and its own sparse samples.

## Training

```bash
python train.py --config unet_b32 --train data/train --val data/val --name baseline
```

`configs/base.yaml` is the recipe: AdamW at 1e-3 with a OneCycle schedule stepped
every batch, 100 epochs of batch 16, L1 plus 0.5 of the gradient loss, bf16. A flag
given on the command line beats the config. `--train` and `--val` take directories
or `*_gt.pt` files. Without `--val`, every `--val-every`-th view is held out.
`--resume checkpoints/<run>_version_N_last.ckpt` continues a run that used the same
`--epochs`. `scripts/train_baseline.sbatch` runs capture and training as one
cluster job.

A run writes `runs/<name>/version_N/` (events, `config.json`, `summary.json`,
`samples/`) and its models to `checkpoints/<name>_version_N_{best,last}.ckpt`.
`best.ckpt` has the lowest `val/mae`.

## Evaluating

```bash
python -m eval.eval --checkpoint checkpoints/baseline_version_0_best.ckpt --val data/val
```

This scores the model, the nearest fill and the constant fill per shape and over the
whole split, with the same pooled metrics training logs. It writes `metrics.json`
and `panel.png` to `output/eval/<run>_<source>/`.

## Exporting

```bash
conda activate executorch
python -m export.export --checkpoint checkpoints/baseline_version_0_best.ckpt --size 256
```

The graph is `torch.export` → `VulkanPartitioner` → `.pte`. It is the whole completion: sparse
depth and its mask go in as `(1, 2, H, W)`, and metric depth comes out, with the
centring inside the graph. It is **GPU only**: if the partitioner leaves any op on
the CPU fallback, the export fails and writes nothing.

## TensorBoard

```bash
tensorboard --logdir runs --port 6006
```

| | |
|---|---|
| `batch/*` | per step: `loss`, `l1`, `grad`, `grad_norm` |
| `train/*` | per epoch |
| `val/loss`, `val/l1`, `val/grad` | every epoch, on the held-out views |
| `val/mae`, `val/rmse`, `val/abs_rel`, `val/delta1`, `val/hole_mae` | pooled over every valid pixel of the split; `hole_mae` counts only pixels with no sparse point |
| `val/*_nearest`, `val/*_constant` | the same five for the two fills that use no network |
| `lr/recon` | the OneCycle rate at the start of each epoch |
| `{train,val}/sparse_pred_gt_error` | sparse input │ prediction │ ground truth │ error, one row per view |

**Read `val/mae` against `val/mae_nearest`.** Copying each pixel's nearest sparse
point already fills a smooth shape well; the gap between the two lines is what the
network is worth.

## Plotting the point clouds

```bash
python -m render.point_cloud   # data/sphere.pt and data/cube.pt
python -m plotting.visualize   # output/plotting/point_clouds.png
python -m plotting.viewer      # output/plotting/viewer.html, opened in a browser
```
