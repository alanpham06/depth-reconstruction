#!/bin/bash
# Phase 2 baseline: generate train/val ground truth (if missing) and train the UNet.
# Edit the #SBATCH lines and the environment setup for your cluster, then:
#   sbatch scripts/train_baseline.sbatch
#SBATCH --job-name=depth-unet
#SBATCH --partition=gpu            # <- your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# <- your environment, e.g. module load anaconda3 && conda activate depth-ml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate depth-ml

TRAIN_VIEWS=${TRAIN_VIEWS:-256}
VAL_VIEWS=${VAL_VIEWS:-24}
RUN=${RUN:-runs/baseline}

# Rendering is CPU-only; different view counts and seeds give val its own
# camera poses and its own sparse samples.
[ -f data/train/cube_gt.pt ] || python generate_gt.py --views "$TRAIN_VIEWS" --seed 0 --out data/train
[ -f data/val/cube_gt.pt ] || python generate_gt.py --views "$VAL_VIEWS" --seed 1 --out data/val

RESUME=()
[ -f "$RUN/last.pt" ] && RESUME=(--resume "$RUN/last.pt")

python train.py \
	--train data/train/*_gt.pt \
	--val data/val/*_gt.pt \
	--out "$RUN" \
	--epochs 100 --batch-size 16 --lr 1e-3 --grad-weight 0.5 \
	--workers "${SLURM_CPUS_PER_TASK:-4}" --amp \
	"${RESUME[@]}"
