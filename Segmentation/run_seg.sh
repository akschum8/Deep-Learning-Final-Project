#!/bin/bash
#SBATCH --job-name=ultrasam_seg
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=seg_runs/logs/%j_train.out
#SBATCH --error=seg_runs/logs/%j_train.err

# ── Environment ──────────────────────────────────────────────────────────────
module unload Python
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/cpsc4520_oh84/project_cpsc4520/cpsc4520_oh84/ultrasam_project/.conda/envs/UltraSam

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=$PYTHONPATH:.

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE=/home/cpsc4520_oh84/project_cpsc4520/cpsc4520_oh84/ultrasam_project
ULTRASAM=$BASE/UltraSam
CSV_DIR=$BASE/data_train7z/train/csv_files
SPLITS_DIR=$BASE/data_train7z/train/splits

cd $ULTRASAM

# ── Step 1: Create splits (skip if already done) ─────────────────────────────
if [ ! -f "$SPLITS_DIR/manifest.json" ]; then
    echo "Creating train/val/test splits..."
    python prepare_splits.py \
        --csv_dir $CSV_DIR \
        --out_dir $SPLITS_DIR
else
    echo "Splits already exist, skipping prepare_splits.py"
fi

# ── Step 2: Train all segmentation datasets ───────────────────────────────────
mkdir -p seg_runs/logs

python train_all_seg.py \
    --manifest   $SPLITS_DIR/manifest.json \
    --checkpoint UltraSam.pth \
    --output_dir seg_runs/all \
    --epochs     5 \
    --batch_size 8 \
    --num_workers 4 \
    --img_size   512 \
    --lr         1e-3 \
    --freeze_backbone

echo "Done. Results in $ULTRASAM/seg_runs/all/summary.json"
