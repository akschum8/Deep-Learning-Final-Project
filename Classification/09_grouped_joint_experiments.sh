module --force purge
module load miniconda/24.11.3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/yr245/project/ultrasam_project/.conda/envs/UltraSam

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /home/yr245

python - <<'PY'
import torch

print("python cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

python - <<'PY'
import pandas as pd
from pathlib import Path

full_root = Path("/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_all")
summary = pd.read_csv(full_root / "summary.csv")

groups = {
    "semantic_pathology": [
        "breast_2cls",
        "breast_3cls",
        "liver_lesion_2cls",
        "lung_2cls",
        "lung_disease_3cls",
        "organ_cls",
        "fetal_plane_cls",
    ],
    "orientation_position": [
        "fetal_head_pos_cls",
        "fetal_sacral_pos_cls",
    ],
}

out_base = Path("/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_grouped")
out_base.mkdir(parents=True, exist_ok=True)

for group_name, prefixes in groups.items():
    out_root = out_base / group_name
    out_root.mkdir(parents=True, exist_ok=True)

    keep = summary[
        summary["task_id"].apply(
            lambda x: any(str(x).startswith(prefix) for prefix in prefixes)
        )
    ].copy()

    keep.to_csv(out_root / "summary.csv", index=False)

    print("\nGROUP:", group_name)
    print(keep[["csv", "task_id", "train_n", "val_n", "num_classes"]].to_string(index=False))
    print("saved:", out_root / "summary.csv")
PY

cat > run_grouped_experiments.sh <<'SH'
#!/bin/bash
set -e

unset PYTHONPATH
unset PYTHONHOME

module --force purge
module load miniconda/24.11.3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/yr245/project/ultrasam_project/.conda/envs/UltraSam

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /home/yr245

CKPT="/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth"
GROUP_ROOT="/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_grouped"

mkdir -p cls_runs/logs

echo "RUN 1: semantic/pathology adapter"

python train_joint_adapter_prototype.py \
  --split_root "$GROUP_ROOT/semantic_pathology" \
  --checkpoint "$CKPT" \
  --output_dir cls_runs/group_semantic_adapter \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256 \
  --adapter_bottleneck 64 \
  --steps_per_epoch 10000

echo "RUN 2: semantic/pathology task-aware"

python train_joint_taskaware_prototype.py \
  --split_root "$GROUP_ROOT/semantic_pathology" \
  --checkpoint "$CKPT" \
  --output_dir cls_runs/group_semantic_taskaware \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256 \
  --adapter_bottleneck 64 \
  --steps_per_epoch 10000

echo "RUN 3: orientation adapter"

python train_joint_adapter_prototype.py \
  --split_root "$GROUP_ROOT/orientation_position" \
  --checkpoint "$CKPT" \
  --output_dir cls_runs/group_orientation_adapter \
  --epochs 8 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 8 \
  --proto_dim 256 \
  --adapter_bottleneck 128 \
  --steps_per_epoch 6000

echo "RUN 4: orientation task-aware"

python train_joint_taskaware_prototype.py \
  --split_root "$GROUP_ROOT/orientation_position" \
  --checkpoint "$CKPT" \
  --output_dir cls_runs/group_orientation_taskaware \
  --epochs 8 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 8 \
  --proto_dim 256 \
  --adapter_bottleneck 128 \
  --steps_per_epoch 6000

echo "ALL RUNS DONE"

python - <<'PY'
import json
from pathlib import Path

runs = [
    "group_semantic_adapter",
    "group_semantic_taskaware",
    "group_orientation_adapter",
    "group_orientation_taskaware",
]

base = Path("/home/yr245/cls_runs")

for run in runs:
    run_dir = base / run
    files = sorted(run_dir.glob("metrics_epoch_*.json"))

    print("\nRUN:", run)

    if not files:
        print("No metric files found.")
        continue

    best_file = max(files, key=lambda f: json.load(open(f))["val_macro_avg"])
    best = json.load(open(best_file))

    print("best:", best_file.name)
    print("val_macro_avg:", best["val_macro_avg"])

    for task, m in sorted(best["val"].items()):
        print(f"{task}: acc={m['acc']:.4f}, macroF1={m['macro_f1']:.4f}, n={m['n']}")
PY
SH

chmod +x run_grouped_experiments.sh

mkdir -p cls_runs/logs

bash run_grouped_experiments.sh 2>&1 | tee cls_runs/logs/grouped_experiments.log

python - <<'PY'
import json
from pathlib import Path

base = Path("/home/yr245/cls_runs")
runs = [
    "group_semantic_adapter",
    "group_semantic_taskaware",
    "group_orientation_adapter",
    "group_orientation_taskaware",
]

for run in runs:
    run_dir = base / run
    files = sorted(run_dir.glob("metrics_epoch_*.json"))

    print("\n" + "=" * 90)
    print("RUN:", run)

    if not files:
        print("No metrics_epoch files found.")
        continue

    print("\nEpoch history:")
    for f in files:
        d = json.load(open(f))
        print(
            f"{f.name}: "
            f"val_macro_avg={d['val_macro_avg']:.4f}, "
            f"val_loss={d['val_loss']:.4f}"
        )

    best_file = max(files, key=lambda f: json.load(open(f))["val_macro_avg"])
    best = json.load(open(best_file))

    print("\nBEST EPOCH")
    print("file:", best_file.name)
    print("val_macro_avg:", round(best["val_macro_avg"], 4))
    print("val_loss:", round(best["val_loss"], 4))

    print("\nPer-task validation metrics:")
    for task, m in sorted(best["val"].items()):
        print(f"{task:60s} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f} n={m['n']}")
PY