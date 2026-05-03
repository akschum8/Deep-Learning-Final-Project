cd /home/yr245/project/ultrasam_project/UltraSam

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PYTHONPATH:."
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

pip install albumentations opencv-python-headless tqdm matplotlib scikit-image segmentation-models-pytorch

python - <<'PY'
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

csv_path = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Six_2.FETAL_PLANES_ZENODO_12,400.csv")
out_dir = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_fetal_plane")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
df["group_id"] = df["image_path"].apply(lambda p: Path(str(p)).stem.rsplit("_", 1)[0])

groups = df["group_id"].drop_duplicates()
train_groups, val_groups = train_test_split(
    groups,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_df = df[df["group_id"].isin(train_groups)].drop(columns=["group_id"])
val_df = df[df["group_id"].isin(val_groups)].drop(columns=["group_id"])

train_df.to_csv(out_dir / "train.csv", index=False)
val_df.to_csv(out_dir / "val.csv", index=False)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Saved splits to: {out_dir}")
PY

rm -rf cls_runs/fetal_plane_split_finetune_5ep

python train_ultrasam_classification.py \
  --train_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_fetal_plane/train.csv" \
  --val_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_fetal_plane/val.csv" \
  --checkpoint UltraSam.pth \
  --output_dir cls_runs/fetal_plane_split_finetune_5ep \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4