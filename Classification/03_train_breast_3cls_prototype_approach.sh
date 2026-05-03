cd /home/yr245/project/ultrasam_project/UltraSam

python - <<'PY'
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

csv_path = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Three_2.BUSI_Dataset_780.csv"
out_dir = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_breast_3cls_strict")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

def group_id(p):
    p = Path(str(p))
    return str(p.parent) + "__" + p.stem

df["group"] = df["image_path"].apply(group_id)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, y=df["mask"], groups=df["group"]))

train = df.iloc[train_idx].drop(columns=["group"])
val = df.iloc[val_idx].drop(columns=["group"])

train.to_csv(out_dir / "train.csv", index=False)
val.to_csv(out_dir / "val.csv", index=False)

print("train:", len(train))
print("val:", len(val))
print("group overlap:", len(set(df.iloc[train_idx]["group"]) & set(df.iloc[val_idx]["group"])))

print("\ntrain labels:")
print(train["mask"].value_counts().sort_index())

print("\nval labels:")
print(val["mask"].value_counts().sort_index())

print("saved:", out_dir)
PY

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

rm -rf cls_runs/breast_3cls_prototype_frozen

python train_ultrasam_prototype_cls.py \
  --train_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_breast_3cls_strict/train.csv" \
  --val_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_breast_3cls_strict/val.csv" \
  --checkpoint UltraSam.pth \
  --output_dir cls_runs/breast_3cls_prototype_frozen \
  --epochs 10 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256

cat cls_runs/breast_3cls_prototype_frozen/val_report.txt
ls cls_runs/breast_3cls_prototype_frozen/prototype_heatmaps | head