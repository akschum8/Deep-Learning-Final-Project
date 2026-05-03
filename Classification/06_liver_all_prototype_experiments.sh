cd /home/yr245/project/ultrasam_project/UltraSam

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python - <<'PY'
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

csv_path = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Two_LIVER_LESION.csv"
out_dir = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_strict")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

def group_id(p):
    p = Path(str(p))
    stem = p.stem
    acq = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return str(p.parent) + "__" + acq

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
PY

rm -rf cls_runs/liver_lesion_prototype

python train_ultrasam_prototype_cls.py \
  --train_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_strict/train.csv" \
  --val_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_strict/val.csv" \
  --checkpoint UltraSam.pth \
  --output_dir cls_runs/liver_lesion_prototype \
  --epochs 10 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256

cat cls_runs/liver_lesion_prototype/val_report.txt
ls cls_runs/liver_lesion_prototype/prototype_heatmaps | head

python - <<'PY'
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

csv_path = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Two_1.HCC_Hemangioma_5466.csv"
out_dir = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_strict")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

def group_id(p):
    p = Path(str(p))
    stem = p.stem
    acq = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return str(p.parent) + "__" + acq

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
print("train labels:")
print(train["mask"].value_counts().sort_index())
print("val labels:")
print(val["mask"].value_counts().sort_index())
PY

rm -rf cls_runs/liver_2cls_prototype_strict_frozen

python train_ultrasam_prototype_cls.py \
  --train_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_strict/train.csv" \
  --val_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_strict/val.csv" \
  --checkpoint UltraSam.pth \
  --output_dir cls_runs/liver_2cls_prototype_strict_frozen \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256

python - <<'PY'
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

csv_path = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Two_1.HCC_Hemangioma_5466.csv"
out_dir = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_stratified")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

def group_id(p):
    p = Path(str(p))
    stem = p.stem
    acq = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return str(p.parent) + "__" + acq

df["group"] = df["image_path"].apply(group_id)

train_parts = []
val_parts = []

for label in sorted(df["mask"].unique()):
    sub = df[df["mask"] == label].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(sub, y=sub["mask"], groups=sub["group"]))
    train_parts.append(sub.iloc[train_idx])
    val_parts.append(sub.iloc[val_idx])

train = pd.concat(train_parts).sample(frac=1, random_state=42).drop(columns=["group"])
val = pd.concat(val_parts).sample(frac=1, random_state=42).drop(columns=["group"])

train.to_csv(out_dir / "train.csv", index=False)
val.to_csv(out_dir / "val.csv", index=False)

train_groups = set(pd.concat(train_parts)["group"])
val_groups = set(pd.concat(val_parts)["group"])

print("train:", len(train))
print("val:", len(val))
print("train labels:")
print(train["mask"].value_counts().sort_index())
print("val labels:")
print(val["mask"].value_counts().sort_index())
print("group overlap:", len(train_groups & val_groups))
PY

rm -rf cls_runs/liver_2cls_prototype_stratified_frozen

python train_ultrasam_prototype_cls.py \
  --train_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_stratified/train.csv" \
  --val_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_stratified/val.csv" \
  --checkpoint UltraSam.pth \
  --output_dir cls_runs/liver_2cls_prototype_stratified_frozen \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256

python - <<'PY'
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

csv_path = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Two_1.HCC_Hemangioma_5466.csv"
out_dir = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_simple")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

train, val = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["mask"]
)

train.to_csv(out_dir / "train.csv", index=False)
val.to_csv(out_dir / "val.csv", index=False)

print("train:", len(train))
print("val:", len(val))
print("train labels:")
print(train["mask"].value_counts().sort_index())
print("val labels:")
print(val["mask"].value_counts().sort_index())
PY

rm -rf cls_runs/liver_2cls_prototype_simple_frozen

python train_ultrasam_prototype_cls.py \
  --train_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_simple/train.csv" \
  --val_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_simple/val.csv" \
  --checkpoint UltraSam.pth \
  --output_dir cls_runs/liver_2cls_prototype_simple_frozen \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256

ls cls_runs/liver_2cls_prototype_simple_frozen/prototype_heatmaps | head

python - <<'PY'
import pandas as pd

csv_path = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Two_1.HCC_Hemangioma_5466.csv"
df = pd.read_csv(csv_path)

for label in sorted(df["mask"].unique()):
    subset = df[df["mask"] == label].head(5)
    print(f"\nLABEL {label}:")
    for _, row in subset.iterrows():
        print(row["image_path"])
PY

cat > analyze_liver_errors_simple.py <<'PY'
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from train_ultrasam_prototype_cls import (
    UltrasoundClassificationCSVDataset,
    UltraSamPrototypeClassifier,
)

LABELS = {
    0: "HCC_cancer",
    1: "Hemangioma_benign",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

val_csv = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_2cls_simple/val.csv"
checkpoint = "UltraSam.pth"
model_path = "cls_runs/liver_2cls_prototype_simple_frozen/best_model.pth"
out_csv = "cls_runs/liver_2cls_prototype_simple_frozen/val_predictions_with_errors.csv"

ds = UltrasoundClassificationCSVDataset(val_csv, img_size=1024, train=False)

model = UltraSamPrototypeClassifier(
    checkpoint_path=checkpoint,
    num_classes=2,
    img_size=1024,
    prototypes_per_class=4,
    proto_dim=256,
    freeze_backbone=True,
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

rows = []

with torch.no_grad():
    for i in range(len(ds)):
        batch = ds[i]
        x = batch["image"].unsqueeze(0).to(device)
        y = int(batch["label"])
        path = batch["path"]

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        pred = int(torch.argmax(probs))

        rows.append({
            "path": path,
            "true": y,
            "pred": pred,
            "true_name": LABELS[y],
            "pred_name": LABELS[pred],
            "correct": y == pred,
            "prob_HCC_cancer": float(probs[0]),
            "prob_Hemangioma_benign": float(probs[1]),
            "confidence": float(probs[pred]),
        })

df = pd.DataFrame(rows)
Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print("Saved:", out_csv)

cm = confusion_matrix(df["true"], df["pred"], labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["true_HCC_cancer", "true_Hemangioma_benign"],
    columns=["pred_HCC_cancer", "pred_Hemangioma_benign"],
)

print("Confusion matrix counts:")
print(cm_df)

print("Row-normalized confusion matrix:")
print((cm_df.div(cm_df.sum(axis=1), axis=0)).round(3))

print("Classification report:")
print(classification_report(
    df["true"],
    df["pred"],
    target_names=["HCC_cancer", "Hemangioma_benign"],
    digits=3
))

missed_cancer = df[(df["true"] == 0) & (df["pred"] == 1)]
false_cancer = df[(df["true"] == 1) & (df["pred"] == 0)]

print("Clinically important errors:")
print("Missed cancers, HCC -> Hemangioma:", len(missed_cancer))
print("False cancer calls, Hemangioma -> HCC:", len(false_cancer))

print("Highest-confidence missed cancers:")
print(missed_cancer.sort_values("confidence", ascending=False)[[
    "path", "confidence", "prob_HCC_cancer", "prob_Hemangioma_benign"
]].head(15).to_string(index=False))

print("Highest-confidence false cancer calls:")
print(false_cancer.sort_values("confidence", ascending=False)[[
    "path", "confidence", "prob_HCC_cancer", "prob_Hemangioma_benign"
]].head(15).to_string(index=False))
PY

python analyze_liver_errors_simple.py

python - <<'PY'
import pandas as pd
from pathlib import Path

csv_path = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Two_1.HCC_Hemangioma_5466.csv"
out_dir = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_official")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

train = df[df["image_path"].str.contains("/train_2/")].copy()
val = df[df["image_path"].str.contains("/test_2/")].copy()

train.to_csv(out_dir / "train.csv", index=False)
val.to_csv(out_dir / "val.csv", index=False)

print("train:", len(train))
print(train["mask"].value_counts().sort_index())
print("val:", len(val))
print(val["mask"].value_counts().sort_index())
PY

rm -rf cls_runs/liver_2cls_prototype_unfrozen

python train_ultrasam_prototype_cls.py \
  --train_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_official/train.csv" \
  --val_csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_official/val.csv" \
  --checkpoint UltraSam.pth \
  --output_dir cls_runs/liver_2cls_prototype_unfrozen \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-5 \
  --img_size 1024 \
  --prototypes_per_class 8 \
  --proto_dim 256 \
  --unfreeze_backbone

cat > analyze_liver_errors_unfrozen.py <<'PY'
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from train_ultrasam_prototype_cls import (
    UltrasoundClassificationCSVDataset,
    UltraSamPrototypeClassifier,
)

LABELS = {
    0: "HCC_cancer",
    1: "Hemangioma_benign",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

val_csv = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_liver_official/val.csv"
checkpoint = "UltraSam.pth"
model_path = "cls_runs/liver_2cls_prototype_unfrozen/best_model.pth"
out_csv = "cls_runs/liver_2cls_prototype_unfrozen/val_predictions_with_errors.csv"

ds = UltrasoundClassificationCSVDataset(val_csv, img_size=1024, train=False)

model = UltraSamPrototypeClassifier(
    checkpoint_path=checkpoint,
    num_classes=2,
    img_size=1024,
    prototypes_per_class=8,
    proto_dim=256,
    freeze_backbone=False,
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

rows = []

with torch.no_grad():
    for i in range(len(ds)):
        batch = ds[i]
        x = batch["image"].unsqueeze(0).to(device)
        y = int(batch["label"])
        path = batch["path"]

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        pred = int(torch.argmax(probs))

        rows.append({
            "path": path,
            "true": y,
            "pred": pred,
            "true_name": LABELS[y],
            "pred_name": LABELS[pred],
            "correct": y == pred,
            "prob_HCC_cancer": float(probs[0]),
            "prob_Hemangioma_benign": float(probs[1]),
            "confidence": float(probs[pred]),
        })

df = pd.DataFrame(rows)
Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print("Saved:", out_csv)

cm = confusion_matrix(df["true"], df["pred"], labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["true_HCC_cancer", "true_Hemangioma_benign"],
    columns=["pred_HCC_cancer", "pred_Hemangioma_benign"],
)

print("Confusion matrix counts:")
print(cm_df)

print("Row-normalized confusion matrix:")
print((cm_df.div(cm_df.sum(axis=1), axis=0)).round(3))

print("Classification report:")
print(classification_report(
    df["true"],
    df["pred"],
    target_names=["HCC_cancer", "Hemangioma_benign"],
    digits=3
))

missed_cancer = df[(df["true"] == 0) & (df["pred"] == 1)]
false_cancer = df[(df["true"] == 1) & (df["pred"] == 0)]

print("Clinically important errors:")
print("Missed cancers, HCC -> Hemangioma:", len(missed_cancer))
print("False cancer calls, Hemangioma -> HCC:", len(false_cancer))

print("Highest-confidence missed cancers:")
print(missed_cancer.sort_values("confidence", ascending=False)[[
    "path", "confidence", "prob_HCC_cancer", "prob_Hemangioma_benign"
]].head(15).to_string(index=False))

print("Highest-confidence false cancer calls:")
print(false_cancer.sort_values("confidence", ascending=False)[[
    "path", "confidence", "prob_HCC_cancer", "prob_Hemangioma_benign"
]].head(15).to_string(index=False))
PY

python analyze_liver_errors_unfrozen.py

cat cls_runs/liver_2cls_prototype_unfrozen/val_report.txt
ls cls_runs/liver_2cls_prototype_unfrozen/prototype_heatmaps | head