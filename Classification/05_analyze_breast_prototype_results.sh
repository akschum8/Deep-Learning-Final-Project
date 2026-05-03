cd /home/yr245/project/ultrasam_project/UltraSam

python - <<'PY'
import pandas as pd
from pathlib import Path

csv_path = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Three_2.BUSI_Dataset_780.csv"
df = pd.read_csv(csv_path)

print("LABEL DISTRIBUTION")
print(df["mask"].value_counts().sort_index())

print("\nLABEL EXAMPLES")

for label in sorted(df["mask"].unique()):
    subset = df[df["mask"] == label]["image_path"].head(5)
    print(f"\nLABEL {label}")
    for p in subset:
        p = Path(p)
        print(f"{p.parent.name} | {p.name}")
PY

cat > analyze_breast_errors.py <<'PY'
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from train_ultrasam_prototype_cls import (
    UltrasoundClassificationCSVDataset,
    UltraSamPrototypeClassifier,
)

LABEL_NAMES = {
    0: "benign",
    1: "malignant",
    2: "normal",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

val_csv = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files_split_breast_3cls_strict/val.csv"
checkpoint = "UltraSam.pth"
model_path = "cls_runs/breast_3cls_prototype_frozen/best_model.pth"
out_csv = "cls_runs/breast_3cls_prototype_frozen/val_predictions_with_errors.csv"

ds = UltrasoundClassificationCSVDataset(val_csv, img_size=1024, train=False)

model = UltraSamPrototypeClassifier(
    checkpoint_path=checkpoint,
    num_classes=3,
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
        sample = ds[i]
        x = sample["image"].unsqueeze(0).to(device)
        y = int(sample["label"])
        path = sample["path"]

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()

        pred = int(torch.argmax(probs))

        rows.append({
            "path": path,
            "true": y,
            "pred": pred,
            "true_name": LABEL_NAMES[y],
            "pred_name": LABEL_NAMES[pred],
            "correct": y == pred,
            "prob_benign": float(probs[0]),
            "prob_malignant": float(probs[1]),
            "prob_normal": float(probs[2]),
            "confidence": float(probs[pred]),
        })

df = pd.DataFrame(rows)
Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print("\nSaved:", out_csv)

cm = confusion_matrix(df["true"], df["pred"], labels=[0,1,2])
print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(
    df["true"], df["pred"],
    target_names=["benign", "malignant", "normal"],
    digits=3
))
PY

python analyze_breast_errors.py

cat > tune_breast_threshold.py <<'PY'
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

pred_csv = "cls_runs/breast_3cls_prototype_frozen/val_predictions_with_errors.csv"
df = pd.read_csv(pred_csv)

rows = []

for threshold in np.arange(0.05, 0.96, 0.01):
    preds = []

    for _, r in df.iterrows():
        probs = {
            0: r["prob_benign"],
            1: r["prob_malignant"],
            2: r["prob_normal"],
        }

        if probs[1] >= threshold:
            pred = 1
        else:
            pred = max(probs, key=probs.get)

        preds.append(pred)

    true = df["true"].astype(int).tolist()
    cm = confusion_matrix(true, preds, labels=[0,1,2])

    malignant_total = cm[1].sum()
    malignant_correct = cm[1,1]

    rows.append({
        "threshold": threshold,
        "accuracy": accuracy_score(true, preds),
        "macro_f1": f1_score(true, preds, average="macro"),
        "malignant_recall": malignant_correct / malignant_total,
    })

res = pd.DataFrame(rows)

best = res.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]

print("\nBest threshold:")
print(best)
PY

python tune_breast_threshold.py