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
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from train_joint_adapter_prototype import (
    JointClsDataset,
    SharedUltraSamAdapterPrototype,
)
from lora_utils import inject_lora

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

SPLIT_ROOT = "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_grouped/semantic_pathology"
CKPT = "/home/yr245/cls_runs/group_semantic_lora/best_model.pth"
ULTRASAM = "/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth"
OUT_DIR = Path("/home/yr245/cls_runs/group_semantic_lora")
OUT_DIR.mkdir(parents=True, exist_ok=True)

val_ds = JointClsDataset(
    SPLIT_ROOT,
    split="val",
    img_size=1024,
    train=False
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=(device == "cuda")
)

print("val samples:", len(val_ds))

ckpt = torch.load(CKPT, map_location="cpu")

if any(".linear.linear." in k for k in ckpt.keys()):
    lora_depth = 2
elif any(".linear." in k for k in ckpt.keys()):
    lora_depth = 1
else:
    lora_depth = 0

print("Detected LoRA depth:", lora_depth)

model = SharedUltraSamAdapterPrototype(
    checkpoint_path=ULTRASAM,
    task_to_num_classes=val_ds.task_to_num_classes,
    img_size=1024,
    prototypes_per_class=4,
    proto_dim=256,
    adapter_bottleneck=64,
    freeze_backbone=True,
)

for _ in range(lora_depth):
    model = inject_lora(model, r=8)

model.load_state_dict(ckpt, strict=False)
model.to(device)
model.eval()

rows = []

with torch.no_grad():
    for batch in val_loader:
        if isinstance(batch, dict):
            x = batch["image"]
            y = batch["label"]
            task_names = batch["task_name"]
        else:
            x, y, task_names = batch[:3]

        x = x.to(device)
        y = y.to(device)

        task_name = task_names[0] if isinstance(task_names, (list, tuple)) else str(task_names)

        logits = model(x, [task_name])[0]
        probs = F.softmax(logits, dim=0).cpu().numpy()

        row = {
            "task": task_name,
            "true": int(y.item()),
            "pred": int(np.argmax(probs)),
        }

        for i, p in enumerate(probs):
            row[f"prob_{i}"] = float(p)

        row["correct"] = int(row["true"] == row["pred"])
        rows.append(row)

pred_df = pd.DataFrame(rows)
pred_path = OUT_DIR / "val_predictions_with_probs.csv"
pred_df.to_csv(pred_path, index=False)

summary = []

for task, sub in pred_df.groupby("task"):
    n_classes = val_ds.task_to_num_classes[task]
    y_true = sub["true"].values
    y_score = sub[[f"prob_{i}" for i in range(n_classes)]].values

    acc = (sub["true"] == sub["pred"]).mean()

    try:
        if n_classes == 2:
            auroc = roc_auc_score(y_true, y_score[:, 1])
        else:
            auroc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except:
        auroc = np.nan

    summary.append({
        "task": task,
        "n": len(sub),
        "num_classes": n_classes,
        "accuracy": acc,
        "auroc": auroc,
    })

summary_df = pd.DataFrame(summary).sort_values("task")
summary_path = OUT_DIR / "auroc_summary.csv"
summary_df.to_csv(summary_path, index=False)

mean_auroc = summary_df["auroc"].mean()
weighted_auroc = np.average(
    summary_df["auroc"].dropna(),
    weights=summary_df.dropna(subset=["auroc"])["n"]
)

overall_acc = pred_df["correct"].mean()

print("\nPer-task AUROC:")
print(summary_df.to_string(index=False))

print("\n==============================")
print("Overall accuracy:", round(overall_acc, 4))
print("Mean AUROC:", round(mean_auroc, 4))
print("Weighted AUROC:", round(weighted_auroc, 4))
print("==============================")
print("Saved:", summary_path)
PY