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

cat > lora_utils.py <<'PY'
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear, r=8):
        super().__init__()
        self.linear = linear
        self.r = r

        in_f = linear.in_features
        out_f = linear.out_features

        self.A = nn.Parameter(torch.randn(in_f, r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, out_f) * 0.01)

    def forward(self, x):
        A = self.A.to(x.device)
        B = self.B.to(x.device)
        return self.linear(x) + x @ A @ B

def inject_lora(model, r=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, LoRALinear(module, r=r))
    return model
PY

python train_joint_adapter_prototype.py \
  --split_root "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_grouped/semantic_pathology" \
  --checkpoint "/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth" \
  --output_dir cls_runs/group_semantic_lora \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 5e-5 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256 \
  --adapter_bottleneck 64 \
  --steps_per_epoch 10000

python train_joint_adapter_prototype.py \
  --split_root "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_grouped/orientation_position" \
  --checkpoint "/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth" \
  --output_dir cls_runs/group_orientation_lora \
  --epochs 8 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 5e-5 \
  --img_size 1024 \
  --prototypes_per_class 8 \
  --proto_dim 256 \
  --adapter_bottleneck 128 \
  --steps_per_epoch 6000

python train_joint_adapter_prototype.py \
  --split_root "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_grouped/semantic_pathology" \
  --checkpoint "/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth" \
  --output_dir cls_runs/group_semantic_lora_proto8 \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 5e-5 \
  --img_size 1024 \
  --prototypes_per_class 8 \
  --proto_dim 256 \
  --adapter_bottleneck 64 \
  --steps_per_epoch 10000

python - <<'PY'
import pandas as pd
from pathlib import Path

full = Path("/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_all")
out = Path("/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_single_orientation")
out.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(full / "summary.csv")

for task_id in ["fetal_head_pos_cls", "fetal_sacral_pos_cls"]:
    sub = summary[summary["task_id"].astype(str).str.startswith(task_id)].copy()
    out_dir = out / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_dir / "summary.csv", index=False)

    print("\nSaved:", out_dir / "summary.csv")
    print(sub[["csv", "task_id", "train_n", "val_n", "num_classes"]].to_string(index=False))
PY

python train_joint_adapter_prototype.py \
  --split_root "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_single_orientation/fetal_head_pos_cls" \
  --checkpoint "/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth" \
  --output_dir cls_runs/single_fetal_head_lora \
  --epochs 10 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 5e-5 \
  --img_size 1024 \
  --prototypes_per_class 8 \
  --proto_dim 256 \
  --adapter_bottleneck 128 \
  --steps_per_epoch 6000

python train_joint_adapter_prototype.py \
  --split_root "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_single_orientation/fetal_sacral_pos_cls" \
  --checkpoint "/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth" \
  --output_dir cls_runs/single_fetal_sacral_lora \
  --epochs 10 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 5e-5 \
  --img_size 1024 \
  --prototypes_per_class 8 \
  --proto_dim 256 \
  --adapter_bottleneck 128 \
  --steps_per_epoch 3000

python - <<'PY'
import json
from pathlib import Path

runs = [
    "group_semantic_adapter",
    "group_semantic_taskaware",
    "group_semantic_lora",
    "group_semantic_lora_proto8",
    "group_orientation_lora",
    "single_fetal_head_lora",
    "single_fetal_sacral_lora",
]

base = Path("/home/yr245/cls_runs")

for run in runs:
    p = base / run
    if not p.exists():
        continue

    files = sorted(p.glob("metrics_epoch_*.json"))

    print("\n" + "=" * 80)
    print("RUN:", run)

    if not files:
        print("No metrics found")
        continue

    best_file = max(files, key=lambda f: json.load(open(f))["val_macro_avg"])
    best = json.load(open(best_file))

    print("BEST:", best_file.name)
    print("Best val_macro_avg:", round(best["val_macro_avg"], 4))
    print("Best val_loss:", round(best["val_loss"], 4))

    print("\nPer-task metrics:")
    for task, m in sorted(best["val"].items()):
        print(f"{task:60s} acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f} n={m['n']}")
PY

python - <<'PY'
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("/home/yr245/cls_runs/group_semantic_lora/val_predictions.csv")

tasks = {
    "Breast BUSI": ("breast_3cls__Cls-Three_2.BUSI_Dataset_780", 1),
    "Breast BUS-UCLM": ("breast_3cls__Cls-Three_1.BUS-UCLM_Dataset_683", 1),
    "Liver HCC": ("liver_lesion_2cls__Cls-Two_1.HCC_Hemangioma_5466", 0),
}

for name, (task, cancer_label) in tasks.items():
    sub = df[df["task"] == task].copy()

    missed = sub[(sub["true"] == cancer_label) & (sub["pred"] != cancer_label)]
    false_alarm = sub[(sub["true"] != cancer_label) & (sub["pred"] == cancer_label)]

    labels = sorted(sub["true"].unique())

    print("\n" + "=" * 80)
    print(name)
    print("n =", len(sub))
    print("labels =", labels)
    print("confusion matrix:")
    print(confusion_matrix(sub["true"], sub["pred"], labels=labels))
    print("\nclassification report:")
    print(classification_report(sub["true"], sub["pred"], digits=4))
    print("missed cancers =", len(missed))
    print("false cancer calls =", len(false_alarm))

    print("\nmissed cancer rows:")
    print(missed[["task", "true", "pred"]].head(20).to_string(index=True))
PY

python - <<'PY'
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms

from train_joint_adapter_prototype import SharedUltraSamAdapterPrototype
from lora_utils import inject_lora

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

out_dir = Path("/home/yr245/prototype_heatmaps_best_lora")
out_dir.mkdir(exist_ok=True)

base_csv_dir = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files")
split_root = Path("/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_all")

checkpoint = "/home/yr245/cls_runs/group_semantic_lora/best_model.pth"
ultrasam_checkpoint = "/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth"

task_to_num_classes = {
    "organ_cls__Cls-Six_1.USAnotAI-master_Dataset": 6,
    "fetal_plane_cls__Cls-Six_2.FETAL_PLANES_ZENODO_12_400": 6,
    "breast_3cls__Cls-Three_1.BUS-UCLM_Dataset_683": 3,
    "breast_3cls__Cls-Three_2.BUSI_Dataset_780": 3,
    "lung_disease_3cls__Cls-Three_3.LUIFA_Dataset": 3,
    "liver_lesion_2cls__Cls-Two_1.HCC_Hemangioma_5466": 2,
    "breast_2cls__Cls-Two_2.BUSI-WHU_Dataset_927": 2,
    "lung_2cls__Cls-Two_3.Lung_Dataset_2756": 2,
}

tasks_to_plot = {
    "breast_3cls__Cls-Three_2.BUSI_Dataset_780": {
        "val_csv": split_root / "breast_3cls__Cls-Three_2.BUSI_Dataset_780" / "val.csv",
        "label_names": {0: "benign", 1: "malignant", 2: "normal"},
        "n": 20,
    },
    "liver_lesion_2cls__Cls-Two_1.HCC_Hemangioma_5466": {
        "val_csv": split_root / "liver_lesion_2cls__Cls-Two_1.HCC_Hemangioma_5466" / "val.csv",
        "label_names": {0: "HCC", 1: "hemangioma"},
        "n": 20,
    },
}

ckpt = torch.load(checkpoint, map_location="cpu")

if any(".linear.linear." in k for k in ckpt.keys()):
    lora_depth = 2
elif any(".linear." in k for k in ckpt.keys()):
    lora_depth = 1
else:
    lora_depth = 0

print("Detected LoRA depth:", lora_depth)

model = SharedUltraSamAdapterPrototype(
    checkpoint_path=ultrasam_checkpoint,
    task_to_num_classes=task_to_num_classes,
    img_size=1024,
    prototypes_per_class=4,
    proto_dim=256,
    adapter_bottleneck=64,
    freeze_backbone=True,
)

for _ in range(lora_depth):
    model = inject_lora(model, r=8)

missing, unexpected = model.load_state_dict(ckpt, strict=False)
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

def prototype_heatmap(model, x, task_name):
    with torch.no_grad():
        fmap = model.extract_feature_map(x)
        key = model.task_keys[task_name]

        fmap_task = model.adapters[key](fmap)
        z = model.projections[key](fmap_task)
        z = F.normalize(z, dim=1)

        p = F.normalize(model.prototypes[key], dim=-1)
        sim = torch.einsum("bdhw,ckd->bckhw", z, p)

        proto_scores = sim.flatten(3).max(dim=-1).values
        logits = proto_scores.mean(dim=-1) * model.temperatures[key].clamp(1.0, 50.0)

        pred = int(logits.argmax(dim=1).item())
        heat = sim[0, pred].max(dim=0).values

        heat = heat.detach().cpu().numpy()
        heat = np.maximum(heat, 0)
        heat = heat / (heat.max() + 1e-8)

    return pred, heat

summary = []

for task_name, info in tasks_to_plot.items():
    val_csv = info["val_csv"]
    label_names = info["label_names"]
    n = info["n"]

    df = pd.read_csv(val_csv).head(n).copy()

    task_out = out_dir / task_name
    task_out.mkdir(exist_ok=True)

    print("\nTask:", task_name)
    print("Val CSV:", val_csv)

    for idx, row in df.iterrows():
        rel_path = Path(str(row["image_path"]))
        img_path = (base_csv_dir / rel_path).resolve()

        if not img_path.exists():
            print("Missing image:", img_path)
            continue

        true_label = int(row["mask"])

        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        pred, heat = prototype_heatmap(model, x, task_name)

        heat_img = Image.fromarray(np.uint8(heat * 255)).resize(img.size)
        heat_resized = np.array(heat_img) / 255.0

        true_name = label_names.get(true_label, str(true_label))
        pred_name = label_names.get(pred, str(pred))

        out_file = task_out / f"proto_heatmap_{idx:03d}_true{true_label}_pred{pred}.png"

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(img)
        ax[0].set_title(f"Original\nTrue={true_label} ({true_name})")
        ax[0].axis("off")

        ax[1].imshow(img)
        ax[1].imshow(heat_resized, cmap="jet", alpha=0.45)
        ax[1].set_title(f"Prototype heatmap\nPred={pred} ({pred_name})")
        ax[1].axis("off")

        plt.tight_layout()
        plt.savefig(out_file, bbox_inches="tight", dpi=200)
        plt.close()

        summary.append({
            "task": task_name,
            "idx": idx,
            "image_path": str(img_path),
            "true": true_label,
            "true_name": true_name,
            "pred": pred,
            "pred_name": pred_name,
            "correct": true_label == pred,
            "out_file": str(out_file),
        })

        print("Saved:", out_file)

pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)

print("\nDone.")
print("Output folder:", out_dir)
print("Summary:", out_dir / "summary.csv")
PY