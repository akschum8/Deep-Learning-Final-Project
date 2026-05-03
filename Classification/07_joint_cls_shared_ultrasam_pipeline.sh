cd /home/yr245/project/ultrasam_project/UltraSam

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cat > inspect_cls_tasks.py <<'PY'
import pandas as pd
from pathlib import Path

CSV_DIR = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files")

for csv in sorted(CSV_DIR.glob("Cls-*.csv")):
    df = pd.read_csv(csv, low_memory=False)
    print("\n" + "="*80)
    print(csv.name)
    print("rows:", len(df))
    print("task_id:", df["task_id"].unique().tolist() if "task_id" in df.columns else "missing")
    print("num_classes:", df["num_classes"].unique().tolist() if "num_classes" in df.columns else "missing")
    print("labels:")
    print(df["mask"].value_counts().sort_index())
    print("example paths:")
    for p in df["image_path"].head(3):
        print(" ", p)
PY

python inspect_cls_tasks.py

cat > make_all_cls_splits.py <<'PY'
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_DIR = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files")
OUT_ROOT = Path("/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_all")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x)

def group_id(p):
    p = Path(str(p))
    stem = p.stem
    acq = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return str(p.parent) + "__" + acq

summary = []

for csv in sorted(CSV_DIR.glob("Cls-*.csv")):
    df = pd.read_csv(csv, low_memory=False).copy()

    if "mask" not in df.columns or "image_path" not in df.columns:
        continue

    task_id = df["task_id"].iloc[0] if "task_id" in df.columns else csv.stem
    split_name = safe_name(task_id + "__" + csv.stem)
    out_dir = OUT_ROOT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df["group"] = df["image_path"].apply(group_id)

    train_parts = []
    val_parts = []

    ok = True
    for label in sorted(df["mask"].dropna().unique()):
        sub = df[df["mask"] == label].copy()
        groups = sub["group"].drop_duplicates()

        if len(groups) < 2 or len(sub) < 5:
            ok = False
            break

        train_g, val_g = train_test_split(groups, test_size=0.2, random_state=42)
        train_parts.append(sub[sub["group"].isin(train_g)])
        val_parts.append(sub[sub["group"].isin(val_g)])

    if not ok:
        train, val = train_test_split(
            df.drop(columns=["group"]),
            test_size=0.2,
            random_state=42,
            stratify=df["mask"],
        )
        split_type = "row_stratified"
        overlap = "NA"
    else:
        train_full = pd.concat(train_parts)
        val_full = pd.concat(val_parts)

        overlap = len(set(train_full["group"]) & set(val_full["group"]))
        train = train_full.drop(columns=["group"])
        val = val_full.drop(columns=["group"])
        split_type = "grouped"

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)

    summary.append({
        "csv": csv.name,
        "split_name": split_name,
        "split_type": split_type,
        "train_n": len(train),
        "val_n": len(val),
        "num_classes": int(df["mask"].nunique()),
        "group_overlap": overlap,
        "out_dir": str(out_dir),
    })

pd.DataFrame(summary).to_csv(OUT_ROOT / "summary.csv", index=False)
PY

python make_all_cls_splits.py

cat > train_joint_shared_ultrasam_prototype.py <<'PY'
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score

from mmengine.runner.checkpoint import _load_checkpoint
from mmpretrain.models.backbones import ViTSAM


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JointDataset(Dataset):
    def __init__(self, split_root, split="train", img_size=1024, train=False):
        summary = pd.read_csv(Path(split_root) / "summary.csv")
        base = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files")

        rows = []
        self.task_to_classes = {}

        for _, r in summary.iterrows():
            csv_path = Path(r["out_dir"]) / f"{split}.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if len(df) == 0:
                continue

            df["task"] = r["split_name"]
            df["label"] = df["mask"].astype(int)
            df["path"] = df["image_path"].apply(lambda p: str((base / str(p)).resolve()))

            self.task_to_classes[r["split_name"]] = int(df["label"].max()) + 1
            rows.append(df)

        self.df = pd.concat(rows).reset_index(drop=True)

        aug = []
        if train:
            aug = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]

        self.transform = transforms.Compose(
            aug + [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "label": int(r["label"]),
            "task": r["task"],
        }


class Model(nn.Module):
    def __init__(self, checkpoint, task_to_classes):
        super().__init__()

        self.backbone = ViTSAM(arch="base", img_size=1024, patch_size=16, out_channels=256)
        ckpt = _load_checkpoint(checkpoint, map_location="cpu")

        state = {k.replace("backbone.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(state, strict=False)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.proj = nn.Conv2d(256, 256, 1)
        self.heads = nn.ModuleDict()

        for t, c in task_to_classes.items():
            self.heads[t] = nn.Linear(256, c)

    def forward(self, x, tasks):
        feat = self.backbone(x)[-1]
        feat = self.proj(feat)
        feat = feat.mean([2,3])

        out = []
        for i, t in enumerate(tasks):
            out.append(self.heads[t](feat[i]))

        return out


def run(model, loader, opt, device, train=True):
    loss_fn = nn.CrossEntropyLoss()
    model.train(train)

    ys, ps = [], []

    for b in loader:
        x = b["image"].to(device)
        y = b["label"].to(device)
        t = b["task"]

        out = model(x, t)

        loss = torch.stack([
            loss_fn(out[i].unsqueeze(0), y[i].unsqueeze(0))
            for i in range(len(y))
        ]).mean()

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        ys += y.cpu().tolist()
        ps += [int(o.argmax()) for o in out]

    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root")
    ap.add_argument("--checkpoint")
    ap.add_argument("--output_dir")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    seed_everything()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = JointDataset(args.split_root, "train", train=True)
    val_ds = JointDataset(args.split_root, "val")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = Model(args.checkpoint, train_ds.task_to_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for e in range(args.epochs):
        tr = run(model, train_loader, opt, device, True)
        va = run(model, val_loader, opt, device, False)

        print({"epoch": e+1, "train": tr, "val": va})


if __name__ == "__main__":
    main()
PY

rm -rf cls_runs/joint_shared_prototype

python train_joint_shared_ultrasam_prototype.py \
  --split_root "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_all" \
  --checkpoint UltraSam.pth \
  --output_dir cls_runs/joint_shared_prototype \
  --epochs 5