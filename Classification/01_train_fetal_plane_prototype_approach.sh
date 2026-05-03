cd /home/yr245/project/ultrasam_project/UltraSam

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PYTHONPATH:."
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cat > make_cls_split.py <<'PY'
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def group_from_path(p):
    p = Path(str(p))
    stem = p.stem
    acq = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return str(p.parent) + "__" + acq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df["group_id"] = df["image_path"].apply(group_from_path)

    groups = df["group_id"].drop_duplicates()

    train_g, temp_g = train_test_split(
        groups,
        test_size=args.val_frac + args.test_frac,
        random_state=args.seed,
        shuffle=True
    )

    relative_test = args.test_frac / (args.val_frac + args.test_frac)

    val_g, test_g = train_test_split(
        temp_g,
        test_size=relative_test,
        random_state=args.seed,
        shuffle=True
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, group_set in [("train", train_g), ("val", val_g), ("test", test_g)]:
        split_df = df[df["group_id"].isin(group_set)].drop(columns=["group_id"])
        split_df.to_csv(out_dir / f"{name}.csv", index=False)
        print(name, len(split_df))
        print(split_df["mask"].value_counts().sort_index())

    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
PY

cat > train_cls_ultrasam_plus.py <<'PY'
import argparse
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_ultrasam_classification import build_model, load_ultrasam_checkpoint

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class UltrasoundClsDataset(Dataset):
    def __init__(self, csv_path, data_root, img_size=256, augment=False):
        self.df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)
        self.data_root = Path(data_root)
        self.csv_base = self.data_root / "csv_files"
        self.img_size = img_size
        self.augment = augment
        self.labels = sorted(self.df["mask"].dropna().astype(int).unique().tolist())
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def augment_image(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12, 12)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        if random.random() < 0.35:
            noise = np.random.normal(0, 8, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.35:
            speckle = np.random.normal(1.0, 0.08, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) * speckle, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = (self.csv_base / str(row["image_path"])).resolve()

        if not os.path.exists(img_path):
            raise FileNotFoundError(str(img_path))

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.augment:
            img = self.augment_image(img)

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        x = torch.from_numpy(img).permute(2, 0, 1).float()
        y = self.label_to_idx[int(row["mask"])]

        return x, torch.tensor(y).long(), str(img_path)

class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class UltraSamClassifier(nn.Module):
    def __init__(self, checkpoint, num_classes, head_type="mlp", freeze_mode="partial"):
        super().__init__()
        self.backbone = build_model()
        load_ultrasam_checkpoint(self.backbone, checkpoint)

        for p in self.backbone.parameters():
            p.requires_grad = False

        if freeze_mode == "partial":
            params = list(self.backbone.parameters())
            for p in params[int(0.8 * len(params)):]:
                p.requires_grad = True

        if freeze_mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True

        self.backbone.eval()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 256, 256)
            feat = self.extract_features(dummy)
            in_dim = feat.shape[1]

        if head_type == "linear":
            self.head = nn.Linear(in_dim, num_classes)
        else:
            self.head = MLPHead(in_dim, num_classes)

    def extract_features(self, x):
        out = self.backbone(x)

        if isinstance(out, (list, tuple)):
            out = out[-1]

        if out.ndim == 4:
            out = torch.nn.functional.adaptive_avg_pool2d(out, 1).flatten(1)

        if out.ndim > 2:
            out = out.flatten(1)

        return out

    def forward(self, x):
        features = self.extract_features(x)
        return self.head(features)

def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    criterion = nn.CrossEntropyLoss()

    losses = []
    labels = []
    preds = []

    for imgs, y, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        labels.extend(y.detach().cpu().numpy().tolist())
        preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return float(np.mean(losses)), acc, macro_f1, labels, preds

def save_confusion_matrix(labels, preds, out_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def export_failures(model, loader, device, out_csv, max_failures=200):
    model.eval()
    rows = []

    with torch.no_grad():
        for imgs, y, paths in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(1).cpu().numpy()
            conf = probs.max(1).values.cpu().numpy()
            true = y.numpy()

            for t, p, c, path in zip(true, pred, conf, paths):
                if t != p:
                    rows.append({
                        "path": path,
                        "true": int(t),
                        "pred": int(p),
                        "confidence": float(c)
                    })

                if len(rows) >= max_failures:
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                    return

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--data_root", default="/home/yr245/project/ultrasam_project/data_train7z/train")
    parser.add_argument("--checkpoint", default="UltraSam.pth")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze_mode", choices=["frozen", "partial", "full"], default="partial")
    parser.add_argument("--head_type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = UltrasoundClsDataset(args.train_csv, args.data_root, augment=True)
    val_ds = UltrasoundClsDataset(args.val_csv, args.data_root, augment=False)

    num_classes = len(train_ds.labels)

    print("Num classes:", num_classes)
    print("Train:", len(train_ds))
    print("Val:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = UltraSamClassifier(
        checkpoint=args.checkpoint,
        num_classes=num_classes,
        head_type=args.head_type,
        freeze_mode=args.freeze_mode
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in trainable))

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss, va_acc, va_f1, va_labels, va_preds = run_epoch(model, val_loader, optimizer, device, train=False)

        print(f"train loss={tr_loss:.4f} acc={tr_acc:.4f} macroF1={tr_f1:.4f}")
        print(f"val   loss={va_loss:.4f} acc={va_acc:.4f} macroF1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            save_confusion_matrix(va_labels, va_preds, out_dir / "val_confusion_matrix.png")

            with open(out_dir / "val_report.txt", "w") as f:
                f.write(classification_report(va_labels, va_preds))

            print(f"Saved best model with val_macro_f1={va_f1:.4f}")

    export_failures(model, val_loader, device, out_dir / "val_failures.csv")

    if args.test_csv:
        model.load_state_dict(torch.load(out_dir / "best_model.pth", map_location=device))

        test_ds = UltrasoundClsDataset(args.test_csv, args.data_root, augment=False)

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        te_loss, te_acc, te_f1, te_labels, te_preds = run_epoch(
            model,
            test_loader,
            optimizer,
            device,
            train=False
        )

        print(f"test  loss={te_loss:.4f} acc={te_acc:.4f} macroF1={te_f1:.4f}")

        save_confusion_matrix(te_labels, te_preds, out_dir / "test_confusion_matrix.png")

        with open(out_dir / "test_report.txt", "w") as f:
            f.write(classification_report(te_labels, te_preds))

        export_failures(model, test_loader, device, out_dir / "test_failures.csv")

if __name__ == "__main__":
    main()
PY

python make_cls_split.py \
  --csv "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Six_2.FETAL_PLANES_ZENODO_12,400.csv" \
  --out_dir "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits/fetal_plane"

rm -rf cls_runs/fetal_plane_partial_mlp

python train_cls_ultrasam_plus.py \
  --train_csv "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits/fetal_plane/train.csv" \
  --val_csv "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits/fetal_plane/val.csv" \
  --test_csv "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits/fetal_plane/test.csv" \
  --checkpoint UltraSam.pth \
  --output_dir cls_runs/fetal_plane_partial_mlp \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 5e-5 \
  --freeze_mode partial \
  --head_type mlp

cat cls_runs/fetal_plane_partial_mlp/test_report.txt
ls cls_runs/fetal_plane_partial_mlp
