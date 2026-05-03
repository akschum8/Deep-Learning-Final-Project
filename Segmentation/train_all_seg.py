"""
train_all_seg.py
----------------
Trains a UltraSam segmentation model for every dataset in the splits manifest.
Each dataset gets its own model (num_classes is read per dataset from the manifest).
Results are saved to a JSON summary at the end.

Automatically skips datasets that already have a completed best_model.pt,
so you can resubmit after a timeout and resume from where it left off.

Usage:
    python train_all_seg.py \
        --manifest  .../splits/manifest.json \
        --checkpoint UltraSam.pth \
        --output_dir seg_runs/all \
        --epochs 5 \
        --batch_size 8 \
        --img_size 512 \
        --freeze_backbone
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from mmengine.runner.checkpoint import _load_checkpoint
from mmpretrain.models.backbones import ViTSAM


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SegDataset(Dataset):
    """
    CSV columns expected: image_path, mask_path
    Mask pixel values are class indices 0..(num_classes-1).
    """
    def __init__(self, csv_path, img_size=512):
        self.csv_path = Path(csv_path)
        self.csv_dir  = self.csv_path.parent
        self.img_size = img_size
        df = pd.read_csv(self.csv_path)

        self.image_paths = self._resolve(df["image_path"].tolist())
        self.mask_paths  = self._resolve(df["mask_path"].tolist())

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.675/255, 116.28/255, 103.53/255],
                std=[ 58.395/255,  57.12/255,  57.375/255],
            ),
        ])

    def _resolve(self, paths):
        out = []
        for p in paths:
            p = Path(p)
            out.append(str(p if p.is_absolute() else (self.csv_dir / p).resolve()))
        return out

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx]).convert("L")

        image = self.img_tf(image)
        mask  = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask  = torch.tensor(np.array(mask), dtype=torch.long)

        return {"image": image, "mask": mask}


# ─── Model ───────────────────────────────────────────────────────────────────

class SegDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=512):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, feat):
        x = self.conv(feat)
        return F.interpolate(x, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)


class UltraSamSeg(nn.Module):
    def __init__(self, checkpoint_path, num_classes,
                 freeze_backbone=True, img_size=512):
        super().__init__()
        self.img_size = img_size

        self.backbone = ViTSAM(
            arch="base", img_size=img_size, patch_size=16,
            out_channels=256, use_abs_pos=True,
            use_rel_pos=True, window_size=14,
        )

        ckpt = _load_checkpoint(checkpoint_path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)
        bsd  = {k[len("backbone."):]: v for k, v in sd.items()
                if k.startswith("backbone.")}
        missing, unexpected = self.backbone.load_state_dict(bsd, strict=False)
        print(f"  Backbone loaded | missing={len(missing)} unexpected={len(unexpected)}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Probe feature channels
        self.backbone.eval()
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, img_size, img_size))
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
        in_ch = feat.shape[1]

        self.decoder = SegDecoder(in_ch, num_classes=num_classes,
                                  img_size=img_size)

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        return self.decoder(feat)


# ─── Losses & Metrics ────────────────────────────────────────────────────────

def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    loss  = 0.0
    for c in range(num_classes):
        p = probs[:, c]
        t = (targets == c).float()
        intersection = (p * t).sum()
        loss += 1.0 - (2.0 * intersection + eps) / (p.sum() + t.sum() + eps)
    return loss / num_classes


def mean_dice(logits, targets, num_classes, eps=1e-6):
    """Mean Dice over foreground classes (skips background=0)."""
    pred = logits.argmax(dim=1)
    scores = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (targets == c).float()
        intersection = (p * t).sum()
        scores.append(((2 * intersection + eps) / (p.sum() + t.sum() + eps)).item())
    return float(np.mean(scores)) if scores else 0.0


# ─── Epoch ───────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, num_classes, train: bool):
    model.train(train)
    ce_fn = nn.CrossEntropyLoss()
    total_loss = total_dice = n = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss   = ce_fn(logits, masks) + dice_loss(logits, masks, num_classes)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs          = images.size(0)
        total_loss += loss.item() * bs
        total_dice += mean_dice(logits, masks, num_classes) * bs
        n          += bs

    return total_loss / n, total_dice / n


# ─── Evaluate on test set ────────────────────────────────────────────────────

def evaluate_test(model, test_csv, device, num_classes, img_size, batch_size, num_workers):
    ds     = SegDataset(test_csv, img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    ce_fn = nn.CrossEntropyLoss()
    total_loss = total_dice = n = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  test", leave=False):
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            logits = model(images)
            loss   = ce_fn(logits, masks) + dice_loss(logits, masks, num_classes)
            bs          = images.size(0)
            total_loss += loss.item() * bs
            total_dice += mean_dice(logits, masks, num_classes) * bs
            n          += bs

    return total_loss / n, total_dice / n


# ─── Train one dataset ───────────────────────────────────────────────────────

def train_one(entry, args, device):
    name        = entry["name"]
    num_classes = entry["num_classes"]
    run_dir     = Path(args.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset : {name}")
    print(f"  Classes : {num_classes}")
    print(f"  Train   : {entry['train']}  Val: {entry['val']}  Test: {entry['test']}")
    print(f"{'='*60}")

    train_ds = SegDataset(entry["train_csv"], img_size=args.img_size)
    val_ds   = SegDataset(entry["val_csv"],   img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = UltraSamSeg(
        checkpoint_path=args.checkpoint,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
        img_size=args.img_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )

    best_val_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, optimizer,
                                     device, num_classes, train=True)
        va_loss, va_dice = run_epoch(model, val_loader,   optimizer,
                                     device, num_classes, train=False)

        row = dict(epoch=epoch,
                   train_loss=round(tr_loss, 4), train_dice=round(tr_dice, 4),
                   val_loss=round(va_loss, 4),   val_dice=round(va_dice, 4))
        history.append(row)
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_dice={tr_dice:.4f}  "
              f"val_loss={va_loss:.4f}  val_dice={va_dice:.4f}")

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "num_classes": num_classes,
                "best_val_dice": best_val_dice,
            }, run_dir / "best_model.pt")
            print(f"    -> Saved best model  val_dice={best_val_dice:.4f}")

    # Test evaluation using best model
    print("  Loading best model for test evaluation...")
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_dice = evaluate_test(
        model, entry["test_csv"], device, num_classes,
        args.img_size, args.batch_size, args.num_workers,
    )
    print(f"  TEST  loss={test_loss:.4f}  dice={test_dice:.4f}")

    return {
        "name":          name,
        "num_classes":   num_classes,
        "best_val_dice": round(best_val_dice, 4),
        "test_loss":     round(test_loss, 4),
        "test_dice":     round(test_dice, 4),
        "run_dir":       str(run_dir),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",        required=True,
                        help="Path to splits/manifest.json from prepare_splits.py")
    parser.add_argument("--checkpoint",      default="UltraSam.pth")
    parser.add_argument("--output_dir",      default="seg_runs/all")
    parser.add_argument("--epochs",          type=int,   default=5)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--img_size",        type=int,   default=512)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"Found {len(manifest)} segmentation datasets in manifest.")

    # Load any existing results so already-completed datasets are preserved
    summary_path = Path(args.output_dir) / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            all_results = json.load(f)
        done_names = {r["name"] for r in all_results}
        print(f"Resuming — {len(done_names)} datasets already completed: "
              f"{', '.join(sorted(done_names))}\n")
    else:
        all_results = []
        done_names  = set()

    for entry in manifest:
        name    = entry["name"]
        run_dir = Path(args.output_dir) / name

        # ── Skip if already completed ────────────────────────────────────────
        if name in done_names and (run_dir / "best_model.pt").exists():
            print(f"  SKIP {name} — already completed")
            continue

        # ── Train ────────────────────────────────────────────────────────────
        result = train_one(entry, args, device)
        all_results.append(result)
        done_names.add(name)

        # Save running summary after each dataset completes
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print final table
    print(f"\n{'='*60}")
    print(f"{'Dataset':<45} {'Classes':>7} {'Val Dice':>9} {'Test Dice':>10}")
    print(f"{'-'*60}")
    for r in all_results:
        test_dice_str = f"{r['test_dice']:>10.4f}" if r['test_dice'] is not None else "        N/A"
        print(f"{r['name']:<45} {r['num_classes']:>7} "
              f"{r['best_val_dice']:>9.4f} {test_dice_str}")
    print(f"{'='*60}")
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
