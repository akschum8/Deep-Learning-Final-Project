"""
UltraSam regression training script.
"""

import ast
import os
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from mmengine.runner.checkpoint import _load_checkpoint
from mmpretrain.models.backbones import ViTSAM


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RegressionDataset(Dataset):

    def __init__(self, csv_path: str, transform=None):
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        self.point_cols = sorted(
            [c for c in self.df.columns if c.startswith("point_") and c.endswith("_xy")]
        )
        assert self.point_cols, "No point_N_xy columns found in CSV"
        self.num_points = len(self.point_cols)

        resolved = []
        for p in self.df["image_path"]:
            p = Path(p)
            resolved.append(str(p if p.is_absolute() else (self.csv_dir / p).resolve()))
        self.df["resolved_path"] = resolved

        missing = [p for p in self.df["resolved_path"] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images not found. First: {missing[0]}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["resolved_path"]).convert("RGB")
        orig_w, orig_h = image.size 

        coords = []
        for col in self.point_cols:
            xy = ast.literal_eval(row[col])  
            coords.extend([xy[0] / orig_w, xy[1] / orig_h])

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "coords": torch.tensor(coords, dtype=torch.float32),
            "orig_wh": torch.tensor([orig_w, orig_h], dtype=torch.float32),
        }


class UltraSamRegressor(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        num_outputs: int,
        freeze_backbone: bool = True,
        img_size: int = 1024,
    ):
        super().__init__()

        self.backbone = ViTSAM(
            arch="base",
            img_size=img_size,
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
        )

        ckpt = _load_checkpoint(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"Backbone loaded | missing={len(missing)} unexpected={len(unexpected)}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = self.backbone(dummy)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            feat_dim = feat.shape[1] if feat.ndim == 4 else feat.shape[-1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_outputs),
            nn.Sigmoid(),  
        )

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.ndim == 4:
            feat = feat.mean(dim=(2, 3))
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)
        return self.head(feat)


def mean_radial_error(pred_norm, gt_norm, orig_wh):
    B = pred_norm.shape[0]
    num_points = pred_norm.shape[1] // 2


    wh = orig_wh.unsqueeze(1).repeat(1, num_points, 1)  # (B, P, 2)
    pred_px = pred_norm.view(B, num_points, 2) * wh
    gt_px = gt_norm.view(B, num_points, 2) * wh

    dists = (pred_px - gt_px).pow(2).sum(dim=-1).sqrt()  # (B, num_points)
    return dists.mean().item()


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_mre = 0.0
    total_seen = 0

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        orig_wh = batch["orig_wh"] 

        with torch.set_grad_enabled(train):
            pred = model(images)
            loss = criterion(pred, coords)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = images.size(0)
        mre = mean_radial_error(pred.detach().cpu(), coords.cpu(), orig_wh)
        total_loss += loss.item() * bs
        total_mre += mre * bs
        total_seen += bs

        pbar.set_postfix(
            loss=f"{total_loss / total_seen:.4f}",
            mre=f"{total_mre / total_seen:.2f}px",
        )

    return total_loss / total_seen, total_mre / total_seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--checkpoint", default="UltraSam.pth")
    parser.add_argument("--output_dir", default="reg_runs/run1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    norm = dict(
        mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
        std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
    )
    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        # no horizontal flip; flipping changes x-coordinates of landmarks
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(**norm),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(**norm),
    ])

    train_ds = RegressionDataset(args.train_csv, transform=transform_train)
    val_ds = RegressionDataset(args.val_csv, transform=transform_val)

    if train_ds.num_points != val_ds.num_points:
        raise ValueError(
            f"Train/val point-count mismatch: {train_ds.num_points} vs {val_ds.num_points}"
        )

    num_outputs = train_ds.num_points * 2
    print(f"Points per image : {train_ds.num_points}")
    print(f"Head outputs     : {num_outputs}")
    print(f"Train size       : {len(train_ds)}")
    print(f"Val size         : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UltraSamRegressor(
        checkpoint_path=args.checkpoint,
        num_outputs=num_outputs,
        freeze_backbone=args.freeze_backbone,
        img_size=args.img_size,
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_mre = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_mre = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_loss, val_mre = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mre_px": train_mre,
            "val_loss": val_loss,
            "val_mre_px": val_mre,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if val_mre < best_val_mre:
            best_val_mre = val_mre
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_outputs": num_outputs,
                    "num_points": train_ds.num_points,
                    "best_val_mre_px": best_val_mre,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )
            print(f"  Saved best model (MRE = {best_val_mre:.2f} px)")

    print(f"\nDone. Best val MRE = {best_val_mre:.2f} px")


if __name__ == "__main__":
    main()
