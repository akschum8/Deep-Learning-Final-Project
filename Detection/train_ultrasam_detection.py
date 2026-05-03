"""
UltraSam detection training script.
"""

import os
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from tqdm import tqdm

from mmengine.runner.checkpoint import _load_checkpoint
from mmpretrain.models.backbones import ViTSAM

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    print("WARNING: torchmetrics not found. Install with: pip install torchmetrics")
    print("         mAP will not be computed; only loss will be tracked.")


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DetectionDataset(Dataset):

    def __init__(self, csv_path: str, img_size: int = 1024, transform=None):
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.img_size = img_size
        self.transform = transform

        df = pd.read_csv(csv_path)

        def resolve(p):
            p = Path(p)
            return str(p if p.is_absolute() else (self.csv_dir / p).resolve())

        df["resolved_path"] = df["image_path"].apply(resolve)

        missing = [p for p in df["resolved_path"].unique() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images not found. First: {missing[0]}"
            )

        all_samples = []
        for path, group in df.groupby("resolved_path", sort=False):
            boxes = group[["x_min", "y_min", "x_max", "y_max"]].values.astype(float)
            orig_h = float(group["height"].iloc[0])
            orig_w = float(group["width"].iloc[0])
            all_samples.append({
                "path": path,
                "boxes": boxes,   # (N, 4) in original pixel coords
                "orig_h": orig_h,
                "orig_w": orig_w,
            })

        self.samples = []
        n_dropped = 0
        for s in all_samples:
            boxes = s["boxes"].copy()
            scale_x = img_size / s["orig_w"]
            scale_y = img_size / s["orig_h"]
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_size)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if valid.any():
                self.samples.append(s)
            else:
                n_dropped += 1
        if n_dropped:
            print(f"Dropped {n_dropped} images with no valid boxes after rescaling")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["path"]).convert("RGB")

        scale_x = self.img_size / s["orig_w"]
        scale_y = self.img_size / s["orig_h"]
        boxes = s["boxes"].copy()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.img_size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.img_size)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.zeros(len(boxes), dtype=torch.int64),  
            "image_id": torch.tensor(idx),
        }


def detection_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    targets = [
        {"boxes": b["boxes"], "labels": b["labels"], "image_id": b["image_id"]}
        for b in batch
    ]
    return images, targets

class ViTSAMDetectionBackbone(nn.Module):
    out_channels: int = 256

    def __init__(self, checkpoint_path: str, img_size: int = 1024, freeze: bool = True):
        super().__init__()
        self.vitsam = ViTSAM(
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
        missing, unexpected = self.vitsam.load_state_dict(backbone_state, strict=False)
        print(f"Backbone loaded | missing={len(missing)} unexpected={len(unexpected)}")

        if freeze:
            for p in self.vitsam.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.vitsam(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        # feat: (B, 256, H/16, W/16) — single spatial scale
        return OrderedDict([("0", feat)])


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_seen = 0

    pbar = tqdm(loader, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        images_list = list(images)
        targets_list = [
            {
                "boxes": t["boxes"].to(device, non_blocking=True),
                "labels": t["labels"].to(device, non_blocking=True),
            }
            for t in targets
        ]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images_list, targets_list)
            losses = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        bs = len(images_list)
        total_loss += losses.item() * bs
        total_seen += bs
        pbar.set_postfix(loss=f"{total_loss / total_seen:.4f}")

    return total_loss / total_seen


@torch.no_grad()
def evaluate(model, loader, device, score_thresh=0.05):
    model.train()  # need train mode for loss computation
    total_loss = 0.0
    total_seen = 0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        images_list = list(images)
        targets_list = [
            {
                "boxes": t["boxes"].to(device, non_blocking=True),
                "labels": t["labels"].to(device, non_blocking=True),
            }
            for t in targets
        ]

        # Get loss
        loss_dict = model(images_list, targets_list)
        losses = sum(loss_dict.values())
        total_loss += losses.item() * len(images_list)
        total_seen += len(images_list)

        # Get predictions for mAP
        model.eval()
        preds = model(images_list)
        model.train()

        for pred, tgt in zip(preds, targets_list):
            all_preds.append({
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu(),
            })
            all_targets.append({
                "boxes": tgt["boxes"].cpu(),
                "labels": tgt["labels"].cpu(),
            })

    val_loss = total_loss / total_seen

    map_val = None
    if HAS_TORCHMETRICS and all_preds:
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metric.update(all_preds, all_targets)
        result = metric.compute()
        map_val = result["map"].item()

    return val_loss, map_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--checkpoint", default="UltraSam.pth")
    parser.add_argument("--output_dir", default="det_runs/run1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--num_classes", type=int, default=1,
                        help="Number of foreground classes (default 1 for single-object detection)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    norm = dict(
        mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
        std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
    )
    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(**norm),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(**norm),
    ])

    train_ds = DetectionDataset(args.train_csv, img_size=args.img_size, transform=transform_train)
    val_ds = DetectionDataset(args.val_csv, img_size=args.img_size, transform=transform_val)
    print(f"Train images: {len(train_ds)}  |  Val images: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=detection_collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build backbone
    backbone = ViTSAMDetectionBackbone(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        freeze=args.freeze_backbone,
    ).to(device)

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),  
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    # image_mean/std = 0/1 because we already normalize in the dataset.
    model = RetinaNet(
        backbone=backbone,
        num_classes=args.num_classes,
        anchor_generator=anchor_generator,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        min_size=args.img_size,
        max_size=args.img_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    best_map = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, map_val = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_map": map_val,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        improved = (map_val is not None and map_val > best_map) or \
                   (map_val is None and val_loss < -best_map)
        if improved:
            best_map = map_val if map_val is not None else -val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": args.num_classes,
                    "best_val_map": map_val,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )
            print(f"  Saved best model (mAP={map_val})")

    print(f"\nDone. Best val mAP = {best_map}")


if __name__ == "__main__":
    main()
