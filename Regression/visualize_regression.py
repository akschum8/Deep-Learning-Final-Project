"""
GradCAM visualization for UltraSam regression models.
"""

import ast
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from mmengine.runner.checkpoint import _load_checkpoint
from mmpretrain.models.backbones import ViTSAM
from peft import get_peft_model, LoraConfig


class UltraSamRegressor(nn.Module):
    def __init__(self, checkpoint_path, num_outputs, freeze_backbone=True, img_size=1024):
        super().__init__()
        self.backbone = ViTSAM(
            arch="base", img_size=img_size, patch_size=16, out_channels=256,
            use_abs_pos=True, use_rel_pos=True, window_size=14,
        )
        ckpt = _load_checkpoint(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_state = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_state, strict=False)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(256, num_outputs), nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)): feat = feat[-1]
        if feat.ndim == 4: feat = feat.mean(dim=(2, 3))
        elif feat.ndim == 3: feat = feat.mean(dim=1)
        return self.head(feat)


class UltraSamRegressorLoRA(nn.Module):
    def __init__(self, checkpoint_path, num_outputs, lora_r=8, lora_alpha=16, img_size=1024):
        super().__init__()
        backbone = ViTSAM(
            arch="base", img_size=img_size, patch_size=16, out_channels=256,
            use_abs_pos=True, use_rel_pos=True, window_size=14,
        )
        ckpt = _load_checkpoint(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_state = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        backbone.load_state_dict(backbone_state, strict=False)
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["qkv", "proj"], lora_dropout=0.1, bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_config)
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(256, num_outputs), nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)): feat = feat[-1]
        if feat.ndim == 4: feat = feat.mean(dim=(2, 3))
        elif feat.ndim == 3: feat = feat.mean(dim=1)
        return self.head(feat)


def gradcam(model, image_tensor, num_outputs, device):
    model.eval()

    feat_captured = [None]

    def _hook(module, inp, out):
        feat_captured[0] = out

    backbone = model.backbone
    handle = backbone.register_forward_hook(_hook)

    with torch.no_grad():
        backbone(image_tensor)

    handle.remove()

    feat_map = feat_captured[0]

    if isinstance(feat_map, (list, tuple)):
        feat_map = feat_map[-1]

    if feat_map.ndim == 3:
        H = W = int(feat_map.shape[1] ** 0.5)
        feat_map = feat_map.permute(0, 2, 1).reshape(1, -1, H, W)

    feat_leaf = feat_map.detach().requires_grad_(True)  # (1, C, H, W)

    pooled = feat_leaf.mean(dim=(2, 3))          # (1, C)
    head_out = model.head(pooled)                # (1, num_outputs)

    cams = []
    for i in range(num_outputs):
        if feat_leaf.grad is not None:
            feat_leaf.grad.zero_()
        head_out[0, i].backward(retain_graph=True)

        grad = feat_leaf.grad                    # (1, C, H, W)
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * feat_leaf).sum(dim=1).squeeze(0).relu()  # (H, W)

        # Normalise to [0, 1]
        cam = cam.detach().cpu().numpy().astype(float)
        vmin, vmax = cam.min(), cam.max()
        if vmax > vmin:
            cam = (cam - vmin) / (vmax - vmin)
        cams.append(cam)

    return cams   # list of (H, W) arrays in [0, 1]


def overlay_heatmap(ax, image_np, cam, alpha=0.45):
    from PIL import Image as PILImage
    h, w = image_np.shape[:2]
    cam_resized = np.array(
        PILImage.fromarray((cam * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR)
    ) / 255.0
    ax.imshow(image_np)
    ax.imshow(cam_resized, cmap="jet", alpha=alpha, vmin=0, vmax=1)


def draw_landmarks(ax, coords_norm, orig_wh, img_size, color, marker, label, size=80):
    W, H = orig_wh
    n = len(coords_norm) // 2
    xs = [coords_norm[2*i]   * W for i in range(n)]
    ys = [coords_norm[2*i+1] * H for i in range(n)]
    ax.scatter(xs, ys, c=color, marker=marker, s=size, zorder=5,
               edgecolors="white", linewidths=1.2, label=label)


def load_frozen_model(frozen_ckpt_path, backbone_ckpt_path, device):
    ckpt = torch.load(frozen_ckpt_path, map_location="cpu")
    num_outputs = ckpt["num_outputs"]
    model = UltraSamRegressor(
        checkpoint_path=backbone_ckpt_path,
        num_outputs=num_outputs,
        freeze_backbone=True,
    )
    model_sd = model.state_dict()
    for k, v in ckpt["model_state_dict"].items():
        if k in model_sd:
            model_sd[k] = v
    model.load_state_dict(model_sd)
    return model.to(device).eval(), num_outputs


def load_lora_model(lora_ckpt_path, backbone_ckpt_path, device):
    ckpt = torch.load(lora_ckpt_path, map_location="cpu")
    num_outputs = ckpt["num_outputs"]
    model = UltraSamRegressorLoRA(
        checkpoint_path=backbone_ckpt_path,
        num_outputs=num_outputs,
    )
    model_sd = model.state_dict()
    for k, v in ckpt["model_state_dict"].items():
        if k in model_sd:
            model_sd[k] = v
    model.load_state_dict(model_sd, strict=False)
    return model.to(device).eval(), num_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv",       required=True)
    parser.add_argument("--frozen_ckpt",   required=True)
    parser.add_argument("--lora_ckpt",     required=True)
    parser.add_argument("--backbone_ckpt", default="UltraSam/UltraSam.pth")
    parser.add_argument("--output_dir",    default="visualizations")
    parser.add_argument("--num_images",    type=int, default=8)
    parser.add_argument("--img_size",      type=int, default=1024)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading frozen model...")
    frozen_model, num_outputs = load_frozen_model(
        args.frozen_ckpt, args.backbone_ckpt, device
    )
    print("Loading LoRA model...")
    lora_model, _ = load_lora_model(
        args.lora_ckpt, args.backbone_ckpt, device
    )
    num_points = num_outputs // 2

    csv_path = Path(args.val_csv)
    csv_dir  = csv_path.parent
    df = pd.read_csv(args.val_csv)
    point_cols = sorted([c for c in df.columns if c.startswith("point_") and c.endswith("_xy")])

    def resolve(p):
        p = Path(p)
        return str(p if p.is_absolute() else (csv_dir / p).resolve())
    df["resolved_path"] = df["image_path"].apply(resolve)

    norm = dict(
        mean=[123.675/255, 116.28/255, 103.53/255],
        std=[58.395/255,  57.12/255,  57.375/255],
    )
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(**norm),
    ])

    np.random.seed(args.seed)
    indices = np.linspace(0, len(df)-1, args.num_images, dtype=int)


    for fig_idx, row_idx in enumerate(indices):
        row = df.iloc[row_idx]
        img_path = row["resolved_path"]

        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        gt_coords = []
        for col in point_cols:
            xy = ast.literal_eval(row[col])
            gt_coords.extend([xy[0]/orig_w, xy[1]/orig_h])
        gt_coords = np.array(gt_coords)

        display_img = np.array(pil_img.resize((args.img_size, args.img_size)))

        with torch.no_grad():
            frozen_pred = frozen_model(img_tensor).squeeze(0).cpu().numpy()
            lora_pred   = lora_model(img_tensor).squeeze(0).cpu().numpy()

        frozen_cams = gradcam(frozen_model, img_tensor, num_outputs, device)
        lora_cams   = gradcam(lora_model,   img_tensor, num_outputs, device)

        frozen_cam_avg = np.mean(frozen_cams, axis=0)
        lora_cam_avg   = np.mean(lora_cams,   axis=0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        disp_wh = (args.img_size, args.img_size)

        # Col 0: original + GT
        axes[0].imshow(display_img)
        draw_landmarks(axes[0], gt_coords, disp_wh, args.img_size,
                       color="lime", marker="*", label="GT", size=350)
        axes[0].set_title("Original + Ground Truth", fontsize=12)
        axes[0].legend(loc="upper right", fontsize=9)
        axes[0].axis("off")

        # Col 1: frozen model
        overlay_heatmap(axes[1], display_img, frozen_cam_avg)
        draw_landmarks(axes[1], gt_coords,     disp_wh, args.img_size,
                       color="lime",   marker="*", label="GT",          size=350)
        draw_landmarks(axes[1], frozen_pred,   disp_wh, args.img_size,
                       color="red",    marker="o", label="Frozen pred", size=80)
        frozen_mre = np.mean([
            np.sqrt(((frozen_pred[2*i]   - gt_coords[2*i])   * orig_w)**2 +
                    ((frozen_pred[2*i+1] - gt_coords[2*i+1]) * orig_h)**2)
            for i in range(num_points)
        ])
        axes[1].set_title(f"Frozen backbone  |  MRE = {frozen_mre:.1f} px", fontsize=12)
        axes[1].legend(loc="upper right", fontsize=9)
        axes[1].axis("off")

        # Col 2: LoRA model
        overlay_heatmap(axes[2], display_img, lora_cam_avg)
        draw_landmarks(axes[2], gt_coords,   disp_wh, args.img_size,
                       color="lime",   marker="*", label="GT",         size=350)
        draw_landmarks(axes[2], lora_pred,   disp_wh, args.img_size,
                       color="orange", marker="o", label="LoRA pred",  size=80)
        lora_mre = np.mean([
            np.sqrt(((lora_pred[2*i]   - gt_coords[2*i])   * orig_w)**2 +
                    ((lora_pred[2*i+1] - gt_coords[2*i+1]) * orig_h)**2)
            for i in range(num_points)
        ])
        axes[2].set_title(f"LoRA fine-tuned  |  MRE = {lora_mre:.1f} px", fontsize=12)
        axes[2].legend(loc="upper right", fontsize=9)
        axes[2].axis("off")

        plt.suptitle(f"GradCAM – {Path(img_path).name}", fontsize=11, y=1.01)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"gradcam_{fig_idx:02d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}  |  frozen MRE={frozen_mre:.1f}  lora MRE={lora_mre:.1f}")

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6*n))
    if n == 1:
        axes = axes[None, :]   

    for fig_idx, row_idx in enumerate(indices):
        out_path = os.path.join(args.output_dir, f"gradcam_{fig_idx:02d}.png")
        img = plt.imread(out_path)
        for col in range(3):
            axes[fig_idx, col].imshow(img[:, col * img.shape[1]//3 : (col+1) * img.shape[1]//3])
            axes[fig_idx, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "gradcam_summary.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSummary grid saved to {args.output_dir}/gradcam_summary.png")


if __name__ == "__main__":
    main()
