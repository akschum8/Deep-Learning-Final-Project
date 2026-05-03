cd /home/yr245/project/ultrasam_project/UltraSam

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cat > train_joint_adapter_prototype.py <<'PY'
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
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score, classification_report

from mmengine.runner.checkpoint import _load_checkpoint
from mmpretrain.models.backbones import ViTSAM


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_key(x):
    return str(x).replace(".", "_").replace("-", "_").replace("/", "_").replace(",", "_")


class JointClsDataset(Dataset):
    def __init__(self, split_root, split="train", img_size=1024, train=False):
        self.split_root = Path(split_root)
        self.summary = pd.read_csv(self.split_root / "summary.csv")
        self.csv_base = Path("/home/yr245/project/ultrasam_project/data_train7z/train/csv_files")

        rows = []
        self.task_to_num_classes = {}
        self.task_to_indices = {}

        for _, r in self.summary.iterrows():
            task = r["split_name"]
            csv_path = Path(r["out_dir"]) / f"{split}.csv"

            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path, low_memory=False)

            if len(df) == 0:
                continue

            df["task_name"] = task
            df["label"] = df["mask"].astype(int)

            resolved = []
            for p in df["image_path"]:
                p = Path(str(p))
                if p.is_absolute():
                    resolved.append(str(p))
                else:
                    resolved.append(str((self.csv_base / p).resolve()))

            df["resolved_image_path"] = resolved

            self.task_to_num_classes[task] = int(df["label"].max()) + 1
            rows.append(df)

        self.df = pd.concat(rows, ignore_index=True).reset_index(drop=True)

        for task in self.df["task_name"].unique():
            self.task_to_indices[task] = self.df.index[self.df["task_name"] == task].tolist()

        aug = []
        if train:
            aug += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]

        self.transform = transforms.Compose(
            aug + [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        print(f"{split} samples:", len(self.df))
        print(f"{split} tasks:", len(self.task_to_num_classes))

        for k, v in self.task_to_num_classes.items():
            print(" ", k, "classes:", v, "samples:", len(self.task_to_indices.get(k, [])))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        image = Image.open(r["resolved_image_path"]).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
            "task_name": r["task_name"],
            "path": r["resolved_image_path"],
        }


class TaskBalancedSampler(Sampler):
    def __init__(self, dataset, steps_per_epoch=None, seed=42):
        self.dataset = dataset
        self.tasks = list(dataset.task_to_indices.keys())
        self.steps_per_epoch = steps_per_epoch or len(dataset)
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)

        for _ in range(self.steps_per_epoch):
            task = rng.choice(self.tasks)
            yield rng.choice(self.dataset.task_to_indices[task])

    def __len__(self):
        return self.steps_per_epoch


class TaskAdapter(nn.Module):
    def __init__(self, channels=256, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SharedUltraSamAdapterPrototype(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        task_to_num_classes,
        img_size=1024,
        prototypes_per_class=4,
        proto_dim=256,
        adapter_bottleneck=64,
        freeze_backbone=True,
        task_aware=False,
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

        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_state[k[len("backbone."):]] = v

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)

        print(f"Loaded shared UltraSam backbone from {checkpoint_path}")
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.task_aware = task_aware
        self.task_to_num_classes = task_to_num_classes
        self.task_keys = {task: safe_key(task) for task in task_to_num_classes}

        if self.task_aware:
            self.task_list = list(task_to_num_classes.keys())
            self.task_to_index = {task: i for i, task in enumerate(self.task_list)}
            self.task_embedding = nn.Embedding(len(self.task_list), 256)
            self.task_film = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
            )

        self.adapters = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.prototypes = nn.ParameterDict()
        self.temperatures = nn.ParameterDict()

        for task, num_classes in task_to_num_classes.items():
            key = self.task_keys[task]

            self.adapters[key] = TaskAdapter(channels=256, bottleneck=adapter_bottleneck)
            self.projections[key] = nn.Conv2d(256, proto_dim, kernel_size=1)

            self.prototypes[key] = nn.Parameter(
                torch.randn(num_classes, prototypes_per_class, proto_dim) * 0.02
            )

            self.temperatures[key] = nn.Parameter(torch.tensor(10.0))

    def extract_feature_map(self, x):
        feat = self.backbone(x)

        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        if feat.ndim == 4:
            return feat

        if feat.ndim == 3:
            b, n, d = feat.shape
            h = w = int(n ** 0.5)

            if h * w != n:
                raise RuntimeError(f"Cannot reshape tokens: {feat.shape}")

            return feat.transpose(1, 2).reshape(b, d, h, w)

        raise RuntimeError(f"Unexpected feature shape: {feat.shape}")

    def apply_task_conditioning(self, fmap, task):
        if not self.task_aware:
            return fmap

        task_idx = torch.tensor(
            [self.task_to_index[task]],
            device=fmap.device,
            dtype=torch.long,
        )

        emb = self.task_embedding(task_idx)
        gamma_beta = self.task_film(emb).view(1, 512, 1, 1)
        gamma, beta = gamma_beta[:, :256], gamma_beta[:, 256:]

        return fmap * (1.0 + 0.1 * gamma) + 0.1 * beta

    def logits_for_task(self, fmap, task):
        key = self.task_keys[task]

        fmap = self.apply_task_conditioning(fmap, task)
        fmap = self.adapters[key](fmap)

        z = self.projections[key](fmap)
        z = F.normalize(z, dim=1)

        p = F.normalize(self.prototypes[key], dim=-1)

        sim = torch.einsum("bdhw,ckd->bckhw", z, p)
        proto_scores = sim.flatten(3).max(dim=-1).values
        logits = proto_scores.mean(dim=-1) * self.temperatures[key].clamp(1.0, 50.0)

        return logits

    def forward(self, x, task_names):
        fmap = self.extract_feature_map(x)
        output_logits = [None] * len(task_names)

        for task in sorted(set(task_names)):
            idxs = [i for i, t in enumerate(task_names) if t == task]
            sub_fmap = fmap[idxs]
            logits = self.logits_for_task(sub_fmap, task)

            for j, idx in enumerate(idxs):
                output_logits[idx] = logits[j]

        return output_logits


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_n = 0

    task_true = {}
    task_pred = {}

    for batch in tqdm(loader, leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        task_names = list(batch["task_name"])

        with torch.set_grad_enabled(train):
            logits_list = model(x, task_names)

            losses = []
            preds = []

            for i, logits in enumerate(logits_list):
                loss = loss_fn(logits.unsqueeze(0), y[i].unsqueeze(0))
                losses.append(loss)
                preds.append(int(logits.argmax().detach().cpu()))

            loss = torch.stack(losses).mean()

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu()) * len(y)
        total_n += len(y)

        for i, task in enumerate(task_names):
            task_true.setdefault(task, []).append(int(y[i].detach().cpu()))
            task_pred.setdefault(task, []).append(preds[i])

    metrics = {}
    for task in task_true:
        metrics[task] = {
            "acc": accuracy_score(task_true[task], task_pred[task]),
            "macro_f1": f1_score(task_true[task], task_pred[task], average="macro", zero_division=0),
            "n": len(task_true[task]),
        }

    return total_loss / max(total_n, 1), metrics, task_true, task_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--checkpoint", default="UltraSam.pth")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--prototypes_per_class", type=int, default=4)
    parser.add_argument("--proto_dim", type=int, default=256)
    parser.add_argument("--adapter_bottleneck", type=int, default=64)
    parser.add_argument("--steps_per_epoch", type=int, default=13012)
    parser.add_argument("--unfreeze_backbone", action="store_true")
    parser.add_argument("--task_aware", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = JointClsDataset(args.split_root, split="train", img_size=args.img_size, train=True)
    val_ds = JointClsDataset(args.split_root, split="val", img_size=args.img_size, train=False)

    sampler = TaskBalancedSampler(
        train_ds,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SharedUltraSamAdapterPrototype(
        checkpoint_path=args.checkpoint,
        task_to_num_classes=train_ds.task_to_num_classes,
        img_size=args.img_size,
        prototypes_per_class=args.prototypes_per_class,
        proto_dim=args.proto_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        freeze_backbone=not args.unfreeze_backbone,
        task_aware=args.task_aware,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable))

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    best_macro = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        tr_loss, tr_metrics, _, _ = run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss, va_metrics, va_true, va_pred = run_epoch(model, val_loader, optimizer, device, train=False)

        macro_avg = float(np.mean([m["macro_f1"] for m in va_metrics.values()]))

        print("train_loss:", tr_loss)
        print("val_loss:", va_loss)
        print("val_macro_avg:", macro_avg)

        for task, m in sorted(va_metrics.items()):
            print(f"{task}: acc={m['acc']:.4f}, macroF1={m['macro_f1']:.4f}, n={m['n']}")

        with open(out_dir / f"metrics_epoch_{epoch}.json", "w") as f:
            json.dump(
                {
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "train": tr_metrics,
                    "val": va_metrics,
                    "val_macro_avg": macro_avg,
                },
                f,
                indent=2,
            )

        if macro_avg > best_macro:
            best_macro = macro_avg
            torch.save(model.state_dict(), out_dir / "best_model.pth")

            rows = []

            for task in va_true:
                for true, pred in zip(va_true[task], va_pred[task]):
                    rows.append({"task": task, "true": true, "pred": pred})

                with open(out_dir / f"report_{safe_key(task)}.txt", "w") as f:
                    f.write(classification_report(
                        va_true[task],
                        va_pred[task],
                        digits=3,
                        zero_division=0,
                    ))

            pd.DataFrame(rows).to_csv(out_dir / "val_predictions.csv", index=False)
            print("Saved best model")

    print("Done:", out_dir)


if __name__ == "__main__":
    main()
PY

python -m py_compile train_joint_adapter_prototype.py

rm -rf cls_runs/joint_adapter_prototype_frozen

python train_joint_adapter_prototype.py \
  --split_root "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_all" \
  --checkpoint "/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth" \
  --output_dir cls_runs/joint_adapter_prototype_frozen \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256 \
  --adapter_bottleneck 64 \
  --steps_per_epoch 13012

rm -rf cls_runs/joint_taskaware_prototype_frozen

python train_joint_adapter_prototype.py \
  --split_root "/home/yr245/project/ultrasam_project/data_train7z/train/classification_splits_all" \
  --checkpoint "/home/yr245/project/ultrasam_project/UltraSam/UltraSam.pth" \
  --output_dir cls_runs/joint_taskaware_prototype_frozen \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 1 \
  --lr 1e-4 \
  --img_size 1024 \
  --prototypes_per_class 4 \
  --proto_dim 256 \
  --adapter_bottleneck 64 \
  --steps_per_epoch 13012 \
  --task_aware

python - <<'PY'
import json
from pathlib import Path

for run_name in [
    "joint_adapter_prototype_frozen",
    "joint_taskaware_prototype_frozen",
]:
    run_dir = Path("cls_runs") / run_name
    files = sorted(run_dir.glob("metrics_epoch_*.json"))

    best_file = max(files, key=lambda f: json.load(open(f))["val_macro_avg"])
    best = json.load(open(best_file))

    print("\nRUN:", run_name)
    print("BEST FILE:", best_file)
    print("BEST VAL MACRO-F1:", best["val_macro_avg"])

    for task, m in sorted(best["val"].items()):
        print(f"{task}: acc={m['acc']:.4f}, macroF1={m['macro_f1']:.4f}, n={m['n']}")
PY