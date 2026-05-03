"""
Splits a single-file detection CSV into train/val by unique image.
Splitting by image prevents the same image appearing in both sets.
"""

import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv",       required=True)
parser.add_argument("--train_out", required=True)
parser.add_argument("--val_out",   required=True)
parser.add_argument("--val_frac",  type=float, default=0.2)
parser.add_argument("--seed",      type=int,   default=42)
args = parser.parse_args()

df = pd.read_csv(args.csv)
unique_images = df["image_path"].unique()

np.random.seed(args.seed)
np.random.shuffle(unique_images)

n_val = max(1, int(len(unique_images) * args.val_frac))
val_images   = set(unique_images[:n_val])
train_images = set(unique_images[n_val:])

df[df["image_path"].isin(train_images)].to_csv(args.train_out, index=False)
df[df["image_path"].isin(val_images)].to_csv(args.val_out,   index=False)

print(f"Total unique images : {len(unique_images)}")
print(f"Train images        : {len(train_images)}")
print(f"Val images          : {len(val_images)}")
print(f"Saved → {args.train_out}")
print(f"Saved → {args.val_out}")
