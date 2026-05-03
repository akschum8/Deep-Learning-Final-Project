"""
prepare_splits.py
-----------------
Reads all Seg-*.csv files from the csv_files directory,
performs a 70/15/15 train/val/test split per CSV,
and writes the splits to a splits/ subdirectory.

Usage:
    python prepare_splits.py \
        --csv_dir /home/cpsc4520_oh84/project_cpsc4520/cpsc4520_oh84/ultrasam_project/data_train7z/train/csv_files \
        --out_dir /home/cpsc4520_oh84/project_cpsc4520/cpsc4520_oh84/ultrasam_project/data_train7z/train/splits
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(csv_dir: Path, out_dir: Path, seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Case-insensitive glob: collect all variations Seg-*, seg-*, SEG-*
    all_csvs = list(csv_dir.glob("*.csv"))
    seg_csvs = sorted([
        p for p in all_csvs
        if p.stem.lower().startswith("seg-")
    ])

    if not seg_csvs:
        raise FileNotFoundError(
            f"No Seg-*.csv files found in {csv_dir}\n"
            f"Files present: {[p.name for p in all_csvs[:10]]}"
        )

    print(f"Found {len(seg_csvs)} segmentation CSVs in {csv_dir}\n")

    manifest = []

    for csv_path in seg_csvs:
        try:
            df = pd.read_csv(csv_path)
            name = csv_path.stem

            # Validate required columns
            missing_cols = [c for c in ("image_path", "mask_path", "num_classes")
                            if c not in df.columns]
            if missing_cols:
                print(f"  SKIP {name} — missing columns: {missing_cols}")
                continue

            # Read num_classes from CSV
            num_classes = int(df["num_classes"].iloc[0])

            # 70 / 15 / 15 split
            train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
            val_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)

            # Save splits
            train_path = out_dir / f"{name}_train.csv"
            val_path   = out_dir / f"{name}_val.csv"
            test_path  = out_dir / f"{name}_test.csv"

            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path,     index=False)
            test_df.to_csv(test_path,   index=False)

            entry = {
                "name":        name,
                "num_classes": num_classes,
                "total":       len(df),
                "train":       len(train_df),
                "val":         len(val_df),
                "test":        len(test_df),
                "train_csv":   str(train_path),
                "val_csv":     str(val_path),
                "test_csv":    str(test_path),
            }
            manifest.append(entry)

            print(f"  OK  {name:<55}  classes={num_classes}  "
                  f"total={len(df):5d}  train={len(train_df):5d}  "
                  f"val={len(val_df):4d}  test={len(test_df):4d}")

        except Exception as e:
            print(f"  ERROR processing {csv_path.name}: {e}")
            continue

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {len(manifest)} datasets to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_dir",
        default="/home/cpsc4520_oh84/project_cpsc4520/cpsc4520_oh84/ultrasam_project/data_train7z/train/csv_files",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/cpsc4520_oh84/project_cpsc4520/cpsc4520_oh84/ultrasam_project/data_train7z/train/splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_splits(Path(args.csv_dir), Path(args.out_dir), seed=args.seed)


if __name__ == "__main__":
    main()
