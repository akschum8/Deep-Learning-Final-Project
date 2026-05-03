cat > make_csv_split.py <<'PY'
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("/home/yr245/project/ultrasam_project/data_train7z/train")
CSV_DIR = ROOT / "csv_files"
OUT_DIR = ROOT / "csv_files_split"
OUT_DIR.mkdir(exist_ok=True)

all_rows = []

for csv_file in sorted(CSV_DIR.glob("*.csv")):
    df = pd.read_csv(csv_file)

    df["source_csv"] = csv_file.name

splits.
    df["group_id"] = df["task_id"].astype(str) + "__" + csv_file.stem

    groups = df["group_id"].drop_duplicates()

    if len(groups) < 3:
        idx = df.index.to_series()
        train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42, shuffle=True)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, shuffle=True)

        df["split"] = "none"
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"
    else:
        train_g, temp_g = train_test_split(groups, test_size=0.30, random_state=42, shuffle=True)
        val_g, test_g = train_test_split(temp_g, test_size=0.50, random_state=42, shuffle=True)

        df["split"] = "none"
        df.loc[df["group_id"].isin(train_g), "split"] = "train"
        df.loc[df["group_id"].isin(val_g), "split"] = "val"
        df.loc[df["group_id"].isin(test_g), "split"] = "test"

    all_rows.append(df)

manifest = pd.concat(all_rows, ignore_index=True)
manifest.to_csv(ROOT / "split_manifest_all.csv", index=False)

for split in ["train", "val", "test"]:
    out = manifest[manifest["split"] == split].drop(columns=["split"])
    out.to_csv(OUT_DIR / f"{split}.csv", index=False)

print("Saved:")
print(ROOT / "split_manifest_all.csv")
print(OUT_DIR / "train.csv")
print(OUT_DIR / "val.csv")
print(OUT_DIR / "test.csv")

print("\nRows per split:")
print(manifest["split"].value_counts())

print("\nRows per split/task:")
print(manifest.groupby(["split", "task_id"]).size().head(50))
PY

python make_csv_split.py




python - <<'PY'
import pandas as pd
from pathlib import Path

ROOT = Path("data_train7z/train")
df = pd.read_csv(ROOT / "split_manifest_all.csv")

print(df["split"].value_counts())

for s in ["train", "val", "test"]:
    x = df[df["split"] == s]
    print("\n", s)
    print("rows:", len(x))
    print("tasks:", x["task_id"].nunique())
    print(x["task_id"].value_counts().head(10))
PY




python - <<'PY'
import pandas as pd
from pathlib import Path

df = pd.read_csv(
    "data_train7z/train/split_manifest_all.csv",
    low_memory=False
)

print(df.columns.tolist())
print("\nExample image paths:")
for p in df["image_path"].head(30):
    print(p)

print("\nPath parts examples:")
for p in df["image_path"].head(10):
    print(Path(str(p)).parts)
PY





python - <<'PY'
import pandas as pd

df = pd.read_csv("data_train7z/train/split_manifest_all.csv", low_memory=False)

dup = df.groupby("image_path")["split"].nunique()
bad = dup[dup > 1]

print("Images appearing in multiple splits:", len(bad))

if len(bad) > 0:
    print(bad.head(20))
    leaked = df[df["image_path"].isin(bad.index)]
    print(leaked[["image_path", "task_id", "source_csv", "split"]].head(30))
PY




cat > make_csv_split_grouped.py <<'PY'
from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("/home/yr245/project/ultrasam_project/data_train7z/train")
CSV_DIR = ROOT / "csv_files"
OUT_DIR = ROOT / "csv_files_split_grouped"
OUT_DIR.mkdir(exist_ok=True)

def make_group_id(row):
    p = str(row["image_path"])
    stem = Path(p).stem

    # group video/frame-like files: xxx_70, xxx_180, etc.
    stem_base = re.sub(r"_\d+$", "", stem)

    return f'{row["task_id"]}__{row["source_csv"]}__{Path(p).parent}__{stem_base}'

def assign_split_by_group(df):
    groups = df["group_id"].drop_duplicates()
    n = len(groups)

    df["split"] = "train"

    if n < 3:
        return df

    if n == 3:
        shuffled = groups.sample(frac=1, random_state=42).tolist()
        df.loc[df["group_id"] == shuffled[0], "split"] = "train"
        df.loc[df["group_id"] == shuffled[1], "split"] = "val"
        df.loc[df["group_id"] == shuffled[2], "split"] = "test"
        return df

    train_g, temp_g = train_test_split(
        groups, test_size=0.30, random_state=42, shuffle=True
    )

    if len(temp_g) < 2:
        df.loc[df["group_id"].isin(train_g), "split"] = "train"
        df.loc[df["group_id"].isin(temp_g), "split"] = "val"
        return df

    val_g, test_g = train_test_split(
        temp_g, test_size=0.50, random_state=42, shuffle=True
    )

    df.loc[df["group_id"].isin(train_g), "split"] = "train"
    df.loc[df["group_id"].isin(val_g), "split"] = "val"
    df.loc[df["group_id"].isin(test_g), "split"] = "test"

    return df

all_rows = []

for csv_file in sorted(CSV_DIR.glob("*.csv")):
    df = pd.read_csv(csv_file, low_memory=False)
    df["source_csv"] = csv_file.name
    df["group_id"] = df.apply(make_group_id, axis=1)
    df = assign_split_by_group(df)
    all_rows.append(df)

manifest = pd.concat(all_rows, ignore_index=True)
manifest.to_csv(ROOT / "split_manifest_grouped.csv", index=False)

for split in ["train", "val", "test"]:
    out = manifest[manifest["split"] == split].drop(columns=["split"])
    out.to_csv(OUT_DIR / f"{split}.csv", index=False)

print("Rows per split:")
print(manifest["split"].value_counts())

print("\nGroups per split:")
print(manifest.groupby("split")["group_id"].nunique())

print("\nSaved to:")
print(OUT_DIR)
PY

python make_csv_split_grouped.py




python - <<'PY'
import pandas as pd

df = pd.read_csv("data_train7z/train/split_manifest_grouped.csv", low_memory=False)

for col in ["image_path", "group_id"]:
    leak = df.groupby(col)["split"].nunique()
    bad = leak[leak > 1]
    print(f"{col} leaking across splits:", len(bad))

print("\nRows:")
print(df["split"].value_counts())

print("\nTasks per split:")
print(df.groupby("split")["task_id"].nunique())
PY




cat > make_csv_split_global_grouped.py <<'PY'
from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("/home/yr245/project/ultrasam_project/data_train7z/train")
CSV_DIR = ROOT / "csv_files"
OUT_DIR = ROOT / "csv_files_split_global_grouped"
OUT_DIR.mkdir(exist_ok=True)

def make_group_id(p):
    p = str(p)
    path = Path(p)
    stem = path.stem

    # group frame-like images: xxx_70, xxx_180 -> xxx
    stem_base = re.sub(r"_\d+$", "", stem)

    return f"{path.parent}__{stem_base}"

all_rows = []

for csv_file in sorted(CSV_DIR.glob("*.csv")):
    df = pd.read_csv(csv_file, low_memory=False)
    df["source_csv"] = csv_file.name
    df["group_id"] = df["image_path"].apply(make_group_id)
    all_rows.append(df)

manifest = pd.concat(all_rows, ignore_index=True)

groups = manifest["group_id"].drop_duplicates()

train_g, temp_g = train_test_split(
    groups, test_size=0.30, random_state=42, shuffle=True
)
val_g, test_g = train_test_split(
    temp_g, test_size=0.50, random_state=42, shuffle=True
)

manifest["split"] = "none"
manifest.loc[manifest["group_id"].isin(train_g), "split"] = "train"
manifest.loc[manifest["group_id"].isin(val_g), "split"] = "val"
manifest.loc[manifest["group_id"].isin(test_g), "split"] = "test"

manifest.to_csv(ROOT / "split_manifest_global_grouped.csv", index=False)

for split in ["train", "val", "test"]:
    out = manifest[manifest["split"] == split].drop(columns=["split"])
    out.to_csv(OUT_DIR / f"{split}.csv", index=False)

print("Rows per split:")
print(manifest["split"].value_counts())

print("\nGroups per split:")
print(manifest.groupby("split")["group_id"].nunique())

print("\nTasks per split:")
print(manifest.groupby("split")["task_id"].nunique())

print("\nSaved to:")
print(OUT_DIR)
PY

python make_csv_split_global_grouped.py



python - <<'PY'
import pandas as pd

df = pd.read_csv("data_train7z/train/split_manifest_global_grouped.csv", low_memory=False)

for col in ["image_path", "group_id"]:
    leak = df.groupby(col)["split"].nunique()
    bad = leak[leak > 1]
    print(f"{col} leaking across splits:", len(bad))

print("\nRows:")
print(df["split"].value_counts())

print("\nTasks per split:")
print(df.groupby("split")["task_id"].nunique())
PY


# check which tasks are missing from val/test
python - <<'PY'
import pandas as pd

df = pd.read_csv("data_train7z/train/split_manifest_global_grouped.csv", low_memory=False)

all_tasks = set(df.task_id.unique())

for split in ["train", "val", "test"]:
    tasks = set(df[df.split == split].task_id.unique())
    print("\nMissing from", split)
    print(sorted(all_tasks - tasks))
PY