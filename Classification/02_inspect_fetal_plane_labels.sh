python - <<'PY'
import pandas as pd
from pathlib import Path

csv_path = "/home/yr245/project/ultrasam_project/data_train7z/train/csv_files/Cls-Six_2.FETAL_PLANES_ZENODO_12,400.csv"

df = pd.read_csv(csv_path)

print("LABEL DISTRIBUTION")
print(df["mask"].value_counts().sort_index())

print("\nLABEL EXAMPLES")

for label in sorted(df["mask"].unique()):
    subset = df[df["mask"] == label]["image_path"].head(5)

    print(f"\nLABEL {int(label)}")
    for p in subset:
        p = Path(p)
        print(f"{p.parent.name} | {p.name}")
PY

python - <<'PY'
label_to_plane = {
    0: "Plane2",
    1: "Plane3",
    2: "Plane5",
    3: "Plane6",
    4: "Plane4",
    5: "Plane1"
}

print(label_to_plane)
PY