import pandas as pd
from pathlib import Path

base = Path("data/processed")

datasets = ["iam", "khatt", "arabic", "muharaf"]

for split in ["train", "validation"]:
    frames = []
    for name in datasets:
        csv_path = base / name / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            frames.append(df)
            print(f"{name}/{split}.csv: {len(df)} rows")
        else:
            print(f"{name}/{split}.csv: not found, skipping")

    combined = pd.concat(frames, ignore_index=True)
    output_path = base / f"combined_{split}.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved {output_path}: {len(combined)} total rows")
    print()