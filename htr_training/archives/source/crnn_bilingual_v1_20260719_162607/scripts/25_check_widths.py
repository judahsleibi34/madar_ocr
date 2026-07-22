import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_width(img_path):
    try:
        with Image.open(img_path) as img:
            return img.width
    except FileNotFoundError:
        return None


def check_widths(csv_paths: list[str], split_name: str):
    frames = [pd.read_csv(p) for p in csv_paths]
    combined = pd.concat(frames, ignore_index=True)

    print(f"--- {split_name} ---")
    print(f"Total rows: {len(combined)}")
    print("Reading image widths (parallel)...")

    widths = []
    missing = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(get_width, p): p for p in combined["image"]}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is None:
                missing += 1
            else:
                widths.append(result)

    widths = np.array(widths)

    print(f"Missing images: {missing}")
    print(f"Checked: {len(widths)}")
    print(f"Min: {widths.min()}")
    print(f"Max: {widths.max()}")
    print(f"Mean: {widths.mean():.0f}")
    print(f"p50: {np.percentile(widths, 50):.0f}")
    print(f"p90: {np.percentile(widths, 90):.0f}")
    print(f"p95: {np.percentile(widths, 95):.0f}")
    print(f"p99: {np.percentile(widths, 99):.0f}")
    print()


if __name__ == "__main__":
    base = Path("data/processed")

    train_csvs = [
        base / "iam" / "train.csv",
        base / "khatt" / "train.csv",
        base / "arabic" / "train.csv",
        base / "muharaf" / "train.csv",
    ]

    val_csvs = [
        base / "iam" / "validation.csv",
        base / "khatt" / "validation.csv",
        base / "arabic" / "validation.csv",
        base / "muharaf" / "validation.csv",
    ]

    check_widths([str(p) for p in train_csvs if p.exists()], "TRAIN")
    check_widths([str(p) for p in val_csvs if p.exists()], "VALIDATION")
