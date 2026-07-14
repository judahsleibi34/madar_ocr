import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMAGE_HEIGHT = 64
MAX_IMAGE_WIDTH = 2560
WIDTH_REDUCTION_FACTOR = 4


def compute_available_timesteps(image_path: str) -> int:
    with Image.open(image_path) as img:
        original_width, original_height = img.size

    scale = IMAGE_HEIGHT / original_height
    resized_width = max(1, round(original_width * scale))
    resized_width = min(resized_width, MAX_IMAGE_WIDTH)

    return resized_width // WIDTH_REDUCTION_FACTOR


def required_timesteps(text: str) -> int:
    # CTC needs enough timesteps for the text length, plus extra
    # room to place blank separators between immediately repeated
    # characters (e.g. "book" needs a blank between the two o's).
    return 2 * len(text) - 1


def filter_manifest(csv_path: str):
    df = pd.read_csv(csv_path)

    keep_mask = []
    dropped_reasons = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Checking {Path(csv_path).name}"):
        available = compute_available_timesteps(row["image"])
        needed = required_timesteps(str(row["text"]))

        if available >= needed:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            dropped_reasons.append({
                "image": row["image"],
                "text_length": len(str(row["text"])),
                "available_timesteps": available,
                "needed_timesteps": needed,
            })

    filtered = df[keep_mask].reset_index(drop=True)

    print(f"{csv_path}: kept {len(filtered)}/{len(df)}, dropped {len(df) - len(filtered)}")

    if dropped_reasons:
        dropped_df = pd.DataFrame(dropped_reasons)
        dropped_path = Path(csv_path).with_name(Path(csv_path).stem + "_dropped.csv")
        dropped_df.to_csv(dropped_path, index=False)
        print(f"  Dropped rows logged to: {dropped_path}")

    filtered.to_csv(csv_path, index=False)
    print(f"  Overwrote {csv_path} with filtered data")
    print()


if __name__ == "__main__":
    filter_manifest("data/processed/combined_train.csv")
    filter_manifest("data/processed/combined_validation.csv")
