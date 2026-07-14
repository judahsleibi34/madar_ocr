from pathlib import Path
import csv
import random

from datasets import load_dataset


RAW_DIR = Path("data/raw/muharaf/data")
OUTPUT_DIR = Path("data/processed/muharaf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initial balanced Arabic expansion.
# KHATT train 1400 + Muharaf train 6000 = 7400 Arabic samples.
LIMITS = {
    "train": None,
    "validation": None,
    "test": None,
}

SEED = 42


def save_split(dataset, split_name: str, limit: int | None) -> None:
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)

    if limit is not None:
        indices = indices[: min(limit, len(indices))]

    image_dir = OUTPUT_DIR / "images" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for output_index, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]

        text = str(sample["text"]).strip()
        if not text:
            continue

        image = sample["image"].convert("RGB")
        image_path = image_dir / f"{split_name}_{output_index:06d}.png"
        image.save(image_path)

        rows.append(
            {
                "image": str(image_path.resolve()),
                "text": text,
                "language": "ar",
                "dataset": "muharaf",
                "split": split_name,
            }
        )

        if (output_index + 1) % 500 == 0:
            print(f"{split_name}: exported {output_index + 1} images")

    csv_path = OUTPUT_DIR / f"{split_name}.csv"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "text",
                "language",
                "dataset",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{split_name}: {len(rows)} samples")
    print(f"Saved: {csv_path.resolve()}")


def main() -> None:
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": [
                str(RAW_DIR / "train-00000-of-00003.parquet"),
                str(RAW_DIR / "train-00001-of-00003.parquet"),
                str(RAW_DIR / "train-00002-of-00003.parquet"),
            ],
            "validation": str(
                RAW_DIR / "validation-00000-of-00001.parquet"
            ),
            "test": str(
                RAW_DIR / "test-00000-of-00001.parquet"
            ),
        },
    )

    print(dataset)

    for split_name, split_data in dataset.items():
        save_split(
            split_data,
            split_name,
            LIMITS.get(split_name),
        )


if __name__ == "__main__":
    main()
