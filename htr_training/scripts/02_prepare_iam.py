from pathlib import Path
import csv

from datasets import load_dataset


RAW_DIR = Path("data/raw/iam_lines/data")
OUTPUT_DIR = Path("data/processed/iam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_split(dataset, split_name: str) -> None:
    image_dir = OUTPUT_DIR / "images" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for index, sample in enumerate(dataset):
        text = sample["text"].strip()

        if not text:
            continue

        image = sample["image"].convert("RGB")
        image_path = image_dir / f"{split_name}_{index:06d}.png"
        image.save(image_path)

        rows.append(
            {
                "image": str(image_path.resolve()),
                "text": text,
                "language": "en",
                "dataset": "iam",
                "split": split_name,
            }
        )

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
            "train": str(RAW_DIR / "train.parquet"),
            "validation": str(RAW_DIR / "validation.parquet"),
            "test": str(RAW_DIR / "test.parquet"),
        },
    )

    for split_name, split_data in dataset.items():
        save_split(split_data, split_name)


if __name__ == "__main__":
    main()
