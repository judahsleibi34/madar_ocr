from pathlib import Path

import pandas as pd


MANIFESTS = [
    Path("data/processed/iam/train.csv"),
    Path("data/processed/iam/validation.csv"),
    Path("data/processed/iam/test.csv"),
    Path("data/processed/arabic/train.csv"),
    Path("data/processed/arabic/validation.csv"),
    Path("data/processed/arabic/test.csv"),
]

REQUIRED_COLUMNS = {
    "image",
    "text",
    "language",
    "dataset",
    "split",
}


def validate_manifest(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"[MISSING CSV] {csv_path}")
        return

    dataframe = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        print(
            f"[INVALID COLUMNS] {csv_path}: "
            f"{sorted(missing_columns)}"
        )
        return

    missing_images = []
    empty_text_count = 0
    invalid_language_count = 0

    for row in dataframe.itertuples(index=False):
        image_path = Path(str(row.image))

        if not image_path.exists():
            missing_images.append(str(image_path))

        text = str(row.text).strip()

        if not text or text.lower() == "nan":
            empty_text_count += 1

        if row.language not in {"ar", "en"}:
            invalid_language_count += 1

    duplicate_count = dataframe.duplicated(
        subset=["image"]
    ).sum()

    print()
    print(f"Manifest: {csv_path}")
    print(f"Rows: {len(dataframe)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Empty transcriptions: {empty_text_count}")
    print(f"Duplicate image paths: {duplicate_count}")
    print(f"Invalid languages: {invalid_language_count}")

    if "dataset" in dataframe.columns:
        print("Datasets:")
        print(dataframe["dataset"].value_counts().to_string())

    if missing_images:
        print("First missing image paths:")

        for path in missing_images[:5]:
            print(f"  {path}")


def main() -> None:
    for manifest in MANIFESTS:
        validate_manifest(manifest)


if __name__ == "__main__":
    main()
