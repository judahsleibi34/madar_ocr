from pathlib import Path
import csv

from PIL import Image


ROOT = Path("data/raw/khatt")
OUTPUT = Path("data/processed/khatt")
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path) -> str:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp1256",
    )

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode transcription file: {path}")


def build_manifest(
    split_name: str,
    input_dir: Path,
) -> None:
    rows = []

    image_dir = OUTPUT / "images" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)

    tif_paths = sorted(input_dir.rglob("*.tif"))

    output_index = 0

    for tif_path in tif_paths:
        text_path = tif_path.with_suffix(".txt")

        if not text_path.exists():
            print(f"Missing transcription: {tif_path}")
            continue

        text = read_text_file(text_path)

        if not text:
            print(f"Empty transcription: {text_path}")
            continue

        try:
            image = Image.open(tif_path).convert("RGB")
        except Exception as exc:
            print(f"Could not open image: {tif_path} ({exc})")
            continue

        image_path = image_dir / f"{split_name}_{output_index:06d}.png"
        image.save(image_path)

        rows.append(
            {
                "image": str(image_path.resolve()),
                "text": text,
                "language": "ar",
                "dataset": "khatt",
                "split": split_name,
            }
        )

        output_index += 1

        if output_index % 500 == 0:
            print(f"{split_name}: exported {output_index} images")

    output_file = OUTPUT / f"{split_name}.csv"

    with output_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
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
    print(f"Saved: {output_file.resolve()}")


def main() -> None:
    build_manifest(
        split_name="train",
        input_dir=ROOT / "train" / "Training",
    )

    build_manifest(
        split_name="validation",
        input_dir=ROOT / "validation" / "Validation",
    )


if __name__ == "__main__":
    main()