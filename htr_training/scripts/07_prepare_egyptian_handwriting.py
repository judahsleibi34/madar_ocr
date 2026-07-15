from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
import csv
import hashlib
import json
import re
import unicodedata

import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "egyptian_handwriting"
    / "data"
    / "egy_handwriting_dataset_set1.parquet"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "egyptian_handwriting"
)

TRAIN_DIR = OUTPUT_ROOT / "train"
IMAGE_DIR = TRAIN_DIR / "images"

NORMALIZED_CSV = TRAIN_DIR / "labels_normalized.csv"
CLEAN_CSV = TRAIN_DIR / "labels_clean.csv"
REVIEW_CSV = TRAIN_DIR / "labels_review.csv"
ERROR_CSV = TRAIN_DIR / "conversion_errors.csv"

SUMMARY_PATH = OUTPUT_ROOT / "preparation_summary.json"
INVENTORY_PATH = OUTPUT_ROOT / "character_inventory.txt"

REVIEW_SYMBOL_PATTERN = re.compile(
    r"[_\[\]\^{}|~<>=]"
)

LATIN_PATTERN = re.compile(r"[A-Za-z]")
DIGIT_PATTERN = re.compile(r"[0-9٠-٩]")

FIELDNAMES = [
    "image_path",
    "text",
    "dataset",
    "split",
    "source_index",
    "image_sha256",
    "original_text",
    "review_reason",
]


def normalize_text(text: str) -> str:
    """Apply conservative OCR-label normalization."""
    text = unicodedata.normalize("NFC", text)

    # Remove invisible and directional formatting characters.
    for character in (
        "\u200b",
        "\ufeff",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2060",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    ):
        text = text.replace(character, "")

    # Collapse tabs, newlines and repeated spaces.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def find_review_reasons(text: str) -> list[str]:
    reasons: list[str] = []

    if not text:
        reasons.append("empty_text")
        return reasons

    unusual_symbols = sorted(
        set(REVIEW_SYMBOL_PATTERN.findall(text))
    )

    if unusual_symbols:
        reasons.append(
            "unusual_symbols:" + "".join(unusual_symbols)
        )

    if LATIN_PATTERN.search(text):
        reasons.append("latin_character")

    # This dataset is described as Arabic handwritten words,
    # so labels containing digits should be checked manually.
    if DIGIT_PATTERN.search(text):
        reasons.append("digit")

    if len(text) > 60:
        reasons.append("unexpectedly_long_label")

    return reasons


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Missing Parquet file: {PARQUET_PATH}"
        )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(
        PARQUET_PATH,
        columns=["image", "label"],
    )

    image_values = table.column("image").to_pylist()
    label_values = table.column("label").to_pylist()

    if len(image_values) != len(label_values):
        raise ValueError(
            "Image and label column lengths do not match."
        )

    normalized_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []

    character_counts: Counter[str] = Counter()

    # Used to avoid exact duplicate image samples.
    seen_images: dict[str, dict[str, object]] = {}

    duplicate_same_label = 0
    duplicate_conflicting_label = 0

    for index, (image_bytes, label_value) in enumerate(
        zip(image_values, label_values)
    ):
        original_text = (
            "" if label_value is None
            else str(label_value)
        )

        text = normalize_text(original_text)
        reasons = find_review_reasons(text)

        if not isinstance(image_bytes, bytes):
            error_rows.append(
                {
                    "source_index": index,
                    "original_text": original_text,
                    "error": (
                        "image value is not bytes: "
                        f"{type(image_bytes).__name__}"
                    ),
                }
            )
            continue

        image_hash = hashlib.sha256(
            image_bytes
        ).hexdigest()

        existing = seen_images.get(image_hash)

        if existing is not None:
            if existing["text"] == text:
                duplicate_same_label += 1
                continue

            duplicate_conflicting_label += 1

            conflict_row = dict(existing)
            conflict_row["source_index"] = index
            conflict_row["original_text"] = original_text
            conflict_row["text"] = text
            conflict_row["review_reason"] = (
                "duplicate_image_conflicting_label"
            )

            normalized_rows.append(conflict_row)
            review_rows.append(conflict_row)
            continue

        filename = (
            f"egyptian_{index:05d}_"
            f"{image_hash[:12]}.png"
        )

        image_path = IMAGE_DIR / filename

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()

                # CRNN input is normally grayscale.
                image = image.convert("L")
                image.save(
                    image_path,
                    format="PNG",
                    optimize=True,
                )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:
            error_rows.append(
                {
                    "source_index": index,
                    "original_text": original_text,
                    "error": str(error),
                }
            )
            continue

        relative_path = image_path.relative_to(
            PROJECT_ROOT
        ).as_posix()

        row: dict[str, object] = {
            "image_path": relative_path,
            "text": text,
            "dataset": "EgyptianHandwriting",
            "split": "train",
            "source_index": index,
            "image_sha256": image_hash,
            "original_text": original_text,
            "review_reason": ",".join(reasons),
        }

        seen_images[image_hash] = row
        normalized_rows.append(row)
        character_counts.update(text)

        if reasons:
            review_rows.append(row)
        else:
            clean_rows.append(row)

    write_csv(
        NORMALIZED_CSV,
        normalized_rows,
        FIELDNAMES,
    )

    write_csv(
        CLEAN_CSV,
        clean_rows,
        FIELDNAMES,
    )

    write_csv(
        REVIEW_CSV,
        review_rows,
        FIELDNAMES,
    )

    write_csv(
        ERROR_CSV,
        error_rows,
        [
            "source_index",
            "original_text",
            "error",
        ],
    )

    character_inventory = "".join(
        sorted(character_counts.keys())
    )

    INVENTORY_PATH.write_text(
        character_inventory + "\n",
        encoding="utf-8",
    )

    summary = {
        "dataset": "EgyptianHandwriting",
        "source_rows": len(image_values),
        "normalized_rows": len(normalized_rows),
        "clean_rows": len(clean_rows),
        "review_rows": len(review_rows),
        "conversion_errors": len(error_rows),
        "duplicate_same_label": duplicate_same_label,
        "duplicate_conflicting_label": (
            duplicate_conflicting_label
        ),
        "unique_characters": len(character_counts),
        "split_policy": (
            "training-only because writer IDs are unavailable"
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Egyptian handwriting preparation completed.")
    print(f"Source rows: {len(image_values)}")
    print(f"Normalized: {len(normalized_rows)}")
    print(f"Clean: {len(clean_rows)}")
    print(f"Review: {len(review_rows)}")
    print(f"Errors: {len(error_rows)}")
    print(
        "Exact duplicate images with same label: "
        f"{duplicate_same_label}"
    )
    print(
        "Duplicate images with conflicting labels: "
        f"{duplicate_conflicting_label}"
    )
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
