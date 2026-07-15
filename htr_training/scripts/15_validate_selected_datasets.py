from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import unicodedata

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dataset_validation"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


SOURCES = [
    {
        "dataset": "IAM",
        "language": "en",
        "domain": "handwriting",
        "splits": {
            "train": "iam/train.csv",
            "val": "iam/validation.csv",
            "test": "iam/test.csv",
        },
    },
    {
        "dataset": "KHATT",
        "language": "ar",
        "domain": "handwriting",
        "splits": {
            "train": "khatt/train.csv",
            "val": "khatt/validation.csv",
        },
    },
    {
        "dataset": "Muharaf",
        "language": "ar",
        "domain": "handwriting",
        "splits": {
            "train": "muharaf/train.csv",
            "val": "muharaf/validation.csv",
            "test": "muharaf/test.csv",
        },
    },
    {
        "dataset": "SAQR",
        "language": "ar",
        "domain": "handwriting",
        "splits": {
            "train": "saqr/train/labels_clean.csv",
            "val": "saqr/val/labels_normalized.csv",
            "test": "saqr/test/labels_normalized.csv",
        },
    },
    {
        "dataset": "EgyptianHandwriting",
        "language": "ar",
        "domain": "handwriting",
        "splits": {
            "train": (
                "egyptian_handwriting/train/"
                "labels_clean_safe.csv"
            ),
        },
    },
    {
        "dataset": "GoBo",
        "language": "en",
        "domain": "handwriting",
        "splits": {
            "train": "gobo/train/labels_clean.csv",
            "val": "gobo/val/labels_normalized.csv",
            "test": "gobo/test/labels_normalized.csv",
        },
    },
    {
        "dataset": "EvArEST",
        "language": None,
        "domain": "scene_text",
        "splits": {
            "train": "evarest/train/labels_clean.csv",
            "test": "evarest/test/labels_normalized.csv",
        },
    },
    {
        "dataset": "KSTRV2",
        "language": None,
        "domain": "scene_text",
        "splits": {
            "train": "kstrv2/train/labels_clean.csv",
            "val": "kstrv2/val/labels_normalized.csv",
            "test": "kstrv2/test/labels_normalized.csv",
        },
    },
]


def resolve_image_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def text_value(row: dict[str, str]) -> str:
    value = (
        row.get("text")
        or row.get("label")
        or row.get("transcription")
        or ""
    )

    return unicodedata.normalize(
        "NFC",
        str(value),
    ).strip()


def image_value(row: dict[str, str]) -> str:
    return str(
        row.get("image_path")
        or row.get("image")
        or ""
    ).strip()


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    errors: list[dict[str, object]] = []
    valid_rows: list[dict[str, object]] = []

    character_counts: Counter[str] = Counter()
    dataset_counts: Counter[tuple[str, str]] = Counter()

    hashes: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    for source in SOURCES:
        dataset_name = source["dataset"]

        for split, relative_manifest in source["splits"].items():
            manifest_path = (
                PROCESSED_ROOT / relative_manifest
            )

            if not manifest_path.exists():
                errors.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "manifest": relative_manifest,
                        "row": "",
                        "error": "manifest_missing",
                    }
                )
                continue

            with manifest_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                reader = csv.DictReader(file)

                for row_number, row in enumerate(
                    reader,
                    start=2,
                ):
                    text = text_value(row)
                    image_text = image_value(row)

                    language = (
                        row.get("language")
                        or source["language"]
                        or ""
                    ).strip().lower()

                    if language == "arabic":
                        language = "ar"
                    elif language == "english":
                        language = "en"

                    base_error = {
                        "dataset": dataset_name,
                        "split": split,
                        "manifest": relative_manifest,
                        "row": row_number,
                        "image": image_text,
                        "text": text,
                    }

                    if not text:
                        errors.append(
                            {
                                **base_error,
                                "error": "empty_text",
                            }
                        )
                        continue

                    if not image_text:
                        errors.append(
                            {
                                **base_error,
                                "error": "empty_image_path",
                            }
                        )
                        continue

                    if language not in {"ar", "en"}:
                        errors.append(
                            {
                                **base_error,
                                "error": (
                                    f"invalid_language:{language}"
                                ),
                            }
                        )
                        continue

                    image_path = resolve_image_path(
                        image_text
                    )

                    if not image_path.exists():
                        errors.append(
                            {
                                **base_error,
                                "error": "image_missing",
                            }
                        )
                        continue

                    try:
                        with Image.open(image_path) as image:
                            image.verify()

                        with Image.open(image_path) as image:
                            width, height = image.size

                    except (
                        UnidentifiedImageError,
                        OSError,
                        ValueError,
                    ) as error:
                        errors.append(
                            {
                                **base_error,
                                "error": (
                                    "invalid_image:"
                                    f"{type(error).__name__}"
                                ),
                            }
                        )
                        continue

                    if width <= 0 or height <= 0:
                        errors.append(
                            {
                                **base_error,
                                "error": "invalid_dimensions",
                            }
                        )
                        continue

                    sha256 = image_hash(image_path)

                    record = {
                        "dataset": dataset_name,
                        "split": split,
                        "domain": source["domain"],
                        "language": language,
                        "manifest": relative_manifest,
                        "row": row_number,
                        "image": image_text,
                        "resolved_image": str(image_path),
                        "text": text,
                        "width": width,
                        "height": height,
                        "sha256": sha256,
                    }

                    valid_rows.append(record)
                    hashes[sha256].append(record)

                    character_counts.update(text)
                    dataset_counts[
                        (dataset_name, split)
                    ] += 1

            print(
                f"Validated {dataset_name}/{split}: "
                f"{dataset_counts[(dataset_name, split)]}"
            )

    duplicate_rows: list[dict[str, object]] = []

    for sha256, records in hashes.items():
        if len(records) < 2:
            continue

        splits = sorted(
            {str(record["split"]) for record in records}
        )
        texts = sorted(
            {str(record["text"]) for record in records}
        )
        datasets = sorted(
            {str(record["dataset"]) for record in records}
        )

        if len(splits) > 1:
            reason = "cross_split_duplicate"
        elif len(texts) > 1:
            reason = "conflicting_label"
        else:
            reason = "exact_duplicate"

        duplicate_rows.append(
            {
                "sha256": sha256,
                "count": len(records),
                "reason": reason,
                "splits": "|".join(splits),
                "datasets": "|".join(datasets),
                "texts": "|".join(texts),
                "images": "|".join(
                    str(record["image"])
                    for record in records
                ),
            }
        )

    error_fields = sorted(
        {
            key
            for row in errors
            for key in row.keys()
        }
    )

    with (
        OUTPUT_ROOT / "validation_errors.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=error_fields or ["error"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(errors)

    duplicate_fields = [
        "sha256",
        "count",
        "reason",
        "splits",
        "datasets",
        "texts",
        "images",
    ]

    with (
        OUTPUT_ROOT / "duplicate_report.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=duplicate_fields,
        )
        writer.writeheader()
        writer.writerows(duplicate_rows)

    character_inventory = "".join(
        sorted(character_counts)
    )

    (
        OUTPUT_ROOT / "character_inventory.txt"
    ).write_text(
        character_inventory + "\n",
        encoding="utf-8",
    )

    duplicate_reason_counts = Counter(
        row["reason"]
        for row in duplicate_rows
    )

    summary = {
        "valid_rows": len(valid_rows),
        "error_rows": len(errors),
        "unique_images": len(hashes),
        "duplicate_groups": len(duplicate_rows),
        "duplicate_reasons": dict(
            duplicate_reason_counts
        ),
        "unique_characters": len(
            character_counts
        ),
        "dataset_split_counts": {
            f"{dataset}/{split}": count
            for (dataset, split), count
            in sorted(dataset_counts.items())
        },
    }

    (
        OUTPUT_ROOT / "validation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Validation complete.")
    print(f"Valid rows: {len(valid_rows)}")
    print(f"Errors: {len(errors)}")
    print(
        f"Duplicate groups: {len(duplicate_rows)}"
    )
    print(
        "Cross-split duplicate groups: "
        f"{duplicate_reason_counts['cross_split_duplicate']}"
    )
    print(
        f"Report directory: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
