from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import os
import re
import shutil
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

INVISIBLE_CHARACTERS = (
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
)

FIELDNAMES = [
    "image_path",
    "text",
    "dataset",
    "language",
    "split",
    "source_type",
    "source_annotation",
    "original_text",
    "review_reason",
]

ERROR_FIELDS = [
    "dataset",
    "language",
    "split",
    "source_type",
    "annotation_file",
    "line_number",
    "raw_line",
    "error",
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)

    for character in INVISIBLE_CHARACTERS:
        text = text.replace(character, "")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_review_reasons(text: str) -> list[str]:
    reasons: list[str] = []

    if not text:
        reasons.append("empty_text")

    for character in text:
        category = unicodedata.category(character)

        if category.startswith("C"):
            reasons.append(
                f"control_character:U+{ord(character):04X}"
            )
            break

    if len(text) > 100:
        reasons.append("unexpectedly_long_label")

    return reasons


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str] = FIELDNAMES,
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


def install_image(
    source: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        return "existing"

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def save_dataset_outputs(
    output_root: Path,
    rows_by_split: dict[str, list[dict[str, str]]],
    errors: list[dict[str, str]],
    character_counts: Counter[str],
    extra_summary: dict[str, object],
) -> None:
    split_summary: dict[str, dict[str, int]] = {}

    for split, normalized_rows in rows_by_split.items():
        clean_rows = [
            row
            for row in normalized_rows
            if not row["review_reason"]
        ]

        review_rows = [
            row
            for row in normalized_rows
            if row["review_reason"]
        ]

        split_root = output_root / split

        write_csv(
            split_root / "labels_normalized.csv",
            normalized_rows,
        )
        write_csv(
            split_root / "labels_clean.csv",
            clean_rows,
        )
        write_csv(
            split_root / "labels_review.csv",
            review_rows,
        )

        for language in ("ar", "en"):
            language_rows = [
                row
                for row in normalized_rows
                if row["language"] == language
            ]

            write_csv(
                split_root / f"labels_{language}.csv",
                language_rows,
            )

        real_rows = [
            row
            for row in normalized_rows
            if row["source_type"] == "real"
        ]

        synthetic_rows = [
            row
            for row in normalized_rows
            if row["source_type"] == "synthetic"
        ]

        if real_rows:
            write_csv(
                split_root / "labels_real.csv",
                real_rows,
            )

        if synthetic_rows:
            write_csv(
                split_root / "labels_synthetic.csv",
                synthetic_rows,
            )

        split_summary[split] = {
            "total": len(normalized_rows),
            "clean": len(clean_rows),
            "review": len(review_rows),
            "arabic": sum(
                row["language"] == "ar"
                for row in normalized_rows
            ),
            "english": sum(
                row["language"] == "en"
                for row in normalized_rows
            ),
            "real": len(real_rows),
            "synthetic": len(synthetic_rows),
        }

        print(
            f"{output_root.name}/{split}: "
            f"total={len(normalized_rows)}, "
            f"clean={len(clean_rows)}, "
            f"review={len(review_rows)}"
        )

    write_csv(
        output_root / "conversion_errors.csv",
        errors,
        ERROR_FIELDS,
    )

    character_inventory = "".join(
        sorted(character_counts.keys())
    )

    (
        output_root / "character_inventory.txt"
    ).write_text(
        character_inventory + "\n",
        encoding="utf-8",
    )

    summary = {
        **extra_summary,
        "splits": split_summary,
        "conversion_errors": len(errors),
        "unique_characters": len(character_counts),
    }

    (
        output_root / "preparation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def prepare_evarest() -> None:
    dataset_name = "EvArEST"
    output_root = PROCESSED_ROOT / "evarest"

    configurations = (
        (
            "train",
            "ar",
            RAW_ROOT
            / "evarest"
            / "train_ar"
            / "Arabic_words_train",
        ),
        (
            "test",
            "ar",
            RAW_ROOT
            / "evarest"
            / "test_ar"
            / "Arabic_words_test",
        ),
        (
            "train",
            "en",
            RAW_ROOT
            / "evarest"
            / "train_en"
            / "English_words",
        ),
        (
            "test",
            "en",
            RAW_ROOT
            / "evarest"
            / "test_en"
            / "English_words_test",
        ),
    )

    rows_by_split: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    errors: list[dict[str, str]] = []
    character_counts: Counter[str] = Counter()
    link_counts: Counter[str] = Counter()

    for split, language, source_root in configurations:
        annotation_path = source_root / "gt.txt"

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Missing EvArEST annotation: {annotation_path}"
            )

        with annotation_path.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            for line_number, raw_line in enumerate(
                file,
                start=1,
            ):
                raw_line = raw_line.rstrip("\r\n")

                if not raw_line.strip():
                    continue

                if "," not in raw_line:
                    errors.append(
                        {
                            "dataset": dataset_name,
                            "language": language,
                            "split": split,
                            "source_type": "real",
                            "annotation_file": str(
                                annotation_path
                            ),
                            "line_number": str(line_number),
                            "raw_line": raw_line,
                            "error": "missing_comma_separator",
                        }
                    )
                    continue

                image_name, original_text = raw_line.split(
                    ",",
                    maxsplit=1,
                )

                image_name = image_name.strip()
                text = normalize_text(original_text)

                source_image = source_root / image_name

                if not source_image.exists():
                    errors.append(
                        {
                            "dataset": dataset_name,
                            "language": language,
                            "split": split,
                            "source_type": "real",
                            "annotation_file": str(
                                annotation_path
                            ),
                            "line_number": str(line_number),
                            "raw_line": raw_line,
                            "error": "image_file_missing",
                        }
                    )
                    continue

                destination_image = (
                    output_root
                    / split
                    / "images"
                    / language
                    / source_image.name
                )

                method = install_image(
                    source_image,
                    destination_image,
                )
                link_counts[method] += 1

                reasons = find_review_reasons(text)

                row = {
                    "image_path": (
                        destination_image
                        .relative_to(PROJECT_ROOT)
                        .as_posix()
                    ),
                    "text": text,
                    "dataset": dataset_name,
                    "language": language,
                    "split": split,
                    "source_type": "real",
                    "source_annotation": (
                        annotation_path
                        .relative_to(PROJECT_ROOT)
                        .as_posix()
                    ),
                    "original_text": original_text,
                    "review_reason": ",".join(reasons),
                }

                rows_by_split[split].append(row)
                character_counts.update(text)

    save_dataset_outputs(
        output_root,
        rows_by_split,
        errors,
        character_counts,
        {
            "dataset": dataset_name,
            "domain": "scene_text",
            "split_policy": (
                "Official train/test split preserved; "
                "dataset has no validation split."
            ),
            "image_installation": dict(link_counts),
        },
    )


def prepare_kstrv2() -> None:
    dataset_name = "KSTRV2"

    source_root = (
        RAW_ROOT
        / "kstrv2"
        / "extracted"
        / "KSTRv2"
        / "recognition"
    )

    output_root = PROCESSED_ROOT / "kstrv2"

    language_directories = {
        "ar": "arabic",
        "en": "english",
    }

    rows_by_split: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    errors: list[dict[str, str]] = []
    character_counts: Counter[str] = Counter()
    link_counts: Counter[str] = Counter()

    for source_type in ("real", "synthetic"):
        for language, directory_name in (
            language_directories.items()
        ):
            language_root = (
                source_root
                / source_type
                / directory_name
            )

            for split in ("train", "val", "test"):
                annotation_path = (
                    language_root / f"gt_{split}.txt"
                )

                if not annotation_path.exists():
                    raise FileNotFoundError(
                        "Missing KSTRV2 annotation: "
                        f"{annotation_path}"
                    )

                with annotation_path.open(
                    "r",
                    encoding="utf-8-sig",
                ) as file:
                    for line_number, raw_line in enumerate(
                        file,
                        start=1,
                    ):
                        raw_line = raw_line.rstrip("\r\n")

                        if not raw_line.strip():
                            continue

                        parts = raw_line.strip().split(
                            maxsplit=1
                        )

                        if len(parts) != 2:
                            errors.append(
                                {
                                    "dataset": dataset_name,
                                    "language": language,
                                    "split": split,
                                    "source_type": source_type,
                                    "annotation_file": str(
                                        annotation_path
                                    ),
                                    "line_number": str(
                                        line_number
                                    ),
                                    "raw_line": raw_line,
                                    "error": (
                                        "expected_path_and_label"
                                    ),
                                }
                            )
                            continue

                        relative_image, original_text = parts
                        text = normalize_text(original_text)

                        source_image = (
                            language_root / relative_image
                        )

                        if not source_image.exists():
                            errors.append(
                                {
                                    "dataset": dataset_name,
                                    "language": language,
                                    "split": split,
                                    "source_type": source_type,
                                    "annotation_file": str(
                                        annotation_path
                                    ),
                                    "line_number": str(
                                        line_number
                                    ),
                                    "raw_line": raw_line,
                                    "error": "image_file_missing",
                                }
                            )
                            continue

                        destination_image = (
                            output_root
                            / split
                            / "images"
                            / source_type
                            / language
                            / source_image.name
                        )

                        method = install_image(
                            source_image,
                            destination_image,
                        )
                        link_counts[method] += 1

                        reasons = find_review_reasons(text)

                        row = {
                            "image_path": (
                                destination_image
                                .relative_to(PROJECT_ROOT)
                                .as_posix()
                            ),
                            "text": text,
                            "dataset": dataset_name,
                            "language": language,
                            "split": split,
                            "source_type": source_type,
                            "source_annotation": (
                                annotation_path
                                .relative_to(PROJECT_ROOT)
                                .as_posix()
                            ),
                            "original_text": original_text,
                            "review_reason": ",".join(reasons),
                        }

                        rows_by_split[split].append(row)
                        character_counts.update(text)

    save_dataset_outputs(
        output_root,
        rows_by_split,
        errors,
        character_counts,
        {
            "dataset": dataset_name,
            "domain": "scene_text",
            "languages_included": ["ar", "en"],
            "languages_excluded": ["kurdish"],
            "split_policy": (
                "Official train/validation/test splits preserved."
            ),
            "image_installation": dict(link_counts),
        },
    )


def main() -> None:
    prepare_evarest()
    print()
    prepare_kstrv2()

    print()
    print("Scene-text preparation completed.")
    print(
        f"EvArEST output: {PROCESSED_ROOT / 'evarest'}"
    )
    print(
        f"KSTRV2 output: {PROCESSED_ROOT / 'kstrv2'}"
    )


if __name__ == "__main__":
    main()
