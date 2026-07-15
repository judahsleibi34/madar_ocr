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

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gobo"
    / "extracted"
    / "GoBo_v1-0"
    / "words"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gobo"
)

SUBSETS = (
    "brown",
    "cedar",
    "domain_A_train",
    "domain_A_test",
    "domain_B_train",
    "domain_B_test",
    "nonwords",
)

FIELDNAMES = [
    "image_path",
    "text",
    "dataset",
    "language",
    "split",
    "writer_id",
    "source_subset",
    "source_annotation",
    "source_status",
    "original_text",
    "review_reason",
]

ERROR_FIELDS = [
    "writer_id",
    "source_subset",
    "annotation_file",
    "line_number",
    "raw_line",
    "error",
]


def normalize_text(text: str) -> str:
    """Apply conservative normalization to English OCR labels."""
    text = unicodedata.normalize("NFC", text)

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

    text = re.sub(r"\s+", " ", text).strip()

    return text


def find_review_reasons(
    text: str,
    status: str,
) -> list[str]:
    reasons: list[str] = []

    if not text:
        reasons.append("empty_text")

    if status.lower() != "ok":
        reasons.append(f"annotation_status:{status}")

    # GoBo contains cropped word images.
    if " " in text:
        reasons.append("multiword_label")

    if len(text) > 80:
        reasons.append("unexpectedly_long_label")

    for character in text:
        category = unicodedata.category(character)

        if category.startswith("C"):
            reasons.append(
                f"control_character:U+{ord(character):04X}"
            )
            break

    return reasons


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
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


def install_image(
    source: Path,
    destination: Path,
) -> str:
    """
    Create a hard link to avoid duplicating disk usage.

    Fall back to copying when hard links are unavailable.
    """
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


def determine_writer_splits(
    writer_ids: list[int],
) -> dict[str, list[int]]:
    if len(writer_ids) < 12:
        raise ValueError(
            "Not enough writers for a writer-independent split."
        )

    # Deterministic split:
    # final 4 writers -> test
    # previous 4 writers -> validation
    # remaining writers -> train
    return {
        "train": writer_ids[:-8],
        "val": writer_ids[-8:-4],
        "test": writer_ids[-4:],
    }


def split_for_writer(
    writer_id: int,
    writer_splits: dict[str, list[int]],
) -> str:
    for split, writers in writer_splits.items():
        if writer_id in writers:
            return split

    raise ValueError(
        f"Writer {writer_id} has no assigned split."
    )


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"GoBo source directory not found: {SOURCE_ROOT}"
        )

    writer_ids = sorted(
        int(path.name)
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and path.name.isdigit()
    )

    writer_splits = determine_writer_splits(writer_ids)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_path = OUTPUT_ROOT / "writer_split.json"
    split_path.write_text(
        json.dumps(
            writer_splits,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows_by_split: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    errors: list[dict[str, str]] = []
    character_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    link_counts: Counter[str] = Counter()

    for writer_id in writer_ids:
        writer_root = SOURCE_ROOT / str(writer_id)
        split = split_for_writer(
            writer_id,
            writer_splits,
        )

        for subset in SUBSETS:
            annotation_path = writer_root / f"{subset}.txt"

            if not annotation_path.exists():
                errors.append(
                    {
                        "writer_id": str(writer_id),
                        "source_subset": subset,
                        "annotation_file": str(
                            annotation_path
                        ),
                        "line_number": "",
                        "raw_line": "",
                        "error": "annotation_file_missing",
                    }
                )
                continue

            with annotation_path.open(
                "r",
                encoding="utf-8-sig",
                errors="replace",
            ) as file:
                for line_number, raw_line in enumerate(
                    file,
                    start=1,
                ):
                    raw_line = raw_line.rstrip("\r\n")

                    if not raw_line.strip():
                        continue

                    parts = raw_line.split(maxsplit=2)

                    if len(parts) != 3:
                        errors.append(
                            {
                                "writer_id": str(writer_id),
                                "source_subset": subset,
                                "annotation_file": str(
                                    annotation_path
                                ),
                                "line_number": str(line_number),
                                "raw_line": raw_line,
                                "error": (
                                    "expected_path_status_label"
                                ),
                            }
                        )
                        continue

                    relative_image, status, original_text = parts
                    text = normalize_text(original_text)

                    source_image = (
                        writer_root / relative_image
                    )

                    if not source_image.exists():
                        errors.append(
                            {
                                "writer_id": str(writer_id),
                                "source_subset": subset,
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
                        OUTPUT_ROOT
                        / split
                        / "images"
                        / f"writer_{writer_id:02d}"
                        / subset
                        / source_image.name
                    )

                    link_method = install_image(
                        source_image,
                        destination_image,
                    )
                    link_counts[link_method] += 1

                    review_reasons = find_review_reasons(
                        text,
                        status,
                    )

                    image_path = (
                        destination_image
                        .relative_to(PROJECT_ROOT)
                        .as_posix()
                    )

                    row = {
                        "image_path": image_path,
                        "text": text,
                        "dataset": "GoBo",
                        "language": "en",
                        "split": split,
                        "writer_id": str(writer_id),
                        "source_subset": subset,
                        "source_annotation": (
                            annotation_path
                            .relative_to(PROJECT_ROOT)
                            .as_posix()
                        ),
                        "source_status": status,
                        "original_text": original_text,
                        "review_reason": ",".join(
                            review_reasons
                        ),
                    }

                    rows_by_split[split].append(row)
                    character_counts.update(text)
                    subset_counts[subset] += 1

    summary_splits: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        normalized_rows = rows_by_split[split]

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

        split_root = OUTPUT_ROOT / split

        write_csv(
            split_root / "labels_normalized.csv",
            normalized_rows,
            FIELDNAMES,
        )

        write_csv(
            split_root / "labels_clean.csv",
            clean_rows,
            FIELDNAMES,
        )

        write_csv(
            split_root / "labels_review.csv",
            review_rows,
            FIELDNAMES,
        )

        summary_splits[split] = {
            "writers": len(writer_splits[split]),
            "total": len(normalized_rows),
            "clean": len(clean_rows),
            "review": len(review_rows),
        }

        print(
            f"{split}: "
            f"writers={len(writer_splits[split])}, "
            f"total={len(normalized_rows)}, "
            f"clean={len(clean_rows)}, "
            f"review={len(review_rows)}"
        )

    write_csv(
        OUTPUT_ROOT / "conversion_errors.csv",
        errors,
        ERROR_FIELDS,
    )

    character_inventory = "".join(
        sorted(character_counts.keys())
    )

    (
        OUTPUT_ROOT / "character_inventory.txt"
    ).write_text(
        character_inventory + "\n",
        encoding="utf-8",
    )

    summary = {
        "dataset": "GoBo",
        "license_note": (
            "Non-commercial research use only; included "
            "for thesis research."
        ),
        "source_writer_count": len(writer_ids),
        "writer_ids": writer_ids,
        "writer_split": writer_splits,
        "splits": summary_splits,
        "subset_counts": dict(subset_counts),
        "conversion_errors": len(errors),
        "image_installation": dict(link_counts),
        "unique_characters": len(character_counts),
    }

    (
        OUTPUT_ROOT / "preparation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("GoBo preparation completed.")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Errors: {len(errors)}")
    print(f"Writer split: {split_path}")


if __name__ == "__main__":
    main()
