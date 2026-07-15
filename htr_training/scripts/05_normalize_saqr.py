from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import re
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAQR_ROOT = PROJECT_ROOT / "data" / "processed" / "saqr"

SPLITS = ("train", "val", "test")

# Characters that should be manually checked.
REVIEW_PATTERN = re.compile(r"[_\[\]\^{}|~<>=]")


def normalize_text(text: str) -> str:
    """Apply conservative OCR-label normalization."""
    text = unicodedata.normalize("NFC", text)

    # Remove invisible formatting characters.
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\u2060", "")

    # Normalize all whitespace to one regular space.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def find_review_reasons(text: str) -> list[str]:
    """Return reasons that require manual inspection."""
    reasons: list[str] = []

    if "_" in text:
        reasons.append("underscore")

    unusual_symbols = sorted(set(REVIEW_PATTERN.findall(text)))

    if unusual_symbols:
        reasons.append(
            "unusual_symbols:" + "".join(unusual_symbols)
        )

    return reasons


def write_csv(
    destination: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open(
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


def process_split(
    split: str,
    character_counts: Counter[str],
) -> dict[str, int]:
    split_dir = SAQR_ROOT / split
    source_path = split_dir / "labels.csv"

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing source manifest: {source_path}"
        )

    with source_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        source_rows = list(csv.DictReader(file))

    if not source_rows:
        raise ValueError(f"No rows found in {source_path}")

    original_fields = list(source_rows[0].keys())

    fieldnames = original_fields.copy()

    for additional_field in (
        "original_text",
        "review_reason",
    ):
        if additional_field not in fieldnames:
            fieldnames.append(additional_field)

    normalized_rows: list[dict[str, str]] = []
    clean_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []

    for source_row in source_rows:
        row = dict(source_row)

        original_text = str(row.get("text", ""))
        normalized_text = normalize_text(original_text)

        if not normalized_text:
            reasons = ["empty_text"]
        else:
            reasons = find_review_reasons(normalized_text)

        character_counts.update(normalized_text)

        row["text"] = normalized_text
        row["original_text"] = original_text
        row["review_reason"] = ",".join(reasons)

        normalized_rows.append(row)

        if reasons:
            review_rows.append(row)
        else:
            clean_rows.append(row)

    write_csv(
        split_dir / "labels_normalized.csv",
        normalized_rows,
        fieldnames,
    )

    write_csv(
        split_dir / "labels_clean.csv",
        clean_rows,
        fieldnames,
    )

    write_csv(
        split_dir / "labels_review.csv",
        review_rows,
        fieldnames,
    )

    print(
        f"{split}: "
        f"total={len(normalized_rows)}, "
        f"clean={len(clean_rows)}, "
        f"review={len(review_rows)}"
    )

    return {
        "total": len(normalized_rows),
        "clean": len(clean_rows),
        "review": len(review_rows),
    }


def combine_review_files() -> int:
    combined_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    for split in SPLITS:
        review_path = (
            SAQR_ROOT / split / "labels_review.csv"
        )

        with review_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))

        for row in rows:
            row["source_split"] = split
            combined_rows.append(row)

        if rows and fieldnames is None:
            fieldnames = ["source_split", *rows[0].keys()]

    if fieldnames is None:
        fieldnames = [
            "source_split",
            "image_path",
            "text",
            "original_text",
            "review_reason",
        ]

    write_csv(
        SAQR_ROOT / "qc_suspicious.csv",
        combined_rows,
        fieldnames,
    )

    return len(combined_rows)


def main() -> None:
    SAQR_ROOT.mkdir(parents=True, exist_ok=True)

    character_counts: Counter[str] = Counter()
    summary: dict[str, object] = {
        "dataset": "SAQR",
        "splits": {},
    }

    for split in SPLITS:
        result = process_split(
            split,
            character_counts,
        )
        summary["splits"][split] = result

    suspicious_count = combine_review_files()
    summary["total_review_rows"] = suspicious_count
    summary["unique_characters"] = len(character_counts)

    character_inventory = "".join(
        sorted(character_counts.keys())
    )

    inventory_path = (
        SAQR_ROOT / "character_inventory.txt"
    )
    inventory_path.write_text(
        character_inventory + "\n",
        encoding="utf-8",
    )

    summary_path = (
        SAQR_ROOT / "normalization_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Normalization completed.")
    print(f"Output directory: {SAQR_ROOT}")
    print(f"Review rows: {suspicious_count}")
    print(f"Character inventory: {inventory_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
