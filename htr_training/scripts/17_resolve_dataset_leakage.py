from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import re
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

OUTPUT_ROOT = (
    PROCESSED_ROOT
    / "master_safe"
)

SPLIT_PRIORITY = {
    "train": 0,
    "val": 1,
    "test": 2,
}

OUTPUT_FIELDS = [
    "image",
    "text",
    "language",
    "dataset",
    "split",
    "domain",
    "source_type",
    "usage_scope",
    "source_manifest",
    "source_row",
    "sha256",
]

REMOVAL_FIELDS = [
    *OUTPUT_FIELDS,
    "reason",
    "kept_image",
    "kept_split",
    "group_splits",
    "group_texts",
]

SOURCES = [
    {
        "dataset": "IAM",
        "language": "en",
        "domain": "handwriting",
        "usage_scope": "dataset_terms",
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
        "usage_scope": "dataset_terms",
        "splits": {
            "train": "khatt/train.csv",
            "val": "khatt/validation.csv",
        },
    },
    {
        "dataset": "Muharaf",
        "language": "ar",
        "domain": "handwriting",
        "usage_scope": "dataset_terms",
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
        "usage_scope": "CC_BY_4_0",
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
        "usage_scope": "CC_BY_4_0",
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
        "usage_scope": "noncommercial_research_only",
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
        "usage_scope": "dataset_terms",
        "splits": {
            "train": "evarest/train/labels_clean.csv",
            "test": "evarest/test/labels_normalized.csv",
        },
    },
    {
        "dataset": "KSTRV2",
        "language": None,
        "domain": "scene_text",
        "usage_scope": "dataset_terms",
        "splits": {
            "train": "kstrv2/train/labels_clean.csv",
            "val": "kstrv2/val/labels_normalized.csv",
            "test": "kstrv2/test/labels_normalized.csv",
        },
    },
]


def normalize_text(text: str) -> str:
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

    return re.sub(r"\s+", " ", text).strip()


def normalize_language(
    value: str | None,
    fallback: str | None,
) -> str:
    language = str(value or fallback or "").strip().lower()

    aliases = {
        "arabic": "ar",
        "ara": "ar",
        "english": "en",
        "eng": "en",
    }

    language = aliases.get(language, language)

    if language not in {"ar", "en"}:
        raise ValueError(
            f"Unsupported language value: {language!r}"
        )

    return language


def resolve_image_path(value: str) -> tuple[Path, str]:
    image_path = Path(value)

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    try:
        stored_path = image_path.relative_to(
            PROJECT_ROOT
        ).as_posix()
    except ValueError:
        stored_path = image_path.as_posix()

    return image_path, stored_path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


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


def load_all_rows() -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []

    for source in SOURCES:
        for split, relative_manifest in (
            source["splits"].items()
        ):
            manifest_path = (
                PROCESSED_ROOT
                / relative_manifest
            )

            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing manifest: {manifest_path}"
                )

            accepted_from_manifest = 0

            with manifest_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                reader = csv.DictReader(file)

                for source_row, row in enumerate(
                    reader,
                    start=2,
                ):
                    image_value = str(
                        row.get("image_path")
                        or row.get("image")
                        or ""
                    ).strip()

                    text = normalize_text(
                        str(
                            row.get("text")
                            or row.get("label")
                            or row.get("transcription")
                            or ""
                        )
                    )

                    if not image_value:
                        raise ValueError(
                            f"Empty image path in "
                            f"{relative_manifest}:{source_row}"
                        )

                    if not text:
                        raise ValueError(
                            f"Empty text in "
                            f"{relative_manifest}:{source_row}"
                        )

                    image_path, stored_path = (
                        resolve_image_path(image_value)
                    )

                    language = normalize_language(
                        row.get("language"),
                        source["language"],
                    )

                    source_type = str(
                        row.get("source_type")
                        or "real"
                    ).strip().lower()

                    output_rows.append(
                        {
                            "image": stored_path,
                            "text": text,
                            "language": language,
                            "dataset": str(
                                source["dataset"]
                            ),
                            "split": split,
                            "domain": str(
                                source["domain"]
                            ),
                            "source_type": source_type,
                            "usage_scope": str(
                                source["usage_scope"]
                            ),
                            "source_manifest": (
                                relative_manifest
                            ),
                            "source_row": str(
                                source_row
                            ),
                            "sha256": hash_file(
                                image_path
                            ),
                        }
                    )

                    accepted_from_manifest += 1

            print(
                f"{source['dataset']}/{split}: "
                f"{accepted_from_manifest}"
            )

    return output_rows


def language_match_score(
    row: dict[str, str],
) -> int:
    text = row["text"]

    has_arabic = bool(
        re.search(
            r"[\u0600-\u06FF"
            r"\u0750-\u077F"
            r"\u08A0-\u08FF]",
            text,
        )
    )

    has_latin = bool(
        re.search(r"[A-Za-z]", text)
    )

    score = 0

    if has_arabic and row["language"] == "ar":
        score += 100

    if has_latin and row["language"] == "en":
        score += 100

    if row["source_type"] == "real":
        score += 10

    if row["domain"] == "handwriting":
        score += 1

    return score


def choose_row(
    candidates: list[dict[str, str]],
) -> dict[str, str]:
    return sorted(
        candidates,
        key=lambda row: (
            -language_match_score(row),
            row["dataset"],
            row["image"],
            row["source_row"],
        ),
    )[0]


def resolve_groups(
    rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    groups: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        groups[row["sha256"]].append(row)

    accepted: list[dict[str, str]] = []
    removed: list[dict[str, str]] = []
    quarantined: list[dict[str, str]] = []

    for group_rows in groups.values():
        texts = sorted(
            {row["text"] for row in group_rows}
        )

        splits = sorted(
            {row["split"] for row in group_rows},
            key=lambda split: SPLIT_PRIORITY[split],
        )

        group_splits = "|".join(splits)
        group_texts = "|".join(texts)

        # Conflicting labels are too ambiguous to trust.
        if len(texts) > 1:
            reason = (
                "cross_split_conflicting_labels"
                if len(splits) > 1
                else "same_split_conflicting_labels"
            )

            for row in group_rows:
                quarantined.append(
                    {
                        **row,
                        "reason": reason,
                        "kept_image": "",
                        "kept_split": "",
                        "group_splits": group_splits,
                        "group_texts": group_texts,
                    }
                )

            continue

        highest_priority_split = max(
            splits,
            key=lambda split: SPLIT_PRIORITY[split],
        )

        highest_priority_rows = [
            row
            for row in group_rows
            if row["split"] == highest_priority_split
        ]

        kept_row = choose_row(
            highest_priority_rows
        )

        accepted.append(kept_row)

        for row in group_rows:
            if row is kept_row:
                continue

            if row["split"] != highest_priority_split:
                reason = "lower_priority_split_removed"
            else:
                reason = "same_split_exact_duplicate_removed"

            removed.append(
                {
                    **row,
                    "reason": reason,
                    "kept_image": kept_row["image"],
                    "kept_split": kept_row["split"],
                    "group_splits": group_splits,
                    "group_texts": group_texts,
                }
            )

    return accepted, removed, quarantined


def save_manifests(
    accepted: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    rows_by_split: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    rows_by_split_domain: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in accepted:
        rows_by_split[row["split"]].append(row)

        rows_by_split_domain[
            (row["split"], row["domain"])
        ].append(row)

    split_summary: dict[
        str,
        dict[str, int],
    ] = {}

    for split in ("train", "val", "test"):
        all_rows = sorted(
            rows_by_split[split],
            key=lambda row: (
                row["dataset"],
                row["image"],
            ),
        )

        handwriting_rows = sorted(
            rows_by_split_domain[
                (split, "handwriting")
            ],
            key=lambda row: (
                row["dataset"],
                row["image"],
            ),
        )

        scene_rows = sorted(
            rows_by_split_domain[
                (split, "scene_text")
            ],
            key=lambda row: (
                row["dataset"],
                row["image"],
            ),
        )

        write_csv(
            OUTPUT_ROOT / f"{split}_all.csv",
            all_rows,
            OUTPUT_FIELDS,
        )

        write_csv(
            OUTPUT_ROOT
            / f"{split}_handwriting.csv",
            handwriting_rows,
            OUTPUT_FIELDS,
        )

        write_csv(
            OUTPUT_ROOT
            / f"{split}_scene_text.csv",
            scene_rows,
            OUTPUT_FIELDS,
        )

        split_summary[split] = {
            "all": len(all_rows),
            "handwriting": len(
                handwriting_rows
            ),
            "scene_text": len(scene_rows),
        }

        print(
            f"{split}: "
            f"all={len(all_rows)}, "
            f"handwriting={len(handwriting_rows)}, "
            f"scene={len(scene_rows)}"
        )

    return split_summary


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_rows = load_all_rows()

    accepted, removed, quarantined = (
        resolve_groups(source_rows)
    )

    split_summary = save_manifests(
        accepted
    )

    write_csv(
        OUTPUT_ROOT / "removed_duplicates.csv",
        removed,
        REMOVAL_FIELDS,
    )

    write_csv(
        OUTPUT_ROOT / "quarantined_conflicts.csv",
        quarantined,
        REMOVAL_FIELDS,
    )

    all_characters = Counter()

    for row in accepted:
        all_characters.update(row["text"])

    (
        OUTPUT_ROOT / "character_inventory.txt"
    ).write_text(
        "".join(sorted(all_characters)) + "\n",
        encoding="utf-8",
    )

    removal_reasons = Counter(
        row["reason"]
        for row in removed
    )

    quarantine_reasons = Counter(
        row["reason"]
        for row in quarantined
    )

    summary = {
        "source_rows": len(source_rows),
        "accepted_rows": len(accepted),
        "removed_duplicate_rows": len(removed),
        "quarantined_rows": len(quarantined),
        "removal_reasons": dict(removal_reasons),
        "quarantine_reasons": dict(
            quarantine_reasons
        ),
        "splits": split_summary,
        "unique_characters": len(
            all_characters
        ),
        "policy": {
            "split_priority": (
                "test > val > train"
            ),
            "same_label_duplicates": (
                "keep one row in the highest-priority split"
            ),
            "conflicting_labels": (
                "quarantine every row in the image group"
            ),
            "source_manifests_modified": False,
        },
    }

    summary_path = (
        OUTPUT_ROOT
        / "leakage_resolution_summary.json"
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
    print("Leakage resolution completed.")
    print(f"Source rows: {len(source_rows)}")
    print(f"Accepted rows: {len(accepted)}")
    print(
        f"Removed duplicates: {len(removed)}"
    )
    print(
        f"Quarantined rows: {len(quarantined)}"
    )
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
