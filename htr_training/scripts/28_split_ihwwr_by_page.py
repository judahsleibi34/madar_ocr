from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


INPUT = Path(
    r"data\processed\online_htr"
    r"\ihwwr_english\words_all.csv"
)

OUTPUT = Path(
    r"data\processed\online_htr"
    r"\ihwwr_english"
)

SEED = "ihwwr_page_split_20260722"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

FILENAME_PATTERN = re.compile(
    r"^(?P<page_id>\d+)_(?P<word_index>\d+)$"
)


def parse_filename(
    image_value: str,
) -> tuple[str, int]:
    stem = Path(
        image_value
    ).stem

    match = FILENAME_PATTERN.fullmatch(
        stem
    )

    if not match:
        raise ValueError(
            f"Unexpected image filename: {stem}"
        )

    return (
        match.group("page_id"),
        int(match.group("word_index")),
    )


def stable_hash(
    page_id: str,
) -> str:
    value = (
        f"{SEED}:{page_id}"
    ).encode("utf-8")

    return hashlib.sha256(
        value
    ).hexdigest()


def character_inventory(
    values: pd.Series,
) -> list[str]:
    characters: set[str] = set()

    for value in values.astype(str):
        characters.update(value)

    return sorted(characters)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input manifest missing: "
            f"{INPUT.resolve()}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # dtype=str is important because labels such as "13113"
    # must remain text rather than becoming numbers.
    data = pd.read_csv(
        INPUT,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "image",
        "text",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    parsed_names = data[
        "image"
    ].map(parse_filename)

    data["page_id"] = parsed_names.map(
        lambda value: value[0]
    )

    data["word_index"] = parsed_names.map(
        lambda value: value[1]
    )

    if data["image"].duplicated().any():
        duplicates = data[
            data["image"].duplicated(
                keep=False
            )
        ]

        raise RuntimeError(
            f"Duplicate image rows found: "
            f"{len(duplicates)}"
        )

    duplicate_positions = data.duplicated(
        subset=[
            "page_id",
            "word_index",
        ],
        keep=False,
    )

    if duplicate_positions.any():
        raise RuntimeError(
            "Duplicate page/word positions found: "
            f"{int(duplicate_positions.sum())}"
        )

    empty_text = (
        data["text"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_text.any():
        raise RuntimeError(
            f"Empty labels found: "
            f"{int(empty_text.sum())}"
        )

    page_counts = (
        data.groupby(
            "page_id"
        )
        .size()
        .rename("word_count")
        .reset_index()
    )

    page_counts["split_hash"] = (
        page_counts["page_id"]
        .map(stable_hash)
    )

    page_counts = (
        page_counts.sort_values(
            [
                "split_hash",
                "page_id",
            ]
        )
        .reset_index(drop=True)
    )

    total_pages = len(page_counts)

    train_page_count = round(
        total_pages * TRAIN_RATIO
    )

    val_page_count = round(
        total_pages * VAL_RATIO
    )

    test_page_count = (
        total_pages
        - train_page_count
        - val_page_count
    )

    page_counts["split"] = "test"

    page_counts.loc[
        : train_page_count - 1,
        "split",
    ] = "train"

    val_start = train_page_count
    val_end = (
        train_page_count
        + val_page_count
    )

    page_counts.loc[
        val_start : val_end - 1,
        "split",
    ] = "val"

    split_map = dict(
        zip(
            page_counts["page_id"],
            page_counts["split"],
        )
    )

    data["split"] = data[
        "page_id"
    ].map(split_map)

    if data["split"].isna().any():
        raise RuntimeError(
            "Some rows did not receive a split."
        )

    # Ensure every page occurs in exactly one split.
    page_split_counts = (
        data.groupby("page_id")[
            "split"
        ].nunique()
    )

    if page_split_counts.max() != 1:
        raise RuntimeError(
            "Page leakage detected."
        )

    page_sets = {
        split: set(
            data.loc[
                data["split"] == split,
                "page_id",
            ]
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    if (
        page_sets["train"]
        & page_sets["val"]
    ):
        raise RuntimeError(
            "Train/validation page overlap."
        )

    if (
        page_sets["train"]
        & page_sets["test"]
    ):
        raise RuntimeError(
            "Train/test page overlap."
        )

    if (
        page_sets["val"]
        & page_sets["test"]
    ):
        raise RuntimeError(
            "Validation/test page overlap."
        )

    data = data.sort_values(
        [
            "split",
            "page_id",
            "word_index",
        ]
    ).reset_index(drop=True)

    all_output = (
        OUTPUT
        / "words_with_split.csv"
    )

    data.to_csv(
        all_output,
        index=False,
        encoding="utf-8-sig",
    )

    split_outputs: dict[str, Path] = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        subset = data[
            data["split"] == split
        ].copy()

        destination = (
            OUTPUT
            / f"{split}.csv"
        )

        subset.to_csv(
            destination,
            index=False,
            encoding="utf-8-sig",
        )

        split_outputs[split] = (
            destination
        )

    assignments_output = (
        OUTPUT
        / "page_split_assignments.csv"
    )

    page_counts[
        [
            "page_id",
            "word_count",
            "split",
            "split_hash",
        ]
    ].to_csv(
        assignments_output,
        index=False,
        encoding="utf-8-sig",
    )

    train = data[
        data["split"] == "train"
    ]

    train_labels = set(
        train["text"]
    )

    train_characters = set(
        character_inventory(
            train["text"]
        )
    )

    report: dict[str, object] = {
        "seed": SEED,
        "total_rows": int(
            len(data)
        ),
        "total_pages": int(
            total_pages
        ),
        "page_leakage_count": 0,
        "duplicate_images": 0,
        "duplicate_page_word_positions": 0,
        "splits": {},
    }

    for split in (
        "train",
        "val",
        "test",
    ):
        subset = data[
            data["split"] == split
        ]

        split_characters = set(
            character_inventory(
                subset["text"]
            )
        )

        unseen_characters = sorted(
            split_characters
            - train_characters
        )

        unseen_labels = (
            set(subset["text"])
            - train_labels
            if split != "train"
            else set()
        )

        words_per_page = (
            subset.groupby(
                "page_id"
            ).size()
        )

        report["splits"][split] = {
            "rows": int(
                len(subset)
            ),
            "row_percent": round(
                100.0
                * len(subset)
                / len(data),
                3,
            ),
            "pages": int(
                subset[
                    "page_id"
                ].nunique()
            ),
            "page_percent": round(
                100.0
                * subset[
                    "page_id"
                ].nunique()
                / total_pages,
                3,
            ),
            "unique_labels": int(
                subset[
                    "text"
                ].nunique()
            ),
            "character_count": int(
                len(split_characters)
            ),
            "unseen_labels_vs_train": int(
                len(unseen_labels)
            ),
            "unseen_characters_vs_train":
                unseen_characters,
            "minimum_words_per_page": int(
                words_per_page.min()
            ),
            "median_words_per_page": float(
                words_per_page.median()
            ),
            "maximum_words_per_page": int(
                words_per_page.max()
            ),
        }

    report_output = (
        OUTPUT
        / "split_report.json"
    )

    report_output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("IHWWR page-separated split complete")
    print(
        "Total words:",
        f"{len(data):,}",
    )
    print(
        "Total source pages:",
        f"{total_pages:,}",
    )
    print()
    print(
        f"Train pages: "
        f"{train_page_count:,}"
    )
    print(
        f"Validation pages: "
        f"{val_page_count:,}"
    )
    print(
        f"Test pages: "
        f"{test_page_count:,}"
    )

    print()

    for split in (
        "train",
        "val",
        "test",
    ):
        subset = data[
            data["split"] == split
        ]

        print(
            f"{split:>5}: "
            f"{len(subset):,} words | "
            f"{subset['page_id'].nunique():,} pages"
        )

    print()
    print("Page leakage: 0")
    print(
        "Combined manifest:",
        all_output.resolve(),
    )
    print(
        "Page assignments:",
        assignments_output.resolve(),
    )
    print(
        "Report:",
        report_output.resolve(),
    )


if __name__ == "__main__":
    main()
