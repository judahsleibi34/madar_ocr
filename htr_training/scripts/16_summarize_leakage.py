from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_validation"
    / "duplicate_report.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_validation"
)


def count_text_variants(value: object) -> int:
    if pd.isna(value):
        return 0

    texts = {
        part.strip()
        for part in str(value).split("|")
        if part.strip()
    }

    return len(texts)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing duplicate report: {INPUT_PATH}"
        )

    duplicates = pd.read_csv(INPUT_PATH)

    required_columns = {
        "sha256",
        "count",
        "reason",
        "splits",
        "datasets",
        "texts",
        "images",
    }

    missing = required_columns - set(duplicates.columns)

    if missing:
        raise ValueError(
            f"Duplicate report is missing: {sorted(missing)}"
        )

    cross_split = duplicates[
        duplicates["reason"] == "cross_split_duplicate"
    ].copy()

    cross_split["text_variant_count"] = (
        cross_split["texts"].apply(count_text_variants)
    )

    cross_split["label_status"] = cross_split[
        "text_variant_count"
    ].apply(
        lambda count: (
            "same_label"
            if count == 1
            else "conflicting_labels"
        )
    )

    summary = (
        cross_split.groupby(
            [
                "datasets",
                "splits",
                "label_status",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="duplicate_groups")
        .sort_values(
            "duplicate_groups",
            ascending=False,
        )
    )

    same_label = cross_split[
        cross_split["label_status"] == "same_label"
    ]

    conflicting = cross_split[
        cross_split["label_status"]
        == "conflicting_labels"
    ]

    summary_path = (
        OUTPUT_ROOT / "cross_split_summary.csv"
    )

    examples_path = (
        OUTPUT_ROOT / "cross_split_examples.csv"
    )

    conflicting_path = (
        OUTPUT_ROOT
        / "cross_split_conflicting_labels.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    cross_split.to_csv(
        examples_path,
        index=False,
        encoding="utf-8-sig",
    )

    conflicting.to_csv(
        conflicting_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("Leakage analysis")
    print("=" * 70)
    print(f"All duplicate groups: {len(duplicates)}")
    print(f"Cross-split groups: {len(cross_split)}")
    print(f"Same-label leakage: {len(same_label)}")
    print(
        "Conflicting-label leakage: "
        f"{len(conflicting)}"
    )

    print()
    print("Largest dataset/split combinations:")
    print(
        summary.head(30).to_string(index=False)
    )

    print()
    print(f"Summary: {summary_path}")
    print(f"All examples: {examples_path}")
    print(f"Conflicting labels: {conflicting_path}")


if __name__ == "__main__":
    main()
