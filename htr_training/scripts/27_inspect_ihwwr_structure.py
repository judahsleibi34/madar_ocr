from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST = Path(
    r"data\processed\online_htr"
    r"\ihwwr_english\words_all.csv"
)

EXTRACTED_ROOT = Path(
    r"data\raw\online_htr"
    r"\ihwwr_english\extracted"
).resolve()

OUTPUT = Path(
    r"data\processed\online_htr"
    r"\ihwwr_english"
)


def safe_relative_path(
    value: str,
) -> str:
    path = Path(value).resolve()

    try:
        return path.relative_to(
            EXTRACTED_ROOT
        ).as_posix()
    except ValueError:
        return path.as_posix()


def tokenize_stem(
    stem: str,
) -> list[str]:
    return [
        token
        for token in re.split(
            r"[_\-\s.]+",
            stem,
        )
        if token
    ]


def first_numeric_token(
    stem: str,
) -> str:
    match = re.search(
        r"\d+",
        stem,
    )

    return (
        match.group(0)
        if match
        else ""
    )


def numeric_tokens(
    stem: str,
) -> list[str]:
    return re.findall(
        r"\d+",
        stem,
    )


def filename_pattern(
    stem: str,
) -> str:
    pattern = re.sub(
        r"\d+",
        "<N>",
        stem,
    )

    pattern = re.sub(
        r"[A-Za-z]+",
        "<A>",
        pattern,
    )

    return pattern


def describe_groups(
    data: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    usable = data[
        data[column]
        .fillna("")
        .astype(str)
        .str.len()
        > 0
    ]

    counts = usable.groupby(
        column
    ).size()

    if counts.empty:
        return {
            "candidate": column,
            "rows_covered": 0,
            "coverage_percent": 0.0,
            "groups": 0,
            "singleton_groups": 0,
            "singleton_percent": 0.0,
            "minimum_group_size": 0,
            "median_group_size": 0.0,
            "mean_group_size": 0.0,
            "maximum_group_size": 0,
        }

    singleton_groups = int(
        (counts == 1).sum()
    )

    return {
        "candidate": column,
        "rows_covered": int(
            len(usable)
        ),
        "coverage_percent": round(
            100.0
            * len(usable)
            / len(data),
            3,
        ),
        "groups": int(
            len(counts)
        ),
        "singleton_groups":
            singleton_groups,
        "singleton_percent": round(
            100.0
            * singleton_groups
            / len(counts),
            3,
        ),
        "minimum_group_size": int(
            counts.min()
        ),
        "median_group_size": float(
            counts.median()
        ),
        "mean_group_size": round(
            float(counts.mean()),
            3,
        ),
        "maximum_group_size": int(
            counts.max()
        ),
    }


def main() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            MANIFEST.resolve()
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        MANIFEST
    )

    required = {
        "image",
        "text",
    }

    missing = required - set(
        data.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    data["relative_path"] = (
        data["image"]
        .astype(str)
        .map(safe_relative_path)
    )

    path_parts = data[
        "relative_path"
    ].map(
        lambda value: Path(
            value
        ).parts
    )

    data["path_depth"] = (
        path_parts.map(len)
    )

    maximum_depth = int(
        data["path_depth"].max()
    )

    for index in range(
        maximum_depth
    ):
        data[
            f"path_component_{index}"
        ] = path_parts.map(
            lambda parts: (
                parts[index]
                if index < len(parts)
                else ""
            )
        )

    data["filename"] = data[
        "relative_path"
    ].map(
        lambda value: Path(
            value
        ).name
    )

    data["stem"] = data[
        "filename"
    ].map(
        lambda value: Path(
            value
        ).stem
    )

    tokens = data[
        "stem"
    ].map(tokenize_stem)

    data["token_count"] = (
        tokens.map(len)
    )

    for index in range(6):
        data[
            f"stem_token_{index}"
        ] = tokens.map(
            lambda values: (
                values[index]
                if index < len(values)
                else ""
            )
        )

    data["first_numeric_token"] = (
        data["stem"]
        .map(first_numeric_token)
    )

    data["numeric_tokens_json"] = (
        data["stem"]
        .map(
            lambda value: json.dumps(
                numeric_tokens(value)
            )
        )
    )

    data["numeric_token_count"] = (
        data["stem"]
        .map(
            lambda value: len(
                numeric_tokens(value)
            )
        )
    )

    data["filename_pattern"] = (
        data["stem"]
        .map(filename_pattern)
    )

    # Common grouping candidates.
    data["first_two_tokens"] = (
        data["stem_token_0"]
        .astype(str)
        + "|"
        + data["stem_token_1"]
        .astype(str)
    ).str.strip("|")

    data["first_three_tokens"] = (
        data["stem_token_0"]
        .astype(str)
        + "|"
        + data["stem_token_1"]
        .astype(str)
        + "|"
        + data["stem_token_2"]
        .astype(str)
    ).str.strip("|")

    audit_path = (
        OUTPUT
        / "filename_structure_audit.csv"
    )

    data.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_columns = [
        column
        for column in data.columns
        if column.startswith(
            "path_component_"
        )
    ]

    candidate_columns += [
        "parent_directory",
        "stem_token_0",
        "stem_token_1",
        "stem_token_2",
        "first_numeric_token",
        "first_two_tokens",
        "first_three_tokens",
    ]

    candidate_columns = [
        column
        for column in candidate_columns
        if column in data.columns
    ]

    summaries = pd.DataFrame(
        [
            describe_groups(
                data,
                column,
            )
            for column in candidate_columns
        ]
    )

    summary_path = (
        OUTPUT
        / "group_candidate_summary.csv"
    )

    summaries.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    pattern_counts = (
        data[
            "filename_pattern"
        ]
        .value_counts()
        .rename_axis(
            "filename_pattern"
        )
        .reset_index(
            name="count"
        )
    )

    pattern_path = (
        OUTPUT
        / "filename_patterns.csv"
    )

    pattern_counts.to_csv(
        pattern_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("IHWWR filename inspection")
    print(
        "Rows:",
        f"{len(data):,}",
    )
    print(
        "Unique relative paths:",
        f"{data['relative_path'].nunique():,}",
    )
    print(
        "Path depths:",
    )
    print(
        data[
            "path_depth"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Most common filename patterns:"
    )
    print(
        pattern_counts.head(
            20
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "Candidate grouping summary:"
    )
    print(
        summaries.to_string(
            index=False
        )
    )

    print()
    print(
        "First 50 paths and labels:"
    )

    sample_columns = [
        "relative_path",
        "text",
        "stem_token_0",
        "stem_token_1",
        "stem_token_2",
        "first_numeric_token",
    ]

    print(
        data[
            sample_columns
        ].head(
            50
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "Detailed audit:",
        audit_path.resolve(),
    )
    print(
        "Group summary:",
        summary_path.resolve(),
    )
    print(
        "Pattern summary:",
        pattern_path.resolve(),
    )


if __name__ == "__main__":
    main()
