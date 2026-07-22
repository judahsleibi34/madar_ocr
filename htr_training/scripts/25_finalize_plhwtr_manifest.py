from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT = Path(
    r"data\processed\online_htr"
    r"\plhwtr_english\pages_clean.csv"
)

TRAINABLE = Path(
    r"data\processed\online_htr"
    r"\plhwtr_english\pages_trainable.csv"
)

QUARANTINE = Path(
    r"data\processed\online_htr"
    r"\plhwtr_english\pages_quarantine.csv"
)

REPORT = Path(
    r"data\processed\online_htr"
    r"\plhwtr_english\finalization_report.json"
)


def as_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input manifest missing: {INPUT.resolve()}"
        )

    data = pd.read_csv(INPUT)

    required_columns = {
        "image",
        "text",
        "starts_with_number",
        "residual_boundary_id",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    starts_with_number = as_boolean(
        data["starts_with_number"]
    )

    residual_boundary_id = as_boolean(
        data["residual_boundary_id"]
    )

    empty_text = (
        data["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    suspicious = (
        starts_with_number
        | residual_boundary_id
        | empty_text
    )

    trainable = data.loc[
        ~suspicious
    ].copy()

    quarantine = data.loc[
        suspicious
    ].copy()

    quarantine["quarantine_reason"] = ""

    quarantine.loc[
        starts_with_number[suspicious],
        "quarantine_reason",
    ] += "starts_with_number;"

    quarantine.loc[
        residual_boundary_id[suspicious],
        "quarantine_reason",
    ] += "residual_boundary_id;"

    quarantine.loc[
        empty_text[suspicious],
        "quarantine_reason",
    ] += "empty_text;"

    quarantine["quarantine_reason"] = (
        quarantine["quarantine_reason"]
        .str.rstrip(";")
    )

    trainable.to_csv(
        TRAINABLE,
        index=False,
        encoding="utf-8-sig",
    )

    quarantine.to_csv(
        QUARANTINE,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "input_pages": int(len(data)),
        "trainable_pages": int(len(trainable)),
        "quarantined_pages": int(len(quarantine)),
        "empty_targets": int(empty_text.sum()),
        "starts_with_number": int(
            starts_with_number.sum()
        ),
        "residual_boundary_ids": int(
            residual_boundary_id.sum()
        ),
        "trainable_total_segments": int(
            trainable["segment_count"].sum()
        ),
        "quarantine_reason_counts": {
            str(key): int(value)
            for key, value in (
                quarantine[
                    "quarantine_reason"
                ].value_counts().items()
            )
        },
    }

    REPORT.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("PLHWTR finalization complete")
    print(
        "Input pages:",
        f"{len(data):,}",
    )
    print(
        "Trainable pages:",
        f"{len(trainable):,}",
    )
    print(
        "Quarantined pages:",
        f"{len(quarantine):,}",
    )
    print(
        "Trainable segments:",
        f"{report['trainable_total_segments']:,}",
    )
    print()
    print(
        "Training manifest:",
        TRAINABLE.resolve(),
    )
    print(
        "Quarantine manifest:",
        QUARANTINE.resolve(),
    )
    print(
        "Report:",
        REPORT.resolve(),
    )


if __name__ == "__main__":
    main()
