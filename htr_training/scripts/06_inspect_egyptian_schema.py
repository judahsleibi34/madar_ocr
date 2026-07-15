from __future__ import annotations

from pathlib import Path
import json

import pyarrow.parquet as pq
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "egyptian_handwriting"
    / "data"
    / "egy_handwriting_dataset_set1.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "egyptian_handwriting_schema.json"
)


def describe_value(value: object) -> dict[str, object]:
    description: dict[str, object] = {
        "python_type": type(value).__name__,
    }

    if isinstance(value, str):
        description["preview"] = value[:200]

    elif isinstance(value, dict):
        description["keys"] = list(value.keys())

        for key, nested_value in value.items():
            description[f"{key}_type"] = type(
                nested_value
            ).__name__

    elif hasattr(value, "size") and hasattr(value, "mode"):
        description["image_size"] = list(value.size)
        description["image_mode"] = value.mode

    elif value is not None:
        description["preview"] = str(value)[:200]

    return description


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {PARQUET_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_file = pq.ParquetFile(PARQUET_PATH)

    dataset = load_dataset(
        "parquet",
        data_files=str(PARQUET_PATH),
        split="train",
    )

    print(f"Rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print()
    print("Features:")

    for column, feature in dataset.features.items():
        print(f"  {column}: {feature}")

    sample = dataset[0]

    sample_description = {
        column: describe_value(value)
        for column, value in sample.items()
    }

    report = {
        "parquet_path": str(PARQUET_PATH),
        "rows": len(dataset),
        "columns": dataset.column_names,
        "features": {
            column: str(feature)
            for column, feature in dataset.features.items()
        },
        "sample": sample_description,
        "arrow_schema": str(parquet_file.schema_arrow),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("First-row value types:")

    for column, description in sample_description.items():
        print(f"  {column}: {description}")

    print()
    print(f"Saved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
