import json
import pandas as pd
from pathlib import Path


def load_extraction_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_text_to_dataframe(extraction_data: dict, source_file: str) -> pd.DataFrame:
    rows = []
    for block in extraction_data.get("text", []):
        rows.append({
            "source_file": source_file,
            "text": block.get("text", ""),
        })
    return pd.DataFrame(rows)


def flatten_tables_to_dataframe(extraction_data: dict, source_file: str) -> list:
    dataframes = []
    for table in extraction_data.get("tables", []):
        rows = table.get("rows", [])
        if not rows:
            continue
        header, *data_rows = rows
        df = pd.DataFrame(data_rows, columns=header)
        df["source_file"] = source_file
        df["page"] = table.get("page")
        dataframes.append(df)
    return dataframes
