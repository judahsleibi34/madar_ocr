from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path("data/processed/arabic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    dataframe = pd.read_csv(file_path)

    required_columns = {
        "image",
        "text",
        "language",
        "dataset",
        "split",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{file_path} is missing columns: {sorted(missing)}"
        )

    return dataframe


def combine_split(
    split_name: str,
    input_files: list[str],
) -> None:
    dataframes = [load_csv(path) for path in input_files]

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    combined = combined.dropna(
        subset=["image", "text"],
    )

    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""]

    combined = combined.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / f"{split_name}.csv"

    combined.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"{split_name}: {len(combined)} samples")
    print(combined["dataset"].value_counts().to_string())
    print(f"Saved: {output_path.resolve()}")


def main() -> None:
    combine_split(
        "train",
        [
            "data/processed/khatt/train.csv",
            "data/processed/muharaf/train.csv",
        ],
    )

    combine_split(
        "validation",
        [
            "data/processed/khatt/validation.csv",
            "data/processed/muharaf/validation.csv",
        ],
    )

    # KHATT has no test split, so Arabic test currently uses Muharaf only.
    combine_split(
        "test",
        [
            "data/processed/muharaf/test.csv",
        ],
    )


if __name__ == "__main__":
    main()
