from pathlib import Path
import json

import pandas as pd


MANIFESTS = [
    Path("data/processed/iam/train.csv"),
    Path("data/processed/arabic/train.csv"),
]

OUTPUT_PATH = Path("data/processed/vocab.json")


def main() -> None:
    characters = set()

    for manifest_path in MANIFESTS:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        dataframe = pd.read_csv(manifest_path)

        if "text" not in dataframe.columns:
            raise ValueError(
                f"'text' column missing in {manifest_path}"
            )

        for text in dataframe["text"].dropna().astype(str):
            characters.update(text)

    sorted_characters = sorted(characters)

    # CTC blank must be index 0.
    char_to_index = {
        "<BLANK>": 0,
    }

    for index, character in enumerate(
        sorted_characters,
        start=1,
    ):
        char_to_index[character] = index

    index_to_char = {
        str(index): character
        for character, index in char_to_index.items()
    }

    output = {
        "char_to_index": char_to_index,
        "index_to_char": index_to_char,
        "blank_index": 0,
        "num_classes": len(char_to_index),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Characters: {len(sorted_characters)}")
    print(f"CTC classes: {len(char_to_index)}")
    print(f"Saved: {OUTPUT_PATH.resolve()}")

    print("\nCharacters:")
    print("".join(sorted_characters))


if __name__ == "__main__":
    main()
