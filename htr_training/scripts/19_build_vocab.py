from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bidi.algorithm import get_display


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_safe"
)

DEFAULT_OUTPUT = MASTER_ROOT / "vocab.json"

DEFAULT_OLD_VOCAB = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vocab.json"
)

MANIFESTS = {
    "train": MASTER_ROOT / "train_all.csv",
    "val": MASTER_ROOT / "val_all.csv",
    "test": MASTER_ROOT / "test_all.csv",
}

BLANK_TOKEN = "<BLANK>"
BLANK_INDEX = 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CRNN/CTC vocabulary from the "
            "leakage-safe master manifests."
        )
    )

    parser.add_argument(
        "--scope",
        choices=("train", "all"),
        default="train",
        help=(
            "Use only training labels, or the union of "
            "train/validation/test labels. Default: train."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--old-vocab",
        type=Path,
        default=DEFAULT_OLD_VOCAB,
        help=(
            "Existing vocabulary used for checkpoint "
            "compatibility comparison."
        ),
    )

    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize(
        "NFC",
        str(value),
    )

    return text.strip()


def to_visual(text: str) -> str:
    """
    Match the current CTCTextEncoder behavior.

    Arabic labels remain logical in CSV manifests. They are
    converted to visual order only when building/encoding the
    CTC target sequence.
    """
    return get_display(text)


def load_manifest(
    split: str,
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split} manifest: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "image",
            "text",
            "language",
            "dataset",
            "split",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                f"{path} is missing columns: "
                f"{sorted(missing)}"
            )

        rows: list[dict[str, str]] = []

        for csv_row, row in enumerate(
            reader,
            start=2,
        ):
            text = normalize_text(
                row.get("text", "")
            )

            if not text:
                raise ValueError(
                    f"Empty text at {path}:{csv_row}"
                )

            declared_split = str(
                row.get("split", "")
            ).strip()

            if declared_split != split:
                raise ValueError(
                    f"Split mismatch at {path}:{csv_row}: "
                    f"{declared_split!r}"
                )

            row["text"] = text
            row["_csv_row"] = str(csv_row)
            rows.append(row)

    return rows


def display_character(character: str) -> str:
    replacements = {
        " ": "<SPACE>",
        "\t": "<TAB>",
        "\n": "<NEWLINE>",
        "\r": "<CARRIAGE_RETURN>",
    }

    return replacements.get(
        character,
        character,
    )


def unicode_name(character: str) -> str:
    return unicodedata.name(
        character,
        "UNKNOWN",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def load_old_vocabulary(
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(file)

    required = {
        "char_to_index",
        "index_to_char",
        "blank_index",
        "num_classes",
    }

    missing = required - set(vocabulary)

    if missing:
        raise ValueError(
            f"Old vocabulary is missing keys: "
            f"{sorted(missing)}"
        )

    return vocabulary


def compare_vocabularies(
    old_vocabulary: dict[str, Any] | None,
    new_char_to_index: dict[str, int],
) -> dict[str, Any]:
    if old_vocabulary is None:
        return {
            "old_vocabulary_found": False,
            "checkpoint_head_compatible": False,
            "reason": "old vocabulary not found",
        }

    old_char_to_index = {
        str(character): int(index)
        for character, index
        in old_vocabulary[
            "char_to_index"
        ].items()
    }

    old_characters = (
        set(old_char_to_index)
        - {BLANK_TOKEN}
    )

    new_characters = (
        set(new_char_to_index)
        - {BLANK_TOKEN}
    )

    shared = (
        old_characters
        & new_characters
    )

    index_changes = sorted(
        character
        for character in shared
        if old_char_to_index[character]
        != new_char_to_index[character]
    )

    added = sorted(
        new_characters
        - old_characters
    )

    removed = sorted(
        old_characters
        - new_characters
    )

    same_class_count = (
        int(old_vocabulary["num_classes"])
        == len(new_char_to_index)
    )

    exact_mapping_match = (
        old_char_to_index
        == new_char_to_index
    )

    return {
        "old_vocabulary_found": True,
        "old_num_classes": int(
            old_vocabulary["num_classes"]
        ),
        "new_num_classes": len(
            new_char_to_index
        ),
        "same_class_count": same_class_count,
        "exact_mapping_match": (
            exact_mapping_match
        ),
        "checkpoint_head_compatible": (
            exact_mapping_match
        ),
        "shared_characters": len(shared),
        "added_characters": len(added),
        "removed_characters": len(removed),
        "index_changed_characters": len(
            index_changes
        ),
        "added": added,
        "removed": removed,
        "index_changed": index_changes,
        "reason": (
            "exact vocabulary mapping match"
            if exact_mapping_match
            else (
                "the classifier head cannot be loaded "
                "directly because the character mapping "
                "changed"
            )
        ),
    }


def main() -> None:
    arguments = parse_arguments()

    rows_by_split = {
        split: load_manifest(split, path)
        for split, path in MANIFESTS.items()
    }

    vocabulary_splits = (
        ("train",)
        if arguments.scope == "train"
        else ("train", "val", "test")
    )

    character_counts: Counter[str] = Counter()

    for split in vocabulary_splits:
        for row in rows_by_split[split]:
            visual_text = to_visual(
                row["text"]
            )

            character_counts.update(
                visual_text
            )

    characters = sorted(
        character_counts
    )

    if not characters:
        raise RuntimeError(
            "No vocabulary characters were found."
        )

    char_to_index: dict[str, int] = {
        BLANK_TOKEN: BLANK_INDEX,
    }

    for index, character in enumerate(
        characters,
        start=1,
    ):
        char_to_index[character] = index

    index_to_char = {
        str(index): character
        for character, index
        in char_to_index.items()
    }

    vocabulary = {
        "char_to_index": char_to_index,
        "index_to_char": index_to_char,
        "blank_index": BLANK_INDEX,
        "num_classes": len(
            char_to_index
        ),
        "scope": arguments.scope,
        "source_manifests": [
            MANIFESTS[split]
            .relative_to(PROJECT_ROOT)
            .as_posix()
            for split in vocabulary_splits
        ],
        "text_order": (
            "visual order produced by "
            "python-bidi get_display"
        ),
    }

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.output.write_text(
        json.dumps(
            vocabulary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_root = (
        arguments.output.parent
        / "vocab_reports"
    )

    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocabulary_rows = [
        {
            "index": char_to_index[
                character
            ],
            "character": (
                display_character(character)
            ),
            "literal_character": character,
            "codepoint": (
                f"U+{ord(character):04X}"
            ),
            "unicode_name": (
                unicode_name(character)
            ),
            "vocabulary_occurrences": (
                character_counts[character]
            ),
        }
        for character in characters
    ]

    write_csv(
        report_root / "vocabulary.csv",
        vocabulary_rows,
        [
            "index",
            "character",
            "literal_character",
            "codepoint",
            "unicode_name",
            "vocabulary_occurrences",
        ],
    )

    vocabulary_character_set = set(
        characters
    )

    oov_character_rows: list[
        dict[str, Any]
    ] = []

    oov_sample_rows: list[
        dict[str, Any]
    ] = []

    oov_summary: dict[
        str,
        dict[str, int]
    ] = {}

    for split in ("train", "val", "test"):
        oov_occurrences: Counter[str] = (
            Counter()
        )

        rows_containing: defaultdict[
            str,
            int,
        ] = defaultdict(int)

        affected_rows = 0

        examples: defaultdict[
            str,
            list[str],
        ] = defaultdict(list)

        for row in rows_by_split[split]:
            visual_text = to_visual(
                row["text"]
            )

            unseen = sorted(
                set(visual_text)
                - vocabulary_character_set
            )

            if not unseen:
                continue

            affected_rows += 1

            for character in visual_text:
                if (
                    character
                    not in vocabulary_character_set
                ):
                    oov_occurrences[
                        character
                    ] += 1

            for character in unseen:
                rows_containing[
                    character
                ] += 1

                if len(
                    examples[character]
                ) < 5:
                    examples[character].append(
                        row["text"]
                    )

            oov_sample_rows.append(
                {
                    "split": split,
                    "csv_row": row[
                        "_csv_row"
                    ],
                    "dataset": row[
                        "dataset"
                    ],
                    "language": row[
                        "language"
                    ],
                    "image": row[
                        "image"
                    ],
                    "logical_text": row[
                        "text"
                    ],
                    "visual_text": (
                        visual_text
                    ),
                    "oov_characters": (
                        "".join(unseen)
                    ),
                    "oov_codepoints": (
                        "|".join(
                            f"U+{ord(character):04X}"
                            for character in unseen
                        )
                    ),
                }
            )

        for character in sorted(
            oov_occurrences
        ):
            oov_character_rows.append(
                {
                    "split": split,
                    "character": (
                        display_character(
                            character
                        )
                    ),
                    "literal_character": (
                        character
                    ),
                    "codepoint": (
                        f"U+{ord(character):04X}"
                    ),
                    "unicode_name": (
                        unicode_name(character)
                    ),
                    "occurrences": (
                        oov_occurrences[
                            character
                        ]
                    ),
                    "rows_containing": (
                        rows_containing[
                            character
                        ]
                    ),
                    "sample_texts": (
                        " | ".join(
                            examples[character]
                        )
                    ),
                }
            )

        oov_summary[split] = {
            "unique_oov_characters": len(
                oov_occurrences
            ),
            "oov_occurrences": sum(
                oov_occurrences.values()
            ),
            "affected_rows": (
                affected_rows
            ),
        }

    write_csv(
        report_root / "oov_report.csv",
        oov_character_rows,
        [
            "split",
            "character",
            "literal_character",
            "codepoint",
            "unicode_name",
            "occurrences",
            "rows_containing",
            "sample_texts",
        ],
    )

    write_csv(
        report_root / "oov_samples.csv",
        oov_sample_rows,
        [
            "split",
            "csv_row",
            "dataset",
            "language",
            "image",
            "logical_text",
            "visual_text",
            "oov_characters",
            "oov_codepoints",
        ],
    )

    old_vocabulary = load_old_vocabulary(
        arguments.old_vocab
    )

    comparison = compare_vocabularies(
        old_vocabulary,
        char_to_index,
    )

    comparison_path = (
        report_root
        / "old_vocab_comparison.json"
    )

    comparison_path.write_text(
        json.dumps(
            comparison,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "scope": arguments.scope,
        "training_rows": len(
            rows_by_split["train"]
        ),
        "validation_rows": len(
            rows_by_split["val"]
        ),
        "test_rows": len(
            rows_by_split["test"]
        ),
        "unique_characters": len(
            characters
        ),
        "ctc_num_classes": len(
            char_to_index
        ),
        "blank_index": BLANK_INDEX,
        "oov": oov_summary,
        "old_vocabulary_comparison": (
            comparison
        ),
        "vocab_path": (
            arguments.output
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
    }

    summary_path = (
        report_root
        / "vocab_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Vocabulary build completed.")
    print(
        f"Scope: {arguments.scope}"
    )
    print(
        f"Unique characters: "
        f"{len(characters)}"
    )
    print(
        f"CTC classes: "
        f"{len(char_to_index)}"
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        values = oov_summary[split]

        print(
            f"{split}: "
            f"OOV characters="
            f"{values['unique_oov_characters']}, "
            f"affected rows="
            f"{values['affected_rows']}"
        )

    print(
        "Old checkpoint vocabulary compatible: "
        f"{comparison['checkpoint_head_compatible']}"
    )

    print(
        f"Vocabulary: {arguments.output}"
    )
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
