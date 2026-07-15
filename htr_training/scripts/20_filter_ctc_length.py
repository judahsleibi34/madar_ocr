from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from bidi.algorithm import get_display
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_SAFE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_safe"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_ctc"
)

DEFAULT_VOCAB = (
    MASTER_SAFE_ROOT
    / "vocab.json"
)

INPUT_MANIFESTS = {
    "train": MASTER_SAFE_ROOT / "train_all.csv",
    "val": MASTER_SAFE_ROOT / "val_all.csv",
    "test": MASTER_SAFE_ROOT / "test_all.csv",
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
    "visual_text",
    "target_length",
    "adjacent_repeats",
    "required_timesteps",
    "available_timesteps",
    "resized_width",
]

EXCLUSION_FIELDS = [
    *OUTPUT_FIELDS,
    "reason",
    "oov_characters",
    "oov_codepoints",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create OOV-safe and CTC-feasible manifests "
            "without modifying the leakage-safe source files."
        )
    )

    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-image-width",
        type=int,
        default=2560,
        help=(
            "Must match the maximum width used during "
            "training."
        ),
    )

    parser.add_argument(
        "--width-reduction-factor",
        type=int,
        default=4,
    )

    return parser.parse_args()


def load_vocabulary(
    path: Path,
) -> tuple[set[str], int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(file)

    required = {
        "char_to_index",
        "blank_index",
        "num_classes",
    }

    missing = required - set(vocabulary)

    if missing:
        raise ValueError(
            f"Vocabulary missing keys: {sorted(missing)}"
        )

    blank_index = int(
        vocabulary["blank_index"]
    )

    character_set = {
        character
        for character, index
        in vocabulary["char_to_index"].items()
        if int(index) != blank_index
    }

    return character_set, blank_index


def normalize_text(value: object) -> str:
    return unicodedata.normalize(
        "NFC",
        str(value),
    ).strip()


def resolve_image_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_manifest(
    split: str,
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {path}"
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
            "domain",
            "source_type",
            "source_manifest",
            "source_row",
            "sha256",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                f"{path.name} missing columns: "
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

            image = str(
                row.get("image", "")
            ).strip()

            declared_split = str(
                row.get("split", "")
            ).strip()

            if not text:
                raise ValueError(
                    f"Empty text at {path}:{csv_row}"
                )

            if not image:
                raise ValueError(
                    f"Empty image at {path}:{csv_row}"
                )

            if declared_split != split:
                raise ValueError(
                    f"Split mismatch at {path}:{csv_row}: "
                    f"{declared_split!r}"
                )

            row["text"] = text
            row["_csv_row"] = str(csv_row)
            rows.append(row)

    return rows


def calculate_resized_width(
    image_path: Path,
    image_height: int,
    max_image_width: int,
) -> int:
    with Image.open(image_path) as image:
        original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            f"Invalid dimensions for {image_path}: "
            f"{original_width}x{original_height}"
        )

    scale = image_height / original_height

    resized_width = max(
        1,
        round(original_width * scale),
    )

    return min(
        resized_width,
        max_image_width,
    )


def count_adjacent_repeats(
    visual_text: str,
) -> int:
    return sum(
        current == previous
        for previous, current in zip(
            visual_text,
            visual_text[1:],
        )
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


def process_split(
    split: str,
    rows: list[dict[str, str]],
    vocabulary_characters: set[str],
    image_height: int,
    max_image_width: int,
    width_reduction_factor: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for row in tqdm(
        rows,
        desc=f"Checking {split}",
    ):
        visual_text = get_display(
            row["text"]
        )

        unseen = sorted(
            set(visual_text)
            - vocabulary_characters
        )

        base_record: dict[str, Any] = {
            key: row.get(key, "")
            for key in OUTPUT_FIELDS
            if key not in {
                "visual_text",
                "target_length",
                "adjacent_repeats",
                "required_timesteps",
                "available_timesteps",
                "resized_width",
            }
        }

        base_record["visual_text"] = (
            visual_text
        )

        if unseen:
            record = {
                **base_record,
                "target_length": len(
                    visual_text
                ),
                "adjacent_repeats": "",
                "required_timesteps": "",
                "available_timesteps": "",
                "resized_width": "",
                "reason": "out_of_vocabulary",
                "oov_characters": "".join(
                    unseen
                ),
                "oov_codepoints": "|".join(
                    f"U+{ord(character):04X}"
                    for character in unseen
                ),
            }

            excluded.append(record)
            continue

        image_path = resolve_image_path(
            row["image"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image missing: {image_path}"
            )

        resized_width = (
            calculate_resized_width(
                image_path=image_path,
                image_height=image_height,
                max_image_width=max_image_width,
            )
        )

        available_timesteps = max(
            1,
            resized_width
            // width_reduction_factor,
        )

        adjacent_repeats = (
            count_adjacent_repeats(
                visual_text
            )
        )

        target_length = len(
            visual_text
        )

        required_timesteps = (
            target_length
            + adjacent_repeats
        )

        record = {
            **base_record,
            "target_length": target_length,
            "adjacent_repeats": (
                adjacent_repeats
            ),
            "required_timesteps": (
                required_timesteps
            ),
            "available_timesteps": (
                available_timesteps
            ),
            "resized_width": resized_width,
        }

        if (
            available_timesteps
            < required_timesteps
        ):
            excluded.append(
                {
                    **record,
                    "reason": (
                        "insufficient_ctc_timesteps"
                    ),
                    "oov_characters": "",
                    "oov_codepoints": "",
                }
            )
            continue

        accepted.append(record)

    return accepted, excluded


def save_domain_manifests(
    output_root: Path,
    split: str,
    rows: list[dict[str, Any]],
) -> None:
    handwriting = [
        row
        for row in rows
        if row["domain"] == "handwriting"
    ]

    scene_text = [
        row
        for row in rows
        if row["domain"] == "scene_text"
    ]

    write_csv(
        output_root
        / f"{split}_all.csv",
        rows,
        OUTPUT_FIELDS,
    )

    write_csv(
        output_root
        / f"{split}_handwriting.csv",
        handwriting,
        OUTPUT_FIELDS,
    )

    write_csv(
        output_root
        / f"{split}_scene_text.csv",
        scene_text,
        OUTPUT_FIELDS,
    )


def main() -> None:
    arguments = parse_arguments()

    if arguments.image_height <= 0:
        raise ValueError(
            "--image-height must be positive."
        )

    if arguments.max_image_width <= 0:
        raise ValueError(
            "--max-image-width must be positive."
        )

    if arguments.width_reduction_factor <= 0:
        raise ValueError(
            "--width-reduction-factor must be positive."
        )

    vocabulary_characters, blank_index = (
        load_vocabulary(arguments.vocab)
    )

    arguments.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_excluded: list[
        dict[str, Any]
    ] = []

    summary: dict[str, Any] = {
        "vocab": str(
            arguments.vocab
            .resolve()
        ),
        "blank_index": blank_index,
        "image_height": (
            arguments.image_height
        ),
        "max_image_width": (
            arguments.max_image_width
        ),
        "width_reduction_factor": (
            arguments.width_reduction_factor
        ),
        "ctc_rule": (
            "target length plus adjacent repeated "
            "target characters"
        ),
        "splits": {},
    }

    for split, manifest_path in (
        INPUT_MANIFESTS.items()
    ):
        rows = load_manifest(
            split,
            manifest_path,
        )

        accepted, excluded = (
            process_split(
                split=split,
                rows=rows,
                vocabulary_characters=(
                    vocabulary_characters
                ),
                image_height=(
                    arguments.image_height
                ),
                max_image_width=(
                    arguments.max_image_width
                ),
                width_reduction_factor=(
                    arguments.width_reduction_factor
                ),
            )
        )

        save_domain_manifests(
            arguments.output_root,
            split,
            accepted,
        )

        all_excluded.extend(
            excluded
        )

        reason_counts = Counter(
            row["reason"]
            for row in excluded
        )

        summary["splits"][split] = {
            "source_rows": len(rows),
            "accepted_rows": len(
                accepted
            ),
            "excluded_rows": len(
                excluded
            ),
            "exclusion_reasons": dict(
                reason_counts
            ),
            "handwriting_rows": sum(
                row["domain"]
                == "handwriting"
                for row in accepted
            ),
            "scene_text_rows": sum(
                row["domain"]
                == "scene_text"
                for row in accepted
            ),
            "coverage_percent": round(
                100
                * len(accepted)
                / len(rows),
                6,
            ),
        }

        print()
        print(
            f"{split}: "
            f"accepted={len(accepted)}/"
            f"{len(rows)}, "
            f"excluded={len(excluded)}"
        )

        for reason, count in sorted(
            reason_counts.items()
        ):
            print(
                f"  {reason}: {count}"
            )

    write_csv(
        arguments.output_root
        / "excluded_samples.csv",
        all_excluded,
        EXCLUSION_FIELDS,
    )

    summary["total_source_rows"] = sum(
        values["source_rows"]
        for values
        in summary["splits"].values()
    )

    summary["total_accepted_rows"] = sum(
        values["accepted_rows"]
        for values
        in summary["splits"].values()
    )

    summary["total_excluded_rows"] = len(
        all_excluded
    )

    summary_path = (
        arguments.output_root
        / "ctc_manifest_summary.json"
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
    print("CTC manifest preparation completed.")
    print(
        f"Accepted rows: "
        f"{summary['total_accepted_rows']}"
    )
    print(
        f"Excluded rows: "
        f"{summary['total_excluded_rows']}"
    )
    print(
        f"Output directory: "
        f"{arguments.output_root}"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
