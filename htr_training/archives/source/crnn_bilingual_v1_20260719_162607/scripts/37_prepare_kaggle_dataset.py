from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_ctc"
)

DEFAULT_VOCAB = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_safe"
    / "vocab.json"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "kaggle"
    / "madar_ocr_ctc"
)

MANIFESTS = {
    "train": "train_all.csv",
    "val": "val_all.csv",
    "test": "test_all.csv",
}

REQUIRED_COLUMNS = {
    "image",
    "text",
    "language",
    "dataset",
    "split",
    "domain",
    "source_type",
    "sha256",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a portable Kaggle dataset directory from the "
            "leakage-safe, CTC-ready manifests."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
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
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help=(
            "Hardlink avoids a second local copy when source and "
            "destination are on the same drive. It falls back to copy."
        ),
    )

    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        help=(
            "Dataset name to exclude. May be supplied more than once."
        ),
    )

    parser.add_argument(
        "--kaggle-id",
        default=(
            "YOUR_KAGGLE_USERNAME/"
            "madar-ocr-bilingual-ctc"
        ),
        help=(
            "Kaggle dataset ID in username/slug form. "
            "The default is a placeholder."
        ),
    )

    parser.add_argument(
        "--title",
        default="MADAR OCR Bilingual CTC Training Package",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def safe_suffix(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }:
        return suffix

    return ".img"


def link_or_copy(
    source: Path,
    destination: Path,
    copy_mode: str,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        return "existing"

    if copy_mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass

    shutil.copy2(source, destination)
    return "copy"


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


def main() -> None:
    arguments = parse_arguments()

    input_root = arguments.input_root.resolve()
    vocab_path = arguments.vocab.resolve()
    output_root = arguments.output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input root not found: {input_root}"
        )

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {vocab_path}"
        )

    if output_root.exists():
        if not arguments.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                "Use --overwrite to rebuild it."
            )

        shutil.rmtree(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_root = output_root / "images"
    exclude_datasets = {
        value.strip()
        for value in arguments.exclude_dataset
        if value.strip()
    }

    copied_by_hash: dict[str, str] = {}
    copy_counts: Counter[str] = Counter()
    split_summaries: dict[str, dict[str, Any]] = {}
    package_rows: list[dict[str, Any]] = []

    for split, filename in MANIFESTS.items():
        manifest_path = input_root / filename

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        dataframe = pd.read_csv(
            manifest_path,
            dtype=str,
            keep_default_na=False,
        )

        missing = (
            REQUIRED_COLUMNS
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                f"{manifest_path.name} missing columns: "
                f"{sorted(missing)}"
            )

        source_rows = len(dataframe)

        if exclude_datasets:
            dataframe = dataframe[
                ~dataframe["dataset"].isin(
                    exclude_datasets
                )
            ].copy()

        output_rows: list[dict[str, Any]] = []
        dataset_counts: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()

        for _, series in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc=f"Packaging {split}",
        ):
            row = {
                str(key): str(value)
                for key, value in series.to_dict().items()
            }

            source_image = resolve_project_path(
                row["image"]
            )

            if not source_image.exists():
                raise FileNotFoundError(
                    f"Missing image: {source_image}"
                )

            expected_hash = row["sha256"].strip().lower()

            if len(expected_hash) != 64:
                raise ValueError(
                    "Invalid SHA-256 in manifest for "
                    f"{source_image}: {expected_hash!r}"
                )

            if expected_hash not in copied_by_hash:
                actual_hash = calculate_sha256(
                    source_image
                )

                if actual_hash != expected_hash:
                    raise RuntimeError(
                        "Image hash mismatch:\n"
                        f"  image: {source_image}\n"
                        f"  manifest: {expected_hash}\n"
                        f"  actual:   {actual_hash}"
                    )

                relative_image = (
                    Path("images")
                    / expected_hash[:2]
                    / (
                        expected_hash
                        + safe_suffix(source_image)
                    )
                )

                destination = (
                    output_root
                    / relative_image
                )

                operation = link_or_copy(
                    source=source_image,
                    destination=destination,
                    copy_mode=arguments.copy_mode,
                )

                copy_counts[operation] += 1
                copied_by_hash[
                    expected_hash
                ] = relative_image.as_posix()

            row["image"] = copied_by_hash[
                expected_hash
            ]

            output_rows.append(row)
            package_rows.append(row)

            dataset_counts[row["dataset"]] += 1
            language_counts[row["language"]] += 1
            domain_counts[row["domain"]] += 1

        output_manifest = (
            output_root / filename
        )

        write_csv(
            output_manifest,
            output_rows,
            list(dataframe.columns),
        )

        split_summaries[split] = {
            "source_rows": source_rows,
            "packaged_rows": len(output_rows),
            "excluded_rows": (
                source_rows - len(output_rows)
            ),
            "datasets": dict(
                sorted(dataset_counts.items())
            ),
            "languages": dict(
                sorted(language_counts.items())
            ),
            "domains": dict(
                sorted(domain_counts.items())
            ),
        }

    shutil.copy2(
        vocab_path,
        output_root / "vocab.json",
    )

    optional_files = [
        input_root / "ctc_manifest_summary.json",
        input_root / "excluded_samples.csv",
    ]

    copied_reports: list[str] = []

    for source in optional_files:
        if source.exists():
            destination = (
                output_root / source.name
            )
            shutil.copy2(source, destination)
            copied_reports.append(source.name)

    dataset_metadata = {
        "title": arguments.title,
        "id": arguments.kaggle_id,
        "licenses": [
            {
                "name": "other",
            }
        ],
        "subtitle": (
            "Leakage-safe Arabic-English OCR manifests "
            "and referenced images for CRNN-CTC training"
        ),
        "description": (
            "Private research training package assembled from "
            "multiple source datasets. Each source remains subject "
            "to its own license and usage terms. Do not publish "
            "this package publicly until redistribution permission "
            "has been verified for every included source."
        ),
        "keywords": [
            "ocr",
            "handwriting",
            "arabic",
            "english",
        ],
    }

    (
        output_root
        / "dataset-metadata.json"
    ).write_text(
        json.dumps(
            dataset_metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    image_files = [
        path
        for path in image_root.rglob("*")
        if path.is_file()
    ]

    total_image_bytes = sum(
        path.stat().st_size
        for path in image_files
    )

    summary = {
        "package_root": str(output_root),
        "input_root": str(input_root),
        "vocab": "vocab.json",
        "kaggle_id": arguments.kaggle_id,
        "excluded_datasets": sorted(
            exclude_datasets
        ),
        "splits": split_summaries,
        "total_manifest_rows": len(
            package_rows
        ),
        "unique_images": len(
            image_files
        ),
        "image_bytes": total_image_bytes,
        "image_gib": round(
            total_image_bytes
            / (1024 ** 3),
            3,
        ),
        "file_operations": dict(
            sorted(copy_counts.items())
        ),
        "copied_reports": copied_reports,
        "path_policy": (
            "Image paths are relative to the package root."
        ),
        "license_warning": (
            "This is a mixed-source research package. "
            "Keep the Kaggle dataset private unless every "
            "source permits public redistribution."
        ),
    }

    (
        output_root
        / "package_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = f"""# MADAR OCR Kaggle training package

This directory contains CTC-ready manifests, a vocabulary, and
the exact image files referenced by the manifests.

## Splits

- train_all.csv
- val_all.csv
- test_all.csv

## Vocabulary

- vocab.json

## Important path note

The `image` column uses paths relative to this package root.
The training dataset loader must resolve relative paths against
the manifest directory, or the notebook must change its working
directory to this package root before constructing the dataset.

## License note

This package combines multiple source datasets. Each source keeps
its original terms. Keep the Kaggle dataset private unless public
redistribution is allowed for every included source.

## Package counts

- rows: {summary["total_manifest_rows"]}
- unique images: {summary["unique_images"]}
- image size: {summary["image_gib"]} GiB
"""

    (
        output_root / "README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    # Final path validation.
    for filename in MANIFESTS.values():
        manifest = pd.read_csv(
            output_root / filename,
            dtype=str,
            keep_default_na=False,
        )

        missing_paths = [
            value
            for value in manifest["image"]
            if not (
                output_root / value
            ).exists()
        ]

        if missing_paths:
            raise RuntimeError(
                f"{filename} has "
                f"{len(missing_paths)} missing package paths."
            )

    print()
    print("Kaggle package completed.")
    print(f"Output: {output_root}")
    print(
        f"Rows: {summary['total_manifest_rows']}"
    )
    print(
        f"Unique images: {summary['unique_images']}"
    )
    print(
        f"Image size: {summary['image_gib']} GiB"
    )
    print(
        "Metadata: "
        f"{output_root / 'dataset-metadata.json'}"
    )
    print(
        "Summary: "
        f"{output_root / 'package_summary.json'}"
    )

    if arguments.kaggle_id.startswith(
        "YOUR_KAGGLE_USERNAME/"
    ):
        print()
        print(
            "WARNING: Replace the placeholder Kaggle ID "
            "before upload."
        )


if __name__ == "__main__":
    main()