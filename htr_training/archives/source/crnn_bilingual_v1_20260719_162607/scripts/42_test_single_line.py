from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def detect_column(
    dataframe: pd.DataFrame,
    preferred_names: list[str],
    contains_words: list[str],
) -> str:
    lower_to_original = {
        str(column).lower(): str(column)
        for column in dataframe.columns
    }

    for name in preferred_names:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]

    for column in dataframe.columns:
        lower = str(column).lower()

        if any(word in lower for word in contains_words):
            return str(column)

    raise ValueError(
        f"Could not detect column. Available columns: "
        f"{dataframe.columns.tolist()}"
    )


def detect_image_column(dataframe: pd.DataFrame) -> str:
    preferred = [
        "image_path",
        "path",
        "image",
        "filepath",
        "file_path",
        "filename",
        "file",
    ]

    lower_to_original = {
        str(column).lower(): str(column)
        for column in dataframe.columns
    }

    for name in preferred:
        if name in lower_to_original:
            return lower_to_original[name]

    for column in dataframe.columns:
        values = (
            dataframe[column]
            .dropna()
            .astype(str)
            .head(20)
        )

        matches = sum(
            Path(value).suffix.lower() in IMAGE_EXTENSIONS
            for value in values
        )

        if matches > 0:
            return str(column)

    raise ValueError(
        "Could not detect the image column. "
        f"Columns: {dataframe.columns.tolist()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test one line image with the final OCR model."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the cropped single-line image.",
    )

    parser.add_argument(
        "--expected",
        default="The lion laughed at the mouse and",
        help="Expected text used only for CER/WER calculation.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/single_line_test",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    image_path = Path(args.image)

    if not image_path.is_absolute():
        image_path = (
            project_root / image_path
        ).resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    template_manifest = (
        project_root
        / "data"
        / "processed"
        / "master_ctc"
        / "val_all.csv"
    )

    evaluator = (
        project_root
        / "scripts"
        / "38_eval_ctc_ngram.py"
    )

    checkpoint = (
        project_root
        / "outputs"
        / "checkpoints"
        / "bilingual_v4_ar70_en30_ft_v3_lr2e6"
        / "best_loss.pt"
    )

    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():
        output_dir = (
            project_root / output_dir
        ).resolve()

    manifest_path = (
        output_dir
        / "single_line_manifest.csv"
    )

    for required_path in [
        template_manifest,
        evaluator,
        checkpoint,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing required file: {required_path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = pd.read_csv(
        template_manifest,
        encoding="utf-8-sig",
    )

    image_column = detect_image_column(template)

    text_column = detect_column(
        template,
        preferred_names=[
            "text",
            "label",
            "transcription",
            "ground_truth",
            "target",
        ],
        contains_words=[
            "text",
            "label",
            "transcript",
            "truth",
        ],
    )

    language_column = None

    for candidate in [
        "language",
        "lang",
    ]:
        if candidate in template.columns:
            language_column = candidate
            break

    dataset_column = None

    for candidate in [
        "dataset",
        "source",
        "dataset_name",
    ]:
        if candidate in template.columns:
            dataset_column = candidate
            break

    if language_column is not None:
        english_rows = template[
            template[language_column]
            .astype(str)
            .str.lower()
            .isin(["en", "eng", "english"])
        ]

        if not english_rows.empty:
            row = english_rows.iloc[[0]].copy()
        else:
            row = template.iloc[[0]].copy()
    else:
        row = template.iloc[[0]].copy()

    row.loc[row.index[0], image_column] = str(
        image_path
    )

    row.loc[row.index[0], text_column] = (
        args.expected
    )

    if language_column is not None:
        row.loc[
            row.index[0],
            language_column,
        ] = "en"

    if dataset_column is not None:
        row.loc[
            row.index[0],
            dataset_column,
        ] = "manual_test"

    row.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Image column: {image_column}")
    print(f"Text column: {text_column}")
    print(f"Image: {image_path}")
    print(f"Expected: {args.expected}")
    print(f"Manifest: {manifest_path}")
    print()

    command = [
        sys.executable,
        str(evaluator),
        "--checkpoint",
        str(checkpoint),
        "--validation-manifest",
        str(manifest_path),
        "--max-samples",
        "0",
        "--beam-width",
        "10",
        "--token-prune",
        "20",
        "--ngram-order",
        "5",
        "--lm-weights",
        "0.6",
        "--word-bonuses",
        "0.25",
        "--output-dir",
        str(output_dir),
    ]

    print("Running OCR...")
    print()

    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "OCR evaluation failed with exit code "
            f"{completed.returncode}."
        )

    predictions_path = (
        output_dir
        / "validation_predictions.csv"
    )

    if predictions_path.exists():
        predictions = pd.read_csv(
            predictions_path,
            encoding="utf-8-sig",
        )

        print()
        print("Prediction record:")
        print(
            predictions
            .tail(1)
            .to_string(index=False)
        )

    print()
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
