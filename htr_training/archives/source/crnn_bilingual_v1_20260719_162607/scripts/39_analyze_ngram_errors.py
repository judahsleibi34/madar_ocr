from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze greedy, beam, and n-gram OCR errors."
    )

    parser.add_argument(
        "--predictions",
        default=(
            "outputs/ngram_full_validation/"
            "validation_predictions.csv"
        ),
    )

    parser.add_argument(
        "--best-result",
        default=(
            "outputs/ngram_full_validation/"
            "best_result.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/error_analysis_full_validation",
    )

    parser.add_argument(
        "--prediction-column",
        default=None,
        help=(
            "Optional explicit n-gram prediction column. "
            "Normally detected from best_result.json."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of common confusions and examples to save.",
    )

    return parser.parse_args()


def decoder_to_column(decoder_name: str) -> str:
    if decoder_name in {"greedy", "beam"}:
        return decoder_name

    safe_name = (
        decoder_name
        .replace("=", "_")
        .replace(",", "__")
        .replace("-", "m")
        .replace(".", "p")
    )

    return f"prediction__{safe_name}"


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def remove_all_spaces(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def visible_token(token: str) -> str:
    if token == " ":
        return "<SPACE>"

    if token == "\t":
        return "<TAB>"

    if token == "\n":
        return "<NEWLINE>"

    if token == "":
        return "<EMPTY>"

    return token


def is_numeric_like(text: str) -> bool:
    compact = remove_all_spaces(text)

    if not compact:
        return False

    allowed_non_digits = set(
        ".,:/\\-+()%[]{}#"
        "٫٬،؛"
    )

    has_digit = any(
        character.isdigit()
        for character in compact
    )

    all_allowed = all(
        character.isdigit()
        or character in allowed_non_digits
        for character in compact
    )

    return has_digit and all_allowed


def levenshtein_alignment(
    reference: Sequence[str],
    prediction: Sequence[str],
) -> dict[str, Any]:
    """
    Exact Levenshtein alignment.

    Returns:
        substitutions
        deletions
        insertions
        matches
        operations
    """

    reference_length = len(reference)
    prediction_length = len(prediction)

    matrix = [
        [0] * (prediction_length + 1)
        for _ in range(reference_length + 1)
    ]

    for reference_index in range(
        reference_length + 1
    ):
        matrix[reference_index][0] = reference_index

    for prediction_index in range(
        prediction_length + 1
    ):
        matrix[0][prediction_index] = prediction_index

    for reference_index in range(
        1,
        reference_length + 1,
    ):
        for prediction_index in range(
            1,
            prediction_length + 1,
        ):
            substitution_cost = (
                0
                if (
                    reference[reference_index - 1]
                    == prediction[prediction_index - 1]
                )
                else 1
            )

            matrix[reference_index][prediction_index] = min(
                matrix[reference_index - 1][prediction_index]
                + 1,
                matrix[reference_index][prediction_index - 1]
                + 1,
                matrix[reference_index - 1][prediction_index - 1]
                + substitution_cost,
            )

    operations: list[
        tuple[str, str, str]
    ] = []

    substitutions = 0
    deletions = 0
    insertions = 0
    matches = 0

    reference_index = reference_length
    prediction_index = prediction_length

    while (
        reference_index > 0
        or prediction_index > 0
    ):
        if (
            reference_index > 0
            and prediction_index > 0
        ):
            reference_token = reference[
                reference_index - 1
            ]

            prediction_token = prediction[
                prediction_index - 1
            ]

            substitution_cost = (
                0
                if reference_token == prediction_token
                else 1
            )

            diagonal_cost = (
                matrix[
                    reference_index - 1
                ][
                    prediction_index - 1
                ]
                + substitution_cost
            )

            if (
                matrix[
                    reference_index
                ][
                    prediction_index
                ]
                == diagonal_cost
            ):
                if substitution_cost == 0:
                    matches += 1

                    operations.append(
                        (
                            "match",
                            reference_token,
                            prediction_token,
                        )
                    )

                else:
                    substitutions += 1

                    operations.append(
                        (
                            "substitution",
                            reference_token,
                            prediction_token,
                        )
                    )

                reference_index -= 1
                prediction_index -= 1
                continue

        if (
            reference_index > 0
            and matrix[
                reference_index
            ][
                prediction_index
            ]
            == matrix[
                reference_index - 1
            ][
                prediction_index
            ]
            + 1
        ):
            reference_token = reference[
                reference_index - 1
            ]

            deletions += 1

            operations.append(
                (
                    "deletion",
                    reference_token,
                    "",
                )
            )

            reference_index -= 1
            continue

        prediction_token = prediction[
            prediction_index - 1
        ]

        insertions += 1

        operations.append(
            (
                "insertion",
                "",
                prediction_token,
            )
        )

        prediction_index -= 1

    operations.reverse()

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "matches": matches,
        "edits": (
            substitutions
            + deletions
            + insertions
        ),
        "operations": operations,
    }


def evaluate_text_pair(
    reference: str,
    prediction: str,
) -> dict[str, Any]:
    reference = str(reference)
    prediction = str(prediction)

    character_result = levenshtein_alignment(
        list(reference),
        list(prediction),
    )

    reference_words = reference.split()
    prediction_words = prediction.split()

    word_result = levenshtein_alignment(
        reference_words,
        prediction_words,
    )

    character_denominator = len(reference)

    word_denominator = len(reference_words)

    if character_denominator:
        character_error_rate = (
            character_result["edits"]
            / character_denominator
        )
    else:
        character_error_rate = float(
            character_result["edits"] > 0
        )

    if word_denominator:
        word_error_rate = (
            word_result["edits"]
            / word_denominator
        )
    else:
        word_error_rate = float(
            word_result["edits"] > 0
        )

    exact_match = reference == prediction

    normalized_space_match = (
        normalize_spaces(reference)
        == normalize_spaces(prediction)
    )

    no_space_match = (
        remove_all_spaces(reference)
        == remove_all_spaces(prediction)
    )

    spacing_only_error = (
        not exact_match
        and no_space_match
    )

    return {
        "reference_characters": character_denominator,
        "reference_words": word_denominator,
        "character_substitutions": (
            character_result["substitutions"]
        ),
        "character_deletions": (
            character_result["deletions"]
        ),
        "character_insertions": (
            character_result["insertions"]
        ),
        "character_edits": (
            character_result["edits"]
        ),
        "cer": character_error_rate,
        "word_substitutions": (
            word_result["substitutions"]
        ),
        "word_deletions": (
            word_result["deletions"]
        ),
        "word_insertions": (
            word_result["insertions"]
        ),
        "word_edits": word_result["edits"],
        "wer": word_error_rate,
        "exact_match": exact_match,
        "normalized_space_match": (
            normalized_space_match
        ),
        "spacing_only_error": (
            spacing_only_error
        ),
        "character_operations": (
            character_result["operations"]
        ),
        "word_operations": (
            word_result["operations"]
        ),
    }


def aggregate_metrics(
    dataframe: pd.DataFrame,
    prefix: str,
) -> dict[str, float | int]:
    reference_characters = int(
        dataframe[
            "reference_characters"
        ].sum()
    )

    reference_words = int(
        dataframe[
            "reference_words"
        ].sum()
    )

    character_substitutions = int(
        dataframe[
            f"{prefix}_character_substitutions"
        ].sum()
    )

    character_deletions = int(
        dataframe[
            f"{prefix}_character_deletions"
        ].sum()
    )

    character_insertions = int(
        dataframe[
            f"{prefix}_character_insertions"
        ].sum()
    )

    word_substitutions = int(
        dataframe[
            f"{prefix}_word_substitutions"
        ].sum()
    )

    word_deletions = int(
        dataframe[
            f"{prefix}_word_deletions"
        ].sum()
    )

    word_insertions = int(
        dataframe[
            f"{prefix}_word_insertions"
        ].sum()
    )

    character_edits = (
        character_substitutions
        + character_deletions
        + character_insertions
    )

    word_edits = (
        word_substitutions
        + word_deletions
        + word_insertions
    )

    cer = (
        character_edits / reference_characters
        if reference_characters
        else 0.0
    )

    wer = (
        word_edits / reference_words
        if reference_words
        else 0.0
    )

    return {
        "count": int(len(dataframe)),
        "reference_characters": (
            reference_characters
        ),
        "reference_words": reference_words,
        "cer": float(cer),
        "wer": float(wer),
        "exact_accuracy": float(
            dataframe[
                f"{prefix}_exact_match"
            ].mean()
        ),
        "space_normalized_accuracy": float(
            dataframe[
                f"{prefix}_normalized_space_match"
            ].mean()
        ),
        "spacing_only_errors": int(
            dataframe[
                f"{prefix}_spacing_only_error"
            ].sum()
        ),
        "character_substitutions": (
            character_substitutions
        ),
        "character_deletions": (
            character_deletions
        ),
        "character_insertions": (
            character_insertions
        ),
        "word_substitutions": (
            word_substitutions
        ),
        "word_deletions": word_deletions,
        "word_insertions": word_insertions,
    }


def grouped_metrics(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for group_value, group in dataframe.groupby(
        group_column,
        dropna=False,
    ):
        metrics = aggregate_metrics(
            group,
            prefix="best",
        )

        metrics[group_column] = str(group_value)
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    columns = [
        group_column,
        "count",
        "cer",
        "wer",
        "exact_accuracy",
        "space_normalized_accuracy",
        "spacing_only_errors",
        "reference_characters",
        "reference_words",
        "character_substitutions",
        "character_deletions",
        "character_insertions",
        "word_substitutions",
        "word_deletions",
        "word_insertions",
    ]

    return (
        pd.DataFrame(rows)[columns]
        .sort_values(
            by=["wer", "cer"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_counter(
    counter: Counter[tuple[str, str]],
    output_path: Path,
    error_type: str,
    top: int,
) -> None:
    rows = []

    for (
        reference_token,
        prediction_token,
    ), count in counter.most_common(top):
        rows.append(
            {
                "error_type": error_type,
                "reference": visible_token(
                    reference_token
                ),
                "prediction": visible_token(
                    prediction_token
                ),
                "count": count,
            }
        )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def create_plots(
    output_directory: Path,
    decoder_metrics: pd.DataFrame,
    language_metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    length_metrics: pd.DataFrame,
    impact_counts: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt

    except ImportError:
        print(
            "Matplotlib is not installed; "
            "CSV analysis was saved without plots."
        )
        return

    plot_directory = (
        output_directory / "plots"
    )

    plot_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    decoder_plot = (
        decoder_metrics
        .set_index("decoder")[["cer", "wer"]]
        * 100
    )

    axis = decoder_plot.plot(
        kind="bar",
        figsize=(9, 6),
    )

    axis.set_title(
        "Decoder CER and WER"
    )

    axis.set_ylabel("Error rate (%)")
    axis.set_xlabel("Decoder")
    axis.tick_params(
        axis="x",
        rotation=20,
    )

    axis.figure.tight_layout()
    axis.figure.savefig(
        plot_directory
        / "decoder_cer_wer.png",
        dpi=180,
    )
    plt.close(axis.figure)

    if not language_metrics.empty:
        language_plot = (
            language_metrics
            .set_index("language")[["cer", "wer"]]
            * 100
        )

        axis = language_plot.plot(
            kind="bar",
            figsize=(8, 6),
        )

        axis.set_title(
            "Best decoder performance by language"
        )

        axis.set_ylabel("Error rate (%)")
        axis.set_xlabel("Language")
        axis.tick_params(
            axis="x",
            rotation=0,
        )

        axis.figure.tight_layout()
        axis.figure.savefig(
            plot_directory
            / "performance_by_language.png",
            dpi=180,
        )
        plt.close(axis.figure)

    if not dataset_metrics.empty:
        dataset_plot = (
            dataset_metrics
            .sort_values(
                "wer",
                ascending=False,
            )
            .head(15)
            .set_index("dataset")["wer"]
            * 100
        )

        axis = dataset_plot.plot(
            kind="barh",
            figsize=(10, 7),
        )

        axis.set_title(
            "Worst datasets by WER"
        )

        axis.set_xlabel("WER (%)")
        axis.set_ylabel("Dataset")

        axis.figure.tight_layout()
        axis.figure.savefig(
            plot_directory
            / "wer_by_dataset.png",
            dpi=180,
        )
        plt.close(axis.figure)

    if not length_metrics.empty:
        length_plot = (
            length_metrics
            .set_index("length_bucket")["wer"]
            * 100
        )

        axis = length_plot.plot(
            kind="bar",
            figsize=(10, 6),
        )

        axis.set_title(
            "WER by reference character length"
        )

        axis.set_ylabel("WER (%)")
        axis.set_xlabel(
            "Reference length"
        )

        axis.tick_params(
            axis="x",
            rotation=25,
        )

        axis.figure.tight_layout()
        axis.figure.savefig(
            plot_directory
            / "wer_by_length.png",
            dpi=180,
        )
        plt.close(axis.figure)

    if not impact_counts.empty:
        axis = (
            impact_counts
            .set_index("impact")["count"]
            .plot(
                kind="bar",
                figsize=(8, 6),
            )
        )

        axis.set_title(
            "N-gram impact compared with greedy"
        )

        axis.set_ylabel("Number of samples")
        axis.set_xlabel("Impact")
        axis.tick_params(
            axis="x",
            rotation=0,
        )

        axis.figure.tight_layout()
        axis.figure.savefig(
            plot_directory
            / "ngram_impact_counts.png",
            dpi=180,
        )
        plt.close(axis.figure)


def main() -> None:
    arguments = parse_arguments()

    predictions_path = Path(
        arguments.predictions
    ).expanduser().resolve()

    best_result_path = Path(
        arguments.best_result
    ).expanduser().resolve()

    output_directory = Path(
        arguments.output_dir
    ).expanduser().resolve()

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions CSV not found: "
            f"{predictions_path}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        predictions_path,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "reference_visual",
        "greedy",
        "beam",
        "language",
        "dataset",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "The predictions CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if arguments.prediction_column:
        best_prediction_column = (
            arguments.prediction_column
        )

        best_decoder = (
            best_prediction_column
        )

    else:
        if not best_result_path.exists():
            raise FileNotFoundError(
                "best_result.json was not found and "
                "--prediction-column was not supplied."
            )

        best_result = json.loads(
            best_result_path.read_text(
                encoding="utf-8",
            )
        )

        best_decoder = str(
            best_result["best_decoder"]
        )

        best_prediction_column = (
            decoder_to_column(best_decoder)
        )

    if (
        best_prediction_column
        not in dataframe.columns
    ):
        possible_columns = [
            column
            for column in dataframe.columns
            if (
                column.startswith("prediction__")
                or column in {"greedy", "beam"}
            )
        ]

        raise ValueError(
            f"Prediction column not found: "
            f"{best_prediction_column}\n"
            f"Available decoder columns: "
            f"{possible_columns}"
        )

    print(
        f"Rows: {len(dataframe):,}"
    )

    print(
        f"Best decoder: {best_decoder}"
    )

    print(
        "Best prediction column: "
        f"{best_prediction_column}"
    )

    character_substitution_counter: Counter[
        tuple[str, str]
    ] = Counter()

    character_deletion_counter: Counter[
        tuple[str, str]
    ] = Counter()

    character_insertion_counter: Counter[
        tuple[str, str]
    ] = Counter()

    word_substitution_counter: Counter[
        tuple[str, str]
    ] = Counter()

    word_deletion_counter: Counter[
        tuple[str, str]
    ] = Counter()

    word_insertion_counter: Counter[
        tuple[str, str]
    ] = Counter()

    analysis_rows: list[
        dict[str, Any]
    ] = []

    total_rows = len(dataframe)

    for row_number, row in enumerate(
        dataframe.to_dict("records"),
        start=1,
    ):
        reference = str(
            row["reference_visual"]
        )

        greedy_prediction = str(
            row["greedy"]
        )

        beam_prediction = str(
            row["beam"]
        )

        best_prediction = str(
            row[best_prediction_column]
        )

        greedy_result = evaluate_text_pair(
            reference,
            greedy_prediction,
        )

        beam_result = evaluate_text_pair(
            reference,
            beam_prediction,
        )

        best_result = evaluate_text_pair(
            reference,
            best_prediction,
        )

        for (
            operation,
            reference_token,
            prediction_token,
        ) in best_result[
            "character_operations"
        ]:
            if operation == "substitution":
                character_substitution_counter[
                    (
                        reference_token,
                        prediction_token,
                    )
                ] += 1

            elif operation == "deletion":
                character_deletion_counter[
                    (
                        reference_token,
                        "",
                    )
                ] += 1

            elif operation == "insertion":
                character_insertion_counter[
                    (
                        "",
                        prediction_token,
                    )
                ] += 1

        for (
            operation,
            reference_token,
            prediction_token,
        ) in best_result[
            "word_operations"
        ]:
            if operation == "substitution":
                word_substitution_counter[
                    (
                        reference_token,
                        prediction_token,
                    )
                ] += 1

            elif operation == "deletion":
                word_deletion_counter[
                    (
                        reference_token,
                        "",
                    )
                ] += 1

            elif operation == "insertion":
                word_insertion_counter[
                    (
                        "",
                        prediction_token,
                    )
                ] += 1

        greedy_word_edits = (
            greedy_result["word_edits"]
        )

        best_word_edits = (
            best_result["word_edits"]
        )

        greedy_character_edits = (
            greedy_result["character_edits"]
        )

        best_character_edits = (
            best_result["character_edits"]
        )

        if (
            best_word_edits
            < greedy_word_edits
        ):
            impact = "improved"

        elif (
            best_word_edits
            > greedy_word_edits
        ):
            impact = "hurt"

        elif (
            best_character_edits
            < greedy_character_edits
        ):
            impact = "improved"

        elif (
            best_character_edits
            > greedy_character_edits
        ):
            impact = "hurt"

        else:
            impact = "unchanged"

        reference_word_count = (
            best_result["reference_words"]
        )

        if is_numeric_like(reference):
            text_type = "numeric_like"

        elif reference_word_count <= 1:
            text_type = "single_word"

        else:
            text_type = "multi_word"

        output_row: dict[str, Any] = {
            "image_path": row.get(
                "image_path",
                "",
            ),
            "dataset": row.get(
                "dataset",
                "",
            ),
            "language": row.get(
                "language",
                "",
            ),
            "reference_logical": row.get(
                "reference_logical",
                "",
            ),
            "reference_visual": reference,
            "greedy": greedy_prediction,
            "beam": beam_prediction,
            "best_prediction": (
                best_prediction
            ),
            "best_decoder": best_decoder,
            "text_type": text_type,
            "reference_characters": (
                best_result[
                    "reference_characters"
                ]
            ),
            "reference_words": (
                reference_word_count
            ),
            "impact": impact,
            "word_edit_improvement": (
                greedy_word_edits
                - best_word_edits
            ),
            "character_edit_improvement": (
                greedy_character_edits
                - best_character_edits
            ),
        }

        for prefix, result in (
            ("greedy", greedy_result),
            ("beam", beam_result),
            ("best", best_result),
        ):
            for key in (
                "character_substitutions",
                "character_deletions",
                "character_insertions",
                "character_edits",
                "cer",
                "word_substitutions",
                "word_deletions",
                "word_insertions",
                "word_edits",
                "wer",
                "exact_match",
                "normalized_space_match",
                "spacing_only_error",
            ):
                output_row[
                    f"{prefix}_{key}"
                ] = result[key]

        analysis_rows.append(
            output_row
        )

        if (
            row_number % 500 == 0
            or row_number == total_rows
        ):
            print(
                f"Analyzed "
                f"{row_number:,}/"
                f"{total_rows:,}"
            )

    analysis_dataframe = pd.DataFrame(
        analysis_rows
    )

    analysis_dataframe[
        "length_bucket"
    ] = pd.cut(
        analysis_dataframe[
            "reference_characters"
        ],
        bins=[
            -1,
            5,
            10,
            20,
            40,
            80,
            math.inf,
        ],
        labels=[
            "0-5",
            "6-10",
            "11-20",
            "21-40",
            "41-80",
            "81+",
        ],
    ).astype(str)

    analysis_dataframe[
        "word_count_bucket"
    ] = pd.cut(
        analysis_dataframe[
            "reference_words"
        ],
        bins=[
            -1,
            1,
            2,
            5,
            10,
            math.inf,
        ],
        labels=[
            "0-1",
            "2",
            "3-5",
            "6-10",
            "11+",
        ],
    ).astype(str)

    decoder_rows = []

    for decoder_name, prefix in (
        ("greedy", "greedy"),
        ("beam", "beam"),
        (best_decoder, "best"),
    ):
        metrics = aggregate_metrics(
            analysis_dataframe,
            prefix=prefix,
        )

        metrics["decoder"] = (
            decoder_name
        )

        decoder_rows.append(metrics)

    decoder_metrics = pd.DataFrame(
        decoder_rows
    )[
        [
            "decoder",
            "count",
            "cer",
            "wer",
            "exact_accuracy",
            "space_normalized_accuracy",
            "spacing_only_errors",
            "character_substitutions",
            "character_deletions",
            "character_insertions",
            "word_substitutions",
            "word_deletions",
            "word_insertions",
        ]
    ]

    language_metrics = grouped_metrics(
        analysis_dataframe,
        "language",
    )

    dataset_metrics = grouped_metrics(
        analysis_dataframe,
        "dataset",
    )

    length_metrics = grouped_metrics(
        analysis_dataframe,
        "length_bucket",
    )

    word_count_metrics = grouped_metrics(
        analysis_dataframe,
        "word_count_bucket",
    )

    text_type_metrics = grouped_metrics(
        analysis_dataframe,
        "text_type",
    )

    impact_counts = (
        analysis_dataframe[
            "impact"
        ]
        .value_counts()
        .rename_axis("impact")
        .reset_index(name="count")
    )

    numeric_rows = analysis_dataframe[
        analysis_dataframe[
            "text_type"
        ]
        == "numeric_like"
    ]

    improved_examples = (
        analysis_dataframe[
            analysis_dataframe[
                "impact"
            ]
            == "improved"
        ]
        .sort_values(
            by=[
                "word_edit_improvement",
                "character_edit_improvement",
            ],
            ascending=False,
        )
        .head(arguments.top)
    )

    hurt_examples = (
        analysis_dataframe[
            analysis_dataframe[
                "impact"
            ]
            == "hurt"
        ]
        .sort_values(
            by=[
                "word_edit_improvement",
                "character_edit_improvement",
            ],
            ascending=True,
        )
        .head(arguments.top)
    )

    worst_examples = (
        analysis_dataframe
        .sort_values(
            by=[
                "best_wer",
                "best_cer",
                "best_word_edits",
                "best_character_edits",
            ],
            ascending=False,
        )
        .head(arguments.top)
    )

    spacing_examples = (
        analysis_dataframe[
            analysis_dataframe[
                "best_spacing_only_error"
            ]
        ]
        .head(arguments.top)
    )

    analysis_dataframe.to_csv(
        output_directory
        / "analysis_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    decoder_metrics.to_csv(
        output_directory
        / "overall_decoder_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    language_metrics.to_csv(
        output_directory
        / "metrics_by_language.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dataset_metrics.to_csv(
        output_directory
        / "metrics_by_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    length_metrics.to_csv(
        output_directory
        / "metrics_by_length.csv",
        index=False,
        encoding="utf-8-sig",
    )

    word_count_metrics.to_csv(
        output_directory
        / "metrics_by_word_count.csv",
        index=False,
        encoding="utf-8-sig",
    )

    text_type_metrics.to_csv(
        output_directory
        / "metrics_by_text_type.csv",
        index=False,
        encoding="utf-8-sig",
    )

    impact_counts.to_csv(
        output_directory
        / "ngram_impact_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    numeric_rows.to_csv(
        output_directory
        / "numeric_like_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    improved_examples.to_csv(
        output_directory
        / "ngram_improved_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hurt_examples.to_csv(
        output_directory
        / "ngram_hurt_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    worst_examples.to_csv(
        output_directory
        / "worst_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    spacing_examples.to_csv(
        output_directory
        / "spacing_only_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_counter(
        character_substitution_counter,
        output_directory
        / "character_substitutions.csv",
        "substitution",
        arguments.top,
    )

    save_counter(
        character_deletion_counter,
        output_directory
        / "character_deletions.csv",
        "deletion",
        arguments.top,
    )

    save_counter(
        character_insertion_counter,
        output_directory
        / "character_insertions.csv",
        "insertion",
        arguments.top,
    )

    save_counter(
        word_substitution_counter,
        output_directory
        / "word_substitutions.csv",
        "substitution",
        arguments.top,
    )

    save_counter(
        word_deletion_counter,
        output_directory
        / "word_deletions.csv",
        "deletion",
        arguments.top,
    )

    save_counter(
        word_insertion_counter,
        output_directory
        / "word_insertions.csv",
        "insertion",
        arguments.top,
    )

    best_overall = aggregate_metrics(
        analysis_dataframe,
        prefix="best",
    )

    numeric_metrics = (
        aggregate_metrics(
            numeric_rows,
            prefix="best",
        )
        if len(numeric_rows)
        else None
    )

    summary = {
        "source_predictions": str(
            predictions_path
        ),
        "rows": int(
            len(analysis_dataframe)
        ),
        "best_decoder": best_decoder,
        "best_prediction_column": (
            best_prediction_column
        ),
        "overall_best": best_overall,
        "numeric_like": numeric_metrics,
        "ngram_impact_counts": {
            str(row["impact"]): int(
                row["count"]
            )
            for row in impact_counts.to_dict(
                "records"
            )
        },
        "worst_language_by_wer": (
            language_metrics.iloc[0][
                "language"
            ]
            if len(language_metrics)
            else None
        ),
        "worst_dataset_by_wer": (
            dataset_metrics.iloc[0][
                "dataset"
            ]
            if len(dataset_metrics)
            else None
        ),
    }

    (
        output_directory
        / "analysis_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    create_plots(
        output_directory=output_directory,
        decoder_metrics=decoder_metrics,
        language_metrics=language_metrics,
        dataset_metrics=dataset_metrics,
        length_metrics=length_metrics,
        impact_counts=impact_counts,
    )

    print()
    print("Overall decoder metrics:")
    print(
        decoder_metrics[
            [
                "decoder",
                "cer",
                "wer",
                "exact_accuracy",
                "spacing_only_errors",
            ]
        ].to_string(index=False)
    )

    print()
    print("N-gram impact:")
    print(
        impact_counts.to_string(
            index=False
        )
    )

    print()
    print("Best decoder by language:")
    print(
        language_metrics[
            [
                "language",
                "count",
                "cer",
                "wer",
                "exact_accuracy",
                "spacing_only_errors",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"Analysis saved to: "
        f"{output_directory}"
    )


if __name__ == "__main__":
    main()
