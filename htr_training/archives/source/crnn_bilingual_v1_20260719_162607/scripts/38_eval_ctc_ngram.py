from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import jiwer
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.crnn_dataset import CRNNDataset, ctc_collate_fn
from lib.crnn_model import CRNN
from lib.ctc_text_encoder import CTCTextEncoder


NEGATIVE_INFINITY = float("-inf")
BOS = "<s>"
EOS = "</s>"


def log_add(*values: float) -> float:
    finite_values = [
        value
        for value in values
        if value != NEGATIVE_INFINITY
    ]

    if not finite_values:
        return NEGATIVE_INFINITY

    maximum = max(finite_values)

    return maximum + math.log(
        sum(
            math.exp(value - maximum)
            for value in finite_values
        )
    )


@dataclass
class BeamState:
    blank: float = NEGATIVE_INFINITY
    nonblank: float = NEGATIVE_INFINITY

    @property
    def total(self) -> float:
        return log_add(self.blank, self.nonblank)


class CharacterNGram:
    """
    Lightweight character n-gram language model.

    It uses only the labels from train_all.csv.
    It does not train a neural network.
    """

    def __init__(
        self,
        order: int = 5,
        smoothing: float = 0.1,
    ) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")

        if smoothing <= 0:
            raise ValueError("smoothing must be positive")

        self.order = order
        self.smoothing = smoothing

        self.context_counts: dict[
            tuple[str, ...],
            Counter[str],
        ] = defaultdict(Counter)

        self.vocabulary: set[str] = {EOS}
        self.fitted = False

    def fit(self, texts: Iterable[str]) -> None:
        token_count = 0

        for text in texts:
            characters = list(text)

            if not characters:
                continue

            self.vocabulary.update(characters)

            history: list[str] = [
                BOS
            ] * (self.order - 1)

            for token in characters + [EOS]:
                max_context_length = min(
                    self.order - 1,
                    len(history),
                )

                for context_length in range(
                    max_context_length + 1
                ):
                    if context_length == 0:
                        context = tuple()
                    else:
                        context = tuple(
                            history[-context_length:]
                        )

                    self.context_counts[context][token] += 1

                history.append(token)
                token_count += 1

        if token_count == 0:
            raise ValueError(
                "No non-empty text was available for n-gram training."
            )

        self.fitted = True

    def token_log_probability(
        self,
        history: Sequence[str],
        token: str,
    ) -> float:
        if not self.fitted:
            raise RuntimeError(
                "The n-gram model has not been fitted."
            )

        vocabulary_size = max(
            1,
            len(self.vocabulary),
        )

        max_context_length = min(
            self.order - 1,
            len(history),
        )

        for context_length in range(
            max_context_length,
            -1,
            -1,
        ):
            if context_length == 0:
                context = tuple()
            else:
                context = tuple(
                    history[-context_length:]
                )

            counts = self.context_counts.get(context)

            if not counts:
                continue

            numerator = (
                counts.get(token, 0)
                + self.smoothing
            )

            denominator = (
                sum(counts.values())
                + self.smoothing * vocabulary_size
            )

            return math.log(
                numerator / denominator
            )

        return -math.log(vocabulary_size)

    def score(self, text: str) -> float:
        history: list[str] = [
            BOS
        ] * (self.order - 1)

        total_score = 0.0

        for token in list(text) + [EOS]:
            total_score += self.token_log_probability(
                history,
                token,
            )

            history.append(token)

        return total_score


def normalize_language(value: str) -> str:
    value = value.strip().lower()

    if value.startswith("ar"):
        return "ar"

    if value.startswith("en"):
        return "en"

    return value or "unknown"


def decode_prefix(
    prefix: Sequence[int],
    index_to_char: dict[int, str],
) -> str:
    return "".join(
        index_to_char[index]
        for index in prefix
        if index in index_to_char
    )


def ctc_prefix_beam_search(
    log_probabilities: torch.Tensor,
    blank_index: int,
    beam_width: int,
    token_prune: int,
) -> list[tuple[tuple[int, ...], float]]:
    """
    Lightweight CTC prefix beam search.

    Input:
        log_probabilities: shape (T, C)

    Output:
        List of collapsed token prefixes and acoustic scores.
    """

    if log_probabilities.ndim != 2:
        raise ValueError(
            "Expected probabilities with shape (T, C), "
            f"got {tuple(log_probabilities.shape)}"
        )

    time_steps, class_count = (
        log_probabilities.shape
    )

    token_prune = min(
        token_prune,
        class_count,
    )

    beams: dict[
        tuple[int, ...],
        BeamState,
    ] = {
        tuple(): BeamState(
            blank=0.0,
            nonblank=NEGATIVE_INFINITY,
        )
    }

    for time_index in range(time_steps):
        frame = log_probabilities[time_index]

        _, top_indices = torch.topk(
            frame,
            k=token_prune,
        )

        candidate_tokens = top_indices.tolist()

        if blank_index not in candidate_tokens:
            candidate_tokens.append(blank_index)

        next_beams: dict[
            tuple[int, ...],
            BeamState,
        ] = defaultdict(BeamState)

        for prefix, state in beams.items():
            for token in candidate_tokens:
                probability = float(
                    frame[token].item()
                )

                if token == blank_index:
                    destination = next_beams[prefix]

                    destination.blank = log_add(
                        destination.blank,
                        state.blank + probability,
                        state.nonblank + probability,
                    )

                    continue

                is_repeated = (
                    bool(prefix)
                    and token == prefix[-1]
                )

                if is_repeated:
                    # Same character without a blank:
                    # keep the same collapsed prefix.
                    destination = next_beams[prefix]

                    destination.nonblank = log_add(
                        destination.nonblank,
                        state.nonblank + probability,
                    )

                    # Same character after a blank:
                    # create a repeated character.
                    extended_prefix = prefix + (token,)
                    destination = next_beams[
                        extended_prefix
                    ]

                    destination.nonblank = log_add(
                        destination.nonblank,
                        state.blank + probability,
                    )

                else:
                    extended_prefix = prefix + (token,)
                    destination = next_beams[
                        extended_prefix
                    ]

                    destination.nonblank = log_add(
                        destination.nonblank,
                        state.blank + probability,
                        state.nonblank + probability,
                    )

        ranked_beams = sorted(
            next_beams.items(),
            key=lambda item: item[1].total,
            reverse=True,
        )[:beam_width]

        beams = dict(ranked_beams)

    return [
        (prefix, state.total)
        for prefix, state in sorted(
            beams.items(),
            key=lambda item: item[1].total,
            reverse=True,
        )
    ]


def choose_balanced_indices(
    dataframe: pd.DataFrame,
    max_samples: int,
) -> list[int]:
    if (
        max_samples <= 0
        or max_samples >= len(dataframe)
    ):
        return list(range(len(dataframe)))

    grouped_indices: dict[
        str,
        list[int],
    ] = defaultdict(list)

    for index, language in enumerate(
        dataframe["language"].astype(str)
    ):
        grouped_indices[
            normalize_language(language)
        ].append(index)

    languages = sorted(grouped_indices)

    samples_per_language = (
        max_samples
        // max(1, len(languages))
    )

    remainder = (
        max_samples
        % max(1, len(languages))
    )

    selected_indices: list[int] = []

    for language_index, language in enumerate(
        languages
    ):
        requested = samples_per_language

        if language_index < remainder:
            requested += 1

        selected_indices.extend(
            grouped_indices[language][:requested]
        )

    if len(selected_indices) < max_samples:
        already_selected = set(selected_indices)

        for index in range(len(dataframe)):
            if index in already_selected:
                continue

            selected_indices.append(index)

            if len(selected_indices) >= max_samples:
                break

    return sorted(selected_indices)


def load_checkpoint_model(
    checkpoint_path: Path,
    vocab_path: Path,
    device: torch.device,
) -> tuple[CRNN, dict]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    with vocab_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(file)

    checkpoint_arguments = checkpoint.get(
        "arguments",
        {},
    )

    hidden_size = int(
        checkpoint_arguments.get(
            "hidden_size",
            checkpoint.get(
                "hidden_size",
                256,
            ),
        )
    )

    dropout = float(
        checkpoint_arguments.get(
            "dropout",
            checkpoint.get(
                "dropout",
                0.1,
            ),
        )
    )

    model = CRNN(
        num_classes=int(
            vocabulary["num_classes"]
        ),
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(device)

    state_dict = checkpoint[
        "model_state_dict"
    ]

    if (
        state_dict
        and all(
            key.startswith("module.")
            for key in state_dict
        )
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model, checkpoint


def build_language_models(
    train_manifest: Path,
    encoder: CTCTextEncoder,
    order: int,
    smoothing: float,
) -> dict[str, CharacterNGram]:
    dataframe = pd.read_csv(
        train_manifest,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "text",
        "language",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in train manifest: "
            f"{sorted(missing_columns)}"
        )

    texts_by_language: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for row in dataframe.itertuples(
        index=False
    ):
        language = normalize_language(
            str(row.language)
        )

        logical_text = str(row.text).strip()

        if not logical_text:
            continue

        # The CRNN was trained with visual-order Arabic.
        visual_text = encoder.to_visual(
            logical_text
        )

        texts_by_language[
            language
        ].append(visual_text)

    models: dict[
        str,
        CharacterNGram,
    ] = {}

    for language, texts in sorted(
        texts_by_language.items()
    ):
        language_model = CharacterNGram(
            order=order,
            smoothing=smoothing,
        )

        language_model.fit(texts)

        models[language] = language_model

        print(
            f"Built {order}-gram for {language}: "
            f"{len(texts):,} labels, "
            f"{len(language_model.context_counts):,} contexts"
        )

    if not models:
        raise RuntimeError(
            "No language models were created."
        )

    return models


def candidate_score(
    acoustic_score: float,
    text: str,
    language_model: CharacterNGram | None,
    language_model_weight: float,
    word_bonus: float,
) -> float:
    if language_model is None:
        lm_score = 0.0
    else:
        lm_score = language_model.score(text)

    word_count = len(text.split())

    return (
        acoustic_score
        + language_model_weight * lm_score
        + word_bonus * word_count
    )


def setting_name(
    lm_weight: float,
    word_bonus: float,
) -> str:
    return (
        f"lm={lm_weight:g},"
        f"word_bonus={word_bonus:g}"
    )


def safe_sample_cer(
    reference: str,
    prediction: str,
) -> float:
    if not reference:
        return 0.0

    return float(
        jiwer.cer(
            reference,
            prediction,
        )
    )


def safe_sample_wer(
    reference: str,
    prediction: str,
) -> float:
    if not reference:
        return 0.0

    return float(
        jiwer.wer(
            reference,
            prediction,
        )
    )


def calculate_metrics(
    references: list[str],
    predictions: list[str],
) -> tuple[float, float]:
    return (
        float(
            jiwer.cer(
                references,
                predictions,
            )
        ),
        float(
            jiwer.wer(
                references,
                predictions,
            )
        ),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare greedy CTC, CTC beam search, "
            "and character n-gram reranking."
        )
    )

    parser.add_argument(
        "--train-manifest",
        default=(
            "data/processed/master_ctc/"
            "train_all.csv"
        ),
    )

    parser.add_argument(
        "--validation-manifest",
        default=(
            "data/processed/master_ctc/"
            "val_all.csv"
        ),
    )

    parser.add_argument(
        "--vocab",
        default=(
            "data/processed/master_safe/"
            "vocab.json"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/checkpoints/"
            "bilingual_v4_master_ctc/"
            "best_cer.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/ngram_experiment"
        ),
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
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--token-prune",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--ngram-order",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lm-weights",
        type=float,
        nargs="+",
        default=[
            0.05,
            0.1,
            0.2,
            0.4,
        ],
    )

    parser.add_argument(
        "--word-bonuses",
        type=float,
        nargs="+",
        default=[
            -0.25,
            0.0,
            0.25,
        ],
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help=(
            "Balanced validation subset size. "
            "Use 0 for the full validation set."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    arguments = parse_arguments()

    train_manifest = Path(
        arguments.train_manifest
    ).expanduser().resolve()

    validation_manifest = Path(
        arguments.validation_manifest
    ).expanduser().resolve()

    vocab_path = Path(
        arguments.vocab
    ).expanduser().resolve()

    checkpoint_path = Path(
        arguments.checkpoint
    ).expanduser().resolve()

    output_directory = Path(
        arguments.output_dir
    ).expanduser().resolve()

    required_paths = [
        train_manifest,
        validation_manifest,
        vocab_path,
        checkpoint_path,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_path}"
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        arguments.cpu
        or not torch.cuda.is_available()
    ):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    use_mixed_precision = (
        device.type == "cuda"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    encoder = CTCTextEncoder(
        vocab_path
    )

    print()
    print("Building character n-gram models...")

    language_models = build_language_models(
        train_manifest=train_manifest,
        encoder=encoder,
        order=arguments.ngram_order,
        smoothing=arguments.smoothing,
    )

    ngram_path = (
        output_directory
        / "char_ngram_models.pkl"
    )

    with ngram_path.open("wb") as file:
        pickle.dump(
            language_models,
            file,
        )

    print(
        f"Saved n-gram models: {ngram_path}"
    )

    print()
    print("Loading validation dataset...")

    validation_dataset = CRNNDataset(
        manifest_path=validation_manifest,
        vocab_path=vocab_path,
        image_height=arguments.image_height,
        max_image_width=arguments.max_image_width,
    )

    selected_indices = choose_balanced_indices(
        validation_dataset.dataframe,
        arguments.max_samples,
    )

    evaluation_dataset = Subset(
        validation_dataset,
        selected_indices,
    )

    validation_loader = DataLoader(
        evaluation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=use_mixed_precision,
        persistent_workers=(
            arguments.num_workers > 0
        ),
        collate_fn=ctc_collate_fn,
    )

    print("Loading checkpoint...")

    model, checkpoint = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        device=device,
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Validation samples:",
        f"{len(evaluation_dataset):,}",
    )

    tested_settings = [
        (lm_weight, word_bonus)
        for lm_weight in arguments.lm_weights
        for word_bonus in arguments.word_bonuses
    ]

    references: list[str] = []
    languages: list[str] = []

    greedy_predictions: list[str] = []
    beam_predictions: list[str] = []

    reranked_predictions: dict[
        str,
        list[str],
    ] = {
        setting_name(
            lm_weight,
            word_bonus,
        ): []
        for (
            lm_weight,
            word_bonus
        ) in tested_settings
    }

    prediction_rows: list[
        dict[str, object]
    ] = []

    for batch in tqdm(
        validation_loader,
        desc="Evaluating",
        unit="image",
    ):
        images = batch[
            "images"
        ].to(
            device,
            non_blocking=True,
        )

        image_widths = batch[
            "image_widths"
        ]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_mixed_precision,
        ):
            logits = model(images)

        log_probabilities = F.log_softmax(
            logits.float(),
            dim=2,
        ).cpu()

        input_lengths = (
            model.calculate_input_lengths(
                image_widths
            )
            .tolist()
        )

        sample_length = int(
            input_lengths[0]
        )

        sample_log_probabilities = (
            log_probabilities[
                0,
                :sample_length,
                :,
            ]
        )

        logical_reference = str(
            batch["texts"][0]
        )

        visual_reference = encoder.to_visual(
            logical_reference
        )

        language = normalize_language(
            str(
                batch["languages"][0]
            )
        )

        greedy_indices = (
            sample_log_probabilities.argmax(
                dim=1
            )
        )

        greedy_text = encoder.decode(
            greedy_indices
        )

        beam_candidates = (
            ctc_prefix_beam_search(
                log_probabilities=(
                    sample_log_probabilities
                ),
                blank_index=(
                    encoder.blank_index
                ),
                beam_width=(
                    arguments.beam_width
                ),
                token_prune=(
                    arguments.token_prune
                ),
            )
        )

        decoded_candidates = [
            (
                decode_prefix(
                    prefix,
                    encoder.index_to_char,
                ),
                acoustic_score,
            )
            for (
                prefix,
                acoustic_score
            ) in beam_candidates
        ]

        if decoded_candidates:
            beam_text = (
                decoded_candidates[0][0]
            )
        else:
            beam_text = greedy_text

        language_model = (
            language_models.get(language)
        )

        selected_predictions: dict[
            str,
            str,
        ] = {}

        for (
            lm_weight,
            word_bonus,
        ) in tested_settings:
            name = setting_name(
                lm_weight,
                word_bonus,
            )

            if decoded_candidates:
                selected_text, _ = max(
                    decoded_candidates,
                    key=lambda item: candidate_score(
                        acoustic_score=item[1],
                        text=item[0],
                        language_model=(
                            language_model
                        ),
                        language_model_weight=(
                            lm_weight
                        ),
                        word_bonus=word_bonus,
                    ),
                )
            else:
                selected_text = greedy_text

            selected_predictions[
                name
            ] = selected_text

            reranked_predictions[
                name
            ].append(selected_text)

        references.append(
            visual_reference
        )

        languages.append(language)

        greedy_predictions.append(
            greedy_text
        )

        beam_predictions.append(
            beam_text
        )

        prediction_row: dict[
            str,
            object,
        ] = {
            "image_path": (
                batch["image_paths"][0]
            ),
            "dataset": (
                batch["datasets"][0]
            ),
            "language": language,
            "reference_logical": (
                logical_reference
            ),
            "reference_visual": (
                visual_reference
            ),
            "greedy": greedy_text,
            "beam": beam_text,
            "greedy_cer": safe_sample_cer(
                visual_reference,
                greedy_text,
            ),
            "greedy_wer": safe_sample_wer(
                visual_reference,
                greedy_text,
            ),
            "beam_cer": safe_sample_cer(
                visual_reference,
                beam_text,
            ),
            "beam_wer": safe_sample_wer(
                visual_reference,
                beam_text,
            ),
        }

        for name, prediction in (
            selected_predictions.items()
        ):
            safe_name = (
                name
                .replace("=", "_")
                .replace(",", "__")
                .replace("-", "m")
                .replace(".", "p")
            )

            prediction_row[
                f"prediction__{safe_name}"
            ] = prediction

        prediction_rows.append(
            prediction_row
        )

    summary_rows: list[
        dict[str, object]
    ] = []

    def add_summary(
        decoder_name: str,
        predictions: list[str],
    ) -> None:
        cer_value, wer_value = (
            calculate_metrics(
                references,
                predictions,
            )
        )

        summary_rows.append(
            {
                "decoder": decoder_name,
                "cer": cer_value,
                "wer": wer_value,
                "count": len(references),
            }
        )

    add_summary(
        "greedy",
        greedy_predictions,
    )

    add_summary(
        "beam",
        beam_predictions,
    )

    for name, predictions in (
        reranked_predictions.items()
    ):
        add_summary(
            name,
            predictions,
        )

    summary_dataframe = (
        pd.DataFrame(summary_rows)
        .sort_values(
            by=["wer", "cer"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    best_decoder = str(
        summary_dataframe.iloc[0][
            "decoder"
        ]
    )

    if best_decoder == "greedy":
        best_predictions = (
            greedy_predictions
        )

    elif best_decoder == "beam":
        best_predictions = (
            beam_predictions
        )

    else:
        best_predictions = (
            reranked_predictions[
                best_decoder
            ]
        )

    language_results: dict[
        str,
        dict[str, float | int],
    ] = {}

    for language in sorted(
        set(languages)
    ):
        language_indices = [
            index
            for index, value in enumerate(
                languages
            )
            if value == language
        ]

        language_references = [
            references[index]
            for index in language_indices
        ]

        language_predictions = [
            best_predictions[index]
            for index in language_indices
        ]

        language_cer, language_wer = (
            calculate_metrics(
                language_references,
                language_predictions,
            )
        )

        language_results[language] = {
            "cer": language_cer,
            "wer": language_wer,
            "count": len(
                language_indices
            ),
        }

    predictions_path = (
        output_directory
        / "validation_predictions.csv"
    )

    summary_path = (
        output_directory
        / "decoder_summary.csv"
    )

    best_result_path = (
        output_directory
        / "best_result.json"
    )

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch": (
            checkpoint.get("epoch")
        ),
        "samples": len(references),
        "beam_width": (
            arguments.beam_width
        ),
        "token_prune": (
            arguments.token_prune
        ),
        "ngram_order": (
            arguments.ngram_order
        ),
        "best_decoder": (
            best_decoder
        ),
        "best_cer": float(
            summary_dataframe.iloc[0][
                "cer"
            ]
        ),
        "best_wer": float(
            summary_dataframe.iloc[0][
                "wer"
            ]
        ),
        "by_language": (
            language_results
        ),
    }

    best_result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Decoder comparison:")
    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    print()
    print(
        f"Best decoder: {best_decoder}"
    )

    print(
        "Best CER: "
        f"{result['best_cer']:.4f}"
    )

    print(
        "Best WER: "
        f"{result['best_wer']:.4f}"
    )

    for (
        language,
        metrics,
    ) in sorted(
        language_results.items()
    ):
        print(
            f"[{language}] "
            f"CER={metrics['cer']:.4f} "
            f"WER={metrics['wer']:.4f} "
            f"n={metrics['count']}"
        )

    print()
    print(
        f"Predictions: {predictions_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Best result: {best_result_path}"
    )


if __name__ == "__main__":
    main()
