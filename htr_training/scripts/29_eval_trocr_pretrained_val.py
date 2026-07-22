from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence, TypeVar

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


DEFAULT_MANIFEST = Path(
    r"data\processed\online_htr"
    r"\ihwwr_english\val.csv"
)

DEFAULT_OUTPUT = Path(
    r"outputs\trocr"
    r"\pretrained_base_ihwwr_val"
)

DEFAULT_MODEL = (
    "microsoft/trocr-base-handwritten"
)

T = TypeVar("T")


def edit_distance(
    reference: Sequence[T],
    prediction: Sequence[T],
) -> int:
    """
    Memory-efficient Levenshtein distance.
    """

    if len(reference) < len(prediction):
        reference, prediction = (
            prediction,
            reference,
        )

    previous = list(
        range(len(prediction) + 1)
    )

    for reference_index, reference_item in enumerate(
        reference,
        start=1,
    ):
        current = [reference_index]

        for prediction_index, prediction_item in enumerate(
            prediction,
            start=1,
        ):
            insertion = (
                current[prediction_index - 1]
                + 1
            )

            deletion = (
                previous[prediction_index]
                + 1
            )

            substitution = (
                previous[prediction_index - 1]
                + (
                    reference_item
                    != prediction_item
                )
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1]


def pad_to_square(
    image: Image.Image,
    margin_fraction: float = 0.08,
) -> Image.Image:
    """
    Preserve the word's aspect ratio before the TrOCR
    processor resizes the image.
    """

    image = image.convert("RGB")

    width, height = image.size

    side = max(
        width,
        height,
        1,
    )

    margin = max(
        2,
        round(side * margin_fraction),
    )

    canvas_side = side + 2 * margin

    canvas = Image.new(
        "RGB",
        (
            canvas_side,
            canvas_side,
        ),
        color="white",
    )

    x = (
        canvas_side - width
    ) // 2

    y = (
        canvas_side - height
    ) // 2

    canvas.paste(
        image,
        (
            x,
            y,
        ),
    )

    return canvas


class WordDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        max_samples: int = 0,
    ) -> None:
        data = pd.read_csv(
            manifest,
            dtype=str,
            keep_default_na=False,
        )

        required = {
            "image",
            "text",
        }

        missing = (
            required
            - set(data.columns)
        )

        if missing:
            raise RuntimeError(
                "Manifest is missing columns: "
                f"{sorted(missing)}"
            )

        data = data[
            data["text"]
            .astype(str)
            .str.len()
            > 0
        ].copy()

        if max_samples > 0:
            data = data.head(
                max_samples
            ).copy()

        if data.empty:
            raise RuntimeError(
                "The evaluation manifest is empty."
            )

        missing_images = [
            value
            for value in data["image"]
            if not Path(value).exists()
        ]

        if missing_images:
            preview = "\n".join(
                missing_images[:10]
            )

            raise FileNotFoundError(
                f"{len(missing_images)} images "
                f"are missing.\n{preview}"
            )

        self.data = data.reset_index(
            drop=True
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, str]:
        row = self.data.iloc[index]

        return {
            "image": row["image"],
            "text": row["text"],
        }


class EvaluationCollator:
    def __init__(
        self,
        processor: TrOCRProcessor,
        preprocessing: str,
    ) -> None:
        self.processor = processor
        self.preprocessing = preprocessing

    def __call__(
        self,
        samples: list[dict[str, str]],
    ) -> dict[str, object]:
        images: list[Image.Image] = []

        for sample in samples:
            with Image.open(
                sample["image"]
            ) as source:
                if self.preprocessing == "square":
                    prepared = pad_to_square(
                        source
                    )
                else:
                    prepared = source.convert(
                        "RGB"
                    ).copy()

                images.append(
                    prepared
                )

        pixel_values = self.processor(
            images=images,
            input_data_format="channels_last",
            return_tensors="pt",
        ).pixel_values

        return {
            "pixel_values": pixel_values,
            "references": [
                sample["text"]
                for sample in samples
            ],
            "image_paths": [
                sample["image"]
                for sample in samples
            ],
        }


def calculate_metrics(
    records: list[dict[str, object]],
) -> dict[str, object]:
    character_errors = 0
    reference_characters = 0

    word_errors = 0
    reference_words = 0

    exact_matches = 0

    for record in records:
        reference = str(
            record["reference"]
        )

        prediction = str(
            record["prediction"]
        )

        character_errors += edit_distance(
            list(reference),
            list(prediction),
        )

        reference_characters += max(
            len(reference),
            1,
        )

        reference_word_list = (
            reference.split()
        )

        prediction_word_list = (
            prediction.split()
        )

        word_errors += edit_distance(
            reference_word_list,
            prediction_word_list,
        )

        reference_words += max(
            len(reference_word_list),
            1,
        )

        exact_matches += int(
            reference == prediction
        )

    count = len(records)

    return {
        "samples": count,
        "character_errors":
            character_errors,
        "reference_characters":
            reference_characters,
        "cer": (
            character_errors
            / reference_characters
        ),
        "word_errors": word_errors,
        "reference_words":
            reference_words,
        "wer": (
            word_errors
            / reference_words
        ),
        "exact_matches":
            exact_matches,
        "exact_match_accuracy": (
            exact_matches / count
        ),
    }


def save_progress(
    records: list[dict[str, object]],
    output: Path,
) -> None:
    pd.DataFrame(
        records
    ).to_csv(
        output / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = calculate_metrics(
        records
    )

    (
        output / "metrics_partial.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pretrained TrOCR Base on "
            "the IHWWR validation split."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--preprocessing",
        choices=[
            "direct",
            "square",
        ],
        default="direct",
        help=(
            "direct uses the official processor on the "
            "original image; square preserves aspect ratio "
            "with white padding."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "0 evaluates the complete manifest."
        ),
    )

    parser.add_argument(
        "--save-every-batches",
        type=int,
        default=100,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.manifest = (
        args.manifest.resolve()
    )

    args.output = (
        args.output.resolve()
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.manifest.exists():
        raise FileNotFoundError(
            args.manifest
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the "
            "scanner environment with the CUDA "
            "PyTorch installation."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "TrOCR pretrained validation"
    )

    print(
        "Manifest:",
        args.manifest,
    )

    print(
        "Output:",
        args.output,
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Model:",
        args.model,
    )

    print(
        "Loading processor..."
    )

    processor = (
        TrOCRProcessor.from_pretrained(
            args.model
        )
    )

    print(
        "Loading model..."
    )

    model = (
        VisionEncoderDecoderModel
        .from_pretrained(
            args.model
        )
    )

    model.to(device)
    model.eval()

    dataset = WordDataset(
        args.manifest,
        max_samples=args.max_samples,
    )

    collator = EvaluationCollator(
        processor,
        preprocessing=args.preprocessing,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    print(
        "Validation samples:",
        f"{len(dataset):,}",
    )

    print(
        "Batches:",
        f"{len(loader):,}",
    )

    generation_settings = {
        "max_new_tokens":
            args.max_new_tokens,
        "num_beams":
            args.num_beams,
        "early_stopping":
            args.num_beams > 1,
    }

    records: list[
        dict[str, object]
    ] = []

    started = time.time()

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            pixel_values = batch[
                "pixel_values"
            ].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                generated_ids = (
                    model.generate(
                        pixel_values,
                        **generation_settings,
                    )
                )

            predictions = (
                processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            )

            for (
                image_path,
                reference,
                prediction,
            ) in zip(
                batch["image_paths"],
                batch["references"],
                predictions,
            ):
                character_distance = (
                    edit_distance(
                        list(reference),
                        list(prediction),
                    )
                )

                word_distance = (
                    edit_distance(
                        reference.split(),
                        prediction.split(),
                    )
                )

                records.append(
                    {
                        "image":
                            image_path,
                        "reference":
                            reference,
                        "prediction":
                            prediction,
                        "character_distance":
                            character_distance,
                        "reference_characters":
                            max(
                                len(reference),
                                1,
                            ),
                        "word_distance":
                            word_distance,
                        "reference_words":
                            max(
                                len(
                                    reference.split()
                                ),
                                1,
                            ),
                        "exact_match":
                            reference
                            == prediction,
                    }
                )

            elapsed = max(
                time.time() - started,
                0.001,
            )

            processed = len(records)

            rate = processed / elapsed

            print(
                f"\rBatch "
                f"{batch_index:,}/"
                f"{len(loader):,} | "
                f"samples "
                f"{processed:,}/"
                f"{len(dataset):,} | "
                f"{rate:.2f} images/s",
                end="",
                flush=True,
            )

            if (
                batch_index
                % args.save_every_batches
                == 0
            ):
                save_progress(
                    records,
                    args.output,
                )

    print()

    save_progress(
        records,
        args.output,
    )

    metrics = calculate_metrics(
        records
    )

    metrics.update(
        {
            "model": args.model,
            "manifest":
                str(args.manifest),
            "batch_size":
                args.batch_size,
            "num_beams":
                args.num_beams,
            "max_new_tokens":
                args.max_new_tokens,
            "elapsed_seconds":
                time.time() - started,
            "gpu":
                torch.cuda.get_device_name(
                    0
                ),
            "preprocessing":
                args.preprocessing,
        }
    )

    (
        args.output / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Pretrained TrOCR validation complete"
    )

    print(
        "Samples:",
        f"{metrics['samples']:,}",
    )

    print(
        "CER:",
        f"{metrics['cer']:.6f}",
    )

    print(
        "WER:",
        f"{metrics['wer']:.6f}",
    )

    print(
        "Exact-match accuracy:",
        f"{metrics['exact_match_accuracy']:.6f}",
    )

    print(
        "Predictions:",
        (
            args.output
            / "predictions.csv"
        ),
    )

    print(
        "Metrics:",
        (
            args.output
            / "metrics.json"
        ),
    )


if __name__ == "__main__":
    main()
