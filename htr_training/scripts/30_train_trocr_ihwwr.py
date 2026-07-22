from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Sequence, TypeVar

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from transformers.trainer_utils import get_last_checkpoint


DEFAULT_MODEL = "microsoft/trocr-base-handwritten"

DEFAULT_TRAIN = Path(
    r"data\processed\online_htr"
    r"\ihwwr_english\train.csv"
)

DEFAULT_VAL = Path(
    r"data\processed\online_htr"
    r"\ihwwr_english\val.csv"
)

DEFAULT_OUTPUT = Path(
    r"outputs\trocr"
    r"\ihwwr_base_ft_v1"
)

T = TypeVar("T")


def edit_distance(
    reference: Sequence[T],
    prediction: Sequence[T],
) -> int:
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
                + int(
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OCRWordDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        processor: TrOCRProcessor,
        max_target_length: int,
        max_samples: int,
        seed: int,
    ) -> None:
        if not manifest.exists():
            raise FileNotFoundError(
                manifest.resolve()
            )

        data = pd.read_csv(
            manifest,
            dtype=str,
            keep_default_na=False,
        )

        required_columns = {
            "image",
            "text",
        }

        missing_columns = (
            required_columns
            - set(data.columns)
        )

        if missing_columns:
            raise RuntimeError(
                "Manifest is missing columns: "
                f"{sorted(missing_columns)}"
            )

        data = data[
            data["text"]
            .astype(str)
            .str.len()
            > 0
        ].copy()

        if (
            max_samples > 0
            and max_samples < len(data)
        ):
            data = data.sample(
                n=max_samples,
                random_state=seed,
            )

        data = data.reset_index(
            drop=True
        )

        missing_images = [
            image
            for image in data["image"]
            if not Path(image).exists()
        ]

        if missing_images:
            preview = "\n".join(
                missing_images[:10]
            )

            raise FileNotFoundError(
                f"{len(missing_images)} images "
                f"are missing:\n{preview}"
            )

        self.data = data
        self.processor = processor
        self.max_target_length = (
            max_target_length
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        row = self.data.iloc[index]

        image_path = Path(
            row["image"]
        )

        text = str(
            row["text"]
        )

        with Image.open(image_path) as source:
            image = source.convert(
                "RGB"
            ).copy()

        # Direct preprocessing performed better than
        # square padding in the pretrained validation.
        pixel_values = self.processor(
            images=image,
            input_data_format="channels_last",
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        tokenized = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenized.input_ids.squeeze(
            0
        )

        labels[
            labels
            == self.processor.tokenizer.pad_token_id
        ] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class OCRDataCollator:
    def __call__(
        self,
        samples: list[
            dict[str, torch.Tensor]
        ],
    ) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack(
                [
                    sample["pixel_values"]
                    for sample in samples
                ]
            ),
            "labels": torch.stack(
                [
                    sample["labels"]
                    for sample in samples
                ]
            ),
        }


def build_metrics_function(
    processor: TrOCRProcessor,
):
    pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    def compute_metrics(
        prediction_output,
    ) -> dict[str, float]:
        predictions = (
            prediction_output.predictions
        )

        if isinstance(
            predictions,
            tuple,
        ):
            predictions = predictions[0]

        labels = np.array(
            prediction_output.label_ids,
            copy=True,
        )

        labels[
            labels == -100
        ] = pad_token_id

        prediction_texts = (
            processor.batch_decode(
                predictions,
                skip_special_tokens=True,
            )
        )

        reference_texts = (
            processor.batch_decode(
                labels,
                skip_special_tokens=True,
            )
        )

        character_errors = 0
        character_total = 0

        word_errors = 0
        word_total = 0

        exact_matches = 0

        for reference, prediction in zip(
            reference_texts,
            prediction_texts,
        ):
            character_errors += edit_distance(
                list(reference),
                list(prediction),
            )

            character_total += max(
                len(reference),
                1,
            )

            reference_words = (
                reference.split()
            )

            prediction_words = (
                prediction.split()
            )

            word_errors += edit_distance(
                reference_words,
                prediction_words,
            )

            word_total += max(
                len(reference_words),
                1,
            )

            exact_matches += int(
                reference == prediction
            )

        count = max(
            len(reference_texts),
            1,
        )

        return {
            "cer": (
                character_errors
                / character_total
            ),
            "wer": (
                word_errors
                / word_total
            ),
            "exact_match_accuracy": (
                exact_matches / count
            ),
        }

    return compute_metrics


def configure_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    max_target_length: int,
) -> None:
    tokenizer = processor.tokenizer

    model.config.decoder_start_token_id = (
        tokenizer.cls_token_id
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.config.eos_token_id = (
        tokenizer.sep_token_id
    )

    model.config.vocab_size = (
        model.config.decoder.vocab_size
    )

    model.config.max_length = (
        max_target_length
    )

    model.config.early_stopping = False
    model.config.no_repeat_ngram_size = 0
    model.config.length_penalty = 1.0

    model.generation_config.decoder_start_token_id = (
        tokenizer.cls_token_id
    )

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.generation_config.eos_token_id = (
        tokenizer.sep_token_id
    )

    model.generation_config.max_length = (
        max_target_length
    )

    model.generation_config.num_beams = 1
    model.generation_config.early_stopping = False

    # Gradient checkpointing and decoder caching are
    # incompatible during training.
    model.config.use_cache = False

    if hasattr(
        model.config.decoder,
        "use_cache",
    ):
        model.config.decoder.use_cache = (
            False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune TrOCR Base on the "
            "page-separated IHWWR English split."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=DEFAULT_TRAIN,
    )

    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=DEFAULT_VAL,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-target-length",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        default="auto",
        help=(
            "auto, none, or a checkpoint path"
        ),
    )

    return parser.parse_args()


def resolve_resume_checkpoint(
    output: Path,
    resume_argument: str,
) -> str | None:
    value = resume_argument.strip()

    if value.lower() == "none":
        return None

    if value.lower() == "auto":
        checkpoint = get_last_checkpoint(
            str(output)
        )

        return checkpoint

    checkpoint_path = Path(
        value
    ).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    return str(
        checkpoint_path
    )


def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    args.train_manifest = (
        args.train_manifest.resolve()
    )

    args.val_manifest = (
        args.val_manifest.resolve()
    )

    args.output = (
        args.output.resolve()
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    print()
    print("TrOCR IHWWR fine-tuning")
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "Base model:",
        args.model,
    )
    print(
        "Train manifest:",
        args.train_manifest,
    )
    print(
        "Validation manifest:",
        args.val_manifest,
    )
    print(
        "Output:",
        args.output,
    )

    print()
    print("Loading processor...")

    processor = (
        TrOCRProcessor.from_pretrained(
            args.model,
            use_fast=False,
        )
    )

    print("Loading model...")

    model = (
        VisionEncoderDecoderModel
        .from_pretrained(
            args.model
        )
    )

    configure_model(
        model,
        processor,
        args.max_target_length,
    )

    train_dataset = OCRWordDataset(
        manifest=args.train_manifest,
        processor=processor,
        max_target_length=(
            args.max_target_length
        ),
        max_samples=(
            args.max_train_samples
        ),
        seed=args.seed,
    )

    validation_dataset = OCRWordDataset(
        manifest=args.val_manifest,
        processor=processor,
        max_target_length=(
            args.max_target_length
        ),
        max_samples=(
            args.max_val_samples
        ),
        seed=args.seed + 1,
    )

    print()
    print(
        "Training samples:",
        f"{len(train_dataset):,}",
    )
    print(
        "Validation samples:",
        f"{len(validation_dataset):,}",
    )

    effective_batch_size = (
        args.train_batch_size
        * args.gradient_accumulation
    )

    print(
        "Per-device training batch:",
        args.train_batch_size,
    )
    print(
        "Gradient accumulation:",
        args.gradient_accumulation,
    )
    print(
        "Effective training batch:",
        effective_batch_size,
    )

    training_arguments = (
        Seq2SeqTrainingArguments(
            output_dir=str(
                args.output
            ),
            overwrite_output_dir=False,

            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="linear",

            per_device_train_batch_size=(
                args.train_batch_size
            ),
            per_device_eval_batch_size=(
                args.eval_batch_size
            ),
            gradient_accumulation_steps=(
                args.gradient_accumulation
            ),

            fp16=True,
            fp16_full_eval=True,
            tf32=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            },

            eval_strategy="steps",
            eval_steps=args.eval_steps,

            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=2,
            save_safetensors=True,

            logging_strategy="steps",
            logging_steps=args.logging_steps,
            logging_first_step=True,
            logging_dir=str(
                args.output / "tensorboard"
            ),

            predict_with_generate=True,
            generation_max_length=(
                args.max_target_length
            ),
            generation_num_beams=1,

            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,

            remove_unused_columns=False,
            label_names=["labels"],

            dataloader_num_workers=(
                args.num_workers
            ),
            dataloader_pin_memory=True,
            dataloader_persistent_workers=(
                args.num_workers > 0
            ),

            optim="adamw_torch",
            max_grad_norm=1.0,

            report_to=["tensorboard"],

            seed=args.seed,
            data_seed=args.seed,
        )
    )

    callbacks = []

    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=(
                    args.early_stopping_patience
                )
            )
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=OCRDataCollator(),
        processing_class=processor,
        compute_metrics=(
            build_metrics_function(
                processor
            )
        ),
        callbacks=callbacks,
    )

    resume_checkpoint = (
        resolve_resume_checkpoint(
            args.output,
            args.resume,
        )
    )

    if resume_checkpoint:
        print(
            "Resuming checkpoint:",
            resume_checkpoint,
        )
    else:
        print(
            "Starting a new training run."
        )

    run_configuration = {
        "model": args.model,
        "train_manifest": str(
            args.train_manifest
        ),
        "validation_manifest": str(
            args.val_manifest
        ),
        "training_samples": len(
            train_dataset
        ),
        "validation_samples": len(
            validation_dataset
        ),
        "epochs": args.epochs,
        "learning_rate": (
            args.learning_rate
        ),
        "weight_decay": (
            args.weight_decay
        ),
        "warmup_ratio": (
            args.warmup_ratio
        ),
        "train_batch_size": (
            args.train_batch_size
        ),
        "eval_batch_size": (
            args.eval_batch_size
        ),
        "gradient_accumulation": (
            args.gradient_accumulation
        ),
        "effective_batch_size": (
            effective_batch_size
        ),
        "max_target_length": (
            args.max_target_length
        ),
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(
            0
        ),
    }

    (
        args.output / "run_config.json"
    ).write_text(
        json.dumps(
            run_configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    train_result = trainer.train(
        resume_from_checkpoint=(
            resume_checkpoint
        )
    )

    trainer.save_state()
    trainer.save_metrics(
        "train",
        train_result.metrics,
    )

    # Re-enable decoder caching for final evaluation
    # and inference.
    model.config.use_cache = True

    if hasattr(
        model.config.decoder,
        "use_cache",
    ):
        model.config.decoder.use_cache = (
            True
        )

    evaluation_metrics = (
        trainer.evaluate()
    )

    trainer.save_metrics(
        "eval",
        evaluation_metrics,
    )

    best_model_directory = (
        args.output / "best_model"
    )

    trainer.save_model(
        str(best_model_directory)
    )

    processor.save_pretrained(
        best_model_directory
    )

    print()
    print("Training complete")
    print(
        "Best checkpoint:",
        trainer.state.best_model_checkpoint,
    )
    print(
        "Best validation metric:",
        trainer.state.best_metric,
    )
    print(
        "Saved best model:",
        best_model_directory,
    )

    print()
    print("Final validation metrics:")

    for key in sorted(
        evaluation_metrics
    ):
        if key.startswith("eval_"):
            print(
                f"{key}: "
                f"{evaluation_metrics[key]}"
            )


if __name__ == "__main__":
    main()


