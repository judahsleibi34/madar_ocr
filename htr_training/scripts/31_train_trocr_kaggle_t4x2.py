#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
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

T = TypeVar("T")
DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
DEFAULT_OUTPUT = Path("/kaggle/working/trocr_ihwwr_t4x2")


def main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def say(message: str = "") -> None:
    if main_process():
        print(message, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def levenshtein(reference: Sequence[T], prediction: Sequence[T]) -> int:
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, prediction_item in enumerate(prediction, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + int(reference_item != prediction_item),
                )
            )
        previous = current
    return previous[-1]


def locate_archive(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    matches = sorted(Path("/kaggle/input").rglob("ihwwr_trocr_data.zip"))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one ihwwr_trocr_data.zip under /kaggle/input. "
            f"Found: {matches}"
        )
    return matches[0]


def extract_dataset(archive: Path) -> Path:
    destination = Path("/kaggle/working/ihwwr_trocr_data")
    marker = destination / ".extract_complete"
    if marker.exists():
        return destination

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        for index, member in enumerate(members, start=1):
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
            package.extract(member, destination)
            if main_process() and (index % 5000 == 0 or index == len(members)):
                print(
                    f"\rExtracted {index:,}/{len(members):,}",
                    end="",
                    flush=True,
                )
    if main_process():
        print()
    marker.write_text("complete\n", encoding="utf-8")
    return destination


def resolve_image(data_root: Path, value: str) -> Path:
    raw = Path(str(value).strip().replace("\\", "/"))
    candidates = [raw, data_root / raw, data_root / "images" / raw.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing image for manifest value: {value}")


class WordDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        processor: TrOCRProcessor,
        max_target_length: int,
        max_samples: int,
        seed: int,
    ) -> None:
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        missing = {"image", "text"} - set(frame.columns)
        if missing:
            raise RuntimeError(f"Missing columns in {manifest}: {sorted(missing)}")
        frame = frame[frame["text"].astype(str).str.strip().ne("")].copy()
        if 0 < max_samples < len(frame):
            frame = frame.sample(n=max_samples, random_state=seed)
        frame["resolved_image"] = frame["image"].map(
            lambda value: str(resolve_image(data_root, value))
        )
        self.frame = frame.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        with Image.open(row["resolved_image"]) as source:
            image = source.convert("RGB").copy()

        pixel_values = self.processor(
            images=image,
            input_data_format="channels_last",
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            str(row["text"]),
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


class Collator:
    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in samples]),
            "labels": torch.stack([item["labels"] for item in samples]),
        }


def metrics_function(processor: TrOCRProcessor):
    pad_id = processor.tokenizer.pad_token_id

    def compute(output) -> dict[str, float]:
        predictions = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        labels = np.array(output.label_ids, copy=True)
        labels[labels == -100] = pad_id
        prediction_texts = processor.batch_decode(predictions, skip_special_tokens=True)
        reference_texts = processor.batch_decode(labels, skip_special_tokens=True)

        character_errors = character_total = 0
        word_errors = word_total = exact = 0
        for reference, prediction in zip(reference_texts, prediction_texts):
            character_errors += levenshtein(list(reference), list(prediction))
            character_total += max(len(reference), 1)
            reference_words = reference.split()
            prediction_words = prediction.split()
            word_errors += levenshtein(reference_words, prediction_words)
            word_total += max(len(reference_words), 1)
            exact += int(reference == prediction)

        count = max(len(reference_texts), 1)
        return {
            "cer": character_errors / character_total,
            "wer": word_errors / word_total,
            "exact_match_accuracy": exact / count,
        }

    return compute


def configure_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    max_target_length: int,
) -> None:
    tokenizer = processor.tokenizer
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = max_target_length
    model.config.early_stopping = False
    model.config.use_cache = False
    if hasattr(model.config.decoder, "use_cache"):
        model.config.decoder.use_cache = False

    generation = model.generation_config
    generation.decoder_start_token_id = tokenizer.cls_token_id
    generation.pad_token_id = tokenizer.pad_token_id
    generation.eos_token_id = tokenizer.sep_token_id
    generation.max_length = max_target_length
    generation.num_beams = 1
    generation.early_stopping = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle T4 x2 TrOCR training")
    parser.add_argument("--data-archive", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="auto", choices=["auto", "none"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.data_dir is not None:
        data_dir = args.data_dir.resolve()

        if not data_dir.exists():
            raise FileNotFoundError(data_dir)

    else:
        archive = locate_archive(
            args.data_archive
        )
        data_dir = extract_archive(
            archive,
            Path("/kaggle/working/ihwwr_trocr_data"),
        )

    train_manifest = data_root / "train_model.csv"
    val_manifest = data_root / "val.csv"
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError("The ZIP must contain train_model.csv and val.csv at its root.")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    say("Kaggle TrOCR training")
    say(f"Visible GPUs: {torch.cuda.device_count()}")
    say(f"DDP world size: {world_size}")
    say(f"Archive: {archive}")
    say(f"Output: {output}")
    if world_size != 2:
        say("WARNING: launch with torchrun --nproc_per_node=2 to use both T4 GPUs.")

    processor = TrOCRProcessor.from_pretrained(args.model, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    configure_model(model, processor, args.max_target_length)

    train_data = WordDataset(
        train_manifest,
        data_root,
        processor,
        args.max_target_length,
        args.max_train_samples,
        args.seed,
    )
    val_data = WordDataset(
        val_manifest,
        data_root,
        processor,
        args.max_target_length,
        args.max_val_samples,
        args.seed + 1,
    )

    global_batch = args.train_batch_size * args.gradient_accumulation * world_size
    say(f"Training samples: {len(train_data):,}")
    say(f"Validation samples: {len(val_data):,}")
    say(f"Global effective batch: {global_batch}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=True,
        fp16_full_eval=True,
        tf32=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        save_safetensors=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_dir=str(output / "tensorboard"),
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=Collator(),
        processing_class=processor,
        compute_metrics=metrics_function(processor),
        callbacks=callbacks,
    )

    checkpoint = None
    if args.resume == "auto":
        checkpoint = get_last_checkpoint(str(output))
    say(f"Resume checkpoint: {checkpoint or 'none'}")

    if main_process():
        (output / "run_config.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "world_size": world_size,
                    "global_effective_batch": global_batch,
                    "train_samples": len(train_data),
                    "validation_samples": len(val_data),
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "eval_steps": args.eval_steps,
                    "gpus": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)

    model.config.use_cache = True
    if hasattr(model.config.decoder, "use_cache"):
        model.config.decoder.use_cache = True

    evaluation = trainer.evaluate()
    trainer.save_metrics("eval", evaluation)
    best_dir = output / "best_model"
    trainer.save_model(str(best_dir))

    if trainer.is_world_process_zero():
        processor.save_pretrained(best_dir)
        archive_path = shutil.make_archive(
            str(output.parent / f"{output.name}_artifacts"),
            "zip",
            root_dir=output.parent,
            base_dir=output.name,
        )
        print("Training complete")
        print("Best checkpoint:", trainer.state.best_model_checkpoint)
        print("Best validation CER:", trainer.state.best_metric)
        print("Best model:", best_dir)
        print("Artifact ZIP:", archive_path)


if __name__ == "__main__":
    main()
