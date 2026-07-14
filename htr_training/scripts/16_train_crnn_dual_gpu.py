from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.crnn_dataset import CRNNDataset, ctc_collate_fn
from lib.crnn_model import CRNN
from lib.bucket_sampler import compute_effective_widths, BucketBatchSampler
from lib.ctc_text_encoder import CTCTextEncoder
from lib.metrics import compute_cer, compute_wer, compute_metrics_by_language
from tqdm import tqdm

import importlib.util as _importlib_util


def _load_plot_module():
    plot_path = Path(__file__).resolve().parent / "16_plot_metrics.py"
    spec = _importlib_util.spec_from_file_location("plot_metrics_module", plot_path)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_plot_module = _load_plot_module()
plot_metrics = _plot_module.plot_metrics

torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocabulary(vocab_path: Path) -> dict[str, Any]:
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    with vocab_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_loader(
    manifest_path: Path,
    vocab_path: Path,
    image_height: int,
    max_image_width: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = CRNNDataset(
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        image_height=image_height,
        max_image_width=max_image_width,
    )

    cache_path = manifest_path.with_suffix(".widths.npy")

    widths = compute_effective_widths(
        dataframe=dataset.dataframe,
        image_height=image_height,
        max_image_width=max_image_width,
        cache_path=cache_path,
    )

    batch_sampler = BucketBatchSampler(
        widths=widths,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=ctc_collate_fn,
    )


def calculate_ctc_loss(
    model: nn.Module,
    raw_model: nn.Module,
    criterion: nn.CTCLoss,
    batch: dict[str, Any],
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)

    image_widths = batch["image_widths"]
    target_lengths = batch["target_lengths"]

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        # logits: (N, T, C) -- batch-first. This is what CRNN.forward now
        # returns, and it's required for nn.DataParallel to correctly
        # split/gather across GPUs (DataParallel always assumes dim 0 is
        # the batch dimension).
        logits = model(images)

    # nn.CTCLoss expects log_probs as (T, N, C) by default, so permute
    # AFTER the model call / DataParallel gather, not inside the model.
    log_probs = F.log_softmax(logits.float(), dim=2).permute(1, 0, 2)
    input_lengths = raw_model.calculate_input_lengths(image_widths)

    if torch.any(target_lengths > input_lengths):
        raise RuntimeError(
            "Target length exceeds the available CTC sequence length."
        )

    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    return loss, logits


def train_one_epoch(
    model: nn.Module,
    raw_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip: float,
    log_every: int,
    accumulation_steps: int,
) -> float:
    model.train()

    total_loss = 0.0
    valid_batches = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(
        enumerate(loader, start=1),
        total=len(loader),
        desc="Training",
        unit="batch",
    )

    for batch_index, batch in progress_bar:
        loss, _ = calculate_ctc_loss(
            model=model, raw_model=raw_model, criterion=criterion, batch=batch,
            device=device, use_amp=use_amp,
        )

        if not torch.isfinite(loss):
            print(f"Skipping non-finite loss at batch {batch_index}: {loss.item()}")
            continue

        scaler.scale(loss / accumulation_steps).backward()

        if batch_index % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        valid_batches += 1

        if valid_batches > 0:
            average_loss = total_loss / valid_batches
            progress_bar.set_postfix(loss=f"{average_loss:.4f}")

    if valid_batches == 0:
        raise RuntimeError("No valid training batches were completed.")

    return total_loss / valid_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    raw_model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
    use_amp: bool,
    encoder: CTCTextEncoder,
) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    valid_batches = 0

    all_references: list[str] = []
    all_hypotheses: list[str] = []
    all_languages: list[str] = []

    for batch in loader:
        loss, logits = calculate_ctc_loss(
            model=model, raw_model=raw_model, criterion=criterion, batch=batch,
            device=device, use_amp=use_amp,
        )

        if torch.isfinite(loss):
            total_loss += loss.item()
            valid_batches += 1

        # logits is (N, T, C) (batch-first, as returned by the model /
        # gathered by DataParallel), so argmax over the class dim (2)
        # already yields (N, T) -- no further permute needed.
        predictions = logits.argmax(dim=2)
        hypotheses = encoder.decode_batch(predictions)

        # decode_batch() returns visual-order text (matching how targets
        # were encoded), so references must be converted to the same
        # visual order before CER/WER comparison -- otherwise Arabic
        # references would still be in logical order and every Arabic
        # sample would score as ~100% wrong regardless of model quality.
        visual_references = [
            encoder.to_visual(text) for text in batch["texts"]
        ]

        all_hypotheses.extend(hypotheses)
        all_references.extend(visual_references)
        all_languages.extend(batch["languages"])

    if valid_batches == 0:
        raise RuntimeError("No valid validation batches were completed.")

    overall_cer = compute_cer(all_references, all_hypotheses)
    overall_wer = compute_wer(all_references, all_hypotheses)
    by_language = compute_metrics_by_language(all_references, all_hypotheses, all_languages)

    return {
        "loss": total_loss / valid_batches,
        "cer": overall_cer,
        "wer": overall_wer,
        "by_language": by_language,
    }


def save_checkpoint(
    checkpoint_path: Path,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    arguments: argparse.Namespace,
    num_classes: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "num_classes": num_classes,
            "image_height": arguments.image_height,
            "max_image_width": arguments.max_image_width,
            "hidden_size": arguments.hidden_size,
            "dropout": arguments.dropout,
            "run_name": arguments.run_name,
        },
        checkpoint_path,
    )


def append_metrics_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CRNN handwriting recognizer with CTC loss."
    )

    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--vocab", default="data/processed/vocab.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--max-image-width", type=int, default=2560)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-multi-gpu", action="store_true",
                         help="Disable DataParallel even if multiple GPUs are available.")

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    set_seed(arguments.seed)

    train_manifest = Path(arguments.train_manifest)
    validation_manifest = Path(arguments.validation_manifest)
    vocab_path = Path(arguments.vocab)

    vocabulary = load_vocabulary(vocab_path)
    num_classes = int(vocabulary["num_classes"])
    blank_index = int(vocabulary["blank_index"])

    encoder = CTCTextEncoder(vocab_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU count: {gpu_count}")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Mixed precision: {use_amp}")
    print(f"Classes: {num_classes}")

    train_loader = create_loader(
        manifest_path=train_manifest, vocab_path=vocab_path,
        image_height=arguments.image_height, max_image_width=arguments.max_image_width,
        batch_size=arguments.batch_size, shuffle=True, num_workers=arguments.num_workers,
    )

    validation_loader = create_loader(
        manifest_path=validation_manifest, vocab_path=vocab_path,
        image_height=arguments.image_height, max_image_width=arguments.max_image_width,
        batch_size=arguments.batch_size, shuffle=False, num_workers=arguments.num_workers,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")

    raw_model = CRNN(
        num_classes=num_classes, hidden_size=arguments.hidden_size, dropout=arguments.dropout,
    ).to(device)

    model: nn.Module = raw_model

    if gpu_count > 1 and not arguments.no_multi_gpu:
        print(f"Using {gpu_count} GPUs via DataParallel")
        model = nn.DataParallel(raw_model)

    optimizer = AdamW(raw_model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    checkpoint_dir = Path("outputs/checkpoints") / arguments.run_name
    metrics_csv_path = Path("outputs/metrics") / f"{arguments.run_name}.csv"

    best_validation_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, arguments.epochs + 1):
        print()
        print(f"Epoch {epoch}/{arguments.epochs}")

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model, raw_model=raw_model, loader=train_loader, optimizer=optimizer,
            criterion=criterion, scaler=scaler, device=device, use_amp=use_amp,
            gradient_clip=arguments.gradient_clip, log_every=arguments.log_every,
            accumulation_steps=arguments.accumulation_steps,
        )

        validation_results = validate(
            model=model, raw_model=raw_model, loader=validation_loader, criterion=criterion,
            device=device, use_amp=use_amp, encoder=encoder,
        )

        validation_loss = validation_results["loss"]
        scheduler.step(validation_loss)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation CER: {validation_results['cer']:.4f}")
        print(f"Validation WER: {validation_results['wer']:.4f}")
        print(f"Learning rate: {current_lr:.2e}")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_cer": validation_results["cer"],
            "validation_wer": validation_results["wer"],
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        }

        for lang, lang_metrics in validation_results["by_language"].items():
            row[f"cer_{lang}"] = lang_metrics["cer"]
            row[f"wer_{lang}"] = lang_metrics["wer"]
            row[f"count_{lang}"] = lang_metrics["count"]
            print(
                f"  [{lang}] CER={lang_metrics['cer']:.4f} "
                f"WER={lang_metrics['wer']:.4f} "
                f"(n={lang_metrics['count']})"
            )

        append_metrics_row(metrics_csv_path, row)

        plots_output_dir = Path("outputs/plots") / arguments.run_name
        try:
            plot_metrics(metrics_csv_path, plots_output_dir)
        except Exception as plot_error:
            print(f"Plot generation failed, continuing training: {plot_error}")

        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(
            checkpoint_path=latest_path, raw_model=raw_model, optimizer=optimizer, epoch=epoch,
            train_loss=train_loss, validation_loss=validation_loss,
            arguments=arguments, num_classes=num_classes,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_counter = 0

            best_path = checkpoint_dir / "best.pt"
            save_checkpoint(
                checkpoint_path=best_path, raw_model=raw_model, optimizer=optimizer, epoch=epoch,
                train_loss=train_loss, validation_loss=validation_loss,
                arguments=arguments, num_classes=num_classes,
            )
            print(f"Saved best checkpoint: {best_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

            if patience_counter >= arguments.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch} - "
                    f"no improvement for {arguments.early_stop_patience} epochs."
                )
                break

    print()
    print("Training complete.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Metrics saved to: {metrics_csv_path}")


if __name__ == "__main__":
    main()