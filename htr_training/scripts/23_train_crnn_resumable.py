from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_base_module():
    script_dir = Path(__file__).resolve().parent
    path = script_dir / "36_legacy_train_crnn_dual_gpu.py"

    spec = importlib.util.spec_from_file_location(
        "base_crnn_training",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load base training script: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    """
    Save to a temporary file first, then replace the destination.

    This lowers the risk of leaving a corrupted checkpoint if the
    process stops during torch.save().
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_metrics_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, path)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(
    state: dict[str, Any] | None,
) -> None:
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def torch_load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def load_kaggle_token(secret_name: str) -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret(
            secret_name
        )
    except Exception as error:
        print(
            f"Kaggle secret {secret_name!r} unavailable: "
            f"{error}"
        )
        return False

    if not token:
        return False

    os.environ["KAGGLE_API_TOKEN"] = token
    return True


def backup_to_kaggle(
    dataset_id: str | None,
    dataset_title: str,
    secret_name: str,
    epoch: int,
    checkpoint_dir: Path,
    metrics_csv: Path,
    upload_dir: Path,
) -> bool:
    """
    Copy local checkpoints to a private Kaggle Dataset.

    Local saving always happens first. A failed upload will not stop
    training or remove the local checkpoint.
    """
    if not dataset_id:
        return False

    if shutil.which("kaggle") is None:
        print(
            "Kaggle CLI not found; remote backup skipped."
        )
        return False

    if not load_kaggle_token(secret_name):
        print(
            "Kaggle API token unavailable; "
            "remote backup skipped."
        )
        return False

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        upload_dir / "dataset-metadata.json"
    )

    check = subprocess.run(
        [
            "kaggle",
            "datasets",
            "metadata",
            dataset_id,
            "-p",
            str(upload_dir),
        ],
        capture_output=True,
        text=True,
    )

    dataset_exists = (
        check.returncode == 0
        and metadata_path.exists()
    )

    if not dataset_exists:
        metadata = {
            "title": dataset_title,
            "id": dataset_id,
            "licenses": [
                {
                    "name": "other",
                }
            ],
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    # Clean old temporary upload files, but preserve metadata.
    for item in upload_dir.iterdir():
        if item == metadata_path:
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    copied_files: list[str] = []

    for filename in (
        "latest.pt",
        "best_cer.pt",
        "best_loss.pt",
    ):
        source = checkpoint_dir / filename

        if source.exists():
            shutil.copy2(
                source,
                upload_dir / filename,
            )
            copied_files.append(filename)

    if metrics_csv.exists():
        shutil.copy2(
            metrics_csv,
            upload_dir / metrics_csv.name,
        )
        copied_files.append(metrics_csv.name)

    if not copied_files:
        print(
            "No checkpoint files were available "
            "for Kaggle backup."
        )
        return False

    if dataset_exists:
        command = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(upload_dir),
            "-m",
            f"Checkpoint after epoch {epoch}",
            "-q",
            "-t",
            "-r",
            "skip",
        ]
    else:
        command = [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(upload_dir),
            "-q",
            "-t",
            "-r",
            "skip",
        ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        print(
            "Remote backup failed; local files "
            f"are still safe: {error}"
        )

        if (
            isinstance(
                error,
                subprocess.CalledProcessError,
            )
            and error.stderr
        ):
            print(error.stderr.strip())

        return False

    print(
        "Remote Kaggle backup complete: "
        + ", ".join(copied_files)
    )

    if result.stdout.strip():
        print(result.stdout.strip())

    return True


def default_output_root() -> str:
    if Path("/kaggle/working").exists():
        return "/kaggle/working/outputs"

    return "outputs"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable bilingual CRNN training with "
            "dynamic LR and Kaggle backup."
        )
    )

    parser.add_argument(
        "--train-manifest",
        required=True,
    )
    parser.add_argument(
        "--validation-manifest",
        required=True,
    )
    parser.add_argument(
        "--run-name",
        required=True,
    )
    parser.add_argument(
        "--vocab",
        default="data/processed/master_safe/vocab.json",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
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
        "--hidden-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no-multi-gpu",
        action="store_true",
    )

    # Dynamic learning-rate settings.
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--lr-threshold",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--lr-cooldown",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--output-root",
        default=default_output_root(),
    )
    parser.add_argument(
        "--resume",
        default=None,
    )

    # Optional Kaggle Dataset backup.
    parser.add_argument(
        "--kaggle-dataset",
        default=None,
    )
    parser.add_argument(
        "--kaggle-dataset-title",
        default="MADAR OCR Checkpoints",
    )
    parser.add_argument(
        "--kaggle-secret-name",
        default="KAGGLE_API_TOKEN",
    )
    parser.add_argument(
        "--kaggle-upload-every",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.kaggle_upload_every < 1:
        raise ValueError(
            "--kaggle-upload-every must be at least 1."
        )

    if args.accumulation_steps < 1:
        raise ValueError(
            "--accumulation-steps must be at least 1."
        )

    base.set_seed(args.seed)

    train_manifest = Path(args.train_manifest)
    validation_manifest = Path(
        args.validation_manifest
    )
    vocab_path = Path(args.vocab)
    output_root = Path(args.output_root)

    vocabulary = base.load_vocabulary(vocab_path)

    num_classes = int(
        vocabulary["num_classes"]
    )
    blank_index = int(
        vocabulary["blank_index"]
    )

    encoder = base.CTCTextEncoder(vocab_path)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = device.type == "cuda"

    gpu_count = (
        torch.cuda.device_count()
        if use_amp
        else 0
    )

    print(f"Device: {device}")

    if use_amp:
        print(f"GPU count: {gpu_count}")

        for index in range(gpu_count):
            print(
                f"  GPU {index}: "
                f"{torch.cuda.get_device_name(index)}"
            )

    print(f"Mixed precision: {use_amp}")
    print(f"Classes: {num_classes}")

    train_loader = base.create_loader(
        train_manifest,
        vocab_path,
        args.image_height,
        args.max_image_width,
        args.batch_size,
        True,
        args.num_workers,
    )

    validation_loader = base.create_loader(
        validation_manifest,
        vocab_path,
        args.image_height,
        args.max_image_width,
        args.batch_size,
        False,
        args.num_workers,
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )
    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )

    raw_model = base.CRNN(
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)

    model: nn.Module = raw_model

    if (
        gpu_count > 1
        and not args.no_multi_gpu
    ):
        print(
            f"Using {gpu_count} GPUs "
            "via DataParallel"
        )
        model = nn.DataParallel(raw_model)

    optimizer = AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    # Validation loss controls dynamic LR reduction.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        threshold_mode="rel",
        cooldown=args.lr_cooldown,
        min_lr=args.min_lr,
    )

    criterion = nn.CTCLoss(
        blank=blank_index,
        reduction="mean",
        zero_infinity=True,
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
    )

    checkpoint_dir = (
        output_root
        / "checkpoints"
        / args.run_name
    )

    metrics_csv = (
        output_root
        / "metrics"
        / f"{args.run_name}.csv"
    )

    plots_dir = (
        output_root
        / "plots"
        / args.run_name
    )

    upload_dir = (
        output_root
        / "kaggle_upload"
        / args.run_name
    )

    start_epoch = 1
    best_loss = float("inf")
    best_cer = float("inf")
    cer_bad_epochs = 0

    history: list[dict[str, Any]] = []

    if args.resume:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: "
                f"{resume_path}"
            )

        checkpoint = torch_load_checkpoint(
            resume_path,
            device,
        )

        raw_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        restore_rng_state(
            checkpoint.get("rng_state")
        )

        start_epoch = (
            int(checkpoint["epoch"]) + 1
        )

        best_loss = float(
            checkpoint.get(
                "best_validation_loss",
                checkpoint["validation_loss"],
            )
        )

        best_cer = float(
            checkpoint.get(
                "best_validation_cer",
                checkpoint["validation_cer"],
            )
        )

        cer_bad_epochs = int(
            checkpoint.get(
                "cer_bad_epochs",
                0,
            )
        )

        history = list(
            checkpoint.get(
                "history",
                [],
            )
        )

        write_metrics_csv(
            metrics_csv,
            history,
        )

        print(
            f"Resumed from {resume_path} "
            f"at epoch {start_epoch}"
        )
        print(
            "Restored learning rate: "
            f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        started = time.time()

        train_loss = base.train_one_epoch(
            model=model,
            raw_model=raw_model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            gradient_clip=args.gradient_clip,
            log_every=100,
            accumulation_steps=(
                args.accumulation_steps
            ),
        )

        results = base.validate(
            model=model,
            raw_model=raw_model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            encoder=encoder,
        )

        validation_loss = float(
            results["loss"]
        )

        validation_cer = float(
            results["cer"]
        )

        old_lr = optimizer.param_groups[0]["lr"]

        # Dynamic LR: lower the LR when validation loss plateaus.
        scheduler.step(validation_loss)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        improved_loss = (
            validation_loss < best_loss
        )

        improved_cer = (
            validation_cer < best_cer
        )

        if improved_loss:
            best_loss = validation_loss

        if improved_cer:
            best_cer = validation_cer
            cer_bad_epochs = 0
        else:
            cer_bad_epochs += 1

        print(
            f"Train loss: {train_loss:.4f}"
        )
        print(
            "Validation loss: "
            f"{validation_loss:.4f}"
        )
        print(
            "Validation CER: "
            f"{validation_cer:.4f}"
        )
        print(
            "Validation WER: "
            f"{results['wer']:.4f}"
        )
        print(
            f"Learning rate: {current_lr:.2e}"
        )

        if current_lr < old_lr:
            print(
                "Learning rate reduced: "
                f"{old_lr:.2e} -> {current_lr:.2e}"
            )

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_cer": validation_cer,
            "validation_wer": results["wer"],
            "learning_rate": current_lr,
            "epoch_time_seconds": (
                time.time() - started
            ),
        }

        for language, metrics in (
            results["by_language"].items()
        ):
            row[f"cer_{language}"] = (
                metrics["cer"]
            )
            row[f"wer_{language}"] = (
                metrics["wer"]
            )
            row[f"count_{language}"] = (
                metrics["count"]
            )

            print(
                f"  [{language}] "
                f"CER={metrics['cer']:.4f} "
                f"WER={metrics['wer']:.4f} "
                f"(n={metrics['count']})"
            )

        history.append(row)

        write_metrics_csv(
            metrics_csv,
            history,
        )

        try:
            base.plot_metrics(
                metrics_csv,
                plots_dir,
            )
        except Exception as error:
            print(
                "Plot generation failed, "
                f"continuing: {error}"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                raw_model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "scaler_state_dict": (
                scaler.state_dict()
            ),
            "rng_state": capture_rng_state(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_cer": validation_cer,
            "validation_wer": results["wer"],
            "best_validation_loss": best_loss,
            "best_validation_cer": best_cer,
            "cer_bad_epochs": cer_bad_epochs,
            "history": history,
            "num_classes": num_classes,
            "arguments": vars(args),
        }

        latest_path = (
            checkpoint_dir / "latest.pt"
        )

        atomic_torch_save(
            checkpoint,
            latest_path,
        )

        print(
            "Saved latest checkpoint: "
            f"{latest_path}"
        )

        if improved_loss:
            best_loss_path = (
                checkpoint_dir
                / "best_loss.pt"
            )

            atomic_torch_save(
                checkpoint,
                best_loss_path,
            )

            print(
                "Saved best-loss checkpoint: "
                f"{best_loss_path}"
            )

        if improved_cer:
            best_cer_path = (
                checkpoint_dir
                / "best_cer.pt"
            )

            atomic_torch_save(
                checkpoint,
                best_cer_path,
            )

            print(
                "Saved best-CER checkpoint: "
                f"{best_cer_path}"
            )
        else:
            print(
                "No CER improvement for "
                f"{cer_bad_epochs} epoch(s)."
            )

        should_upload = (
            epoch % args.kaggle_upload_every == 0
            or improved_loss
            or improved_cer
        )

        if should_upload:
            backup_to_kaggle(
                dataset_id=args.kaggle_dataset,
                dataset_title=(
                    args.kaggle_dataset_title
                ),
                secret_name=(
                    args.kaggle_secret_name
                ),
                epoch=epoch,
                checkpoint_dir=checkpoint_dir,
                metrics_csv=metrics_csv,
                upload_dir=upload_dir,
            )

        if (
            cer_bad_epochs
            >= args.early_stop_patience
        ):
            print(
                "Early stopping: no CER "
                "improvement for "
                f"{args.early_stop_patience} "
                "epochs."
            )
            break

    print("\nTraining complete.")
    print(
        "Best validation loss: "
        f"{best_loss:.4f}"
    )
    print(
        "Best validation CER: "
        f"{best_cer:.4f}"
    )
    print(
        f"Checkpoints: {checkpoint_dir}"
    )


if __name__ == "__main__":
    main()
