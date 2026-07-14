from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def plot_metrics(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Loss curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y="train_loss", label="Train Loss", ax=ax, linewidth=2.5)
    sns.lineplot(data=df, x="epoch", y="validation_loss", label="Validation Loss", ax=ax, linewidth=2.5)
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CTC Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=200)
    plt.close(fig)

    # --- CER / WER curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y="validation_cer", label="CER", ax=ax, linewidth=2.5)
    sns.lineplot(data=df, x="epoch", y="validation_wer", label="WER", ax=ax, linewidth=2.5)
    ax.set_title("Validation Character/Word Error Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_rates.png", dpi=200)
    plt.close(fig)

    # --- Per-language CER/WER, if present ---
    lang_cer_cols = [c for c in df.columns if c.startswith("cer_")]
    lang_wer_cols = [c for c in df.columns if c.startswith("wer_")]

    if lang_cer_cols:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for col in lang_cer_cols:
            lang = col.replace("cer_", "")
            sns.lineplot(data=df, x="epoch", y=col, label=lang, ax=axes[0], linewidth=2.5)
        axes[0].set_title("CER by Language")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("CER")
        axes[0].legend()

        for col in lang_wer_cols:
            lang = col.replace("wer_", "")
            sns.lineplot(data=df, x="epoch", y=col, label=lang, ax=axes[1], linewidth=2.5)
        axes[1].set_title("WER by Language")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("WER")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(output_dir / "error_rates_by_language.png", dpi=200)
        plt.close(fig)

    # --- Learning rate schedule ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y="learning_rate", ax=ax, linewidth=2.5, color="darkorange")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_rate.png", dpi=200)
    plt.close(fig)

    # --- Epoch duration (useful for spotting slowdowns) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="epoch", y="epoch_time_seconds", ax=ax, color="steelblue")
    ax.set_title("Epoch Duration")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(output_dir / "epoch_duration.png", dpi=200)
    plt.close(fig)

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    csv_path = Path("outputs/metrics") / f"{args.run_name}.csv"
    output_dir = Path("outputs/plots") / args.run_name

    plot_metrics(csv_path, output_dir)
