from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.crnn_model import CRNN


ARABIC_DATASETS = [
    "SAQR",
    "KHATT",
    "Muharaf",
]

ENGLISH_DATASETS = [
    "IAM",
    "GoBo",
    "KSTRV2",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Arabic-focused fine-tuning manifest "
            "and a fresh optimizer checkpoint."
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
        "--vocab",
        default=(
            "data/processed/master_safe/"
            "vocab.json"
        ),
    )

    parser.add_argument(
        "--source-checkpoint",
        default=(
            "outputs/checkpoints/"
            "bilingual_v4_master_ctc/"
            "best_cer.pt"
        ),
    )

    parser.add_argument(
        "--output-manifest",
        default=(
            "data/processed/master_ctc/"
            "train_finetune_ar70_en30.csv"
        ),
    )

    parser.add_argument(
        "--init-checkpoint",
        default=(
            "outputs/checkpoints/"
            "finetune_initializers/"
            "ar70_en30_epoch22_init.pt"
        ),
    )

    parser.add_argument(
        "--stats-output",
        default=(
            "outputs/"
            "finetune_ar70_en30_distribution.csv"
        ),
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=60000,
    )

    parser.add_argument(
        "--arabic-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def torch_load(
    path: Path,
) -> dict:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def normalize_language(
    value: object,
) -> str:
    value = str(value).strip().lower()

    if value.startswith("ar"):
        return "ar"

    if value.startswith("en"):
        return "en"

    return value


def allocate_rows(
    total: int,
    groups: int,
) -> list[int]:
    quotient, remainder = divmod(
        total,
        groups,
    )

    return [
        quotient + (1 if index < remainder else 0)
        for index in range(groups)
    ]


def sample_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    expected_language: str,
    rows: int,
    seed: int,
) -> pd.DataFrame:
    mask = (
        dataframe["__dataset_norm"].eq(
            dataset_name.lower()
        )
        & dataframe["__language_norm"].eq(
            expected_language
        )
    )

    subset = dataframe.loc[mask]

    if subset.empty:
        available = (
            dataframe.groupby(
                [
                    "__language_norm",
                    "__dataset_norm",
                ]
            )
            .size()
            .to_string()
        )

        raise ValueError(
            f"No rows found for dataset={dataset_name!r}, "
            f"language={expected_language!r}.\n\n"
            f"Available groups:\n{available}"
        )

    replace = rows > len(subset)

    sampled = subset.sample(
        n=rows,
        replace=replace,
        random_state=seed,
    )

    print(
        f"{dataset_name:8s} "
        f"source={len(subset):6,d} "
        f"sampled={rows:6,d} "
        f"replacement={replace}"
    )

    return sampled


def strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if (
        state_dict
        and all(
            key.startswith("module.")
            for key in state_dict
        )
    ):
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    return state_dict


def atomic_torch_save(
    payload: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
        output_path
    )


def main() -> None:
    args = parse_arguments()

    if args.rows < 6:
        raise ValueError(
            "--rows must be at least 6."
        )

    if not 0.0 < args.arabic_ratio < 1.0:
        raise ValueError(
            "--arabic-ratio must be between 0 and 1."
        )

    train_manifest = Path(
        args.train_manifest
    ).expanduser().resolve()

    vocab_path = Path(
        args.vocab
    ).expanduser().resolve()

    source_checkpoint_path = Path(
        args.source_checkpoint
    ).expanduser().resolve()

    output_manifest = Path(
        args.output_manifest
    ).expanduser().resolve()

    init_checkpoint_path = Path(
        args.init_checkpoint
    ).expanduser().resolve()

    stats_output = Path(
        args.stats_output
    ).expanduser().resolve()

    for required_path in (
        train_manifest,
        vocab_path,
        source_checkpoint_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_path}"
            )

    dataframe = pd.read_csv(
        train_manifest,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "text",
        "language",
        "dataset",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    original_columns = list(
        dataframe.columns
    )

    dataframe["__language_norm"] = (
        dataframe["language"].map(
            normalize_language
        )
    )

    dataframe["__dataset_norm"] = (
        dataframe["dataset"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    arabic_rows = round(
        args.rows * args.arabic_ratio
    )

    english_rows = (
        args.rows - arabic_rows
    )

    arabic_allocations = allocate_rows(
        arabic_rows,
        len(ARABIC_DATASETS),
    )

    english_allocations = allocate_rows(
        english_rows,
        len(ENGLISH_DATASETS),
    )

    print(
        f"Original rows: {len(dataframe):,}"
    )

    print(
        f"Fine-tuning rows: {args.rows:,}"
    )

    print(
        f"Arabic rows: {arabic_rows:,}"
    )

    print(
        f"English rows: {english_rows:,}"
    )

    print()
    print("Sampling Arabic datasets...")

    sampled_parts: list[
        pd.DataFrame
    ] = []

    for index, (
        dataset_name,
        row_count,
    ) in enumerate(
        zip(
            ARABIC_DATASETS,
            arabic_allocations,
        )
    ):
        sampled_parts.append(
            sample_dataset(
                dataframe=dataframe,
                dataset_name=dataset_name,
                expected_language="ar",
                rows=row_count,
                seed=args.seed + index,
            )
        )

    print()
    print("Sampling English datasets...")

    for index, (
        dataset_name,
        row_count,
    ) in enumerate(
        zip(
            ENGLISH_DATASETS,
            english_allocations,
        )
    ):
        sampled_parts.append(
            sample_dataset(
                dataframe=dataframe,
                dataset_name=dataset_name,
                expected_language="en",
                rows=row_count,
                seed=(
                    args.seed
                    + 100
                    + index
                ),
            )
        )

    selected = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    selected = selected.sample(
        frac=1.0,
        random_state=args.seed + 1000,
    ).reset_index(drop=True)

    selected = selected[
        original_columns
    ]

    output_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        output_manifest,
        index=False,
        encoding="utf-8-sig",
    )

    distribution = (
        selected.groupby(
            [
                "language",
                "dataset",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .sort_values(
            [
                "language",
                "dataset",
            ]
        )
    )

    stats_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution.to_csv(
        stats_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Created distribution:")
    print(
        distribution.to_string(
            index=False
        )
    )

    vocabulary = json.loads(
        vocab_path.read_text(
            encoding="utf-8-sig",
        )
    )

    source_checkpoint = torch_load(
        source_checkpoint_path
    )

    source_arguments = (
        source_checkpoint.get(
            "arguments",
            {},
        )
    )

    hidden_size = int(
        source_arguments.get(
            "hidden_size",
            source_checkpoint.get(
                "hidden_size",
                256,
            ),
        )
    )

    dropout = float(
        source_arguments.get(
            "dropout",
            source_checkpoint.get(
                "dropout",
                0.1,
            ),
        )
    )

    num_classes = int(
        vocabulary["num_classes"]
    )

    model = CRNN(
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout=dropout,
    )

    state_dict = strip_module_prefix(
        source_checkpoint[
            "model_state_dict"
        ]
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    initializer = {
        "epoch": 0,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "validation_loss": (
            float("inf")
        ),
        "validation_cer": (
            float("inf")
        ),
        "validation_wer": (
            float("inf")
        ),
        "best_validation_loss": (
            float("inf")
        ),
        "best_validation_cer": (
            float("inf")
        ),
        "cer_bad_epochs": 0,
        "history": [],
        "num_classes": num_classes,
        "arguments": {
            "fine_tuning_initializer": True,
            "source_checkpoint": str(
                source_checkpoint_path
            ),
            "source_epoch": (
                source_checkpoint.get(
                    "epoch"
                )
            ),
            "hidden_size": hidden_size,
            "dropout": dropout,
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "arabic_ratio": (
                args.arabic_ratio
            ),
            "manifest_rows": (
                args.rows
            ),
        },
    }

    atomic_torch_save(
        initializer,
        init_checkpoint_path,
    )

    print()
    print(
        f"Manifest saved: "
        f"{output_manifest}"
    )

    print(
        f"Distribution saved: "
        f"{stats_output}"
    )

    print(
        f"Initializer saved: "
        f"{init_checkpoint_path}"
    )

    print(
        "Source checkpoint epoch: "
        f"{source_checkpoint.get('epoch')}"
    )

    print(
        "Fresh learning rate: "
        f"{args.learning_rate:.2e}"
    )


if __name__ == "__main__":
    main()
