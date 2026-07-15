from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"

# lib/crnn_dataset.py imports lib.ctc_text_encoder, so the
# scripts directory must be available as an import root.
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lib.crnn_dataset import CRNNDataset, ctc_collate_fn


DEFAULT_MANIFESTS = [
    (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "master_ctc"
        / "train_all.csv"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "master_ctc"
        / "val_all.csv"
    ),
    (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "master_ctc"
        / "test_all.csv"
    ),
]

DEFAULT_VOCAB = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_safe"
    / "vocab.json"
)

WIDTH_REDUCTION_FACTOR = 4


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the CRNN dataset loader against the "
            "new CTC-ready master manifests."
        )
    )

    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        dest="manifests",
        help=(
            "Manifest to test. May be supplied multiple "
            "times. Defaults to train, val, and test."
        ),
    )

    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB,
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
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--samples-per-manifest",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def select_indices(
    dataset_length: int,
    requested_samples: int,
) -> list[int]:
    if dataset_length < 1:
        raise ValueError(
            "Cannot select samples from an empty dataset."
        )

    sample_count = min(
        dataset_length,
        max(1, requested_samples),
    )

    if sample_count == 1:
        return [0]

    indices = {
        round(
            position
            * (dataset_length - 1)
            / (sample_count - 1)
        )
        for position in range(sample_count)
    }

    return sorted(indices)


def count_adjacent_repeats(text: str) -> int:
    return sum(
        previous == current
        for previous, current in zip(
            text,
            text[1:],
        )
    )


def validate_sample(
    dataset: CRNNDataset,
    index: int,
) -> dict[str, object]:
    sample = dataset[index]

    image = sample["image"]
    target = sample["target"]
    text = sample["text"]

    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"Sample {index}: image is not a tensor."
        )

    if image.ndim != 3:
        raise ValueError(
            f"Sample {index}: expected image shape "
            f"[C,H,W], received {tuple(image.shape)}."
        )

    if image.shape[0] != 1:
        raise ValueError(
            f"Sample {index}: expected one channel, "
            f"received {image.shape[0]}."
        )

    if image.shape[1] != dataset.image_height:
        raise ValueError(
            f"Sample {index}: expected height "
            f"{dataset.image_height}, "
            f"received {image.shape[1]}."
        )

    if image.shape[2] > dataset.max_image_width:
        raise ValueError(
            f"Sample {index}: image width "
            f"{image.shape[2]} exceeds "
            f"{dataset.max_image_width}."
        )

    if target.ndim != 1:
        raise ValueError(
            f"Sample {index}: target must be 1D."
        )

    if target.numel() == 0:
        raise ValueError(
            f"Sample {index}: empty encoded target."
        )

    if int(target.min()) < 0:
        raise ValueError(
            f"Sample {index}: negative target index."
        )

    if int(target.max()) >= dataset.num_classes:
        raise ValueError(
            f"Sample {index}: target index exceeds "
            f"the vocabulary size."
        )

    visual_text = dataset.text_encoder.to_visual(
        text
    )

    if len(visual_text) != target.numel():
        raise ValueError(
            f"Sample {index}: visual-text length "
            f"{len(visual_text)} differs from encoded "
            f"length {target.numel()}."
        )

    adjacent_repeats = count_adjacent_repeats(
        visual_text
    )

    required_timesteps = (
        len(visual_text)
        + adjacent_repeats
    )

    available_timesteps = max(
        1,
        image.shape[2]
        // WIDTH_REDUCTION_FACTOR,
    )

    if required_timesteps > available_timesteps:
        raise ValueError(
            f"Sample {index}: CTC requires "
            f"{required_timesteps} timesteps but only "
            f"{available_timesteps} are available."
        )

    return {
        "index": index,
        "image_width": image.shape[2],
        "target_length": target.numel(),
        "required_timesteps": required_timesteps,
        "available_timesteps": available_timesteps,
        "language": sample["language"],
        "dataset": sample["dataset"],
        "text": text,
    }


def validate_batch(
    dataset: CRNNDataset,
    indices: list[int],
    batch_size: int,
    num_workers: int,
) -> None:
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=min(
            batch_size,
            len(subset),
        ),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
    )

    batch = next(iter(loader))

    required_keys = {
        "images",
        "targets",
        "target_lengths",
        "image_widths",
        "texts",
        "languages",
        "datasets",
        "image_paths",
    }

    missing = required_keys - set(batch)

    if missing:
        raise ValueError(
            f"Collated batch is missing keys: "
            f"{sorted(missing)}"
        )

    images = batch["images"]
    image_widths = batch["image_widths"]
    target_lengths = batch["target_lengths"]

    if images.ndim != 4:
        raise ValueError(
            "Collated images must have shape [N,C,H,W]."
        )

    if images.shape[0] != len(
        target_lengths
    ):
        raise ValueError(
            "Batch size and target-length count differ."
        )

    if images.shape[0] != len(
        image_widths
    ):
        raise ValueError(
            "Batch size and image-width count differ."
        )

    if images.shape[2] != dataset.image_height:
        raise ValueError(
            "Unexpected collated image height."
        )

    if images.shape[3] % 64 != 0:
        raise ValueError(
            "Collated width was not padded to a "
            "multiple of 64."
        )

    if int(image_widths.max()) > images.shape[3]:
        raise ValueError(
            "An image width exceeds the padded width."
        )

    if int(target_lengths.sum()) != int(
        batch["targets"].numel()
    ):
        raise ValueError(
            "Concatenated target length is inconsistent."
        )

    print(
        "Batch images: "
        f"{tuple(images.shape)}"
    )
    print(
        "Batch targets: "
        f"{tuple(batch['targets'].shape)}"
    )
    print(
        "Image widths: "
        f"{image_widths.tolist()}"
    )
    print(
        "Target lengths: "
        f"{target_lengths.tolist()}"
    )


def test_manifest(
    manifest_path: Path,
    vocab_path: Path,
    image_height: int,
    max_image_width: int,
    batch_size: int,
    samples_per_manifest: int,
    num_workers: int,
) -> None:
    manifest_path = resolve_path(
        manifest_path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    dataset = CRNNDataset(
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        image_height=image_height,
        max_image_width=max_image_width,
    )

    indices = select_indices(
        len(dataset),
        samples_per_manifest,
    )

    print()
    print("=" * 78)
    print(
        f"Manifest: "
        f"{manifest_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Dataset rows: {len(dataset)}")
    print(f"Vocabulary classes: {dataset.num_classes}")
    print(f"Sample indices: {indices}")

    sample_reports = [
        validate_sample(dataset, index)
        for index in indices
    ]

    for report in sample_reports:
        print(
            f"  [{report['index']}] "
            f"{report['dataset']}/"
            f"{report['language']} "
            f"width={report['image_width']} "
            f"target={report['target_length']} "
            f"required={report['required_timesteps']} "
            f"available={report['available_timesteps']}"
        )

    validate_batch(
        dataset=dataset,
        indices=indices,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("Manifest test passed.")


def main() -> None:
    arguments = parse_arguments()

    if arguments.image_height != 64:
        raise ValueError(
            "The current CRNN architecture requires "
            "--image-height 64."
        )

    if arguments.max_image_width <= 0:
        raise ValueError(
            "--max-image-width must be positive."
        )

    if arguments.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    vocab_path = resolve_path(
        arguments.vocab
    )

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {vocab_path}"
        )

    manifests = (
        arguments.manifests
        if arguments.manifests
        else DEFAULT_MANIFESTS
    )

    for manifest_path in manifests:
        test_manifest(
            manifest_path=manifest_path,
            vocab_path=vocab_path,
            image_height=arguments.image_height,
            max_image_width=(
                arguments.max_image_width
            ),
            batch_size=arguments.batch_size,
            samples_per_manifest=(
                arguments.samples_per_manifest
            ),
            num_workers=arguments.num_workers,
        )

    print()
    print("=" * 78)
    print("ALL CRNN DATASET TESTS PASSED")


if __name__ == "__main__":
    main()
