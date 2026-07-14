from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from lib.ctc_text_encoder import CTCTextEncoder


class CRNNDataset(Dataset):
    """
    Dataset for handwritten text recognition using CRNN + CTC.

    Each item contains:
        image: grayscale tensor with shape [1, height, width]
        target: encoded character IDs
        target_length: number of target characters
        text: original transcription
        language: ar or en
        dataset: source dataset name
        image_path: original image path
    """

    def __init__(
        self,
        manifest_path: str | Path,
        vocab_path: str | Path,
        image_height: int = 64,
        max_image_width: int = 1024,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.vocab_path = Path(vocab_path)
        self.image_height = image_height
        self.max_image_width = max_image_width

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        if not self.vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary not found: {self.vocab_path}"
            )

        self.dataframe = pd.read_csv(self.manifest_path)

        required_columns = {
            "image",
            "text",
            "language",
            "dataset",
            "split",
        }

        missing_columns = required_columns - set(
            self.dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                f"{self.manifest_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        self.dataframe = self.dataframe.dropna(
            subset=["image", "text"]
        ).reset_index(drop=True)

        # Single source of truth for text <-> index encoding, including
        # RTL-aware reversal for Arabic text. Both training-time encoding
        # (here) and validation-time decoding (in the training script) go
        # through this same object, so the visual left-to-right ordering
        # stays consistent everywhere.
        self.text_encoder = CTCTextEncoder(self.vocab_path)

        self.blank_index = self.text_encoder.blank_index
        self.num_classes = self.text_encoder.num_classes

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def normalize_text(text: str) -> str:
        # NFC preserves Arabic and English characters while normalizing
        # equivalent Unicode character sequences.
        return unicodedata.normalize("NFC", text).strip()

    def encode_text(self, text: str) -> torch.Tensor:
        return self.text_encoder.encode(text)

    def resize_image(self, image: Image.Image) -> Image.Image:
        original_width, original_height = image.size

        if original_width <= 0 or original_height <= 0:
            raise ValueError(
                f"Invalid image size: {image.size}"
            )

        scale = self.image_height / original_height

        resized_width = max(
            1,
            round(original_width * scale),
        )

        resized_width = min(
            resized_width,
            self.max_image_width,
        )

        return image.resize(
            (resized_width, self.image_height),
            Image.Resampling.BILINEAR,
        )

    def prepare_image(
        self,
        image_path: Path,
    ) -> torch.Tensor:
        with Image.open(image_path) as image:
            image = image.convert("L")
            image = self.resize_image(image)

            tensor = TF.to_tensor(image)

        # Convert the pixel range from [0, 1] to [-1, 1].
        tensor = (tensor - 0.5) / 0.5

        return tensor

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.dataframe.iloc[index]

        image_path = Path(str(row["image"]))
        text = self.normalize_text(str(row["text"]))

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not text:
            raise ValueError(
                f"Empty transcription for image: {image_path}"
            )

        image = self.prepare_image(image_path)
        target = self.encode_text(text)

        return {
            "image": image,
            "target": target,
            "target_length": len(target),
            "text": text,
            "language": str(row["language"]),
            "dataset": str(row["dataset"]),
            "image_path": str(image_path),
        }


def round_up(
    value: int,
    multiple: int,
) -> int:
    return int(
        math.ceil(value / multiple) * multiple
    )


def ctc_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Pads images to the widest image in the batch.

    White padding is represented by 1.0 because images were normalized
    from [0, 1] into [-1, 1].
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    batch_size = len(batch)
    channels = batch[0]["image"].shape[0]
    image_height = batch[0]["image"].shape[1]

    image_widths = torch.tensor(
        [
            sample["image"].shape[2]
            for sample in batch
        ],
        dtype=torch.long,
    )

    maximum_width = round_up(
        int(image_widths.max().item()),
        multiple=64,
    )

    images = torch.ones(
        (
            batch_size,
            channels,
            image_height,
            maximum_width,
        ),
        dtype=torch.float32,
    )

    for index, sample in enumerate(batch):
        image = sample["image"]
        width = image.shape[2]

        images[index, :, :, :width] = image

    targets = torch.cat(
        [
            sample["target"]
            for sample in batch
        ],
        dim=0,
    )

    target_lengths = torch.tensor(
        [
            sample["target_length"]
            for sample in batch
        ],
        dtype=torch.long,
    )

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "image_widths": image_widths,
        "texts": [
            sample["text"]
            for sample in batch
        ],
        "languages": [
            sample["language"]
            for sample in batch
        ],
        "datasets": [
            sample["dataset"]
            for sample in batch
        ],
        "image_paths": [
            sample["image_path"]
            for sample in batch
        ],
    }