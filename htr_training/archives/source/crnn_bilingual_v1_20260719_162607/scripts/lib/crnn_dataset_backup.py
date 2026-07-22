from __future__ import annotations

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
    Dataset for Arabic-English text recognition using CRNN + CTC.

    Relative image paths are resolved in this order:

    1. Relative to the manifest directory.
    2. Relative to the project root.
    3. Relative to the current working directory.

    Resolved paths are stored in dataframe["image"] so the dataset
    loader and bucket-width calculator use the same paths.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        vocab_path: str | Path,
        image_height: int = 64,
        max_image_width: int = 1024,
    ) -> None:
        self.manifest_path = (
            Path(manifest_path)
            .expanduser()
            .resolve()
        )

        self.vocab_path = (
            Path(vocab_path)
            .expanduser()
            .resolve()
        )

        # scripts/lib/crnn_dataset.py -> htr_training
        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        self.image_height = image_height
        self.max_image_width = max_image_width

        if image_height <= 0:
            raise ValueError(
                "image_height must be positive."
            )

        if max_image_width <= 0:
            raise ValueError(
                "max_image_width must be positive."
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{self.manifest_path}"
            )

        if not self.vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary not found: "
                f"{self.vocab_path}"
            )

        # keep_default_na=False prevents valid labels such as "NA"
        # from being interpreted as missing values.
        self.dataframe = pd.read_csv(
            self.manifest_path,
            dtype=str,
            keep_default_na=False,
        )

        required_columns = {
            "image",
            "text",
            "language",
            "dataset",
            "split",
        }

        missing_columns = (
            required_columns
            - set(self.dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{self.manifest_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        self.dataframe = (
            self.dataframe
            .reset_index(drop=True)
        )

        self.dataframe["image"] = (
            self.dataframe["image"]
            .astype(str)
            .str.strip()
        )

        self.dataframe["text"] = (
            self.dataframe["text"]
            .astype(str)
        )

        empty_images = (
            self.dataframe["image"] == ""
        )

        normalized_texts = (
            self.dataframe["text"]
            .map(self.normalize_text)
        )

        empty_texts = (
            normalized_texts == ""
        )

        if empty_images.any():
            indexes = (
                self.dataframe.index[
                    empty_images
                ]
                .tolist()
            )

            csv_rows = [
                index + 2
                for index in indexes[:20]
            ]

            raise ValueError(
                "Manifest contains empty image paths "
                f"at CSV rows: {csv_rows}"
            )

        if empty_texts.any():
            indexes = (
                self.dataframe.index[
                    empty_texts
                ]
                .tolist()
            )

            csv_rows = [
                index + 2
                for index in indexes[:20]
            ]

            raise ValueError(
                "Manifest contains empty transcriptions "
                f"at CSV rows: {csv_rows}"
            )

        self.dataframe["text"] = (
            normalized_texts
        )

        resolved_paths: list[str] = []

        for dataframe_index, value in enumerate(
            self.dataframe["image"]
        ):
            try:
                resolved_path = (
                    self.resolve_image_path(
                        str(value)
                    )
                )

            except FileNotFoundError as error:
                csv_row = dataframe_index + 2

                raise FileNotFoundError(
                    f"{error}\n"
                    f"Manifest: {self.manifest_path}\n"
                    f"CSV row: {csv_row}"
                ) from error

            resolved_paths.append(
                str(resolved_path)
            )

        # The bucket-width calculator reads this same column.
        self.dataframe["image"] = (
            resolved_paths
        )

        self.text_encoder = CTCTextEncoder(
            self.vocab_path
        )

        self.blank_index = (
            self.text_encoder.blank_index
        )

        self.num_classes = (
            self.text_encoder.num_classes
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        return unicodedata.normalize(
            "NFC",
            text,
        ).strip()

    def resolve_image_path(
        self,
        value: str,
    ) -> Path:
        """
        Resolve both local project paths and Kaggle package paths.
        """
        raw_path = (
            Path(value)
            .expanduser()
        )

        if raw_path.is_absolute():
            candidate = (
                raw_path.resolve()
            )

            if candidate.exists():
                return candidate

            raise FileNotFoundError(
                f"Image not found: {candidate}"
            )

        candidates = [
            (
                self.manifest_path.parent
                / raw_path
            ).resolve(),
            (
                self.project_root
                / raw_path
            ).resolve(),
            (
                Path.cwd()
                / raw_path
            ).resolve(),
        ]

        unique_candidates: list[Path] = []

        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(
                    candidate
                )

        for candidate in unique_candidates:
            if candidate.exists():
                return candidate

        searched_paths = "\n".join(
            f"  - {candidate}"
            for candidate in unique_candidates
        )

        raise FileNotFoundError(
            "Image path could not be resolved.\n"
            f"Manifest value: {value!r}\n"
            f"Searched:\n{searched_paths}"
        )

    def encode_text(
        self,
        text: str,
    ) -> torch.Tensor:
        return self.text_encoder.encode(
            text
        )

    def resize_image(
        self,
        image: Image.Image,
    ) -> Image.Image:
        original_width, original_height = (
            image.size
        )

        if (
            original_width <= 0
            or original_height <= 0
        ):
            raise ValueError(
                f"Invalid image size: {image.size}"
            )

        scale = (
            self.image_height
            / original_height
        )

        resized_width = max(
            1,
            round(
                original_width
                * scale
            ),
        )

        resized_width = min(
            resized_width,
            self.max_image_width,
        )

        return image.resize(
            (
                resized_width,
                self.image_height,
            ),
            Image.Resampling.BILINEAR,
        )

    def prepare_image(
        self,
        image_path: Path,
    ) -> torch.Tensor:
        with Image.open(
            image_path
        ) as image:
            image = image.convert("L")

            image = self.resize_image(
                image
            )

            tensor = TF.to_tensor(
                image
            )

        # Convert pixel range from [0, 1] to [-1, 1].
        return (
            tensor - 0.5
        ) / 0.5

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.dataframe.iloc[
            index
        ]

        image_path = Path(
            str(row["image"])
        )

        text = str(
            row["text"]
        )

        image = self.prepare_image(
            image_path
        )

        target = self.encode_text(
            text
        )

        if target.numel() == 0:
            raise ValueError(
                "Encoded target is empty for image: "
                f"{image_path}"
            )

        return {
            "image": image,
            "target": target,
            "target_length": int(
                target.numel()
            ),
            "text": text,
            "language": str(
                row["language"]
            ),
            "dataset": str(
                row["dataset"]
            ),
            "image_path": str(
                image_path
            ),
        }


def round_up(
    value: int,
    multiple: int,
) -> int:
    if multiple <= 0:
        raise ValueError(
            "multiple must be positive."
        )

    return int(
        math.ceil(
            value / multiple
        )
        * multiple
    )


def ctc_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Pad images to the widest image in the batch.

    White padding is represented by 1.0 because image pixels are
    normalized from [0, 1] into [-1, 1].
    """
    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    batch_size = len(batch)

    channels = (
        batch[0]["image"]
        .shape[0]
    )

    image_height = (
        batch[0]["image"]
        .shape[1]
    )

    image_widths = torch.tensor(
        [
            sample["image"].shape[2]
            for sample in batch
        ],
        dtype=torch.long,
    )

    maximum_width = round_up(
        int(
            image_widths
            .max()
            .item()
        ),
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

    for index, sample in enumerate(
        batch
    ):
        image = sample["image"]
        width = image.shape[2]

        images[
            index,
            :,
            :,
            :width,
        ] = image

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
