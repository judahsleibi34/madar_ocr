from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Sampler
from tqdm import tqdm


def compute_effective_widths(
    dataframe: pd.DataFrame,
    image_height: int,
    max_image_width: int,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return np.load(cache_path)

    widths = []

    for image_path in tqdm(dataframe["image"], desc="Computing effective widths"):
        with Image.open(image_path) as img:
            original_width, original_height = img.size

        scale = image_height / original_height
        resized_width = max(1, round(original_width * scale))
        resized_width = min(resized_width, max_image_width)

        widths.append(resized_width)

    widths = np.array(widths, dtype=np.int32)

    if cache_path is not None:
        np.save(cache_path, widths)

    return widths


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        widths: np.ndarray,
        batch_size: int,
        bucket_multiplier: int = 50,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.widths = widths
        self.batch_size = batch_size
        self.bucket_multiplier = bucket_multiplier
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        indices = np.argsort(self.widths)

        mega_batch_size = self.batch_size * self.bucket_multiplier
        mega_batches = [
            indices[i : i + mega_batch_size]
            for i in range(0, len(indices), mega_batch_size)
        ]

        if self.shuffle:
            for mega_batch in mega_batches:
                np.random.shuffle(mega_batch)
            random.shuffle(mega_batches)

        all_batches = []
        for mega_batch in mega_batches:
            for i in range(0, len(mega_batch), self.batch_size):
                batch = mega_batch[i : i + self.batch_size].tolist()
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.widths) // self.batch_size
        return (len(self.widths) + self.batch_size - 1) // self.batch_size
