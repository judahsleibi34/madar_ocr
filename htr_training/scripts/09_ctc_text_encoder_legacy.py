from __future__ import annotations

from pathlib import Path
import json

import torch


class CTCTextEncoder:
    def __init__(self, vocab_path: str | Path) -> None:
        self.vocab_path = Path(vocab_path)

        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        with self.vocab_path.open("r", encoding="utf-8") as file:
            vocabulary = json.load(file)

        self.char_to_index = vocabulary["char_to_index"]
        self.index_to_char = {
            int(index): character
            for index, character in vocabulary["index_to_char"].items()
        }
        self.blank_index = int(vocabulary["blank_index"])
        self.num_classes = int(vocabulary["num_classes"])

    def encode(self, text: str) -> torch.Tensor:
        indices = []
        for character in text:
            if character not in self.char_to_index:
                raise ValueError(f"Unknown character {character!r}")
            indices.append(self.char_to_index[character])
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: list[int] | torch.Tensor) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()

        characters = []
        previous_index = None

        for index in indices:
            if index == self.blank_index:
                previous_index = index
                continue

            if index == previous_index:
                continue

            character = self.index_to_char.get(index)
            if character is not None:
                characters.append(character)
            else:
                print(f"Warning: index {index} not found in vocabulary during decode.")

            previous_index = index

        return "".join(characters)

    def decode_batch(self, batch_indices: torch.Tensor) -> list[str]:
        """
        Decodes a full batch of greedy predictions.
        Expects shape [batch, time] (already argmax'd over classes).
        """
        return [self.decode(row) for row in batch_indices]
