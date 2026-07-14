from __future__ import annotations

from pathlib import Path
import json

import torch


# Unicode code point ranges covering Arabic script (base block, supplement,
# extended-A, and the presentation-form blocks used by some fonts/renderers).
_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def _is_arabic_character(character: str) -> bool:
    code_point = ord(character)
    return any(low <= code_point <= high for low, high in _ARABIC_RANGES)


def _is_rtl_text(text: str) -> bool:
    """
    Heuristic: a string is treated as RTL if it contains at least one
    Arabic-script character. This matches this dataset's structure, where
    each sample belongs to a single-script corpus (khatt/muharaf = Arabic,
    iam = English) rather than mixed-script lines.
    """
    return any(_is_arabic_character(character) for character in text)


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
        """
        Encodes ground-truth text into target indices for CTC training.

        Arabic (RTL) text is stored in the manifest in logical reading
        order (first character = first character a human reads, which sits
        on the RIGHT side of the rendered image). The CNN backbone scans
        the image left-to-right in time, so CTC needs the target sequence
        in the same left-to-right visual order. We reverse RTL text here
        so index 0 of the target corresponds to the leftmost glyph in the
        image, matching frame 0 of the CNN's time axis.

        English (LTR) text is already in the correct visual order and is
        left untouched.
        """
        if _is_rtl_text(text):
            text = text[::-1]

        indices = []
        for character in text:
            if character not in self.char_to_index:
                raise ValueError(f"Unknown character {character!r}")
            indices.append(self.char_to_index[character])
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: list[int] | torch.Tensor) -> str:
        """
        Greedy CTC decode: collapses repeats and drops blanks, producing
        characters in the same left-to-right visual order the model
        predicted them in (matching the time axis).

        If the collapsed result is Arabic (RTL), we reverse it back to
        logical reading order before returning, so it's directly
        comparable to the ground-truth reference strings (which are
        stored in logical order) for CER/WER computation.
        """
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

        decoded_text = "".join(characters)

        if _is_rtl_text(decoded_text):
            decoded_text = decoded_text[::-1]

        return decoded_text

    def decode_batch(self, batch_indices: torch.Tensor) -> list[str]:
        """
        Decodes a full batch of greedy predictions.
        Expects shape [batch, time] (already argmax'd over classes).
        """
        return [self.decode(row) for row in batch_indices]