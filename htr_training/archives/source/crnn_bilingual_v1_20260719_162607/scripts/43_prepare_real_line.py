from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def tight_crop(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 12,
) -> np.ndarray:
    points = cv2.findNonZero(mask)

    if points is None:
        return image

    x, y, width, height = cv2.boundingRect(points)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + width + padding)
    y2 = min(image.shape[0], y + height + padding)

    return image[y1:y2, x1:x2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--output-dir",
        default="samples/preprocessed",
    )
    args = parser.parse_args()

    source = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(source))

    if image is None:
        raise FileNotFoundError(source)

    # Variant 1: grayscale with local contrast enhancement.
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray_contrast = clahe.apply(gray)

    cv2.imwrite(
        str(output_dir / "line_gray_contrast.png"),
        gray_contrast,
    )

    # Variant 2: isolate blue handwriting and suppress gray rules.
    blue, green, red = cv2.split(
        image.astype(np.float32)
    )

    blue_strength = (
        blue - ((red + green) / 2.0)
    )

    ink_strength = np.clip(
        (blue_strength - 5.0) * 6.0,
        0,
        255,
    )

    blue_ink = (
        255 - ink_strength
    ).astype(np.uint8)

    cv2.imwrite(
        str(output_dir / "line_blue_ink.png"),
        blue_ink,
    )

    # Variant 3: tighter blue-ink crop.
    ink_mask = (
        blue_strength > 10
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    ink_mask = cv2.morphologyEx(
        ink_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    tight_blue = tight_crop(
        blue_ink,
        ink_mask,
        padding=10,
    )

    tight_blue = cv2.copyMakeBorder(
        tight_blue,
        8,
        8,
        12,
        12,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    cv2.imwrite(
        str(output_dir / "line_blue_ink_tight.png"),
        tight_blue,
    )

    print("Created:")
    for path in sorted(output_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
