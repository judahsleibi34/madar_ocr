#!/usr/bin/env python
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_MANIFEST_DIR = Path(r"data\processed\online_htr\ihwwr_english")
DEFAULT_OUTPUT = Path(r"kaggle_upload\ihwwr_trocr_data.zip")


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path.resolve())

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"image", "text"} - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path.name} is missing columns: {sorted(missing)}")

    empty = frame["text"].astype(str).str.strip().eq("")
    if empty.any():
        raise RuntimeError(f"{path.name} contains {int(empty.sum())} empty labels.")

    missing_images = [value for value in frame["image"] if not Path(value).exists()]
    if missing_images:
        raise FileNotFoundError(
            f"{path.name} references {len(missing_images)} missing images. "
            f"First: {missing_images[0]}"
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the private Kaggle data ZIP for TrOCR training."
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also include the untouched test split. The training script never reads it.",
    )
    args = parser.parse_args()

    manifest_dir = args.manifest_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest_names = ["train_model.csv", "val.csv"]
    if args.include_test:
        manifest_names.append("test.csv")

    manifests = {
        name: load_manifest(manifest_dir / name)
        for name in manifest_names
    }

    images: dict[str, Path] = {}
    portable: dict[str, pd.DataFrame] = {}

    for name, frame in manifests.items():
        converted = frame.copy()
        relative_paths: list[str] = []

        for value in frame["image"]:
            image_path = Path(value).resolve()
            filename = image_path.name
            previous = images.get(filename)
            if previous is not None and previous != image_path:
                raise RuntimeError(
                    f"Filename collision for {filename}:\n{previous}\n{image_path}"
                )
            images[filename] = image_path
            relative_paths.append(f"images/{filename}")

        converted["image"] = relative_paths
        portable[name] = converted

    if output.exists():
        output.unlink()

    print("Creating:", output)
    print("Manifests:", ", ".join(manifest_names))
    print("Unique images:", f"{len(images):,}")

    # The images are already JPEG-compressed, so ZIP_STORED is faster.
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as package:
        for name, frame in portable.items():
            package.writestr(
                name,
                frame.to_csv(index=False).encode("utf-8-sig"),
            )

        for audit_name in (
            "page_split_assignments.csv",
            "split_report.json",
            "preparation_report.json",
        ):
            source = manifest_dir / audit_name
            if source.exists():
                package.write(source, arcname=f"audit/{audit_name}")

        for index, (filename, image_path) in enumerate(
            sorted(images.items()),
            start=1,
        ):
            package.write(image_path, arcname=f"images/{filename}")
            if index % 5000 == 0 or index == len(images):
                print(
                    f"\rPacked {index:,}/{len(images):,}",
                    end="",
                    flush=True,
                )

    print()
    print("Kaggle data ZIP ready")
    print("Path:", output)
    print("Size:", f"{output.stat().st_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
