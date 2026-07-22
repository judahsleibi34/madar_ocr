from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests


DATASET_URL = (
    "https://ilocr.iiit.ac.in/icdar_2024_hwd/assets/dataset/"
    "PLHWTR-1.0/Training-Set/Page_Level_English_Training_Set.zip"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def download_file(url: str, destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        print(f"Using existing ZIP: {destination}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    resume_from = partial.stat().st_size if partial.exists() and not force else 0
    headers = {"User-Agent": "Mozilla/5.0 PLHWTR downloader"}

    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    with requests.get(
        url,
        stream=True,
        timeout=(30, 300),
        allow_redirects=True,
        headers=headers,
    ) as response:
        response.raise_for_status()

        append = resume_from > 0 and response.status_code == 206
        if not append:
            resume_from = 0

        content_length = response.headers.get("Content-Length")
        total = int(content_length) + resume_from if content_length else None
        mode = "ab" if append else "wb"
        downloaded = resume_from
        started = time.time()

        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                handle.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.time() - started, 0.001)
                speed = downloaded / elapsed / (1024**2)

                if total:
                    percent = downloaded * 100.0 / total
                    message = (
                        f"\rDownloaded {downloaded / 1024**2:,.1f}/"
                        f"{total / 1024**2:,.1f} MB "
                        f"({percent:5.1f}%) at {speed:,.1f} MB/s"
                    )
                else:
                    message = (
                        f"\rDownloaded {downloaded / 1024**2:,.1f} MB "
                        f"at {speed:,.1f} MB/s"
                    )

                print(message, end="", flush=True)

    print()
    partial.replace(destination)
    print(f"Saved ZIP: {destination}")


def safe_extract(archive: Path, destination: Path, force: bool = False) -> None:
    marker = destination / ".extract_complete"

    if marker.exists() and not force:
        print(f"Using existing extraction: {destination}")
        return

    if force and destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()

        for index, member in enumerate(members, start=1):
            target = (destination / member.filename).resolve()

            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")

            handle.extract(member, destination)

            if index % 500 == 0 or index == len(members):
                print(
                    f"\rExtracted {index:,}/{len(members):,}",
                    end="",
                    flush=True,
                )

    print()
    marker.write_text("ok\n", encoding="utf-8")
    print(f"Extracted to: {destination}")


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"Could not decode: {path}")


def split_name(key: str) -> str:
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100

    if value < 80:
        return "train"
    if value < 90:
        return "val"
    return "test"


def find_image_text_pairs(root: Path) -> list[tuple[Path, Path]]:
    text_by_stem: dict[str, list[Path]] = {}

    for text_path in root.rglob("*.txt"):
        text_by_stem.setdefault(text_path.stem.lower(), []).append(text_path)

    pairs: list[tuple[Path, Path]] = []

    for image_path in root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        candidates = text_by_stem.get(image_path.stem.lower(), [])
        same_directory = [path for path in candidates if path.parent == image_path.parent]

        if same_directory:
            pairs.append((image_path, same_directory[0]))
        elif len(candidates) == 1:
            pairs.append((image_path, candidates[0]))

    return sorted(pairs, key=lambda item: str(item[0]))


def remove_long_horizontal_rules(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    kernel_width = max(30, width // 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(mask, horizontal)


def crop_page_lines(
    image_path: Path,
    expected_lines: int,
    output_directory: Path,
) -> tuple[list[Path], str]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        return [], "image_read_failed"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    mask = remove_long_horizontal_rules(mask)

    height, width = mask.shape
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[tuple[int, int, int, int, float]] = []
    minimum_area = max(8, int(height * width * 0.000002))

    for component_index in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[component_index]]

        if area < minimum_area or w < 2 or h < 3:
            continue

        if w > width * 0.75 and h < max(5, height * 0.015):
            continue

        components.append((x, y, w, h, float(centroids[component_index][1])))

    if expected_lines <= 0:
        return [], "no_text_lines"

    if len(components) < expected_lines:
        return [], f"too_few_components:{len(components)}<{expected_lines}"

    y_points = np.array([[component[4]] for component in components], dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.1,
    )

    _, labels, centers = cv2.kmeans(
        y_points,
        expected_lines,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    clusters: list[dict[str, float]] = []

    for cluster_index in range(expected_lines):
        members = [
            components[index]
            for index, label in enumerate(labels.ravel())
            if int(label) == cluster_index
        ]

        if not members:
            return [], "empty_cluster"

        clusters.append(
            {
                "x1": float(min(item[0] for item in members)),
                "y1": float(min(item[1] for item in members)),
                "x2": float(max(item[0] + item[2] for item in members)),
                "y2": float(max(item[1] + item[3] for item in members)),
                "center": float(centers[cluster_index][0]),
            }
        )

    clusters.sort(key=lambda item: item["center"])

    for previous, current in zip(clusters, clusters[1:]):
        if current["center"] <= previous["center"]:
            return [], "invalid_reading_order"

    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for line_number, cluster in enumerate(clusters, start=1):
        pad_x = max(8, int(width * 0.01))
        pad_y = max(5, int(height * 0.008))

        x1 = max(0, int(cluster["x1"]) - pad_x)
        y1 = max(0, int(cluster["y1"]) - pad_y)
        x2 = min(width, int(cluster["x2"]) + pad_x)
        y2 = min(height, int(cluster["y2"]) + pad_y)

        crop_height = y2 - y1
        crop_width = x2 - x1

        if crop_height < 8 or crop_width < 20:
            return [], "crop_too_small"

        if crop_height > height * 0.45:
            return [], "crop_too_tall"

        crop = image[y1:y2, x1:x2]
        destination = output_directory / f"{image_path.stem}_line_{line_number:03d}.png"

        if not cv2.imwrite(str(destination), crop):
            return [], "write_failed"

        written.append(destination)

    return written, "ok"


def prepare_dataset(
    extracted_directory: Path,
    processed_directory: Path,
    max_pages: int,
) -> None:
    pairs = find_image_text_pairs(extracted_directory)

    if not pairs:
        raise RuntimeError(
            "No matching image/text pairs were found after extraction. "
            "Inspect the extracted folder structure."
        )

    if max_pages > 0:
        pairs = pairs[:max_pages]

    page_rows: list[dict[str, str]] = []
    line_rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, str | int]] = []
    lines_root = processed_directory / "line_images"

    for index, (image_path, text_path) in enumerate(pairs, start=1):
        raw_text = read_text(text_path)
        text_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        split = split_name(image_path.stem)

        page_rows.append(
            {
                "image": str(image_path.resolve()),
                "text": "\n".join(text_lines),
                "language": "en",
                "dataset": "PLHWTR-1.0-English",
                "split": split,
                "source_page": str(image_path.resolve()),
                "text_file": str(text_path.resolve()),
            }
        )

        crop_paths, status = crop_page_lines(
            image_path,
            len(text_lines),
            lines_root / split,
        )

        if status != "ok" or len(crop_paths) != len(text_lines):
            for crop_path in crop_paths:
                crop_path.unlink(missing_ok=True)

            rejected_rows.append(
                {
                    "image": str(image_path.resolve()),
                    "text_file": str(text_path.resolve()),
                    "expected_lines": len(text_lines),
                    "created_lines": len(crop_paths),
                    "reason": status,
                }
            )
        else:
            for crop_path, label in zip(crop_paths, text_lines):
                line_rows.append(
                    {
                        "image": str(crop_path.resolve()),
                        "text": label,
                        "language": "en",
                        "dataset": "PLHWTR-1.0-English",
                        "split": split,
                        "source_page": str(image_path.resolve()),
                        "crop_type": "automatic_line_crop",
                    }
                )

        if index % 25 == 0 or index == len(pairs):
            print(
                f"\rPrepared {index:,}/{len(pairs):,} pages | "
                f"accepted lines: {len(line_rows):,} | "
                f"rejected pages: {len(rejected_rows):,}",
                end="",
                flush=True,
            )

    print()
    processed_directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(page_rows).to_csv(
        processed_directory / "pages.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(line_rows).to_csv(
        processed_directory / "lines.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(rejected_rows).to_csv(
        processed_directory / "rejected_pages.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Pages manifest: {processed_directory / 'pages.csv'}")
    print(f"Lines manifest: {processed_directory / 'lines.csv'}")
    print(f"Rejected pages: {processed_directory / 'rejected_pages.csv'}")
    print(f"Prepared pages: {len(page_rows):,}")
    print(f"Prepared lines: {len(line_rows):,}")
    print(f"Rejected pages: {len(rejected_rows):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the PLHWTR-1.0 English training set."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/online_htr/plhwtr_english"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/online_htr/plhwtr_english"),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Prepare only the first N pages. Zero means all pages.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--delete-zip",
        action="store_true",
        help="Delete the downloaded ZIP after successful preparation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    archive = raw_root / "downloads" / "Page_Level_English_Training_Set.zip"
    extracted = raw_root / "extracted"

    print("PLHWTR-1.0 English downloader/preparer")
    print(f"Raw root: {raw_root}")
    print(f"Processed root: {processed_root}")
    print()

    download_file(DATASET_URL, archive, force=args.force_download)
    safe_extract(archive, extracted, force=args.force_extract)
    prepare_dataset(extracted, processed_root, max_pages=args.max_pages)

    if args.delete_zip:
        archive.unlink(missing_ok=True)
        print(f"Deleted ZIP: {archive}")


if __name__ == "__main__":
    main()
