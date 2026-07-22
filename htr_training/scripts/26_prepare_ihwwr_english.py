from __future__ import annotations

import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd


ARCHIVE = Path(
    r"data\raw\online_htr\ihwwr_english"
    r"\downloads\Word_Level_English_Training_Set.zip"
)

EXTRACTED = Path(
    r"data\raw\online_htr\ihwwr_english"
    r"\extracted"
)

OUTPUT = Path(
    r"data\processed\online_htr"
    r"\ihwwr_english"
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def safe_extract(
    archive: Path,
    destination: Path,
) -> None:
    marker = destination / ".extract_complete"

    if marker.exists():
        print(
            "Using existing extraction:",
            destination.resolve(),
        )
        return

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    root = destination.resolve()

    with zipfile.ZipFile(archive) as package:
        members = package.infolist()

        for index, member in enumerate(
            members,
            start=1,
        ):
            target = (
                destination / member.filename
            ).resolve()

            if (
                target != root
                and root not in target.parents
            ):
                raise RuntimeError(
                    f"Unsafe ZIP path: "
                    f"{member.filename}"
                )

            package.extract(
                member,
                destination,
            )

            if (
                index % 1000 == 0
                or index == len(members)
            ):
                print(
                    f"\rExtracted "
                    f"{index:,}/{len(members):,}",
                    end="",
                    flush=True,
                )

    print()

    marker.write_text(
        "complete\n",
        encoding="utf-8",
    )

    print(
        "Extracted to:",
        destination.resolve(),
    )


def read_manifest(
    path: Path,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    for raw_line in path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    ).splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if "\t" in line:
            image_value, label = line.split(
                "\t",
                1,
            )
        else:
            parts = line.split(
                maxsplit=1,
            )

            if len(parts) != 2:
                continue

            image_value, label = parts

        image_value = (
            image_value
            .strip()
            .strip('"')
            .replace("\\", "/")
        )

        # Preserve capitalization, punctuation and spelling.
        label = label.strip()

        if image_value and label:
            rows.append(
                (
                    image_value,
                    label,
                )
            )

    return rows


def create_image_indexes(
    root: Path,
) -> tuple[
    dict[str, Path],
    dict[str, list[Path]],
]:
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}

    for image_path in root.rglob("*"):
        if (
            not image_path.is_file()
            or image_path.suffix.lower()
            not in IMAGE_SUFFIXES
        ):
            continue

        relative = image_path.relative_to(
            root
        ).as_posix().lower()

        by_relative[relative] = image_path

        by_name.setdefault(
            image_path.name.lower(),
            [],
        ).append(image_path)

    return by_relative, by_name


def resolve_image(
    image_value: str,
    manifest_path: Path,
    root: Path,
    by_relative: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> Path | None:
    normalized = (
        image_value
        .replace("\\", "/")
        .lstrip("./")
    )

    direct_candidates = [
        manifest_path.parent / normalized,
        root / normalized,
    ]

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    normalized_lower = normalized.lower()

    for relative, image_path in by_relative.items():
        if (
            relative == normalized_lower
            or relative.endswith(
                "/" + normalized_lower
            )
        ):
            return image_path.resolve()

    name_matches = by_name.get(
        Path(normalized).name.lower(),
        [],
    )

    if len(name_matches) == 1:
        return name_matches[0].resolve()

    return None


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Archive missing: "
            f"{ARCHIVE.resolve()}"
        )

    if not zipfile.is_zipfile(ARCHIVE):
        raise RuntimeError(
            f"Invalid ZIP archive: "
            f"{ARCHIVE.resolve()}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_extract(
        ARCHIVE,
        EXTRACTED,
    )

    manifests = sorted(
        EXTRACTED.rglob("train.txt")
    )

    if not manifests:
        manifests = sorted(
            path
            for path in EXTRACTED.rglob("*.txt")
            if "train" in path.name.lower()
        )

    if not manifests:
        raise RuntimeError(
            "No train.txt annotation file found."
        )

    print("Annotation files:")

    for manifest in manifests:
        print("-", manifest.resolve())

    by_relative, by_name = (
        create_image_indexes(EXTRACTED)
    )

    prepared_rows: list[dict[str, object]] = []
    problem_rows: list[dict[str, object]] = []

    seen_images: dict[str, str] = {}

    for manifest_path in manifests:
        annotation_rows = read_manifest(
            manifest_path
        )

        print(
            f"Rows in {manifest_path.name}: "
            f"{len(annotation_rows):,}"
        )

        for image_value, label in annotation_rows:
            image_path = resolve_image(
                image_value,
                manifest_path,
                EXTRACTED,
                by_relative,
                by_name,
            )

            if image_path is None:
                problem_rows.append(
                    {
                        "source_manifest":
                            str(
                                manifest_path.resolve()
                            ),
                        "image_value":
                            image_value,
                        "label":
                            label,
                        "reason":
                            "image_not_found",
                    }
                )
                continue

            image_key = str(
                image_path.resolve()
            ).lower()

            previous_label = seen_images.get(
                image_key
            )

            if (
                previous_label is not None
                and previous_label != label
            ):
                problem_rows.append(
                    {
                        "source_manifest":
                            str(
                                manifest_path.resolve()
                            ),
                        "image_value":
                            image_value,
                        "label":
                            label,
                        "reason":
                            "conflicting_duplicate_label",
                    }
                )
                continue

            if previous_label is not None:
                continue

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                problem_rows.append(
                    {
                        "source_manifest":
                            str(
                                manifest_path.resolve()
                            ),
                        "image_value":
                            image_value,
                        "label":
                            label,
                        "reason":
                            "image_unreadable",
                    }
                )
                continue

            height, width = image.shape[:2]

            seen_images[image_key] = label

            prepared_rows.append(
                {
                    "image":
                        str(image_path.resolve()),
                    "text":
                        label,
                    "language":
                        "en",
                    "dataset":
                        "IHWWR-1.0-English",
                    "unit":
                        "word",
                    "split":
                        "official_train",
                    "width":
                        width,
                    "height":
                        height,
                    "source_manifest":
                        str(
                            manifest_path.resolve()
                        ),
                    "source_image_value":
                        image_value,
                    "image_name":
                        image_path.name,
                    "image_stem":
                        image_path.stem,
                    "parent_directory":
                        image_path.parent.name,
                }
            )

            count = len(prepared_rows)

            if count % 5000 == 0:
                print(
                    f"Prepared "
                    f"{count:,} word images"
                )

    if not prepared_rows:
        raise RuntimeError(
            "No valid word samples were prepared."
        )

    prepared = pd.DataFrame(
        prepared_rows
    )

    problems = pd.DataFrame(
        problem_rows,
        columns=[
            "source_manifest",
            "image_value",
            "label",
            "reason",
        ],
    )

    manifest_output = (
        OUTPUT / "words_all.csv"
    )

    problems_output = (
        OUTPUT / "preparation_problems.csv"
    )

    prepared.to_csv(
        manifest_output,
        index=False,
        encoding="utf-8-sig",
    )

    problems.to_csv(
        problems_output,
        index=False,
        encoding="utf-8-sig",
    )

    character_counts = Counter(
        character
        for text in prepared["text"]
        for character in str(text)
    )

    report = {
        "prepared_words": int(
            len(prepared)
        ),
        "problem_rows": int(
            len(problems)
        ),
        "unique_labels": int(
            prepared["text"].nunique()
        ),
        "unique_characters": int(
            len(character_counts)
        ),
        "minimum_width": int(
            prepared["width"].min()
        ),
        "maximum_width": int(
            prepared["width"].max()
        ),
        "minimum_height": int(
            prepared["height"].min()
        ),
        "maximum_height": int(
            prepared["height"].max()
        ),
        "duplicate_image_rows_removed": int(
            sum(
                1
                for reason in problems.get(
                    "reason",
                    []
                )
                if reason
                == "conflicting_duplicate_label"
            )
        ),
        "character_inventory": "".join(
            sorted(character_counts)
        ),
    }

    report_output = (
        OUTPUT / "preparation_report.json"
    )

    report_output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("IHWWR English preparation complete")
    print(
        "Prepared words:",
        f"{len(prepared):,}",
    )
    print(
        "Problem rows:",
        f"{len(problems):,}",
    )
    print(
        "Unique labels:",
        f"{prepared['text'].nunique():,}",
    )
    print(
        "Manifest:",
        manifest_output.resolve(),
    )
    print(
        "Report:",
        report_output.resolve(),
    )
    print()
    print(
        "No train/validation/test split was "
        "created yet."
    )
    print(
        "We must inspect filenames/directories "
        "before creating a leakage-safe split."
    )


if __name__ == "__main__":
    main()
