from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "scene_text_annotation_samples.txt"
)

ANNOTATION_FILES = {
    "EvArEST Arabic train": (
        PROJECT_ROOT
        / "data/raw/evarest/train_ar"
        / "Arabic_words_train/gt.txt"
    ),
    "EvArEST Arabic test": (
        PROJECT_ROOT
        / "data/raw/evarest/test_ar"
        / "Arabic_words_test/gt.txt"
    ),
    "EvArEST English train": (
        PROJECT_ROOT
        / "data/raw/evarest/train_en"
        / "English_words/gt.txt"
    ),
    "EvArEST English test": (
        PROJECT_ROOT
        / "data/raw/evarest/test_en"
        / "English_words_test/gt.txt"
    ),
    "KSTRV2 real Arabic train": (
        PROJECT_ROOT
        / "data/raw/kstrv2/extracted/KSTRv2"
        / "recognition/real/arabic/gt_train.txt"
    ),
    "KSTRV2 real English train": (
        PROJECT_ROOT
        / "data/raw/kstrv2/extracted/KSTRv2"
        / "recognition/real/english/gt_train.txt"
    ),
    "KSTRV2 synthetic Arabic train": (
        PROJECT_ROOT
        / "data/raw/kstrv2/extracted/KSTRv2"
        / "recognition/synthetic/arabic/gt_train.txt"
    ),
    "KSTRV2 synthetic English train": (
        PROJECT_ROOT
        / "data/raw/kstrv2/extracted/KSTRv2"
        / "recognition/synthetic/english/gt_train.txt"
    ),
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def read_lines(
    path: Path,
    limit: int = 20,
) -> tuple[str, list[str]]:
    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1256",
        "cp1252",
        "latin-1",
    ):
        try:
            with path.open(
                "r",
                encoding=encoding,
            ) as file:
                lines = []

                for line in file:
                    line = line.rstrip("\r\n")

                    if line.strip():
                        lines.append(line)

                    if len(lines) >= limit:
                        break

            return encoding, lines

        except UnicodeDecodeError:
            continue

    return "unknown", []


def find_sample_images(
    directory: Path,
    limit: int = 10,
) -> list[Path]:
    images = []

    for path in directory.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

            if len(images) >= limit:
                break

    return images


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report: list[str] = []

    for name, annotation_path in ANNOTATION_FILES.items():
        report.append("=" * 80)
        report.append(name)
        report.append(f"PATH: {annotation_path}")
        report.append("=" * 80)

        if not annotation_path.exists():
            report.append("FILE NOT FOUND")
            report.append("")
            continue

        encoding, lines = read_lines(annotation_path)

        report.append(f"Encoding: {encoding}")
        report.append(
            f"File size: {annotation_path.stat().st_size:,} bytes"
        )
        report.append("")
        report.append("ANNOTATION LINES:")

        for index, line in enumerate(lines, start=1):
            report.append(f"{index:02d}: {line}")

        report.append("")
        report.append("SAMPLE IMAGES UNDER SAME DIRECTORY:")

        sample_images = find_sample_images(
            annotation_path.parent
        )

        if not sample_images:
            report.append("No images found under this directory.")
        else:
            for image_path in sample_images:
                relative = image_path.relative_to(
                    annotation_path.parent
                )
                report.append(f"  {relative.as_posix()}")

        report.append("")

    output_text = "\n".join(report)

    OUTPUT_PATH.write_text(
        output_text,
        encoding="utf-8",
    )

    print(output_text)
    print()
    print(f"Saved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
