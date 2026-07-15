from __future__ import annotations

from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATASETS = (
    "egyptian_handwriting",
    "gobo",
    "ahla",
    "kstrv2",
    "evarest",
    "saqr",
)

ANNOTATION_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".txt",
    ".xml",
    ".tru",
    ".tsv",
    ".parquet",
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


def inspect_dataset(dataset_name: str) -> list[str]:
    dataset_root = RAW_ROOT / dataset_name
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(f"DATASET: {dataset_name}")
    lines.append(f"PATH: {dataset_root}")

    if not dataset_root.exists():
        lines.append("STATUS: directory not found")
        return lines

    files = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and ".cache" not in path.parts
    ]

    extension_counts = Counter(
        path.suffix.lower() or "<no extension>"
        for path in files
    )

    image_files = [
        path for path in files
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    annotation_files = [
        path for path in files
        if path.suffix.lower() in ANNOTATION_EXTENSIONS
    ]

    lines.append(f"TOTAL FILES: {len(files)}")
    lines.append(f"IMAGE FILES: {len(image_files)}")
    lines.append(
        f"ANNOTATION CANDIDATES: {len(annotation_files)}"
    )

    lines.append("")
    lines.append("FILE EXTENSIONS:")

    for extension, count in extension_counts.most_common(30):
        lines.append(f"  {extension}: {count}")

    lines.append("")
    lines.append("SAMPLE IMAGE PATHS:")

    for path in image_files[:10]:
        lines.append(
            "  " + path.relative_to(dataset_root).as_posix()
        )

    lines.append("")
    lines.append("ANNOTATION FILES:")

    for path in annotation_files[:50]:
        relative = path.relative_to(dataset_root).as_posix()
        size = path.stat().st_size
        lines.append(f"  {relative} ({size:,} bytes)")

    lines.append("")
    lines.append("TOP-LEVEL CONTENT:")

    for path in sorted(dataset_root.iterdir()):
        item_type = "DIR " if path.is_dir() else "FILE"
        lines.append(f"  [{item_type}] {path.name}")

    return lines


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []

    for dataset_name in DATASETS:
        report_lines.extend(
            inspect_dataset(dataset_name)
        )
        report_lines.append("")

    report = "\n".join(report_lines)
    report_path = OUTPUT_DIR / "new_datasets_inventory.txt"

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(report)
    print()
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
