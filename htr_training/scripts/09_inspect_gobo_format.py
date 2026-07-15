from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOBO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gobo"
    / "extracted"
    / "GoBo_v1-0"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "gobo_annotation_samples.txt"
)


def read_lines(path: Path, limit: int = 20) -> list[str]:
    if not path.exists():
        return [f"FILE NOT FOUND: {path}"]

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with path.open(
                "r",
                encoding=encoding,
            ) as file:
                return [
                    line.rstrip("\n\r")
                    for _, line in zip(range(limit), file)
                ]
        except UnicodeDecodeError:
            continue

    return [f"Could not decode: {path}"]


def add_section(
    report: list[str],
    title: str,
    path: Path,
) -> None:
    report.append("=" * 80)
    report.append(title)
    report.append(f"PATH: {path}")
    report.append("=" * 80)
    report.extend(read_lines(path))
    report.append("")


def main() -> None:
    if not GOBO_ROOT.exists():
        raise FileNotFoundError(
            f"GoBo root not found: {GOBO_ROOT}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report: list[str] = []

    add_section(
        report,
        "README",
        GOBO_ROOT / "README.txt",
    )

    writer_root = GOBO_ROOT / "words" / "0"

    for filename in (
        "brown.txt",
        "cedar.txt",
        "domain_A_train.txt",
        "domain_A_test.txt",
        "domain_B_train.txt",
        "domain_B_test.txt",
        "nonwords.txt",
    ):
        add_section(
            report,
            f"WRITER 0: {filename}",
            writer_root / filename,
        )

    report.append("=" * 80)
    report.append("SAMPLE IMAGE FILES")
    report.append("=" * 80)

    for category in (
        "brown",
        "cedar",
        "domain_A_train",
        "domain_A_test",
        "domain_B_train",
        "domain_B_test",
        "nonwords",
    ):
        directory = writer_root / category
        report.append(f"[{category}]")

        if not directory.exists():
            report.append("  directory not found")
            continue

        for image_path in sorted(directory.glob("*.png"))[:10]:
            report.append(f"  {image_path.name}")

    text = "\n".join(report)

    OUTPUT_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print(text)
    print()
    print(f"Saved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
