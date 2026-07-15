from __future__ import annotations

from pathlib import Path
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_manifests_inventory.json"
)


def inspect_csv(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "rows": 0,
        "columns": [],
        "missing_image_paths": 0,
        "empty_text_rows": 0,
        "sample_rows": [],
    }

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        result["columns"] = reader.fieldnames or []

        rows = []

        for index, row in enumerate(reader):
            result["rows"] += 1

            text = str(
                row.get(
                    "text",
                    row.get(
                        "label",
                        row.get("transcription", ""),
                    ),
                )
            ).strip()

            if not text:
                result["empty_text_rows"] += 1

            image_value = (
                row.get("image_path")
                or row.get("path")
                or row.get("image")
                or row.get("file")
                or row.get("filename")
            )

            if image_value:
                image_path = Path(image_value)

                if not image_path.is_absolute():
                    image_path = PROJECT_ROOT / image_path

                if not image_path.exists():
                    result["missing_image_paths"] += 1

            if index < 3:
                rows.append(row)

        result["sample_rows"] = rows

    return result


def main() -> None:
    if not PROCESSED_ROOT.exists():
        raise FileNotFoundError(
            f"Processed directory not found: {PROCESSED_ROOT}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifests = []

    for path in sorted(PROCESSED_ROOT.rglob("*.csv")):
        # Skip reports that are not training manifests.
        if path.name in {
            "conversion_errors.csv",
            "duplicate_conflicts.csv",
            "qc_suspicious.csv",
        }:
            continue

        manifests.append(inspect_csv(path))

    report = {
        "processed_root": str(PROCESSED_ROOT),
        "manifest_count": len(manifests),
        "manifests": manifests,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Found {len(manifests)} CSV manifests.")
    print()

    for manifest in manifests:
        print("=" * 80)
        print(manifest["path"])
        print(f"Rows: {manifest['rows']}")
        print(
            "Columns: "
            + ", ".join(manifest["columns"])
        )
        print(
            "Missing image paths: "
            f"{manifest['missing_image_paths']}"
        )
        print(
            "Empty text rows: "
            f"{manifest['empty_text_rows']}"
        )

    print()
    print(f"Saved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
