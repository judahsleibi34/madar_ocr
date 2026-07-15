from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "egyptian_handwriting"
    / "train"
)

NORMALIZED_PATH = TRAIN_DIR / "labels_normalized.csv"
CLEAN_PATH = TRAIN_DIR / "labels_clean.csv"
SAFE_CLEAN_PATH = TRAIN_DIR / "labels_clean_safe.csv"
CONFLICT_PATH = TRAIN_DIR / "duplicate_conflicts.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    normalized_rows = read_csv(NORMALIZED_PATH)
    clean_rows = read_csv(CLEAN_PATH)

    conflict_hashes = {
        row["image_sha256"]
        for row in normalized_rows
        if "duplicate_image_conflicting_label"
        in row.get("review_reason", "")
    }

    conflict_rows = [
        row
        for row in normalized_rows
        if row["image_sha256"] in conflict_hashes
    ]

    safe_clean_rows = [
        row
        for row in clean_rows
        if row["image_sha256"] not in conflict_hashes
    ]

    normalized_fields = list(normalized_rows[0].keys())
    clean_fields = list(clean_rows[0].keys())

    write_csv(
        CONFLICT_PATH,
        conflict_rows,
        normalized_fields,
    )

    write_csv(
        SAFE_CLEAN_PATH,
        safe_clean_rows,
        clean_fields,
    )

    print(f"Conflicting hashes: {len(conflict_hashes)}")
    print(f"Conflict rows: {len(conflict_rows)}")
    print(f"Original clean rows: {len(clean_rows)}")
    print(f"Safe clean rows: {len(safe_clean_rows)}")
    print(f"Conflict report: {CONFLICT_PATH}")
    print(f"Training manifest: {SAFE_CLEAN_PATH}")


if __name__ == "__main__":
    main()
