from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_safe"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "master_validation"
)

MANIFESTS = {
    "train": MASTER_ROOT / "train_all.csv",
    "val": MASTER_ROOT / "val_all.csv",
    "test": MASTER_ROOT / "test_all.csv",
}

REQUIRED_COLUMNS = {
    "image",
    "text",
    "language",
    "dataset",
    "split",
    "domain",
    "source_type",
    "source_manifest",
    "source_row",
    "sha256",
}


def resolve_image_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

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
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames: list[pd.DataFrame] = []
    issues: list[dict[str, object]] = []
    split_summary: dict[str, dict[str, int]] = {}

    for manifest_split, path in MANIFESTS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing manifest: {path}"
            )

        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

        missing_columns = (
            REQUIRED_COLUMNS - set(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{path.name} missing columns: "
                f"{sorted(missing_columns)}"
            )

        frame = frame.copy()
        frame["manifest_split"] = manifest_split

        empty_image_count = 0
        empty_text_count = 0
        missing_file_count = 0
        split_mismatch_count = 0

        for index, row in frame.iterrows():
            csv_row = index + 2

            image_value = str(row["image"]).strip()
            text_value = str(row["text"]).strip()
            declared_split = str(row["split"]).strip()

            base_issue = {
                "manifest": path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "manifest_split": manifest_split,
                "csv_row": csv_row,
                "dataset": row["dataset"],
                "image": image_value,
                "text": text_value,
                "source_manifest": row[
                    "source_manifest"
                ],
                "source_row": row["source_row"],
                "sha256": row["sha256"],
            }

            if not image_value:
                empty_image_count += 1
                issues.append(
                    {
                        **base_issue,
                        "issue": "empty_image_path",
                    }
                )

            if not text_value:
                empty_text_count += 1
                issues.append(
                    {
                        **base_issue,
                        "issue": "empty_text",
                    }
                )

            if image_value:
                image_path = resolve_image_path(
                    image_value
                )

                if not image_path.exists():
                    missing_file_count += 1
                    issues.append(
                        {
                            **base_issue,
                            "issue": "image_file_missing",
                        }
                    )

            if declared_split != manifest_split:
                split_mismatch_count += 1
                issues.append(
                    {
                        **base_issue,
                        "issue": (
                            "split_mismatch:"
                            f"{declared_split}"
                        ),
                    }
                )

        split_summary[manifest_split] = {
            "rows": len(frame),
            "empty_images": empty_image_count,
            "empty_texts": empty_text_count,
            "missing_files": missing_file_count,
            "split_mismatches": split_mismatch_count,
        }

        frames.append(frame)

        print(
            f"{manifest_split}: "
            f"rows={len(frame)}, "
            f"empty_images={empty_image_count}, "
            f"empty_texts={empty_text_count}, "
            f"missing_files={missing_file_count}, "
            f"split_mismatches={split_mismatch_count}"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    hash_split_counts = (
        combined.groupby("sha256")[
            "manifest_split"
        ].nunique()
    )

    cross_split_hashes = set(
        hash_split_counts[
            hash_split_counts > 1
        ].index
    )

    hash_text_counts = (
        combined.groupby("sha256")[
            "text"
        ].nunique()
    )

    conflicting_hashes = set(
        hash_text_counts[
            hash_text_counts > 1
        ].index
    )

    same_split_groups: dict[
        tuple[str, str],
        list[int],
    ] = defaultdict(list)

    for index, row in combined.iterrows():
        same_split_groups[
            (
                str(row["manifest_split"]),
                str(row["sha256"]),
            )
        ].append(index)

    same_split_duplicate_groups = {
        key: indexes
        for key, indexes in same_split_groups.items()
        if len(indexes) > 1
    }

    for sha256 in sorted(cross_split_hashes):
        rows = combined[
            combined["sha256"] == sha256
        ]

        issues.append(
            {
                "issue": "cross_split_duplicate",
                "sha256": sha256,
                "manifest_split": "|".join(
                    sorted(
                        set(rows["manifest_split"])
                    )
                ),
                "dataset": "|".join(
                    sorted(set(rows["dataset"]))
                ),
                "image": "|".join(
                    rows["image"].tolist()
                ),
                "text": "|".join(
                    sorted(set(rows["text"]))
                ),
            }
        )

    for sha256 in sorted(conflicting_hashes):
        rows = combined[
            combined["sha256"] == sha256
        ]

        issues.append(
            {
                "issue": "conflicting_labels",
                "sha256": sha256,
                "manifest_split": "|".join(
                    sorted(
                        set(rows["manifest_split"])
                    )
                ),
                "dataset": "|".join(
                    sorted(set(rows["dataset"]))
                ),
                "image": "|".join(
                    rows["image"].tolist()
                ),
                "text": "|".join(
                    sorted(set(rows["text"]))
                ),
            }
        )

    for (
        manifest_split,
        sha256,
    ), indexes in same_split_duplicate_groups.items():
        rows = combined.loc[indexes]

        issues.append(
            {
                "issue": "same_split_duplicate",
                "sha256": sha256,
                "manifest_split": manifest_split,
                "dataset": "|".join(
                    sorted(set(rows["dataset"]))
                ),
                "image": "|".join(
                    rows["image"].tolist()
                ),
                "text": "|".join(
                    sorted(set(rows["text"]))
                ),
            }
        )

    issue_path = (
        OUTPUT_ROOT
        / "master_manifest_issues.csv"
    )

    save_csv(issue_path, issues)

    summary = {
        "total_rows": len(combined),
        "splits": split_summary,
        "issue_rows": len(issues),
        "cross_split_duplicate_hashes": len(
            cross_split_hashes
        ),
        "conflicting_label_hashes": len(
            conflicting_hashes
        ),
        "same_split_duplicate_groups": len(
            same_split_duplicate_groups
        ),
        "passed": len(issues) == 0,
    }

    summary_path = (
        OUTPUT_ROOT
        / "master_validation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Total rows: {len(combined)}")
    print(
        "Cross-split duplicate hashes: "
        f"{len(cross_split_hashes)}"
    )
    print(
        "Conflicting-label hashes: "
        f"{len(conflicting_hashes)}"
    )
    print(
        "Same-split duplicate groups: "
        f"{len(same_split_duplicate_groups)}"
    )
    print(f"Total issues: {len(issues)}")
    print(f"Issue report: {issue_path}")
    print(f"Summary: {summary_path}")

    if issues:
        raise RuntimeError(
            "MASTER MANIFEST VALIDATION FAILED. "
            "Review master_manifest_issues.csv."
        )

    print()
    print("MASTER MANIFEST VALIDATION PASSED")


if __name__ == "__main__":
    main()
