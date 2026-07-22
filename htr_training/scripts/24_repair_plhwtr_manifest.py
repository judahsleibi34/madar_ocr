from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


INPUT = Path(
    r"data\processed\online_htr"
    r"\plhwtr_english\pages_all.csv"
)

OUTPUT = Path(
    r"data\processed\online_htr"
    r"\plhwtr_english\pages_clean.csv"
)

AUDIT = Path(
    r"data\processed\online_htr"
    r"\plhwtr_english\repair_audit.csv"
)

REPORT = Path(
    r"data\processed\online_htr"
    r"\plhwtr_english\repair_report.json"
)


# Main record IDs found in this dataset are commonly 4–8 digits.
#
# A token is considered a record ID only when it:
# 1. appears at the start of the transcription, or
# 2. follows likely sentence-ending punctuation, and
# 3. is followed by likely sentence text.
ID_TOKEN = re.compile(
    r"(?<!\d)(\d{4,8})\s+"
    r"(?=[A-Z\"'“‘(])"
)

# Some pages begin with a shorter ID, such as:
# 36 She was tall at 5 feet 9 inches.
SHORT_LEADING_ID = re.compile(
    r"^(\d{1,3})\s+"
    r"(?=[A-Z\"'“‘(])"
)

VALID_PREVIOUS_CHARACTERS = set(
    ".!?;:)]}\"'”’"
)

RESIDUAL_BOUNDARY_ID = re.compile(
    r"(?:^|[.!?]\s+)"
    r"\d{4,8}\s+"
    r"(?=[A-Z\"'“‘(])"
)

STARTING_NUMBER = re.compile(
    r"^\d{1,8}\s+"
)


def normalize(text: object) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def find_record_ids(
    text: str,
) -> list[re.Match[str]]:
    accepted: list[re.Match[str]] = []

    for match in ID_TOKEN.finditer(text):
        if match.start() == 0:
            accepted.append(match)
            continue

        preceding = text[:match.start()].rstrip()

        if (
            preceding
            and preceding[-1]
            in VALID_PREVIOUS_CHARACTERS
        ):
            accepted.append(match)

    return accepted


def parse_transcription(
    raw_text: str,
) -> tuple[
    list[str],
    list[str],
    str,
]:
    text = normalize(raw_text)
    matches = find_record_ids(text)

    segments: list[str] = []
    record_ids: list[str] = []
    status_parts: list[str] = []

    if not matches:
        short_match = SHORT_LEADING_ID.match(text)

        if short_match:
            remaining = normalize(
                text[short_match.end():]
            )

            if remaining:
                segments.append(remaining)
                record_ids.append(
                    short_match.group(1)
                )
                status_parts.append(
                    "short_leading_id_removed"
                )
        else:
            segments.append(text)
            record_ids.append("")
            status_parts.append("no_ids_found")

        return (
            record_ids,
            segments,
            "+".join(status_parts),
        )

    prefix = normalize(
        text[:matches[0].start()]
    )

    if prefix:
        short_match = SHORT_LEADING_ID.match(
            prefix
        )

        if short_match:
            prefix_text = normalize(
                prefix[short_match.end():]
            )

            if prefix_text:
                segments.append(prefix_text)
                record_ids.append(
                    short_match.group(1)
                )

            status_parts.append(
                "leading_short_id_removed"
            )
        else:
            segments.append(prefix)
            record_ids.append("")
            status_parts.append(
                "unnumbered_prefix_preserved"
            )

    for index, match in enumerate(matches):
        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        segment = normalize(
            text[start:end]
        )

        if not segment:
            continue

        segments.append(segment)
        record_ids.append(
            match.group(1)
        )

    status_parts.append("record_ids_removed")

    return (
        record_ids,
        segments,
        "+".join(status_parts),
    )


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT.resolve())

    data = pd.read_csv(INPUT)

    required = {
        "image",
        "text_raw",
    }

    missing = required - set(data.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    repaired_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for index, row in data.iterrows():
        raw_text = normalize(row["text_raw"])

        record_ids, segments, status = (
            parse_transcription(raw_text)
        )

        clean_text = normalize(
            " ".join(segments)
        )

        residual_boundary = bool(
            RESIDUAL_BOUNDARY_ID.search(
                clean_text
            )
        )

        starts_with_number = bool(
            STARTING_NUMBER.match(
                clean_text
            )
        )

        repaired = row.to_dict()

        repaired["text"] = clean_text
        repaired["segment_count"] = len(
            segments
        )
        repaired["segment_ids_json"] = (
            json.dumps(
                record_ids,
                ensure_ascii=False,
            )
        )
        repaired["segments_json"] = (
            json.dumps(
                segments,
                ensure_ascii=False,
            )
        )
        repaired["parse_status"] = status
        repaired["residual_boundary_id"] = (
            residual_boundary
        )
        repaired["starts_with_number"] = (
            starts_with_number
        )

        repaired_rows.append(repaired)

        if (
            residual_boundary
            or starts_with_number
            or not clean_text
        ):
            audit_rows.append(
                {
                    "row": index,
                    "image": row["image"],
                    "parse_status": status,
                    "residual_boundary_id":
                        residual_boundary,
                    "starts_with_number":
                        starts_with_number,
                    "text": clean_text,
                    "text_raw": raw_text,
                }
            )

        if (
            (index + 1) % 500 == 0
            or index + 1 == len(data)
        ):
            print(
                f"\rRepaired {index + 1:,}/"
                f"{len(data):,} pages | "
                f"audit rows: {len(audit_rows):,}",
                end="",
                flush=True,
            )

    print()

    repaired_data = pd.DataFrame(
        repaired_rows
    )

    repaired_data.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    audit_data = pd.DataFrame(
        audit_rows,
        columns=[
            "row",
            "image",
            "parse_status",
            "residual_boundary_id",
            "starts_with_number",
            "text",
            "text_raw",
        ],
    )

    audit_data.to_csv(
        AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "input_pages": len(data),
        "output_pages": len(repaired_data),
        "audit_rows": len(audit_data),
        "empty_targets": int(
            repaired_data["text"]
            .fillna("")
            .eq("")
            .sum()
        ),
        "residual_boundary_ids": int(
            repaired_data[
                "residual_boundary_id"
            ].sum()
        ),
        "targets_starting_with_number": int(
            repaired_data[
                "starts_with_number"
            ].sum()
        ),
        "status_counts": {
            str(key): int(value)
            for key, value in (
                repaired_data[
                    "parse_status"
                ].value_counts()
                .to_dict()
                .items()
            )
        },
        "total_segments": int(
            repaired_data[
                "segment_count"
            ].sum()
        ),
    }

    REPORT.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Repair complete")
    print("Pages:", f"{len(repaired_data):,}")
    print(
        "Total segments:",
        f"{report['total_segments']:,}",
    )
    print(
        "Residual boundary IDs:",
        report["residual_boundary_ids"],
    )
    print(
        "Targets starting with numbers:",
        report[
            "targets_starting_with_number"
        ],
    )
    print(
        "Rows requiring audit:",
        report["audit_rows"],
    )
    print("Clean manifest:", OUTPUT.resolve())
    print("Audit:", AUDIT.resolve())
    print("Report:", REPORT.resolve())


if __name__ == "__main__":
    main()
