from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2


ROOT = Path(
    r"data\raw\online_htr\plhwtr_english\extracted"
)

OUTPUT = Path(
    r"data\processed\online_htr\plhwtr_english"
)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
)

# PLHWTR examples look like:
# 70691 First sentence. 70692 Second sentence. 70693 Third sentence.
#
# We require at least five digits so normal years such as 1979 and 2009
# are not treated as record identifiers.
RECORD_ID_PATTERN = re.compile(
    r"(?<!\d)(\d{5,8})\s+"
)


def read_text(path: Path) -> str:
    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ):
        try:
            return path.read_text(
                encoding=encoding
            ).replace(
                "\r\n",
                "\n",
            ).strip()

        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"Could not decode transcription: {path}"
    )


def normalize_spaces(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def parse_records(
    raw_text: str,
) -> tuple[list[str], list[str], str]:
    """
    Return:
        record IDs
        sentence transcriptions
        parse status
    """

    raw_text = normalize_spaces(raw_text)

    matches = list(
        RECORD_ID_PATTERN.finditer(raw_text)
    )

    if not matches:
        return (
            [],
            [raw_text] if raw_text else [],
            "no_record_ids",
        )

    # A valid PLHWTR record should begin with the first ID.
    prefix = raw_text[:matches[0].start()].strip()

    if prefix:
        return (
            [],
            [raw_text],
            "text_before_first_id",
        )

    record_ids: list[str] = []
    sentences: list[str] = []

    for index, match in enumerate(matches):
        record_id = match.group(1)

        text_start = match.end()

        if index + 1 < len(matches):
            text_end = matches[index + 1].start()
        else:
            text_end = len(raw_text)

        sentence = normalize_spaces(
            raw_text[text_start:text_end]
        )

        if not sentence:
            continue

        record_ids.append(record_id)
        sentences.append(sentence)

    if not sentences:
        return (
            [],
            [],
            "no_sentences",
        )

    return (
        record_ids,
        sentences,
        "parsed",
    )


def find_image(
    text_path: Path,
) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = text_path.with_suffix(
            extension
        )

        if candidate.exists():
            return candidate

    return None


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(
            f"Extracted dataset not found: "
            f"{ROOT.resolve()}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []
    problem_rows: list[dict[str, object]] = []

    text_files = sorted(
        ROOT.rglob("*.txt")
    )

    for index, text_path in enumerate(
        text_files,
        start=1,
    ):
        image_path = find_image(text_path)

        if image_path is None:
            problem_rows.append(
                {
                    "source_text": str(
                        text_path.resolve()
                    ),
                    "image": "",
                    "reason": "matching_image_missing",
                }
            )
            continue

        raw_text = read_text(text_path)

        if not raw_text:
            problem_rows.append(
                {
                    "source_text": str(
                        text_path.resolve()
                    ),
                    "image": str(
                        image_path.resolve()
                    ),
                    "reason": "empty_transcription",
                }
            )
            continue

        record_ids, sentences, status = (
            parse_records(raw_text)
        )

        if not sentences:
            problem_rows.append(
                {
                    "source_text": str(
                        text_path.resolve()
                    ),
                    "image": str(
                        image_path.resolve()
                    ),
                    "reason": status,
                }
            )
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            problem_rows.append(
                {
                    "source_text": str(
                        text_path.resolve()
                    ),
                    "image": str(
                        image_path.resolve()
                    ),
                    "reason": "image_unreadable",
                }
            )
            continue

        height, width = image.shape[:2]

        clean_page_text = normalize_spaces(
            " ".join(sentences)
        )

        rows.append(
            {
                "image": str(
                    image_path.resolve()
                ),
                "text": clean_page_text,
                "text_raw": normalize_spaces(
                    raw_text
                ),
                "language": "en",
                "dataset": "PLHWTR-1.0-English",
                "unit": "page",
                "split": "official_train",
                "width": width,
                "height": height,
                "segment_count": len(sentences),
                "segment_ids_json": json.dumps(
                    record_ids,
                    ensure_ascii=False,
                ),
                "segments_json": json.dumps(
                    sentences,
                    ensure_ascii=False,
                ),
                "parse_status": status,
                "source_text": str(
                    text_path.resolve()
                ),
            }
        )

        if index % 250 == 0 or index == len(
            text_files
        ):
            print(
                f"\rProcessed {index:,}/"
                f"{len(text_files):,} files | "
                f"accepted: {len(rows):,} | "
                f"problems: {len(problem_rows):,}",
                end="",
                flush=True,
            )

    print()

    if not rows:
        raise RuntimeError(
            "No page pairs were prepared."
        )

    manifest_path = OUTPUT / "pages_all.csv"

    fieldnames = [
        "image",
        "text",
        "text_raw",
        "language",
        "dataset",
        "unit",
        "split",
        "width",
        "height",
        "segment_count",
        "segment_ids_json",
        "segments_json",
        "parse_status",
        "source_text",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    problems_path = (
        OUTPUT / "page_preparation_problems.csv"
    )

    with problems_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_text",
                "image",
                "reason",
            ],
        )

        writer.writeheader()
        writer.writerows(problem_rows)

    parsed_count = sum(
        row["parse_status"] == "parsed"
        for row in rows
    )

    fallback_count = len(rows) - parsed_count

    segment_counts = [
        int(row["segment_count"])
        for row in rows
    ]

    report = {
        "total_text_files": len(text_files),
        "accepted_pages": len(rows),
        "problem_pages": len(problem_rows),
        "parsed_id_format": parsed_count,
        "fallback_format": fallback_count,
        "total_segments": sum(segment_counts),
        "minimum_segments_per_page": min(
            segment_counts
        ),
        "maximum_segments_per_page": max(
            segment_counts
        ),
        "average_segments_per_page": (
            sum(segment_counts)
            / len(segment_counts)
        ),
    }

    report_path = (
        OUTPUT / "preparation_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("PLHWTR page preparation complete")
    print(
        "Accepted pages:",
        f"{len(rows):,}",
    )
    print(
        "Parsed ID format:",
        f"{parsed_count:,}",
    )
    print(
        "Fallback format:",
        f"{fallback_count:,}",
    )
    print(
        "Problem pages:",
        f"{len(problem_rows):,}",
    )
    print(
        "Total sentence segments:",
        f"{sum(segment_counts):,}",
    )
    print(
        "Manifest:",
        manifest_path.resolve(),
    )
    print(
        "Report:",
        report_path.resolve(),
    )


if __name__ == "__main__":
    main()
