from __future__ import annotations

from collections import defaultdict
from typing import Any

import jiwer


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    valid_pairs = [
        (r, h) for r, h in zip(references, hypotheses) if r.strip()
    ]
    if not valid_pairs:
        return 0.0

    refs, hyps = zip(*valid_pairs)
    return jiwer.cer(list(refs), list(hyps))


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    valid_pairs = [
        (r, h) for r, h in zip(references, hypotheses) if r.strip()
    ]
    if not valid_pairs:
        return 0.0

    refs, hyps = zip(*valid_pairs)
    return jiwer.wer(list(refs), list(hyps))


def compute_metrics_by_language(
    references: list[str],
    hypotheses: list[str],
    languages: list[str],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"refs": [], "hyps": []}
    )

    for ref, hyp, lang in zip(references, hypotheses, languages):
        grouped[lang]["refs"].append(ref)
        grouped[lang]["hyps"].append(hyp)

    results = {}
    for lang, data in grouped.items():
        results[lang] = {
            "cer": compute_cer(data["refs"], data["hyps"]),
            "wer": compute_wer(data["refs"], data["hyps"]),
            "count": len(data["refs"]),
        }

    return results
