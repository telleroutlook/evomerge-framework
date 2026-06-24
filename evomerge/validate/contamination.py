"""Contamination check: flag training records whose output n-grams overlap
with evaluation items.

Algorithm: 8-gram Jaccard similarity between training record outputs and
eval items.  Configurable threshold (default 0.2).  Returns a report with
flagged record indices so callers can filter before exporting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


def _ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", text.lower())
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class ContaminationReport:
    threshold: float
    n_training: int
    n_eval: int
    flagged: list[dict] = field(default_factory=list)

    @property
    def n_flagged(self) -> int:
        return len(self.flagged)

    @property
    def flag_rate(self) -> float:
        return self.n_flagged / self.n_training if self.n_training else 0.0


def check_contamination(
    training_texts: Sequence[str],
    eval_texts: Sequence[str],
    *,
    threshold: float = 0.2,
    ngram_size: int = 8,
) -> ContaminationReport:
    """Check training outputs against eval items for n-gram overlap.

    Args:
        training_texts: output texts from training records (e.g. SftTrainingRecord
            messages[-1].content).
        eval_texts: reference eval item texts.
        threshold: Jaccard similarity above which a record is flagged.
        ngram_size: token n-gram size.

    Returns:
        ContaminationReport with indices and scores of flagged records.
    """
    eval_ngrams = [_ngrams(t, ngram_size) for t in eval_texts]
    report = ContaminationReport(
        threshold=threshold,
        n_training=len(training_texts),
        n_eval=len(eval_texts),
    )
    for idx, text in enumerate(training_texts):
        train_ng = _ngrams(text, ngram_size)
        max_score = max(
            (_jaccard(train_ng, eng) for eng in eval_ngrams), default=0.0
        )
        if max_score >= threshold:
            report.flagged.append({"index": idx, "jaccard": round(max_score, 4)})
    return report
