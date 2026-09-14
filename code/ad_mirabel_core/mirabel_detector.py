"""Canonical and explicitly named legacy query-level MIRABEL baselines."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from .canonical_mirabel import canonical_mirabel_from_full_scores


class MirabelDetector:
    """Official MIRABEL formula evaluated on the complete corpus scores."""

    def __init__(self, confidence: float = 0.95) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = confidence

    def detect_query(
        self, sorted_scores: Sequence[float], sorted_doc_ids: Sequence[str]
    ) -> dict:
        scores = np.asarray(sorted_scores, dtype=float)
        if len(scores) == 0 or len(scores) != len(sorted_doc_ids):
            raise ValueError("Scores and document IDs must be non-empty and aligned.")
        stats = canonical_mirabel_from_full_scores(scores, confidence=self.confidence)
        return {
            "decision": "suspicious" if stats.margin > 0.0 else "benign",
            "top_similarity": stats.top1,
            "top_doc_id": str(sorted_doc_ids[0]),
            "threshold": stats.threshold,
            "mean": stats.background_mean,
            "std": stats.background_std,
            "margin": stats.margin,
            "corpus_size": stats.corpus_size,
            "formula_version": "canonical_official_v2",
        }


class LegacyMirabelDetector:
    """Historical approximation retained solely for frozen-result reproduction."""

    def __init__(self, confidence: float = 0.95) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = confidence

    def detect_query(
        self, sorted_scores: Sequence[float], sorted_doc_ids: Sequence[str]
    ) -> dict:
        scores = np.asarray(sorted_scores, dtype=float)
        if len(scores) == 0 or len(scores) != len(sorted_doc_ids):
            raise ValueError("Scores and document IDs must be non-empty and aligned.")
        top1 = float(scores[0])
        conservative = {
            "decision": "benign", "top_similarity": top1,
            "top_doc_id": str(sorted_doc_ids[0]), "threshold": float("inf"),
            "mean": 0.0, "std": 0.0,
        }
        if len(scores) < 4:
            return conservative
        background = scores[1:]
        mean = float(background.mean())
        std = float(background.std()) + 1e-12
        n = len(background)
        root = math.sqrt(2.0 * math.log(n))
        threshold = mean + std * root + (-math.log(-math.log(self.confidence))) * std / root
        return {
            "decision": "suspicious" if top1 > threshold else "benign",
            "top_similarity": top1, "top_doc_id": str(sorted_doc_ids[0]),
            "threshold": float(threshold), "mean": mean, "std": std,
        }
