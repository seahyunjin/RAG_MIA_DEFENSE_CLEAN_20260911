"""Evidence-accumulating Mirabel detector for one-to-four-query sessions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
import math

import numpy as np

from .mirabel_detector import LegacyMirabelDetector


def stable_softmax(values: Sequence[float], temperature: float = 0.08) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    scaled = (array - np.max(array)) / max(float(temperature), 1e-6)
    weights = np.exp(np.clip(scaled, -60.0, 0.0))
    return weights / max(float(weights.sum()), 1e-12)


@dataclass(frozen=True)
class EarlyMirabelConfig:
    max_queries: int = 4
    top_k: int = 5
    temperature: float = 0.08
    tail_threshold: float = 0.80
    thresholds_by_length: dict[int, float] = field(
        default_factory=lambda: {1: 0.76, 2: 0.70, 3: 0.66, 4: 0.62}
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "top1": 0.28,
            "gap": 0.15,
            "softmax_concentration": 0.12,
            "gumbel_margin": 0.12,
            "tail_fraction": 0.18,
            "doc_concentration": 0.15,
        }
    )


def per_query_retrieval_features(
    sorted_scores: Sequence[float],
    sorted_doc_ids: Sequence[str],
    *,
    top_k: int = 5,
    temperature: float = 0.08,
) -> dict[str, float | str]:
    if not sorted_scores or not sorted_doc_ids:
        raise ValueError("At least one score and document ID are required.")
    usable = min(len(sorted_scores), len(sorted_doc_ids))
    scores = np.asarray(sorted_scores[:usable], dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be finite.")
    top1 = float(scores[0])
    top2 = float(scores[1]) if scores.size > 1 else top1
    probabilities = stable_softmax(scores[:top_k], temperature)
    return {
        "top1": top1,
        "top2": top2,
        "gap": max(0.0, top1 - top2),
        "softmax_concentration": float(probabilities[0]) if probabilities.size else 1.0,
        "top_doc_id": str(sorted_doc_ids[0]),
    }


class EarlyMirabelDetector:
    """Accumulate weak retrieval evidence before AD-Mirabel has enough samples."""

    def __init__(self, config: EarlyMirabelConfig | None = None) -> None:
        self.config = config or EarlyMirabelConfig()
        # This historical detector is part of the frozen V16 reproduction
        # path. New experiments consume versioned canonical cache margins.
        self.mirabel = LegacyMirabelDetector()
        self.sessions: dict[str, list[dict[str, float | str]]] = defaultdict(list)

    @staticmethod
    def _clip_feature(name: str, value: float) -> float:
        if name == "gap":
            return float(np.clip(value / 0.35, 0.0, 1.0))
        if name == "gumbel_margin":
            return float(np.clip(0.5 + value / 0.30, 0.0, 1.0))
        return float(np.clip(value, 0.0, 1.0))

    def update(
        self,
        session_id: str,
        query: str,
        sorted_scores: Sequence[float],
        sorted_doc_ids: Sequence[str],
    ) -> dict:
        if len(self.sessions[session_id]) >= self.config.max_queries:
            raise ValueError("EarlyMirabelDetector only supports prefixes of length 1-4.")
        features = per_query_retrieval_features(
            sorted_scores,
            sorted_doc_ids,
            top_k=self.config.top_k,
            temperature=self.config.temperature,
        )
        mirabel = self.mirabel.detect_query(sorted_scores, sorted_doc_ids)
        threshold = float(mirabel["threshold"])
        features["gumbel_margin"] = (
            float(features["top1"]) - threshold if math.isfinite(threshold) else 0.0
        )
        features["query"] = query or ""
        self.sessions[session_id].append(features)

        history = self.sessions[session_id]
        top_scores = np.asarray([float(item["top1"]) for item in history], dtype=float)
        doc_counts = Counter(str(item["top_doc_id"]) for item in history)
        doc_concentration = doc_counts.most_common(1)[0][1] / len(history)
        tail_fraction = float(np.mean(top_scores > self.config.tail_threshold))
        aggregate = {
            "top1": float(np.mean(top_scores)),
            "gap": float(np.mean([float(item["gap"]) for item in history])),
            "softmax_concentration": float(
                np.mean([float(item["softmax_concentration"]) for item in history])
            ),
            "gumbel_margin": float(
                np.mean([float(item["gumbel_margin"]) for item in history])
            ),
            "tail_fraction": tail_fraction,
            "doc_concentration": float(doc_concentration),
        }
        weight_total = sum(max(0.0, value) for value in self.config.weights.values())
        risk = sum(
            max(0.0, self.config.weights.get(name, 0.0))
            * self._clip_feature(name, value)
            for name, value in aggregate.items()
        ) / max(weight_total, 1e-12)
        length = len(history)
        threshold_by_length = self.config.thresholds_by_length.get(
            length, self.config.thresholds_by_length[self.config.max_queries]
        )
        decision = "suspicious" if risk >= threshold_by_length else "benign"
        return {
            "decision": decision,
            "base_risk": float(np.clip(risk, 0.0, 1.0)),
            "risk_threshold": float(threshold_by_length),
            "num_queries": length,
            "detector_stage": "early",
            "tail_mass": tail_fraction,
            "max_doc_concentration": float(doc_concentration),
            "latest_top_similarity": float(features["top1"]),
            "latest_top_doc_id": str(features["top_doc_id"]),
            "gumbel_margin": float(aggregate["gumbel_margin"]),
            "top1_top2_gap": float(aggregate["gap"]),
            "softmax_concentration": float(aggregate["softmax_concentration"]),
            "session_id": session_id,
        }

    def get_session_buffer(self, session_id: str) -> list[dict[str, float | str]]:
        return [dict(item) for item in self.sessions[session_id]]
