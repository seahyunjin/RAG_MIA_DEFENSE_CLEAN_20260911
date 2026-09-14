"""Sequential Early-to-CCT detector built from cached retrieval features.

This module is deliberately separate from the original Mirabel, Early-Mirabel,
and Experiment 22 code.  It contains no model or API calls.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import atan, pi, tan
from typing import Iterable, Mapping, Sequence

import numpy as np


FEATURES = (
    "top1_similarity",
    "top1_top2_gap",
    "softmax_concentration",
    "gumbel_margin",
    "high_score_fraction",
    "same_document_concentration",
)

HEURISTIC_WEIGHTS = {
    "top1_similarity": 0.28,
    "top1_top2_gap": 0.15,
    "softmax_concentration": 0.12,
    "gumbel_margin": 0.12,
    "high_score_fraction": 0.18,
    "same_document_concentration": 0.15,
}


def aggregate_prefix(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Reproduce the six conceptual Early-Mirabel prefix features."""
    if not rows:
        raise ValueError("A prefix needs at least one query")
    top1 = np.asarray([float(row["top1_similarity"]) for row in rows], dtype=float)
    gaps = np.asarray([float(row["top1_top2_gap"]) for row in rows], dtype=float)
    concentration = np.asarray(
        [float(row["softmax_concentration"]) for row in rows], dtype=float
    )
    margins = np.asarray([float(row["gumbel_margin"]) for row in rows], dtype=float)
    docs = [str(row["top1_document_id"]) for row in rows]
    if not all(np.all(np.isfinite(values)) for values in (top1, gaps, concentration, margins)):
        raise ValueError("Early features must be finite")
    return {
        "top1_similarity": float(top1.mean()),
        "top1_top2_gap": float(gaps.mean()),
        "softmax_concentration": float(concentration.mean()),
        "gumbel_margin": float(margins.mean()),
        "high_score_fraction": float(np.mean(top1 > 0.80)),
        "same_document_concentration": float(Counter(docs).most_common(1)[0][1] / len(docs)),
    }


def current_clip(name: str, value: float) -> float:
    """Exact conceptual mapping used by the current Early-Mirabel code."""
    if name == "top1_top2_gap":
        return float(np.clip(value / 0.35, 0.0, 1.0))
    if name == "gumbel_margin":
        return float(np.clip(0.5 + value / 0.30, 0.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class NormalizationSpec:
    method: str
    feature: str
    center: float = 0.0
    scale: float = 1.0
    reference: tuple[float, ...] = ()

    def transform(self, value: float) -> float:
        if self.method == "current_clip":
            return current_clip(self.feature, value)
        if self.method in {"zscore", "robust_mad"}:
            return float((value - self.center) / max(self.scale, 1e-12))
        if self.method == "empirical_percentile":
            ordered = np.asarray(self.reference, dtype=float)
            if not len(ordered):
                raise ValueError("Empirical normalization needs a reference")
            return float(np.searchsorted(ordered, value, side="right") / (len(ordered) + 1.0))
        raise KeyError(self.method)


def fit_normalization(
    benign_rows: Sequence[Mapping[str, float]], method: str
) -> dict[str, NormalizationSpec]:
    if method not in {"current_clip", "zscore", "empirical_percentile", "robust_mad"}:
        raise KeyError(method)
    output: dict[str, NormalizationSpec] = {}
    for feature in FEATURES:
        values = np.asarray([float(row[feature]) for row in benign_rows], dtype=float)
        if not len(values):
            raise ValueError("Benign calibration is empty")
        if method == "current_clip":
            output[feature] = NormalizationSpec(method, feature)
        elif method == "zscore":
            output[feature] = NormalizationSpec(
                method, feature, float(values.mean()), float(max(values.std(ddof=0), 1e-8))
            )
        elif method == "robust_mad":
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            output[feature] = NormalizationSpec(
                method, feature, median, max(1.4826 * mad, 1e-8)
            )
        else:
            output[feature] = NormalizationSpec(
                method, feature, reference=tuple(float(value) for value in np.sort(values))
            )
    return output


def normalized_vector(
    row: Mapping[str, float], specs: Mapping[str, NormalizationSpec], features: Sequence[str]
) -> np.ndarray:
    return np.asarray([specs[name].transform(float(row[name])) for name in features], dtype=float)


def renormalize_weights(features: Sequence[str], weights: Mapping[str, float]) -> dict[str, float]:
    selected = {name: max(0.0, float(weights.get(name, 0.0))) for name in features}
    total = sum(selected.values())
    if total <= 0:
        raise ValueError("At least one selected feature needs positive weight")
    return {name: value / total for name, value in selected.items()}


def early_score(
    row: Mapping[str, float],
    specs: Mapping[str, NormalizationSpec],
    weights: Mapping[str, float],
) -> float:
    features = tuple(weights)
    vector = normalized_vector(row, specs, features)
    w = np.asarray([float(weights[name]) for name in features], dtype=float)
    return float(vector @ w)


def empirical_upper_p(value: float, reference: Sequence[float]) -> float:
    ordered = np.sort(np.asarray(reference, dtype=float))
    if not len(ordered):
        raise ValueError("p-value reference is empty")
    ge = len(ordered) - int(np.searchsorted(ordered, float(value), side="left"))
    return float((ge + 1.0) / (len(ordered) + 1.0))


def cct_risk(p_values: Iterable[float]) -> float:
    values = np.clip(np.asarray(list(p_values), dtype=float), 1e-12, 1.0 - 1e-12)
    if not len(values):
        raise ValueError("CCT needs at least one p-value")
    statistic = float(np.mean(np.tan((0.5 - values) * np.pi)))
    p_value = 0.5 - atan(statistic) / pi
    return float(-np.log10(max(p_value, 1e-15)))


def mahalanobis_distance(vector: Sequence[float], location: Sequence[float], precision: Sequence[Sequence[float]]) -> float:
    delta = np.asarray(vector, dtype=float) - np.asarray(location, dtype=float)
    matrix = np.asarray(precision, dtype=float)
    return float(delta @ matrix @ delta)


@dataclass(frozen=True)
class SequentialArtifacts:
    features: tuple[str, ...]
    weights: dict[str, float]
    normalization_by_prefix: dict[int, dict[str, NormalizationSpec]]
    mirabel_threshold: float
    early_thresholds: dict[int, float]
    cct_threshold: float
    ordered_location: tuple[float, ...]
    ordered_precision: tuple[tuple[float, ...], ...]
    calibration_ordered_distances: tuple[float, ...]
    calibration_early4_scores: tuple[float, ...]
    max_early_prefix: int = 4
    rolling_window: int = 5


class SequentialEarlyCCTDetector:
    """Query-by-query Mirabel + selected Early + rolling five-query CCT."""

    def __init__(self, artifacts: SequentialArtifacts) -> None:
        if artifacts.rolling_window != 5:
            raise ValueError("The primary rolling policy requires a five-query window")
        self.artifacts = artifacts
        self.rows: list[dict[str, object]] = []
        self.blocked = False
        self.first_detection_turn: int | None = None
        self.first_detection_branch: str | None = None

    def _score_prefix(self, rows: Sequence[Mapping[str, object]], prefix: int) -> float:
        aggregate = aggregate_prefix(rows)
        return early_score(
            aggregate,
            self.artifacts.normalization_by_prefix[prefix],
            self.artifacts.weights,
        )

    def update(self, query_features: Mapping[str, object]) -> dict[str, object]:
        if self.blocked:
            self.rows.append(dict(query_features))
            return {
                "blocked": True,
                "sticky_block": True,
                "turn": len(self.rows),
                "branches": (),
                "first_detection_turn": self.first_detection_turn,
                "first_detection_branch": self.first_detection_branch,
            }

        self.rows.append(dict(query_features))
        turn = len(self.rows)
        branches: list[str] = []
        margin = float(query_features["gumbel_margin"])
        if margin > self.artifacts.mirabel_threshold:
            branches.append("Mirabel")

        early_value = None
        if turn <= self.artifacts.max_early_prefix:
            early_value = self._score_prefix(self.rows, turn)
            if early_value > self.artifacts.early_thresholds[turn]:
                branches.append("Early")

        cct_value = None
        if turn >= self.artifacts.rolling_window:
            window = self.rows[-self.artifacts.rolling_window :]
            ordered = [float(row["top1_similarity"]) for row in window]
            distance = mahalanobis_distance(
                ordered,
                self.artifacts.ordered_location,
                self.artifacts.ordered_precision,
            )
            p_m = empirical_upper_p(distance, self.artifacts.calibration_ordered_distances)
            early4 = self._score_prefix(window[:4], 4)
            p_e = empirical_upper_p(early4, self.artifacts.calibration_early4_scores)
            cct_value = cct_risk((p_m, p_e))
            if cct_value > self.artifacts.cct_threshold:
                branches.append("CCT")

        if branches:
            self.blocked = True
            self.first_detection_turn = turn
            self.first_detection_branch = branches[0] if len(branches) == 1 else "multiple"
        return {
            "blocked": self.blocked,
            "sticky_block": False,
            "turn": turn,
            "branches": tuple(branches),
            "early_score": early_value,
            "cct_score": cct_value,
            "first_detection_turn": self.first_detection_turn,
            "first_detection_branch": self.first_detection_branch,
        }


def cumulative_session_fpr(decisions: Sequence[Mapping[str, object]], horizon: int) -> float:
    """Fraction of benign sessions blocked at least once by ``horizon``."""
    by_session: dict[str, bool] = {}
    for row in decisions:
        if str(row.get("label")) != "benign" or int(row.get("turn", 0)) > horizon:
            continue
        session_id = str(row["session_id"])
        by_session[session_id] = by_session.get(session_id, False) or bool(row["blocked"])
    return float(np.mean(list(by_session.values()))) if by_session else float("nan")
