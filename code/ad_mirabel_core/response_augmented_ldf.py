"""Response-augmented learned direction with calibrated Adaptive Fisher.

The scorer is one learned path over retrieval and semantic response features.
There are no query-budget, ordering, dataset, or attack-family branches.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import numpy as np

from .detector_calibration import HorizonCalibration, adaptive_score, empirical_risk, empirical_upper_pvalue
from .order_robust_fisher_detector import query_vector


@dataclass(frozen=True)
class ResponseQueryModel:
    response_feature_dimension: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    query_score_reference: np.ndarray

    def __post_init__(self) -> None:
        dimension = 11 + int(self.response_feature_dimension)
        for value in (self.scaler_mean, self.scaler_scale, self.coefficients):
            if np.asarray(value).shape != (dimension,):
                raise ValueError("response query-model dimension mismatch")
        reference = np.asarray(self.query_score_reference, dtype=float)
        if reference.ndim != 1 or not len(reference) or np.any(reference[:-1] > reference[1:]):
            raise ValueError("query_score_reference must be nonempty and sorted")

    def vector(self, retrieval_row: Mapping[str, object], response_feature: Sequence[float]) -> np.ndarray:
        response = np.asarray(response_feature, dtype=float)
        if response.shape != (self.response_feature_dimension,):
            raise ValueError("response feature dimension mismatch")
        vector = np.r_[query_vector(retrieval_row), response]
        if not np.isfinite(vector).all():
            raise ValueError("combined feature vector must be finite")
        return vector

    def raw_score(self, retrieval_row: Mapping[str, object], response_feature: Sequence[float]) -> float:
        vector = self.vector(retrieval_row, response_feature)
        scale = np.where(np.asarray(self.scaler_scale) == 0.0, 1.0, self.scaler_scale)
        return float(((vector - self.scaler_mean) / scale) @ self.coefficients + self.intercept)

    def p_value(self, retrieval_row: Mapping[str, object], response_feature: Sequence[float]) -> float:
        return empirical_upper_pvalue(self.raw_score(retrieval_row, response_feature), self.query_score_reference)


@dataclass(frozen=True)
class ResponseAugmentedCalibration:
    query_model: ResponseQueryModel
    horizons: Mapping[int, HorizonCalibration]
    retriever_id: str
    retrieval_model_version: str
    response_model_version: str
    calibration_version: str
    threshold_version: str
    retriever_schema_version: str

    def for_horizon(self, horizon: int) -> HorizonCalibration:
        try:
            return self.horizons[int(horizon)]
        except KeyError as error:
            raise ValueError(f"unsupported horizon: {horizon}") from error


def score_response_set(
    rows: Sequence[tuple[Mapping[str, object], Sequence[float]]],
    calibration: ResponseAugmentedCalibration,
    *,
    horizon: int,
) -> dict[str, float | int | bool]:
    if not rows or len(rows) > int(horizon):
        raise ValueError("query-response set must be nonempty and no longer than its horizon")
    h = calibration.for_horizon(horizon)
    pvalues = [calibration.query_model.p_value(retrieval, response) for retrieval, response in rows]
    selected, statistic, adaptive = adaptive_score(pvalues, h.partial_fisher_references)
    risk = empirical_risk(adaptive, h.adaptive_score_reference)
    return {
        "selected_k": int(selected), "partial_fisher_statistic": float(statistic),
        "adaptive_score": float(adaptive), "calibrated_risk": float(risk),
        "score": float(risk), "threshold": float(h.threshold),
        "blocked": bool(risk > h.threshold),
    }


def mask_future_responses(candidates: Sequence[Mapping[str, object]], submitted_query_ids: Sequence[str]) -> list[dict[str, object]]:
    """Return an attacker view in which unsubmitted future responses are absent."""

    allowed = {str(value) for value in submitted_query_ids}
    output = []
    for candidate in candidates:
        row = dict(candidate)
        if str(row.get("query_id")) not in allowed:
            for name in ("response_text", "response_feature", "response_text_hash", "generation_latency"):
                row.pop(name, None)
        output.append(row)
    return output


def sequential_response_attack(
    candidates: Sequence[Mapping[str, object]],
    *,
    budget: int,
    choose_next: Callable[[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]], str],
    submit: Callable[[str], Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Realizable adaptive loop: a response becomes visible only after submit."""

    remaining = {str(row["query_id"]): dict(row) for row in candidates}
    prefix: list[Mapping[str, object]] = []
    while remaining and len(prefix) < int(budget):
        masked = mask_future_responses(list(remaining.values()), [str(row["query_id"]) for row in prefix])
        selected = str(choose_next(masked, prefix))
        if selected not in remaining:
            raise ValueError("attacker selected an unavailable query")
        observed = dict(submit(selected))
        if str(observed.get("query_id")) != selected:
            raise ValueError("submitted query and observed response are misaligned")
        prefix.append(observed)
        remaining.pop(selected)
    return prefix


def paired_bootstrap_difference(left: Sequence[bool], right: Sequence[bool], *, seed: int, draws: int = 5000) -> tuple[float, float, float]:
    a = np.asarray(left, dtype=float); b = np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired outcomes must be aligned nonempty vectors")
    difference = a - b; rng = np.random.default_rng(seed)
    samples = difference[rng.integers(0, len(difference), size=(int(draws), len(difference)))].mean(1)
    return float(difference.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))
