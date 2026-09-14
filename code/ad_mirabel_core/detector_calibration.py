"""Calibration primitives for label-free and learned-direction Adaptive Fisher.

Every empirical calibration stage is represented separately so callers cannot
silently reuse the same benign sessions for feature, query, partial-Fisher,
adaptive-maximum, and threshold calibration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from .order_robust_fisher_detector import COMMON_TOP_K, query_vector, threshold_at_fpr


def _ordered(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.sort(np.asarray(values, dtype=float))
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    return array


def empirical_upper_pvalue(value: float, ordered_reference: np.ndarray) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("empirical p-value input must be finite")
    reference = np.asarray(ordered_reference, dtype=float)
    count = len(reference) - int(np.searchsorted(reference, value, side="left"))
    return float((count + 1.0) / (len(reference) + 1.0))


def empirical_two_sided_pvalue(value: float, ordered_reference: np.ndarray) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("empirical p-value input must be finite")
    reference = np.asarray(ordered_reference, dtype=float)
    lower = (int(np.searchsorted(reference, value, side="right")) + 1.0) / (len(reference) + 1.0)
    upper = empirical_upper_pvalue(value, reference)
    return float(min(1.0, 2.0 * min(lower, upper)))


def empirical_risk(value: float, ordered_reference: np.ndarray) -> float:
    """High-score risk in [1/(n+1), 1], preserving the score ordering."""

    reference = np.asarray(ordered_reference, dtype=float)
    rank = int(np.searchsorted(reference, float(value), side="right"))
    return float((rank + 1.0) / (len(reference) + 1.0))


@dataclass(frozen=True)
class QueryEvidenceModel:
    kind: str
    common_top_k: int
    feature_references: tuple[np.ndarray, ...]
    query_score_reference: np.ndarray
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    intercept: float | None = None
    logistic_c: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"uf_two_sided", "ldf_upper"}:
            raise ValueError(f"unsupported query model: {self.kind}")
        _ordered(self.query_score_reference, name="query_score_reference")
        if self.kind == "uf_two_sided" and len(self.feature_references) != self.common_top_k + 1:
            raise ValueError("UF feature references do not match query schema")
        if self.kind == "ldf_upper":
            arrays = (self.scaler_mean, self.scaler_scale, self.coefficients)
            if any(value is None for value in arrays) or self.intercept is None:
                raise ValueError("LDF parameters are incomplete")
            if any(len(np.asarray(value)) != self.common_top_k + 1 for value in arrays):
                raise ValueError("LDF parameters do not match query schema")

    def raw_score(self, row: Mapping[str, object]) -> float:
        vector = query_vector(row, common_top_k=self.common_top_k)
        if self.kind == "uf_two_sided":
            p_values = [
                empirical_two_sided_pvalue(value, reference)
                for value, reference in zip(vector, self.feature_references)
            ]
            return float(-2.0 * np.log(np.asarray(p_values, dtype=float)).sum())
        mean = np.asarray(self.scaler_mean, dtype=float)
        scale = np.asarray(self.scaler_scale, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        standardized = (vector - mean) / np.where(scale == 0.0, 1.0, scale)
        return float(standardized @ coefficients + float(self.intercept))

    def p_value(self, row: Mapping[str, object]) -> float:
        return empirical_upper_pvalue(self.raw_score(row), self.query_score_reference)


@dataclass(frozen=True)
class HorizonCalibration:
    horizon: int
    partial_fisher_references: tuple[np.ndarray, ...]
    adaptive_score_reference: np.ndarray
    threshold: float
    target_fpr: float
    partial_reference_sessions: int
    adaptive_reference_sessions: int
    threshold_sessions: int

    def __post_init__(self) -> None:
        if self.horizon < 1 or len(self.partial_fisher_references) != self.horizon:
            raise ValueError("partial Fisher references must cover k=1..horizon")
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0,1)")
        for index, reference in enumerate(self.partial_fisher_references, 1):
            _ordered(reference, name=f"partial_fisher_reference_k{index}")
        _ordered(self.adaptive_score_reference, name="adaptive_score_reference")


@dataclass(frozen=True)
class AdaptiveFisherCalibration:
    query_model: QueryEvidenceModel
    horizons: Mapping[int, HorizonCalibration]
    model_version: str
    threshold_version: str
    feature_schema_version: str
    retriever_id: str
    retriever_schema_version: str
    mirabel_formula_version: str

    def for_horizon(self, horizon: int) -> HorizonCalibration:
        try:
            return self.horizons[int(horizon)]
        except KeyError as error:
            raise ValueError(f"unsupported horizon: {horizon}") from error


def fit_uf_query_model(
    feature_calibration_rows: Sequence[Mapping[str, object]],
    query_score_reference_rows: Sequence[Mapping[str, object]],
    *,
    common_top_k: int = COMMON_TOP_K,
) -> QueryEvidenceModel:
    feature_matrix = np.asarray(
        [query_vector(row, common_top_k=common_top_k) for row in feature_calibration_rows],
        dtype=float,
    )
    if feature_matrix.ndim != 2 or not len(feature_matrix):
        raise ValueError("feature calibration rows cannot be empty")
    references = tuple(np.sort(feature_matrix[:, index]) for index in range(feature_matrix.shape[1]))

    def statistic(row: Mapping[str, object]) -> float:
        vector = query_vector(row, common_top_k=common_top_k)
        values = [
            empirical_two_sided_pvalue(value, reference)
            for value, reference in zip(vector, references)
        ]
        return float(-2.0 * np.log(np.asarray(values)).sum())

    query_reference = _ordered(
        [statistic(row) for row in query_score_reference_rows],
        name="UF query-score reference",
    )
    return QueryEvidenceModel(
        kind="uf_two_sided",
        common_top_k=int(common_top_k),
        feature_references=references,
        query_score_reference=query_reference,
    )


def query_model_from_sklearn_pipeline(
    pipeline: object,
    query_score_reference_rows: Sequence[Mapping[str, object]],
    *,
    common_top_k: int = COMMON_TOP_K,
    logistic_c: float | None = None,
) -> QueryEvidenceModel:
    scaler = pipeline[0]
    logistic = pipeline[-1]
    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    coefficients = np.asarray(logistic.coef_[0], dtype=float)
    intercept = float(logistic.intercept_[0])

    def score(row: Mapping[str, object]) -> float:
        vector = query_vector(row, common_top_k=common_top_k)
        return float(((vector - mean) / np.where(scale == 0.0, 1.0, scale)) @ coefficients + intercept)

    reference = _ordered(
        [score(row) for row in query_score_reference_rows],
        name="LDF query-score reference",
    )
    return QueryEvidenceModel(
        kind="ldf_upper",
        common_top_k=int(common_top_k),
        feature_references=(),
        query_score_reference=reference,
        scaler_mean=mean,
        scaler_scale=scale,
        coefficients=coefficients,
        intercept=intercept,
        logistic_c=logistic_c,
    )


def partial_fisher_statistics(query_p_values: Sequence[float]) -> np.ndarray:
    values = np.sort(np.asarray(query_p_values, dtype=float))
    if values.ndim != 1 or not len(values) or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("query p-values must be a nonempty vector in (0,1]")
    return np.cumsum(-2.0 * np.log(values))


def adaptive_score(
    query_p_values: Sequence[float],
    partial_references: Sequence[np.ndarray],
) -> tuple[int, float, float]:
    partial = partial_fisher_statistics(query_p_values)
    if len(partial) > len(partial_references):
        raise ValueError("query count exceeds calibrated horizon")
    transformed = np.asarray(
        [
            -math.log(empirical_upper_pvalue(value, partial_references[index]))
            for index, value in enumerate(partial)
        ],
        dtype=float,
    )
    selected = int(np.argmax(transformed))
    return selected + 1, float(partial[selected]), float(transformed[selected])


def fit_horizon_calibration(
    query_model: QueryEvidenceModel,
    *,
    horizon: int,
    partial_fisher_reference_sessions: Sequence[Sequence[Mapping[str, object]]],
    adaptive_score_reference_sessions: Sequence[Sequence[Mapping[str, object]]],
    threshold_lock_sessions: Sequence[Sequence[Mapping[str, object]]],
    target_fpr: float = 0.01,
) -> HorizonCalibration:
    horizon = int(horizon)

    def qvalues(session: Sequence[Mapping[str, object]]) -> list[float]:
        if len(session) < horizon:
            raise ValueError(f"calibration session shorter than horizon {horizon}")
        return [query_model.p_value(row) for row in session[:horizon]]

    partial_matrix = np.asarray(
        [partial_fisher_statistics(qvalues(session)) for session in partial_fisher_reference_sessions],
        dtype=float,
    )
    if partial_matrix.ndim != 2 or partial_matrix.shape[1] != horizon:
        raise ValueError("invalid partial-Fisher calibration matrix")
    partial_references = tuple(np.sort(partial_matrix[:, index]) for index in range(horizon))
    adaptive_reference = _ordered(
        [adaptive_score(qvalues(session), partial_references)[2] for session in adaptive_score_reference_sessions],
        name="adaptive score reference",
    )
    threshold_risks = [
        empirical_risk(adaptive_score(qvalues(session), partial_references)[2], adaptive_reference)
        for session in threshold_lock_sessions
    ]
    threshold = threshold_at_fpr(threshold_risks, float(target_fpr))
    return HorizonCalibration(
        horizon=horizon,
        partial_fisher_references=partial_references,
        adaptive_score_reference=adaptive_reference,
        threshold=float(threshold),
        target_fpr=float(target_fpr),
        partial_reference_sessions=len(partial_fisher_reference_sessions),
        adaptive_reference_sessions=len(adaptive_score_reference_sessions),
        threshold_sessions=len(threshold_lock_sessions),
    )


def score_query_set(
    rows: Sequence[Mapping[str, object]],
    calibration: AdaptiveFisherCalibration,
    *,
    horizon: int,
) -> dict[str, float | int | bool]:
    horizon_calibration = calibration.for_horizon(horizon)
    if not rows or len(rows) > horizon:
        raise ValueError("query set must be nonempty and no longer than the horizon")
    query_p_values = [calibration.query_model.p_value(row) for row in rows]
    selected_k, statistic, raw_adaptive = adaptive_score(
        query_p_values, horizon_calibration.partial_fisher_references
    )
    risk = empirical_risk(raw_adaptive, horizon_calibration.adaptive_score_reference)
    return {
        "query_p_value": float(query_p_values[-1]),
        "selected_k": int(selected_k),
        "partial_fisher_statistic": float(statistic),
        "adaptive_score": float(raw_adaptive),
        "calibrated_risk": float(risk),
        "score": float(risk),
        "threshold": float(horizon_calibration.threshold),
        "blocked": bool(risk > horizon_calibration.threshold),
    }
