"""Single-path order-robust Fisher detectors for canonical retrieval rows.

No analytic chi-square approximation is used. Feature, query-statistic and
session-boundary nulls are estimated from distinct benign roles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import norm

from .canonical_mirabel import require_canonical_cache_row


COMMON_TOP_K = 10
TAILS = ("upper", "two-sided")
OPERATING_POLICIES = ("strict-q5", "strict-q30", "prefix-adaptive")


def empirical_tail_pvalues(
    value: float,
    ordered_reference: np.ndarray,
) -> tuple[float, float, float]:
    """Finite-sample lower, upper and two-sided empirical p-values."""

    reference = np.asarray(ordered_reference, dtype=float)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("ordered_reference must be a nonempty vector")
    if not np.isfinite(reference).all() or np.any(reference[:-1] > reference[1:]):
        raise ValueError("ordered_reference must be finite and sorted")
    if not math.isfinite(float(value)):
        raise ValueError("value must be finite")
    n = len(reference)
    lower_count = int(np.searchsorted(reference, value, side="right"))
    upper_count = n - int(np.searchsorted(reference, value, side="left"))
    p_lower = (lower_count + 1.0) / (n + 1.0)
    p_upper = (upper_count + 1.0) / (n + 1.0)
    p_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))
    return float(p_lower), float(p_upper), float(p_two_sided)


def empirical_upper_pvalue(value: float, ordered_reference: np.ndarray) -> float:
    return empirical_tail_pvalues(value, ordered_reference)[1]


def signed_empirical_normal_score(value: float, ordered_reference: np.ndarray) -> float:
    """Diagnostic rank-normal score that retains low/high direction."""

    reference = np.asarray(ordered_reference, dtype=float)
    if reference.ndim != 1 or not len(reference):
        raise ValueError("ordered_reference must be a nonempty vector")
    rank = int(np.searchsorted(reference, value, side="right"))
    probability = (rank + 0.5) / (len(reference) + 1.0)
    return float(norm.ppf(np.clip(probability, np.finfo(float).eps, 1.0 - np.finfo(float).eps)))


def query_vector(
    row: Mapping[str, object], *, common_top_k: int = COMMON_TOP_K
) -> np.ndarray:
    """Return [D1,...,Dk, canonical margin] without target-aware features."""

    require_canonical_cache_row(row)
    scores = np.asarray(row.get("scores", ()), dtype=float)
    if scores.ndim != 1 or len(scores) < common_top_k:
        raise ValueError(f"canonical row must contain at least top-{common_top_k} scores")
    vector = np.r_[scores[:common_top_k], float(row["mirabel_margin"])]
    if not np.isfinite(vector).all():
        raise ValueError("query representation must be finite")
    return vector.astype(float, copy=False)


def fit_feature_references(
    rows: Sequence[Mapping[str, object]], *, common_top_k: int = COMMON_TOP_K
) -> tuple[np.ndarray, ...]:
    if not rows:
        raise ValueError("feature_calibration rows cannot be empty")
    matrix = np.asarray([query_vector(row, common_top_k=common_top_k) for row in rows])
    return tuple(np.sort(matrix[:, index]) for index in range(matrix.shape[1]))


def query_fisher_statistic(
    row: Mapping[str, object],
    feature_references: Sequence[np.ndarray],
    *,
    tail: str,
    common_top_k: int = COMMON_TOP_K,
) -> float:
    if tail not in TAILS:
        raise KeyError(f"unknown empirical tail: {tail}")
    vector = query_vector(row, common_top_k=common_top_k)
    if len(feature_references) != len(vector):
        raise ValueError("feature references do not match the query vector")
    p_values = []
    for value, reference in zip(vector, feature_references):
        # References are validated/sorted once by ``fit_feature_references``.
        # Revalidating every 30k-element reference for every query dominates
        # real evaluation time without adding a runtime safety property.
        n = len(reference)
        lower_count = int(np.searchsorted(reference, value, side="right"))
        upper_count = n - int(np.searchsorted(reference, value, side="left"))
        upper = (upper_count + 1.0) / (n + 1.0)
        two_sided = min(1.0, 2.0 * min(
            (lower_count + 1.0) / (n + 1.0), upper
        ))
        p_values.append(upper if tail == "upper" else two_sided)
    # Finite-sample empirical p-values are positive, so no clipping is needed.
    return float(-2.0 * np.log(np.asarray(p_values, dtype=float)).sum())


def fit_query_statistic_reference(
    rows: Sequence[Mapping[str, object]],
    feature_references: Sequence[np.ndarray],
    *,
    tail: str,
    common_top_k: int = COMMON_TOP_K,
) -> np.ndarray:
    if not rows:
        raise ValueError("branch_p_reference rows cannot be empty")
    return np.sort(
        np.asarray(
            [
                query_fisher_statistic(
                    row, feature_references, tail=tail, common_top_k=common_top_k
                )
                for row in rows
            ],
            dtype=float,
        )
    )


def query_empirical_pvalue(
    row: Mapping[str, object],
    feature_references: Sequence[np.ndarray],
    query_statistic_reference: np.ndarray,
    *,
    tail: str,
    common_top_k: int = COMMON_TOP_K,
) -> tuple[float, float]:
    statistic = query_fisher_statistic(
        row, feature_references, tail=tail, common_top_k=common_top_k
    )
    return statistic, empirical_upper_pvalue(statistic, query_statistic_reference)


def cumulative_fisher_from_query_pvalues(query_pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(query_pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("query p-values must be a nonempty vector in (0, 1]")
    return np.cumsum(-2.0 * np.log(p))


def threshold_at_fpr(values: Sequence[float], target_fpr: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("threshold values must be a finite nonempty vector")
    if not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr must be strictly between zero and one")
    return float(np.quantile(array, 1.0 - float(target_fpr), method="higher"))


@dataclass(frozen=True)
class FisherCalibration:
    tail: str
    common_top_k: int
    feature_references: tuple[np.ndarray, ...]
    query_statistic_reference: np.ndarray
    strict_thresholds: Mapping[str, Mapping[float, float]]
    prefix_thresholds: Mapping[int, Mapping[float, float]]
    feature_calibration_count: int
    query_reference_count: int
    threshold_session_count: int


def session_query_evidence(
    rows: Sequence[Mapping[str, object]], calibration: FisherCalibration
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = []
    p_values = []
    for row in rows:
        statistic, p_value = query_empirical_pvalue(
            row,
            calibration.feature_references,
            calibration.query_statistic_reference,
            tail=calibration.tail,
            common_top_k=calibration.common_top_k,
        )
        raw.append(statistic)
        p_values.append(p_value)
    p = np.asarray(p_values, dtype=float)
    return np.asarray(raw, dtype=float), p, cumulative_fisher_from_query_pvalues(p)


def fit_fisher_calibration(
    *,
    feature_calibration_rows: Sequence[Mapping[str, object]],
    branch_p_reference_rows: Sequence[Mapping[str, object]],
    threshold_lock_sessions: Sequence[Sequence[Mapping[str, object]]],
    tail: str,
    targets: Sequence[float] = (0.01,),
    strict_horizons: Sequence[int] = (5, 30),
    common_top_k: int = COMMON_TOP_K,
) -> FisherCalibration:
    """Fit the three distinct benign stages required by the protocol."""

    if tail not in TAILS:
        raise KeyError(tail)
    feature_refs = fit_feature_references(
        feature_calibration_rows, common_top_k=common_top_k
    )
    query_ref = fit_query_statistic_reference(
        branch_p_reference_rows,
        feature_refs,
        tail=tail,
        common_top_k=common_top_k,
    )
    curves = []
    for session in threshold_lock_sessions:
        q_values = [
            query_empirical_pvalue(
                row, feature_refs, query_ref, tail=tail, common_top_k=common_top_k
            )[1]
            for row in session
        ]
        curves.append(cumulative_fisher_from_query_pvalues(q_values))
    if not curves:
        raise ValueError("threshold_lock sessions cannot be empty")
    maximum = max(len(curve) for curve in curves)
    prefix_thresholds: dict[int, dict[float, float]] = {}
    for query_count in range(1, maximum + 1):
        values = [curve[query_count - 1] for curve in curves if len(curve) >= query_count]
        prefix_thresholds[query_count] = {
            float(target): threshold_at_fpr(values, float(target)) for target in targets
        }
    strict_thresholds: dict[str, dict[float, float]] = {}
    for horizon in strict_horizons:
        if int(horizon) > maximum:
            continue
        values = [curve[int(horizon) - 1] for curve in curves if len(curve) >= int(horizon)]
        strict_thresholds[f"strict-q{int(horizon)}"] = {
            float(target): threshold_at_fpr(values, float(target)) for target in targets
        }
    return FisherCalibration(
        tail=tail,
        common_top_k=int(common_top_k),
        feature_references=feature_refs,
        query_statistic_reference=query_ref,
        strict_thresholds=strict_thresholds,
        prefix_thresholds=prefix_thresholds,
        feature_calibration_count=len(feature_calibration_rows),
        query_reference_count=len(branch_p_reference_rows),
        threshold_session_count=len(threshold_lock_sessions),
    )


def boundary_for(
    calibration: FisherCalibration,
    *,
    query_count: int,
    operating_policy: str,
    target_fpr: float,
) -> float:
    target = float(target_fpr)
    if operating_policy == "prefix-adaptive":
        return float(calibration.prefix_thresholds[int(query_count)][target])
    if operating_policy not in calibration.strict_thresholds:
        raise KeyError(f"unconfigured operating policy: {operating_policy}")
    horizon = int(operating_policy.rsplit("q", 1)[1])
    if int(query_count) > horizon:
        raise ValueError(f"{operating_policy} does not accept Q{query_count}")
    return float(calibration.strict_thresholds[operating_policy][target])


@dataclass(frozen=True)
class FisherObservation:
    query_count: int
    query_raw_statistic: float
    query_empirical_p: float
    cumulative_fisher: float
    terminal_risk: float | None
    blocked: bool
    first_detection_turn: int | None


def score_rows(
    rows: Sequence[Mapping[str, object]],
    calibration: FisherCalibration,
    *,
    operating_policy: str,
    target_fpr: float,
) -> list[FisherObservation]:
    raw, p_values, cumulative = session_query_evidence(rows, calibration)
    output = []
    blocked = False
    first = None
    for index, (statistic, p_value, score) in enumerate(
        zip(raw, p_values, cumulative), 1
    ):
        boundary = boundary_for(
            calibration,
            query_count=index,
            operating_policy=operating_policy,
            target_fpr=target_fpr,
        )
        if float(score) > boundary and first is None:
            first = index
        blocked = blocked or float(score) > boundary
        output.append(
            FisherObservation(
                query_count=index,
                query_raw_statistic=float(statistic),
                query_empirical_p=float(p_value),
                cumulative_fisher=float(score),
                terminal_risk=float(score),
                blocked=blocked,
                first_detection_turn=first,
            )
        )
    return output


class OrderRobustFisherDetector:
    """In-memory stateful wrapper; a stable linkage key preserves evidence."""

    def __init__(self, calibration: FisherCalibration) -> None:
        self.calibration = calibration
        self._rows: dict[str, list[Mapping[str, object]]] = {}
        self._blocked: dict[tuple[str, str, float], int] = {}

    def observe(
        self,
        linkage_key: str,
        row: Mapping[str, object],
        *,
        operating_policy: str,
        target_fpr: float,
    ) -> FisherObservation:
        key = str(linkage_key)
        history = self._rows.setdefault(key, [])
        history.append(row)
        observations = score_rows(
            history,
            self.calibration,
            operating_policy=operating_policy,
            target_fpr=target_fpr,
        )
        current = observations[-1]
        state_key = (key, operating_policy, float(target_fpr))
        if current.blocked and state_key not in self._blocked:
            self._blocked[state_key] = current.first_detection_turn or current.query_count
        if state_key not in self._blocked:
            return current
        return FisherObservation(
            query_count=current.query_count,
            query_raw_statistic=current.query_raw_statistic,
            query_empirical_p=current.query_empirical_p,
            cumulative_fisher=current.cumulative_fisher,
            terminal_risk=current.terminal_risk,
            blocked=True,
            first_detection_turn=self._blocked[state_key],
        )

    def observe_many(
        self,
        linkage_key: str,
        rows: Sequence[Mapping[str, object]],
        *,
        operating_policy: str,
        target_fpr: float,
    ) -> list[FisherObservation]:
        return [
            self.observe(
                linkage_key,
                row,
                operating_policy=operating_policy,
                target_fpr=target_fpr,
            )
            for row in rows
        ]

    def client_reset(self, linkage_key: str) -> None:
        """A UI reset does not erase server-linked detector state."""

        self._rows.setdefault(str(linkage_key), [])

    def unlinkable_reset(self, new_linkage_key: str) -> None:
        self._rows.setdefault(str(new_linkage_key), [])

    def clear_for_retention_policy(self, linkage_key: str) -> None:
        key = str(linkage_key)
        self._rows.pop(key, None)
        self._blocked = {item: turn for item, turn in self._blocked.items() if item[0] != key}
