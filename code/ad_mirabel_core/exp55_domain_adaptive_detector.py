"""Pure detector primitives for Exp55 DA-AOMD.

The deployed scorer is attack-family and query-budget agnostic.  A deployment
object may contain a benign reference fitted for one RAG service, but scoring
does not accept domain, retriever, family, membership, turn, or budget fields.
No network, API, or filesystem access occurs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np


EARLY6_FEATURES = (
    "mean_top1_similarity",
    "mean_top1_top2_gap",
    "mean_standardized_retrieval_concentration",
    "mean_gumbel_margin",
    "max_top1_similarity",
    "same_document_concentration",
)

FORBIDDEN_INFERENCE_FIELDS = frozenset(
    {
        "attack_family", "attack_method", "family_id", "domain", "domain_id",
        "corpus", "retriever", "retriever_id", "member_label", "membership",
        "query_budget", "native_budget", "remaining_queries", "turn_index",
    }
)


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite and nonempty")
    return vector


def early6_raw(
    scores: Sequence[float],
    gumbel_margin: float,
    top_document_id: str,
) -> np.ndarray:
    """Return the frozen six Early-Mirabel statistics for a single query.

    At Q1 the prefix mean and maximum top-1 statistics coincide and document
    concentration is one.  These degeneracies are retained intentionally: the
    primary method is the frozen All6 symmetry rule, not a post-hoc subset.
    """

    values = _finite_vector(scores, name="top-k scores")
    if values.size < 2 or not math.isfinite(float(gumbel_margin)):
        raise ValueError("Early6 needs top-2 scores and a finite Gumbel margin")
    if not str(top_document_id):
        raise ValueError("top_document_id cannot be empty")
    top1 = float(values[0])
    background = values[1:]
    spread = max(float(background.std(ddof=0)), np.finfo(float).eps)
    concentration = (top1 - float(background.mean())) / spread
    return np.asarray(
        [top1, top1 - float(values[1]), concentration, float(gumbel_margin), top1, 1.0],
        dtype=np.float64,
    )


def empirical_upper_p(value: float, sorted_reference: Sequence[float]) -> float:
    reference = _finite_vector(sorted_reference, name="normal reference")
    if np.any(reference[:-1] > reference[1:]):
        raise ValueError("normal reference must be sorted")
    if not math.isfinite(float(value)):
        raise ValueError("value must be finite")
    ge = reference.size - int(np.searchsorted(reference, float(value), side="left"))
    return float((1 + ge) / (reference.size + 1))


def empirical_upper_p_array(values: Sequence[float], sorted_reference: Sequence[float]) -> np.ndarray:
    reference = _finite_vector(sorted_reference, name="normal reference")
    array = _finite_vector(values, name="values")
    if np.any(reference[:-1] > reference[1:]):
        raise ValueError("normal reference must be sorted")
    ge = reference.size - np.searchsorted(reference, array, side="left")
    return (1.0 + ge.astype(np.float64)) / (reference.size + 1.0)


def equal_fisher_statistic(pvalues: Sequence[float]) -> float:
    """Standard unweighted Fisher statistic ``-2 sum(log(p_j))``."""

    p = _finite_vector(pvalues, name="p-values")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("p-values must be in (0,1]")
    return float(-2.0 * np.log(p).sum())


def equal_fisher_array(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] == 0 or not np.isfinite(p).all():
        raise ValueError("pvalues must be a finite [N,K] matrix")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("p-values must be in (0,1]")
    return -2.0 * np.log(p).sum(axis=1)


def cct_p(pvalues: Sequence[float]) -> float:
    p = _finite_vector(pvalues, name="p-values")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("p-values must be in (0,1]")
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    statistic = float(np.mean(np.tan(np.pi * (0.5 - clipped))))
    return float(np.clip(0.5 - math.atan(statistic) / math.pi, 0.0, 1.0))


def minp(pvalues: Sequence[float]) -> float:
    p = _finite_vector(pvalues, name="p-values")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("p-values must be in (0,1]")
    return float(np.min(p))


def anomaly_from_p(pvalue: float) -> float:
    if not math.isfinite(float(pvalue)) or not 0 < float(pvalue) <= 1:
        raise ValueError("p-value must be finite and in (0,1]")
    return float(-math.log10(max(float(pvalue), 1e-300)))


def threshold_higher(values: Sequence[float], target_fpr: float = 0.01) -> float:
    array = _finite_vector(values, name="threshold values")
    if not 0 < float(target_fpr) < 1:
        raise ValueError("target_fpr must lie in (0,1)")
    return float(np.quantile(array, 1.0 - float(target_fpr), method="higher"))


def strict_alarm(score: float | np.ndarray, threshold: float) -> bool | np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    if not np.isfinite(values).all() or not math.isfinite(float(threshold)):
        raise ValueError("score and threshold must be finite")
    result = values > float(threshold)
    return bool(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class Early6Calibration:
    feature_references: tuple[np.ndarray, ...]
    fisher_reference: np.ndarray
    b3_reference: np.ndarray
    b3_mean: float
    b3_std: float

    def __post_init__(self) -> None:
        if len(self.feature_references) != 6:
            raise ValueError("Early6 requires exactly six feature references")
        for reference in (*self.feature_references, self.fisher_reference, self.b3_reference):
            array = _finite_vector(reference, name="calibration reference")
            if np.any(array[:-1] > array[1:]):
                raise ValueError("calibration references must be sorted")
        if not math.isfinite(self.b3_mean) or not math.isfinite(self.b3_std) or self.b3_std <= 0:
            raise ValueError("B3 location/scale must be finite with positive scale")

    def score(self, early_raw_values: Sequence[float], b2: float) -> dict[str, float]:
        raw = _finite_vector(early_raw_values, name="Early6 raw features")
        if raw.size != 6 or not math.isfinite(float(b2)):
            raise ValueError("scorer requires Early6 and finite B2")
        feature_p = np.asarray(
            [empirical_upper_p(value, ref) for value, ref in zip(raw, self.feature_references)],
            dtype=np.float64,
        )
        fisher = equal_fisher_statistic(feature_p)
        p_early = empirical_upper_p(fisher, self.fisher_reference)
        p_b3 = empirical_upper_p(float(b2), self.b3_reference)
        p_min = minp((p_b3, p_early))
        p_cct = cct_p((p_b3, p_early))
        return {
            "b3": (float(b2) - self.b3_mean) / self.b3_std,
            "early6_fisher": fisher,
            "p_b3": p_b3,
            "p_early6": p_early,
            "p_minp": p_min,
            "p_cct": p_cct,
            "score_b3": anomaly_from_p(p_b3),
            "score_early6": anomaly_from_p(p_early),
            "score_minp": anomaly_from_p(p_min),
            "score_cct": anomaly_from_p(p_cct),
        }


def fit_early6_calibration(
    feature_reference: np.ndarray,
    fisher_reference_rows: np.ndarray,
    b3_feature_reference: Sequence[float],
) -> Early6Calibration:
    features = np.asarray(feature_reference, dtype=np.float64)
    fisher_rows = np.asarray(fisher_reference_rows, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 6 or not np.isfinite(features).all():
        raise ValueError("feature_reference must be finite [N,6]")
    if fisher_rows.ndim != 2 or fisher_rows.shape[1] != 6 or not np.isfinite(fisher_rows).all():
        raise ValueError("fisher_reference_rows must be finite [M,6]")
    b3 = _finite_vector(b3_feature_reference, name="B3 reference")
    feature_refs = tuple(np.sort(features[:, index]) for index in range(6))
    p_matrix = np.column_stack(
        [empirical_upper_p_array(fisher_rows[:, index], feature_refs[index]) for index in range(6)]
    )
    fisher_reference = np.sort(equal_fisher_array(p_matrix))
    std = max(float(b3.std(ddof=0)), np.finfo(float).eps)
    return Early6Calibration(
        feature_references=feature_refs,
        fisher_reference=fisher_reference,
        b3_reference=np.sort(b3),
        b3_mean=float(b3.mean()),
        b3_std=std,
    )


def family_macro_adr(
    attack_family: Sequence[str],
    member_label: Sequence[int],
    alarm: Sequence[bool],
) -> dict[str, object]:
    family = np.asarray(attack_family, dtype=str)
    member = np.asarray(member_label, dtype=int)
    prediction = np.asarray(alarm, dtype=bool)
    if not (len(family) == len(member) == len(prediction)) or not len(family):
        raise ValueError("aligned nonempty attack arrays required")
    details: dict[str, dict[str, float | int]] = {}
    member_rates, nonmember_rates, adr_rates = [], [], []
    for name in sorted(set(family)):
        mask = family == name
        member_mask = mask & (member == 1)
        nonmember_mask = mask & (member == 0)
        if not member_mask.any() or not nonmember_mask.any():
            raise ValueError(f"family {name} lacks member/nonmember attacks")
        mtpr = float(prediction[member_mask].mean())
        ntpr = float(prediction[nonmember_mask].mean())
        adr = (mtpr + ntpr) / 2.0
        details[name] = {
            "member_tpr": mtpr, "nonmember_tpr": ntpr, "adr": adr,
            "member_n": int(member_mask.sum()), "nonmember_n": int(nonmember_mask.sum()),
        }
        member_rates.append(mtpr); nonmember_rates.append(ntpr); adr_rates.append(adr)
    macro_member = float(np.mean(member_rates))
    macro_nonmember = float(np.mean(nonmember_rates))
    return {
        "fm_adr": float(np.mean(adr_rates)),
        "member_tpr": macro_member,
        "nonmember_tpr": macro_nonmember,
        "member_nonmember_gap": abs(macro_member - macro_nonmember),
        "families": details,
    }


def session_pvalue_curve(pvalues: Sequence[float], method: str) -> np.ndarray:
    p = _finite_vector(pvalues, name="query p-values")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("p-values must be in (0,1]")
    output = []
    for end in range(1, len(p) + 1):
        combined = minp(p[:end]) if method == "MINP" else cct_p(p[:end]) if method == "CCT" else None
        if combined is None:
            raise ValueError("method must be MINP or CCT")
        output.append(anomaly_from_p(combined))
    return np.asarray(output, dtype=np.float64)


@dataclass
class StickyOnlineDetector:
    method: str
    threshold: float
    _pvalues: list[float] = field(default_factory=list, init=False, repr=False)
    _alarm: bool = field(default=False, init=False, repr=False)
    _first_alarm: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method not in {"MINP", "CCT"} or not math.isfinite(float(self.threshold)):
            raise ValueError("valid method and finite threshold required")

    def reset(self) -> None:
        self._pvalues.clear(); self._alarm = False; self._first_alarm = None

    def update(self, pvalue: float) -> dict[str, object]:
        if not math.isfinite(float(pvalue)) or not 0 < float(pvalue) <= 1:
            raise ValueError("query p-value must be in (0,1]")
        self._pvalues.append(float(pvalue))
        score = float(session_pvalue_curve(self._pvalues, self.method)[-1])
        if not self._alarm and bool(strict_alarm(score, self.threshold)):
            self._alarm = True; self._first_alarm = len(self._pvalues)
        return {
            "query_count": len(self._pvalues), "score": score,
            "alarm": self._alarm, "first_alarm": self._first_alarm,
        }

    @property
    def final_score(self) -> float:
        if not self._pvalues:
            raise ValueError("no observations")
        return float(session_pvalue_curve(self._pvalues, self.method)[-1])


def validate_inference_payload(payload: Mapping[str, object]) -> None:
    forbidden = sorted(FORBIDDEN_INFERENCE_FIELDS & set(payload))
    if forbidden:
        raise ValueError(f"forbidden inference metadata: {forbidden}")


__all__ = [
    "EARLY6_FEATURES", "FORBIDDEN_INFERENCE_FIELDS", "Early6Calibration",
    "StickyOnlineDetector", "anomaly_from_p", "cct_p", "early6_raw",
    "empirical_upper_p", "empirical_upper_p_array", "equal_fisher_array",
    "equal_fisher_statistic", "family_macro_adr", "fit_early6_calibration",
    "minp", "session_pvalue_curve", "strict_alarm", "threshold_higher",
    "validate_inference_payload",
]
