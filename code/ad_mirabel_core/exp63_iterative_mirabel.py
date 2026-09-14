"""Minimal statistical and retrieval primitives for Exp63.

Exp63 keeps the Exp61 intent model frozen.  This module adds no learned
parameters: it only provides risk-controlled benign calibration, exact
iterative use of the canonical Mirabel statistic, and the optional
Bonferroni-corrected MinP session control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from src.canonical_mirabel import CanonicalMirabelStats, canonical_mirabel_from_full_scores
from src.exp61_intent_mirabel import wilson_interval


def empirical_upper_pvalue(scores: Sequence[float] | float, reference: Sequence[float]) -> np.ndarray:
    ref=np.sort(np.asarray(reference,dtype=np.float64))
    query=np.asarray(scores,dtype=np.float64)
    if ref.ndim!=1 or not len(ref) or not np.isfinite(ref).all() or not np.isfinite(query).all():
        raise ValueError("finite non-empty reference and finite scores required")
    counts=len(ref)-np.searchsorted(ref,query,side="left")
    return (1.0+counts.astype(np.float64))/(len(ref)+1.0)


def strict_alarm(scores: Sequence[float], threshold: float) -> np.ndarray:
    values=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(values).all() or not math.isfinite(float(threshold)):
        raise ValueError("finite scores and threshold required")
    return values>float(threshold)


def risk_controlled_threshold(calibration_scores: Sequence[float], *, upper_bound: float=.02,
                              confidence: float=.95) -> dict[str,float|int]:
    """Most permissive observed-score threshold whose Wilson upper is bounded.

    The strict comparator and a threshold selected from an observed score make
    ties conservative.  No attack labels are used.
    """
    values=np.asarray(calibration_scores,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite non-empty calibration scores required")
    if confidence!=.95:
        raise ValueError("the audited implementation currently fixes 95% Wilson confidence")
    candidates=np.unique(values)
    best=None
    for threshold in candidates[::-1]:
        positives=int(np.count_nonzero(values>threshold))
        low,high=wilson_interval(positives,len(values))
        if high<=upper_bound:
            record={"threshold":float(threshold),"false_positives":positives,"n":len(values),
                    "observed_fpr":positives/len(values),"wilson_low":float(low),
                    "wilson_upper":float(high),"required_upper_bound":float(upper_bound)}
            if best is None or positives>int(best["false_positives"]):
                best=record
    if best is None:
        # The maximum score yields zero alarms and therefore always has a
        # finite Wilson upper; reaching this branch indicates an impossible
        # sample-size/upper-bound combination rather than a tunable failure.
        threshold=float(np.max(values)); low,high=wilson_interval(0,len(values))
        raise ValueError(f"no risk-controlled threshold: zero-event Wilson upper={high:.6f}")
    return best


def empirical_conservative_threshold(calibration_scores: Sequence[float], target_fpr: float) -> float:
    values=np.asarray(calibration_scores,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not 0<target_fpr<1:
        raise ValueError("valid calibration scores and target FPR required")
    allowed=int(math.floor(target_fpr*len(values)))
    ordered=np.sort(values)[::-1]
    return float(ordered[min(allowed,len(ordered)-1)])


@dataclass(frozen=True)
class IterativeHideResult:
    original_top_k: tuple[str,...]
    safe_top_k: tuple[str,...]
    hidden_documents: tuple[str,...]
    first_stats: CanonicalMirabelStats
    residual_stats: CanonicalMirabelStats
    residual_alarm: bool
    backfilled: bool


def iterative_mirabel_hide(sorted_document_ids: Sequence[str], sorted_scores: Sequence[float], *,
                            intent_alarm: bool, top_k: int=3, max_hidden: int=2,
                            fixed_top2: bool=False) -> IterativeHideResult:
    ids=np.asarray(list(map(str,sorted_document_ids)),dtype=object)
    scores=np.asarray(sorted_scores,dtype=np.float64)
    if ids.ndim!=1 or scores.ndim!=1 or len(ids)!=len(scores) or len(ids)<top_k+max_hidden:
        raise ValueError("aligned full ranking with k+max_hidden documents required")
    if not np.isfinite(scores).all() or np.any(scores[:-1]<scores[1:]):
        raise ValueError("scores must be finite and descending")
    if max_hidden!=2:
        raise ValueError("Exp63 fixes max_hidden=2")
    first=canonical_mirabel_from_full_scores(scores)
    residual=canonical_mirabel_from_full_scores(scores[1:])
    residual_alarm=bool(residual.margin>0)
    hidden: list[str]=[]
    if intent_alarm:
        hidden.append(str(ids[0]))
        if fixed_top2 or residual_alarm:
            hidden.append(str(ids[1]))
    hidden_set=set(hidden)
    safe=tuple(str(value) for value in ids if str(value) not in hidden_set)[:top_k]
    if len(safe)!=top_k or any(value in safe for value in hidden):
        raise RuntimeError("iterative hide/backfill failed")
    return IterativeHideResult(tuple(map(str,ids[:top_k])),safe,tuple(hidden),first,residual,
                               residual_alarm,bool(hidden))


def bonferroni_minp_curve(intent_pvalues: Sequence[float]) -> np.ndarray:
    p=np.asarray(intent_pvalues,dtype=np.float64)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or np.any((p<=0)|(p>1)):
        raise ValueError("finite p-values in (0,1] required")
    running=np.minimum.accumulate(p)
    return np.minimum(1.0,np.arange(1,len(p)+1,dtype=np.float64)*running)


def session_anomaly_curve(intent_pvalues: Sequence[float]) -> np.ndarray:
    return -np.log10(np.maximum(bonferroni_minp_curve(intent_pvalues),np.finfo(np.float64).tiny))


__all__=["IterativeHideResult","bonferroni_minp_curve","empirical_conservative_threshold",
         "empirical_upper_pvalue","iterative_mirabel_hide","risk_controlled_threshold",
         "session_anomaly_curve","strict_alarm"]
