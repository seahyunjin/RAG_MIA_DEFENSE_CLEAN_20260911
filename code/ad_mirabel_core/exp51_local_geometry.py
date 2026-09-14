"""Exp51 retrieval-local geometry and normal conditional calibration primitives."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

EPS = 1e-6
GEOMETRIES = ("G0_GUMBEL", "G1_LOCAL_GAP", "G2_LOCAL_ROBUST_Z", "G3_LOCAL_CONCENTRATION", "G4_RANK_CURVATURE")
CALIBRATORS = ("C0_UNCONDITIONAL", "C1_CONDITIONAL_QUANTILE", "C2_CROSSFITTED_CONFORMAL")
AGGREGATORS = ("A0_MINP", "A1_CCT")


@dataclass(frozen=True)
class LocalGeometryOutput:
    g0_gumbel: float
    g1_local_gap: float
    g2_local_robust_z: float
    g3_local_concentration: float
    g4_rank_curvature: float
    spread: float
    local_mad: float
    local_entropy: float
    top1_rank_margin: float


def _validate(scores: Sequence[float]) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all():
        raise ValueError("native top-k scores must be finite and contain at least three values")
    x = np.sort(x)[::-1]
    return x


def local_geometry(scores: Sequence[float], *, gumbel_tail: float = math.nan, epsilon: float = EPS) -> LocalGeometryOutput:
    """Compute G0-G4 from the native top-k score vector only."""
    x = _validate(scores)
    top, rest = float(x[0]), x[1:]
    gaps = x[:-1] - x[1:]
    bulk_gaps = gaps[1:] if gaps.size > 1 else gaps
    gap_denom = float(np.median(np.abs(rest[:-1] - rest[1:]))) if rest.size > 1 else 0.0
    local_mad = float(np.median(np.abs(rest - np.median(rest))))
    g1 = (top - float(x[1])) / (gap_denom + epsilon)
    g2 = (top - float(np.median(rest))) / (1.4826 * local_mad + epsilon)
    shifted = x - top
    expv = np.exp(np.clip(shifted, -80.0, 0.0)); weights = expv / max(float(expv.sum()), epsilon)
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, epsilon))))
    concentration = float(1.0 - entropy / math.log(len(x))) if len(x) > 1 else 0.0
    g4 = float(gaps[0] / (float(np.median(bulk_gaps)) + epsilon)) if bulk_gaps.size else 0.0
    return LocalGeometryOutput(
        g0_gumbel=float(gumbel_tail), g1_local_gap=float(g1), g2_local_robust_z=float(g2),
        g3_local_concentration=float(concentration), g4_rank_curvature=float(g4),
        spread=float(top - x[-1]), local_mad=local_mad, local_entropy=entropy,
        top1_rank_margin=float(top - np.mean(x[1:])),
    )


def statistic_value(output: LocalGeometryOutput, geometry: str) -> float:
    mapping = {
        "G0_GUMBEL": output.g0_gumbel, "G1_LOCAL_GAP": output.g1_local_gap,
        "G2_LOCAL_ROBUST_Z": output.g2_local_robust_z,
        "G3_LOCAL_CONCENTRATION": output.g3_local_concentration,
        "G4_RANK_CURVATURE": output.g4_rank_curvature,
    }
    if geometry not in mapping: raise ValueError(f"unknown geometry {geometry}")
    value = float(mapping[geometry])
    return value if math.isfinite(value) else 0.0


def empirical_upper_p(value: float, reference: Sequence[float]) -> float:
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    if ref.size == 0 or not np.isfinite(ref).all(): raise ValueError("invalid normal reference")
    ge = int(ref.size - np.searchsorted(ref, float(value), side="left"))
    return float(np.clip((1.0 + ge) / (ref.size + 1.0), 1e-12, 1.0 - 1e-12))


def cct_p(values: Sequence[float]) -> float:
    p = np.clip(np.asarray(values, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(np.clip(0.5 - np.arctan(np.mean(np.tan(np.pi * (0.5 - p)))) / np.pi, 0.0, 1.0))


def aggregate_pvalues(pvalues: Sequence[float], aggregator: str) -> float:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.size == 0: return 1.0
    if aggregator == "A0_MINP": return float(np.min(p))
    if aggregator == "A1_CCT": return cct_p(np.sort(p))
    raise ValueError(f"unknown aggregator {aggregator}")


def online_prefix_pvalues(pvalues: Sequence[float], aggregator: str) -> np.ndarray:
    values=[]
    for i in range(1, len(pvalues)+1): values.append(aggregate_pvalues(pvalues[:i], aggregator))
    return np.asarray(values, dtype=np.float64)


__all__ = ["GEOMETRIES", "CALIBRATORS", "AGGREGATORS", "LocalGeometryOutput", "local_geometry", "statistic_value", "empirical_upper_p", "cct_p", "aggregate_pvalues", "online_prefix_pvalues"]
