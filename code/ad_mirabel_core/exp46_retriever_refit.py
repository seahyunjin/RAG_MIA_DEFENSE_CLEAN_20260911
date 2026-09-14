"""Pure, testable primitives for Exp46 retriever-specific LDF refitting.

This module has no network or filesystem side effects.  The experiment driver
uses these helpers to keep transfer-condition semantics and statistical rules
fixed before any evaluation result is observed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np


CONDITIONS = ("A_DIRECT", "B_THRESHOLD_ONLY", "C_TARGET_REFERENCE", "D_FULL_REFIT")
RETRIEVERS = ("MPNet", "GTE", "BGE-M3")
TARGET_FPRS = (0.01, 0.03, 0.05)


@dataclass(frozen=True)
class TransferComponents:
    condition: str
    source: str
    target: str
    coefficient_source: str
    orientation_source: str
    reference_source: str
    threshold_source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def resolve_components(condition: str, source: str, target: str) -> TransferComponents:
    """Resolve the preregistered A/B/C/D component provenance exactly."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if source not in RETRIEVERS or target not in RETRIEVERS:
        raise ValueError("source and target must be registered retrievers")
    if condition == "A_DIRECT":
        coefficient = orientation = reference = threshold = source
    elif condition == "B_THRESHOLD_ONLY":
        coefficient = orientation = reference = source
        threshold = target
    elif condition == "C_TARGET_REFERENCE":
        coefficient = orientation = source
        reference = threshold = target
    else:
        coefficient = orientation = reference = threshold = target
    return TransferComponents(
        condition, source, target, coefficient, orientation, reference, threshold
    )


def sorted_identifier_hash(values: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key(metadata: Mapping[str, object]) -> str:
    required = {
        "retriever_name", "model_id", "model_revision", "tokenizer_hash",
        "pooling_method", "embedding_normalization", "dataset_hash",
        "session_id_hash", "query_budget", "feature_version", "code_commit",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"missing cache-key fields: {missing}")
    return canonical_json_hash({key: metadata[key] for key in sorted(required)})


def assert_aligned_session_ids(*collections: Sequence[str]) -> None:
    if not collections:
        return
    reference = tuple(str(value) for value in collections[0])
    for index, values in enumerate(collections[1:], start=1):
        candidate = tuple(str(value) for value in values)
        if candidate != reference:
            raise ValueError(f"session order mismatch at collection {index}")


def choose_orientation(member_scores: Sequence[float], normal_scores: Sequence[float]) -> int:
    """Choose orientation on development data only; +1 means higher is riskier."""
    member = np.asarray(member_scores, dtype=float)
    normal = np.asarray(normal_scores, dtype=float)
    if not len(member) or not len(normal) or not np.isfinite(member).all() or not np.isfinite(normal).all():
        raise ValueError("finite nonempty development scores are required")
    return 1 if float(member.mean()) >= float(normal.mean()) else -1


def threshold_for_fpr(normal_scores: Sequence[float], target_fpr: float) -> float:
    """Conservative empirical threshold for the frozen strict `score > threshold` policy."""
    scores = np.asarray(normal_scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("finite nonempty normal scores are required")
    if not 0 < float(target_fpr) < 1:
        raise ValueError("target_fpr must lie in (0,1)")
    try:
        return float(np.quantile(scores, 1.0 - float(target_fpr), method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(scores, 1.0 - float(target_fpr), interpolation="higher"))


def strict_positive(scores: Sequence[float], threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    return values > float(threshold)


def wilson_interval(positives: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0 or positives < 0 or positives > total:
        raise ValueError("require 0 <= positives <= total and total > 0")
    # 1.959963984540054 is the two-sided 95% standard-normal quantile.
    if not math.isclose(confidence, 0.95):
        raise ValueError("Exp46 preregisters a 95% Wilson interval")
    z = 1.959963984540054
    p = positives / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def macro_tpr(family_labels: Mapping[str, Sequence[bool]]) -> float:
    if not family_labels:
        raise ValueError("at least one attack family is required")
    rates = []
    for family, labels in family_labels.items():
        values = np.asarray(labels, dtype=bool)
        if not len(values):
            raise ValueError(f"empty attack family: {family}")
        rates.append(float(values.mean()))
    return float(np.mean(rates))


def effective_auc(raw_auc: float) -> float:
    value = float(raw_auc)
    if not 0.0 <= value <= 1.0:
        raise ValueError("AUROC must lie in [0,1]")
    return max(value, 1.0 - value)


def attack_advantage(raw_auc: float) -> float:
    return abs(float(raw_auc) - 0.5)


def neutralization_ratio(no_defense_auc: float, defended_auc: float) -> float:
    denominator = attack_advantage(no_defense_auc)
    if denominator <= 1e-12:
        return math.nan
    return 1.0 - attack_advantage(defended_auc) / denominator


def score_diagnostics(scores: Sequence[float], threshold: float, orientation: int) -> dict[str, float | int | bool]:
    values = np.asarray(scores, dtype=float)
    finite = values[np.isfinite(values)]
    if orientation not in (-1, 1):
        raise ValueError("orientation must be -1 or +1")
    quantiles = np.quantile(finite, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]) if len(finite) else np.full(7, np.nan)
    unique = len(np.unique(finite)) if len(finite) else 0
    return {
        "min": float(np.min(finite)) if len(finite) else math.nan,
        "max": float(np.max(finite)) if len(finite) else math.nan,
        "mean": float(np.mean(finite)) if len(finite) else math.nan,
        "std": float(np.std(finite)) if len(finite) else math.nan,
        "p0_1": float(quantiles[0]), "p1": float(quantiles[1]), "p5": float(quantiles[2]),
        "p50": float(quantiles[3]), "p95": float(quantiles[4]), "p99": float(quantiles[5]),
        "p99_9": float(quantiles[6]),
        "tie_rate": 1.0 - unique / len(finite) if len(finite) else math.nan,
        "nan_count": int(np.isnan(values).sum()), "inf_count": int(np.isinf(values).sum()),
        "threshold": float(threshold), "orientation": int(orientation),
        "positive_rate": float(strict_positive(finite, threshold).mean()) if len(finite) else math.nan,
        "all_scores_equal": bool(unique <= 1),
        "threshold_above_max": bool(len(finite) and threshold >= float(np.max(finite))),
    }


def checkpoint_is_reusable(progress: Mapping[str, object], name: str, input_hash: str) -> bool:
    checkpoint = dict(progress.get("checkpoints", {})).get(name, {})
    return checkpoint.get("status") == "COMPLETED" and checkpoint.get("input_hash") == input_hash

