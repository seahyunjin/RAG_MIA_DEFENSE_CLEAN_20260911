"""Statistical and protocol guards for frozen multi-attack evaluation.

The helpers in this module are deliberately independent of Experiment 43's
data-loading code.  They encode the rules that must remain true even when a
new attack source is added later: native budgets are never padded, attacker
inputs cannot contain labels, blocking has one fixed response, and model
selection is lexicographic rather than a post-hoc weighted sum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


BLOCKED_RESPONSE = "REQUEST_BLOCKED"
ALLOWED_ATTACKER_FIELDS = frozenset(
    {
        "session_id",
        "query_indices",
        "query_scores",
        "retrieval_scores",
        "retrieval_doc_ids",
        "responses_seen",
        "native_utility_proxy",
        "target_document_id",
        "detector_public_score",
    }
)
FORBIDDEN_ATTACKER_FIELDS = frozenset(
    {
        "member_label",
        "membership_label",
        "ground_truth_label",
        "evaluation_auc",
        "normal_calibration_scores",
        "future_response",
    }
)


@dataclass(frozen=True)
class NativeBudget:
    attack_name: str
    native_budget: int
    evaluated_budgets: tuple[int, ...]

    def __post_init__(self) -> None:
        if int(self.native_budget) < 1:
            raise ValueError("native_budget must be positive")
        values = tuple(int(value) for value in self.evaluated_budgets)
        if not values or any(value < 1 or value > int(self.native_budget) for value in values):
            raise ValueError("evaluated budgets must lie within the native budget")
        if tuple(sorted(set(values))) != values:
            raise ValueError("evaluated budgets must be unique and increasing")


def validate_attacker_view(payload: Mapping[str, object]) -> None:
    fields = set(map(str, payload))
    forbidden = fields & FORBIDDEN_ATTACKER_FIELDS
    if forbidden:
        raise ValueError(f"attacker view contains forbidden fields: {sorted(forbidden)}")
    unknown = fields - ALLOWED_ATTACKER_FIELDS
    if unknown:
        raise ValueError(f"attacker view contains undeclared fields: {sorted(unknown)}")


def blocked_response(blocked: bool, response: str | None) -> str:
    """Apply the single pre-registered response policy for every attack."""

    if bool(blocked):
        return BLOCKED_RESPONSE
    if response is None:
        raise ValueError("an unblocked query requires its original response")
    return str(response)


def membership_blindness_aurocs(
    normal_scores: Sequence[float],
    member_attack_scores: Sequence[float],
    nonmember_attack_scores: Sequence[float],
) -> dict[str, float]:
    normal = np.asarray(normal_scores, dtype=float)
    member = np.asarray(member_attack_scores, dtype=float)
    nonmember = np.asarray(nonmember_attack_scores, dtype=float)
    if any(values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all()
           for values in (normal, member, nonmember)):
        raise ValueError("all score groups must be nonempty finite vectors")

    def auc(positive: np.ndarray, negative: np.ndarray) -> float:
        labels = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)]
        scores = np.r_[positive, negative]
        return float(roc_auc_score(labels, scores))

    return {
        "normal_vs_member_attack_auroc": auc(member, normal),
        "normal_vs_nonmember_attack_auroc": auc(nonmember, normal),
        "member_vs_nonmember_attack_auroc": auc(member, nonmember),
    }


def family_statistics(values: Mapping[str, float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("at least one family is required")
    array = np.asarray(list(values.values()), dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("family values must be finite")
    return {
        "family_count": len(array),
        "macro_mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "standard_deviation": float(array.std(ddof=0)),
    }


def gate_worst_family(*, target_fpr: float, median_tpr: float, worst_tpr: float) -> bool:
    requirements = {0.01: (0.40, 0.20), 0.03: (0.60, 0.30), 0.05: (0.70, 0.40)}
    key = round(float(target_fpr), 2)
    if key not in requirements:
        raise ValueError("unsupported target FPR")
    median_required, worst_required = requirements[key]
    return float(median_tpr) >= median_required and float(worst_tpr) >= worst_required


def select_final_candidate(candidates: Sequence[Mapping[str, object]]) -> str | None:
    """Select by the pre-registered lexicographic rule, never a weighted sum."""

    eligible = [row for row in candidates if bool(row.get("all_required_gates_pass", False))]
    if not eligible:
        return None
    ordered = sorted(
        eligible,
        key=lambda row: (
            -float(row["worst_family_tpr_1pct"]),
            -float(row["macro_tpr_1pct"]),
            -float(row["post_defense_auc_reduction"]),
            float(row["observed_normal_fpr_1pct"]),
            float(row["latency_p95_ms"]),
            str(row["candidate"]),
        ),
    )
    return str(ordered[0]["candidate"])


def validate_external_config(config: Mapping[str, object]) -> None:
    if bool(config.get("enabled", False)):
        raise ValueError("external API must be disabled by default")
    if "api_key" in config or "api_key_value" in config:
        raise ValueError("API key values must never be stored")
    if str(config.get("api_key_env", "")) != "EXTERNAL_LLM_API_KEY":
        raise ValueError("external API key must be referenced only by environment-variable name")


def validate_code_only_export(root: str | Path) -> None:
    base = Path(root)
    forbidden_parts = {
        "results_raw", "results_publication", "private_cache", "raw_datasets",
        "raw_attacks", "raw_responses", "retrieval_cache", "logs",
    }
    for path in base.rglob("*"):
        lowered = {part.lower() for part in path.parts}
        if lowered & forbidden_parts:
            raise ValueError(f"code-only export contains forbidden path: {path}")
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".npy", ".npz"}:
            if "synthetic" not in path.name.lower():
                raise ValueError(f"code-only export contains a result/data artifact: {path}")
