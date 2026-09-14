"""Pure components for Exp60's paraphrase-invariant linear Intent audit.

The primitives in this module keep the encoder frozen and return linear heads.
No attack family, member state, domain, retriever, or query budget is accepted by
an inference function; those fields are used only to construct development
groups and aggregate evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from src.exp58_intent_exposure_detector import LinearHead, sigmoid


CANDIDATES = (
    "P0_BASELINE",
    "P1_MULTIVIEW",
    "P2_FAMILY_BALANCED",
    "P3_GROUPDRO",
    "P4_CONSISTENCY",
)

SELECTION_PROTOCOL: dict[str, Any] = {
    "version": "exp60-paraphrase-invariant-v1",
    "feasibility": {
        "normal_fpr_max": .02,
        "wilson_upper_max": .03,
        "proxy_hard_normal_fpr_max": .03,
        "member_nonmember_gap_max": .10,
        "lofo_worst_min": .25,
    },
    "robust_score": "min(worst_domain_fm_adr/0.40, lofo_worst/0.30, paraphrase_retention)",
    "tie_break": [
        "robust_score_desc",
        "worst_domain_fm_adr_desc",
        "paraphrase_retention_desc",
        "lofo_median_desc",
        "fm_adr_desc",
        "parameter_count_asc",
    ],
    "final_evaluation_used_for_selection": False,
    "conditional_lr_trigger": "max_P0_to_P4_paraphrase_retention_below_0.70",
    "lr_ranks": [4, 8, 16],
}


def protocol_hash(protocol: Mapping[str, Any] = SELECTION_PROTOCOL) -> str:
    payload = json.dumps(dict(protocol), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def within_original_view_weights(original_id: Sequence[str]) -> np.ndarray:
    """Give every original unit mass, divided evenly across its retained views."""
    ids = np.asarray(original_id, str)
    if not len(ids):
        raise ValueError("non-empty original IDs required")
    unique, inverse, count = np.unique(ids, return_inverse=True, return_counts=True)
    if not len(unique) or np.any(count <= 0):
        raise ValueError("invalid original IDs")
    return 1.0 / count[inverse]


def family_balanced_original_weights(group: Sequence[str]) -> np.ndarray:
    """Assign equal total sampling mass to every observed family/member group."""
    values = np.asarray(group, str)
    if not len(values):
        raise ValueError("non-empty groups required")
    names, inverse, count = np.unique(values, return_inverse=True, return_counts=True)
    weights = 1.0 / count[inverse]
    # Mean one keeps regularization comparable with the unweighted objective.
    return weights / weights.mean()


def weighted_pairwise_fit(attack_features: np.ndarray, normal_features: np.ndarray,
                          pair_weight: np.ndarray, *, regularization: float = 1.0,
                          offset_difference: np.ndarray | None = None,
                          max_iter: int = 80) -> LinearHead:
    attack = np.asarray(attack_features, float)
    normal = np.asarray(normal_features, float)
    weights = np.asarray(pair_weight, float)
    if attack.shape != normal.shape or attack.ndim != 2 or len(attack) != len(weights) or not len(attack):
        raise ValueError("non-empty aligned pair arrays required")
    if not np.isfinite(attack).all() or not np.isfinite(normal).all() or np.any(weights < 0):
        raise ValueError("finite features and nonnegative weights required")
    difference = attack - normal
    doubled_x = np.vstack([difference, -difference])
    labels = np.r_[np.ones(len(difference)), np.zeros(len(difference))]
    doubled_w = np.r_[weights, weights]
    doubled_w /= max(doubled_w.mean(), 1e-12)
    if offset_difference is None:
        offset = np.zeros(len(labels), float)
    else:
        delta = np.asarray(offset_difference, float)
        if delta.shape != (len(difference),):
            raise ValueError("offset differences must align")
        offset = np.r_[delta, -delta]

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        score = offset + doubled_x @ coefficient
        loss = np.logaddexp(0.0, score) - labels * score
        residual = doubled_w * (sigmoid(score) - labels)
        value = float(np.mean(doubled_w * loss) + .5 * regularization * coefficient @ coefficient)
        gradient = doubled_x.T @ residual / len(labels) + regularization * coefficient
        return value, gradient

    result = minimize(objective, np.zeros(attack.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": int(max_iter), "ftol": 1e-9, "gtol": 1e-6})
    if not np.isfinite(result.x).all():
        raise RuntimeError(f"weighted pairwise fit failed: {result.message}")
    return LinearHead(result.x, 0.0)


def consistency_pairwise_fit(attack_features: np.ndarray, normal_features: np.ndarray,
                             pair_weight: np.ndarray, consistency_difference: np.ndarray,
                             consistency_weight: np.ndarray, *, regularization: float = 1.0,
                             offset_difference: np.ndarray | None = None,
                             consistency_offset: np.ndarray | None = None,
                             max_iter: int = 80) -> LinearHead:
    """Equal normalized pairwise and original/view consistency objectives."""
    attack = np.asarray(attack_features, float)
    normal = np.asarray(normal_features, float)
    pair_w = np.asarray(pair_weight, float)
    consistency = np.asarray(consistency_difference, float)
    consistency_w = np.asarray(consistency_weight, float)
    if attack.shape != normal.shape or attack.ndim != 2 or len(pair_w) != len(attack):
        raise ValueError("invalid ranking pairs")
    if consistency.ndim != 2 or consistency.shape[1] != attack.shape[1] or len(consistency_w) != len(consistency):
        raise ValueError("invalid consistency pairs")
    pair_w = pair_w / max(pair_w.mean(), 1e-12)
    consistency_w = consistency_w / max(consistency_w.mean(), 1e-12)
    difference = attack - normal
    pair_offset = np.zeros(len(difference), float) if offset_difference is None else np.asarray(offset_difference, float)
    cons_offset = np.zeros(len(consistency), float) if consistency_offset is None else np.asarray(consistency_offset, float)

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        margin = pair_offset + difference @ coefficient
        rank_loss = np.logaddexp(0.0, -margin)
        rank_gradient = difference.T @ (-pair_w * sigmoid(-margin)) / len(margin)
        residual = cons_offset + consistency @ coefficient
        consistency_loss = residual * residual
        consistency_gradient = 2.0 * consistency.T @ (consistency_w * residual) / max(len(residual), 1)
        # Each empirical component has coefficient 0.5 after its own normalization.
        value = float(.5 * np.mean(pair_w * rank_loss) +
                      .5 * np.mean(consistency_w * consistency_loss) +
                      .5 * regularization * coefficient @ coefficient)
        gradient = .5 * rank_gradient + .5 * consistency_gradient + regularization * coefficient
        return value, gradient

    result = minimize(objective, np.zeros(attack.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": int(max_iter), "ftol": 1e-9, "gtol": 1e-6})
    if not np.isfinite(result.x).all():
        raise RuntimeError(f"consistency fit failed: {result.message}")
    return LinearHead(result.x, 0.0)


def multiview_group_dro_fit(features: np.ndarray, label: np.ndarray, groups: Sequence[str],
                            original_id: Sequence[str], *, regularization: float = 1.0,
                            offset: np.ndarray | None = None, rounds: int = 5,
                            eta: float = .25, max_iter: int = 50) -> tuple[LinearHead, dict[str, float]]:
    """GroupDRO with each original's view mass normalized before group reweighting."""
    x = np.asarray(features, float)
    y = np.asarray(label, int)
    group = np.asarray(groups, str)
    ids = np.asarray(original_id, str)
    if x.ndim != 2 or not (len(x) == len(y) == len(group) == len(ids)) or set(np.unique(y)) != {0, 1}:
        raise ValueError("aligned binary multi-view data required")
    view = within_original_view_weights(ids)
    names = sorted(np.unique(group).tolist())
    group_mass = np.ones(len(names), float) / len(names)
    base = np.zeros(len(y), float) if offset is None else np.asarray(offset, float)
    head = LinearHead(np.zeros(x.shape[1]), 0.0)
    for _ in range(int(rounds)):
        sample_weight = np.zeros(len(y), float)
        for position, name in enumerate(names):
            mask = group == name
            local = view[mask]
            sample_weight[mask] = group_mass[position] * local / max(local.sum(), 1e-12)
        sample_weight *= len(sample_weight) / max(sample_weight.sum(), 1e-12)

        def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
            coefficient, intercept = parameter[:-1], parameter[-1]
            score = base + x @ coefficient + intercept
            loss = np.logaddexp(0.0, score) - y * score
            residual = sample_weight * (sigmoid(score) - y)
            value = float(np.mean(sample_weight * loss) + .5 * regularization * coefficient @ coefficient)
            gradient = np.r_[x.T @ residual / len(y) + regularization * coefficient,
                             residual.mean()]
            return value, gradient

        result = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B",
                          options={"maxiter": int(max_iter), "ftol": 1e-9, "gtol": 1e-6})
        if not np.isfinite(result.x).all():
            raise RuntimeError(f"multi-view GroupDRO failed: {result.message}")
        head = LinearHead(result.x[:-1], float(result.x[-1]))
        raw_loss = np.logaddexp(0.0, base + head.score(x)) - y * (base + head.score(x))
        losses = np.asarray([np.average(raw_loss[group == name], weights=view[group == name])
                             for name in names], float)
        group_mass *= np.exp(eta * losses)
        group_mass /= group_mass.sum()
    return head, {name: float(group_mass[index]) for index, name in enumerate(names)}


def feasibility(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    limits = {
        "normal_fpr": (.02, "NORMAL_FPR"),
        "wilson_upper": (.03, "WILSON_UPPER"),
        "hard_normal_fpr": (.03, "HARD_NORMAL_FPR"),
        "member_nonmember_gap": (.10, "MEMBER_NONMEMBER_GAP"),
    }
    failures: list[str] = []
    numeric = ("normal_fpr", "wilson_upper", "hard_normal_fpr", "member_nonmember_gap",
               "lofo_worst", "worst_domain_fm_adr", "lofo_median", "fm_adr", "paraphrase_retention")
    if not all(np.isfinite(float(row.get(key, math.nan))) for key in numeric):
        failures.append("NONFINITE")
    for key, (maximum, label) in limits.items():
        if float(row.get(key, math.inf)) > maximum:
            failures.append(label)
    if float(row.get("lofo_worst", -math.inf)) < .25:
        failures.append("LOFO_WORST")
    return not failures, failures


def robust_score(row: Mapping[str, Any]) -> float:
    return min(float(row["worst_domain_fm_adr"]) / .40,
               float(row["lofo_worst"]) / .30,
               float(row["paraphrase_retention"]))


def select_candidate(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audited: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        passed, failures = feasibility(row)
        row["constraint_pass"] = passed
        row["constraint_failures"] = ";".join(failures)
        row["robust_score"] = robust_score(row) if passed else -math.inf
        audited.append(row)
    eligible = [row for row in audited if row["constraint_pass"]]
    if not eligible:
        raise RuntimeError("NO_FEASIBLE_EXP60_CANDIDATE")
    selected = sorted(eligible, key=lambda row: (
        -float(row["robust_score"]), -float(row["worst_domain_fm_adr"]),
        -float(row["paraphrase_retention"]), -float(row["lofo_median"]),
        -float(row["fm_adr"]), int(row["parameter_count"]), str(row["candidate"]),
    ))[0]
    return dict(selected), audited


def q1_tier(row: Mapping[str, Any]) -> str:
    common = (float(row["normal_fpr"]) <= .02 and float(row["hard_normal_fpr"]) <= .03 and
              float(row["member_nonmember_gap"]) <= .10)
    research = (common and float(row["worst_domain_fm_adr"]) >= .25 and
                float(row["lofo_median"]) >= .45 and float(row["lofo_worst"]) >= .20 and
                float(row["paraphrase_retention"]) >= .70)
    deployment = (common and float(row["wilson_upper"]) <= .03 and
                  float(row["worst_domain_fm_adr"]) >= .35 and float(row["lofo_median"]) >= .50 and
                  float(row["lofo_worst"]) >= .25 and float(row["paraphrase_retention"]) >= .80 and
                  float(row["lofo_p_median"]) >= .40)
    strong = (float(row["normal_fpr"]) <= .015 and float(row["wilson_upper"]) <= .025 and
              float(row["hard_normal_fpr"]) <= .02 and float(row["member_nonmember_gap"]) <= .10 and
              float(row["fm_adr"]) >= .60 and float(row["worst_domain_fm_adr"]) >= .40 and
              float(row["lofo_median"]) >= .55 and float(row["lofo_worst"]) >= .30 and
              float(row["paraphrase_retention"]) >= .85 and float(row["lofo_p_median"]) >= .50)
    if strong:
        return "Q1_SURFACE_STRONG"
    if deployment:
        return "Q1_SURFACE_DEPLOYMENT_CANDIDATE"
    if research:
        return "Q1_SURFACE_RESEARCH_PROMISING"
    return "Q1_SURFACE_FAILED"


def automated_fidelity_label(row: Mapping[str, Any]) -> str:
    valid = (float(row["accepted_rate"]) >= .80 and float(row["cosine_median"]) >= .80 and
             float(row["entity_preservation_rate"]) >= .95 and
             float(row["number_preservation_rate"]) >= .98 and
             float(row["target_reference_preservation_rate"]) >= .95 and
             float(row["retrieval_top5_overlap_mean"]) >= .50)
    partial = (float(row["accepted_rate"]) >= .50 and float(row["cosine_median"]) >= .70 and
               float(row["entity_preservation_rate"]) >= .80 and
               float(row["number_preservation_rate"]) >= .90 and
               float(row["target_reference_preservation_rate"]) >= .80 and
               float(row["retrieval_top5_overlap_mean"]) >= .30)
    if valid:
        return "PARAPHRASE_FIDELITY_VALID"
    if partial:
        return "PARAPHRASE_FIDELITY_PARTIAL"
    return "PARAPHRASE_FIDELITY_INVALID"


@dataclass(frozen=True)
class LowRankHead:
    projection: np.ndarray
    coefficient: np.ndarray
    intercept: float = 0.0

    def __post_init__(self) -> None:
        projection = np.asarray(self.projection, float)
        coefficient = np.asarray(self.coefficient, float)
        if projection.ndim != 2 or coefficient.shape != (projection.shape[1],):
            raise ValueError("invalid low-rank head shapes")
        if not np.isfinite(projection).all() or not np.isfinite(coefficient).all() or not np.isfinite(self.intercept):
            raise ValueError("low-rank parameters must be finite")
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "coefficient", coefficient)

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, float)
        return (values @ self.projection) @ self.coefficient + self.intercept

    @property
    def parameter_count(self) -> int:
        return int(self.projection.size + self.coefficient.size + 1)

