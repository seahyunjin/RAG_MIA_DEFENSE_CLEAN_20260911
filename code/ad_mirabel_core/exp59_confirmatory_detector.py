"""Core primitives for Exp59's confirmatory linear Intent audit.

The module deliberately contains no data-loading or experiment-specific paths.
It preserves the Exp58 two-head architecture: one frozen-embedding linear Intent
head and one separately reported frozen Exposure head.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPSILON = 1e-6
REPRESENTATIONS = ("R0_RAW", "R1_CENTERED", "R2_STANDARDIZED")
OBJECTIVES = ("I0_ERM", "I1_GROUPDRO", "I2_PAIRWISE")


def domain_family_macro_metrics(domain: Sequence[str], family: Sequence[str],
                                member: Sequence[int], detected: Sequence[bool]) -> dict[str, float]:
    """Match Exp59's final metric hierarchy: domain, then family/member.

    A surface transform must be compared with S0 using the same estimand as the
    confirmatory final evaluation.  Pooling all domains before computing the
    family macro changes domain weights and is therefore not comparable.
    """
    domains = np.asarray(domain, str)
    families = np.asarray(family, str)
    members = np.asarray(member, int)
    decisions = np.asarray(detected, bool)
    if not (len(domains) == len(families) == len(members) == len(decisions)):
        raise ValueError("aligned metric inputs required")
    if not len(domains):
        raise ValueError("non-empty metric inputs required")

    rows: list[dict[str, float]] = []
    for domain_name in sorted(np.unique(domains)):
        domain_mask = domains == domain_name
        family_adr: list[float] = []
        member_rates: list[float] = []
        nonmember_rates: list[float] = []
        gaps: list[float] = []
        for family_name in sorted(np.unique(families[domain_mask])):
            rates: dict[int, float] = {}
            for label in (0, 1):
                mask = domain_mask & (families == family_name) & (members == label)
                if mask.any():
                    rates[label] = float(decisions[mask].mean())
            if rates:
                family_adr.append(float(np.mean(list(rates.values()))))
            if 0 in rates and 1 in rates:
                member_rates.append(rates[1])
                nonmember_rates.append(rates[0])
                gaps.append(abs(rates[1] - rates[0]))
        rows.append({
            "fm_adr": float(np.mean(family_adr)) if family_adr else float("nan"),
            "member_tpr": float(np.mean(member_rates)) if member_rates else float("nan"),
            "nonmember_tpr": float(np.mean(nonmember_rates)) if nonmember_rates else float("nan"),
            "member_nonmember_gap": float(np.mean(gaps)) if gaps else float("nan"),
        })
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


@dataclass(frozen=True)
class BenignRepresentationStats:
    """Target-domain nuisance statistics estimated from benign references only."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    reference_count: int
    source_role: str = "benign_reference"
    attack_rows_used: int = 0

    def validate(self) -> None:
        if self.reference_count <= 0 or self.attack_rows_used != 0:
            raise ValueError("representation statistics must use benign references only")
        if self.mean.ndim != 1 or self.standard_deviation.shape != self.mean.shape:
            raise ValueError("invalid representation-statistic shape")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.standard_deviation).all():
            raise ValueError("non-finite representation statistics")
        if (self.standard_deviation < 0).any():
            raise ValueError("negative standard deviation")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    denominator = np.linalg.norm(values, axis=1, keepdims=True)
    denominator = np.maximum(denominator, EPSILON)
    output = values / denominator
    if not np.isfinite(output).all():
        raise ValueError("non-finite normalized representation")
    return output.astype(np.float32)


def fit_benign_stats(embeddings: np.ndarray, benign_reference_mask: Sequence[bool],
                     attack_mask: Sequence[bool] | None = None) -> BenignRepresentationStats:
    values = np.asarray(embeddings, dtype=np.float64)
    benign = np.asarray(benign_reference_mask, dtype=bool)
    if values.ndim != 2 or len(values) != len(benign) or not benign.any():
        raise ValueError("non-empty aligned benign reference mask required")
    if attack_mask is not None and np.any(benign & np.asarray(attack_mask, dtype=bool)):
        raise ValueError("attack example included in benign representation statistics")
    reference = values[benign]
    stats = BenignRepresentationStats(
        mean=reference.mean(axis=0).astype(np.float32),
        standard_deviation=reference.std(axis=0, ddof=0).astype(np.float32),
        reference_count=int(len(reference)),
    )
    stats.validate()
    return stats


def transform_representation(embeddings: np.ndarray, representation: str,
                             stats: BenignRepresentationStats | None = None) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("2-D embeddings required")
    if representation == "R0_RAW":
        return l2_normalize(values)
    if stats is None:
        raise ValueError("benign statistics required for R1/R2")
    stats.validate()
    if values.shape[1] != len(stats.mean):
        raise ValueError("embedding dimension mismatch")
    if representation == "R1_CENTERED":
        return l2_normalize(values - stats.mean)
    if representation == "R2_STANDARDIZED":
        return l2_normalize((values - stats.mean) / (stats.standard_deviation + EPSILON))
    raise ValueError(f"unsupported representation: {representation}")


def representation_parameter_count(representation: str, dimension: int) -> int:
    """Deterministic nuisance statistics are not trainable parameters."""
    if representation not in REPRESENTATIONS or dimension <= 0:
        raise ValueError("invalid representation")
    return 0


SELECTION_PROTOCOL: dict[str, Any] = {
    "version": "exp59-confirmatory-v1",
    "feasibility": {
        "normal_fpr_max": 0.02,
        "wilson_upper_max": 0.03,
        "proxy_hard_normal_fpr_max": 0.05,
        "member_nonmember_gap_max": 0.15,
        "human_benign_hard_normal_fpr_max_if_available": 0.05,
        "finite_required": True,
        "score_collapse_forbidden": True,
    },
    "generalization": "min(worst_domain_fm_adr, lofo_median)",
    "tie_break": [
        "generalization_score_desc",
        "lofo_worst_desc",
        "member_nonmember_gap_asc",
        "hard_normal_fpr_asc",
        "fm_adr_desc",
        "parameter_count_asc",
    ],
    "final_evaluation_used_for_selection": False,
}


def selection_protocol_hash(protocol: Mapping[str, Any] = SELECTION_PROTOCOL) -> str:
    payload = json.dumps(dict(protocol), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_candidate(row: Mapping[str, Any]) -> bool:
    keys = ("normal_fpr", "wilson_upper", "hard_normal_fpr", "member_nonmember_gap",
            "worst_domain_fm_adr", "lofo_median", "lofo_worst", "fm_adr")
    return all(np.isfinite(float(row[key])) for key in keys)


def feasibility(row: Mapping[str, Any], *, human_fpr: float | None = None) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not _finite_candidate(row):
        failures.append("NONFINITE")
    if float(row.get("normal_fpr", math.inf)) > .02:
        failures.append("NORMAL_FPR")
    if float(row.get("wilson_upper", math.inf)) > .03:
        failures.append("WILSON_UPPER")
    if float(row.get("hard_normal_fpr", math.inf)) > .05:
        failures.append("HARD_NORMAL_FPR")
    if float(row.get("member_nonmember_gap", math.inf)) > .15:
        failures.append("MEMBER_NONMEMBER_GAP")
    if bool(row.get("score_collapse", False)):
        failures.append("SCORE_COLLAPSE")
    if human_fpr is not None and (not np.isfinite(human_fpr) or human_fpr > .05):
        failures.append("HUMAN_HARD_NORMAL_FPR")
    return not failures, failures


def candidate_generalization_score(row: Mapping[str, Any]) -> float:
    return min(float(row["worst_domain_fm_adr"]), float(row["lofo_median"]))


def prospective_select(candidates: Sequence[Mapping[str, Any]], *,
                       human_fpr_by_candidate: Mapping[str, float] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply frozen constraints first, then maximize generalization.

    Lower FPR never receives preference once candidates satisfy feasibility.
    """
    audited: list[dict[str, Any]] = []
    for source in candidates:
        row = dict(source)
        key = str(row.get("candidate", f"{row.get('representation')}__{row.get('objective')}"))
        human = None if human_fpr_by_candidate is None else human_fpr_by_candidate.get(key)
        passed, failures = feasibility(row, human_fpr=human)
        row["candidate"] = key
        row["constraint_pass"] = passed
        row["constraint_failures"] = ";".join(failures)
        row["generalization_score"] = candidate_generalization_score(row) if passed else -math.inf
        audited.append(row)
    eligible = [row for row in audited if row["constraint_pass"]]
    if not eligible:
        raise RuntimeError("NO_FEASIBLE_CONFIRMATORY_CANDIDATE")
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["generalization_score"]),
            -float(row["lofo_worst"]),
            float(row["member_nonmember_gap"]),
            float(row["hard_normal_fpr"]),
            -float(row["fm_adr"]),
            int(row.get("parameter_count", 10**9)),
            str(row["candidate"]),
        ),
    )
    return dict(ranked[0]), audited


PROTECTED_TOKEN = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]*|\d+(?:\.\d+)?)\b")


def protected_entities(text: str) -> tuple[str, ...]:
    candidates = list(dict.fromkeys(PROTECTED_TOKEN.findall(str(text))))
    sentence_openers = {
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "does", "do", "did", "is", "are", "was", "were", "can", "could", "would",
        "explain", "describe", "identify", "tell", "give", "please", "answer",
    }
    return tuple(value for value in candidates if value.casefold() not in sentence_openers)


def entity_preserving_syntax_rewrite(text: str) -> str:
    """Fixed, label-independent syntax rewrite used only at evaluation."""
    original = " ".join(str(text).split())
    body = original[:-1] if original.endswith("?") else original
    rules = (
        (r"(?i)^what\s+(?:is|are)\s+(.+)$", r"Identify \1."),
        (r"(?i)^how\s+(?:does|do|did)\s+(.+)$", r"Explain how \1."),
        (r"(?i)^why\s+(.+)$", r"Give the reason that \1."),
        (r"(?i)^where\s+(?:is|are|was|were)\s+(.+)$", r"Identify the location of \1."),
        (r"(?i)^when\s+(?:is|was|did)\s+(.+)$", r"Identify the time associated with \1."),
    )
    for pattern, replacement in rules:
        if re.match(pattern, body):
            return re.sub(pattern, replacement, body)
    return f"Regarding the same request, provide an answer to this: {original}"


NONCRITICAL_MODIFIERS = re.compile(
    r"\b(?:please|kindly|carefully|briefly|quickly|simply|actually|really|exactly|just|perhaps|possibly)\b",
    flags=re.IGNORECASE,
)


def noncritical_lexical_mask(text: str) -> str:
    original = " ".join(str(text).split())
    masked = NONCRITICAL_MODIFIERS.sub("[MOD]", original)
    return masked if masked != original else f"[MOD] {original}"


def semantic_equivalence_accept(original: str, transformed: str, cosine: float,
                                threshold: float = .80) -> bool:
    left, right = " ".join(str(original).split()), " ".join(str(transformed).split())
    if not left or not right or left.casefold() == right.casefold():
        return False
    if not np.isfinite(cosine) or cosine < threshold:
        return False
    lowered = right.casefold()
    return all(entity.casefold() in lowered for entity in protected_entities(left))


def surface_label(retentions: Iterable[float]) -> str:
    values = np.asarray(list(retentions), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return "SURFACE_SHORTCUT_SUSPECTED"
    worst = float(values.min())
    if worst >= .85:
        return "SURFACE_ROBUST"
    if worst >= .70:
        return "SURFACE_PARTIAL"
    return "SURFACE_SHORTCUT_SUSPECTED"


def q1_tier(row: Mapping[str, Any], *, human_labels_available: bool = False) -> str:
    surface = float(row.get("surface_retention", -math.inf))
    common = (float(row["normal_fpr"]) <= .02 and float(row["wilson_upper"]) <= .03 and
              float(row["member_nonmember_gap"]) <= .15)
    research = (common and float(row["hard_normal_fpr"]) <= .05 and
                float(row["worst_domain_fm_adr"]) >= .25 and float(row["lofo_median"]) >= .45 and
                float(row["lofo_worst"]) >= .20)
    human_ok = (not human_labels_available) or float(row.get("human_hard_normal_fpr", math.inf)) <= .05
    deployment = (common and human_ok and float(row["hard_normal_fpr"]) <= .03 and
                  float(row["member_nonmember_gap"]) <= .10 and float(row["worst_domain_fm_adr"]) >= .35 and
                  float(row["lofo_median"]) >= .50 and float(row["lofo_worst"]) >= .25 and surface >= .80)
    strong = (float(row["normal_fpr"]) <= .015 and float(row["wilson_upper"]) <= .025 and
              float(row["hard_normal_fpr"]) <= .02 and float(row["member_nonmember_gap"]) <= .10 and
              float(row["worst_domain_fm_adr"]) >= .40 and float(row["fm_adr"]) >= .60 and
              float(row["lofo_median"]) >= .55 and float(row["lofo_worst"]) >= .30 and surface >= .85 and
              human_ok)
    if strong:
        return "Q1_CONFIRMATORY_STRONG"
    if deployment:
        return "Q1_CONFIRMATORY_DEPLOYMENT_CANDIDATE"
    if research:
        return "Q1_CONFIRMATORY_RESEARCH_PROMISING"
    return "Q1_CONFIRMATORY_FAILED"


def no_inference_metadata(metadata: Sequence[str]) -> bool:
    forbidden = {"attack_family", "member_label", "domain", "domain_id", "retriever",
                 "retriever_id", "query_budget", "budget"}
    return not (set(map(str, metadata)) & forbidden)
