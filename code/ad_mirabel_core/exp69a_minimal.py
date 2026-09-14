"""Pure statistical helpers for Exp69A's minimal detector.

No attack label, family identifier, domain identifier, or query budget is an
inference feature.  Metadata is used only for calibration audits and reports.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np


def normalized_tokens(text: object) -> tuple[str, ...]:
    """Frozen containment normalization: NFKC, lowercase, punctuation to space.

    Stopwords are deliberately retained and repeated query tokens are retained.
    The tokenizer is the Unicode-aware ``\w+`` regular expression.
    """
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return tuple(re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def lexical_containment(query: object, document: object) -> float:
    query_tokens = normalized_tokens(query)
    if not query_tokens:
        return 0.0
    document_tokens = set(normalized_tokens(document))
    return float(sum(token in document_tokens for token in query_tokens) / len(query_tokens))


def upper_tail_pvalue(observed: Sequence[float] | float,
                      benign_reference: Sequence[float]) -> np.ndarray:
    """Finite-sample upper-tail empirical p-value with +1 correction."""
    values = np.atleast_1d(np.asarray(observed, dtype=np.float64))
    reference = np.sort(np.asarray(benign_reference, dtype=np.float64))
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("finite non-empty reference and observations required")
    greater_equal = len(reference) - np.searchsorted(reference, values, side="left")
    return (greater_equal + 1.0) / (len(reference) + 1.0)


def retrieval_pvalue(mirabel_p: Sequence[float], containment_rank_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mirabel = np.asarray(mirabel_p, dtype=np.float64)
    containment = np.asarray(containment_rank_p, dtype=np.float64)
    if containment.ndim != 2 or len(mirabel) != len(containment):
        raise ValueError("aligned Mirabel and rank-containment p-values required")
    if np.any((mirabel <= 0) | (mirabel > 1)) or np.any((containment <= 0) | (containment > 1)):
        raise ValueError("p-values must be in (0,1]")
    corrected_containment = np.minimum(1.0, containment.shape[1] * containment.min(axis=1))
    retrieval = np.minimum(1.0, 2.0 * np.minimum(mirabel, corrected_containment))
    return retrieval, corrected_containment


def dual_path_pvalue(intent_p: Sequence[float], retrieval_p: Sequence[float]) -> np.ndarray:
    intent, retrieval = np.asarray(intent_p, float), np.asarray(retrieval_p, float)
    if intent.shape != retrieval.shape or np.any((intent <= 0) | (intent > 1)) or np.any((retrieval <= 0) | (retrieval > 1)):
        raise ValueError("aligned p-values in (0,1] required")
    return np.minimum(1.0, 2.0 * np.minimum(intent, retrieval))


def strict_score_threshold(scores: Sequence[float], target_fpr: float) -> dict[str, float | int]:
    """Largest-alarm threshold obeying strict ``score > threshold`` FPR."""
    values = np.asarray(scores, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or not 0 < target_fpr < 1:
        raise ValueError("finite scores and target in (0,1) required")
    candidates = np.unique(values)
    feasible = [(int(np.count_nonzero(values > threshold)), float(threshold)) for threshold in candidates
                if np.mean(values > threshold) <= target_fpr + 1e-15]
    alarms, threshold = max(feasible, key=lambda row: (row[0], -row[1]))
    return {"threshold": threshold, "alarms": alarms, "n": len(values), "observed_fpr": alarms / len(values),
            "strict_comparator": ">"}


def running_minp(values: Sequence[float]) -> np.ndarray:
    pvalues = np.asarray(values, dtype=np.float64)
    if not len(pvalues) or np.any((pvalues <= 0) | (pvalues > 1)) or not np.isfinite(pvalues).all():
        raise ValueError("valid p-values required")
    return np.minimum.accumulate(pvalues)


def session_threshold(calibration_minima: Sequence[float], target_fpr: float = .02) -> dict[str, Any]:
    """Threshold on anomaly ``-log10(minP)`` using benign sessions only."""
    minima = np.asarray(calibration_minima, dtype=np.float64)
    if np.any((minima <= 0) | (minima > 1)):
        raise ValueError("session minima must be p-values")
    result = strict_score_threshold(-np.log10(minima), target_fpr)
    return {**result, "statistic": "-log10(min_{i<=t} p_Q1_i)", "attack_examples_used": 0}


def link_equal_label_q15_sessions(sessions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically link two disjoint frozen Q15 IA segments.

    This helper never creates a query.  It forms a Q30 *multi-target linked
    campaign* from two protocol-aligned Q15 sessions with the same membership
    label.  Component targets and normalized query keys must be disjoint so the
    resulting horizon cannot be described as duplicated or same-document Q30.
    """
    by_label: dict[int, list[dict[str, Any]]] = {}
    for source in sessions:
        row = dict(source)
        values = tuple(float(value) for value in row["values"])
        keys = tuple(str(value) for value in row["query_keys"])
        if len(values) != 15 or len(keys) != 15 or len(set(keys)) != 15:
            raise ValueError("each IA component must contain 15 unique frozen queries")
        if np.any((np.asarray(values) <= 0) | (np.asarray(values) > 1)):
            raise ValueError("IA component values must be p-values")
        row["values"], row["query_keys"] = values, keys
        by_label.setdefault(int(row["member_label"]), []).append(row)

    linked: list[dict[str, Any]] = []
    for label, source_rows in sorted(by_label.items()):
        pool = sorted(source_rows, key=lambda row: hashlib.sha256(
            f"exp69a-ia-link|{row['session_id']}".encode()).hexdigest())
        if len(pool) % 2:
            raise ValueError(f"IA label {label} has an odd number of Q15 sessions")
        while pool:
            first = pool.pop(0)
            first_keys = set(first["query_keys"])
            partner_index = next((index for index, candidate in enumerate(pool)
                                  if str(candidate["target_document_id"]) != str(first["target_document_id"])
                                  and first_keys.isdisjoint(candidate["query_keys"])), None)
            if partner_index is None:
                raise ValueError(f"no disjoint IA Q15 partner for {first['session_id']}")
            second = pool.pop(partner_index)
            component_ids = (str(first["session_id"]), str(second["session_id"]))
            linked_id = hashlib.sha256(("exp69a-linked-ia30|" + "|".join(component_ids)).encode()).hexdigest()
            linked.append({
                "session_id": linked_id,
                "member_label": label,
                "values": first["values"] + second["values"],
                "query_keys": first["query_keys"] + second["query_keys"],
                "component_session_ids": component_ids,
                "target_document_ids": (str(first["target_document_id"]), str(second["target_document_id"])),
                "construction": "two frozen protocol-aligned Q15 IA segments; deterministic multi-target linkage",
            })
    return linked


def candidate_feasible(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    for key in ("known_mean_fpr", "known_max_fpr", "finance_fpr", "qrecc_rewrite_fpr", "quac_fpr", "topiocqa_fpr"):
        if float(row.get(key, math.inf)) > .02:
            failures.append(key)
    if float(row.get("max_member_nonmember_gap", math.inf)) > .15:
        failures.append("member_nonmember_gap")
    return not failures, failures


def select_minimal_candidate(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    complexity = {"M0": 0, "M2": 1, "M1": 2, "M3": 3}
    audited = []
    for source in rows:
        row = dict(source); feasible, failures = candidate_feasible(row)
        row["feasible"] = feasible; row["failed_constraints"] = ";".join(failures)
        audited.append(row)
    eligible = [row for row in audited if row["feasible"]]
    if not eligible:
        return None, audited
    eligible.sort(key=lambda row: (-float(row["worst_domain_adr"]), -float(row["worst_family_adr"]),
                                   -float(row["finance_fm_adr"]), -float(row["lofo_worst"]),
                                   -float(row["known_fm_adr"]), complexity[str(row["model"])]))
    return dict(eligible[0]), audited


__all__ = ["candidate_feasible", "dual_path_pvalue", "lexical_containment", "link_equal_label_q15_sessions", "normalized_tokens",
           "retrieval_pvalue", "running_minp", "select_minimal_candidate", "session_threshold",
           "strict_score_threshold", "upper_tail_pvalue"]
