"""Metadata-free counterfactual sensitivity primitives for Exp54-U.

Only quantities observable from the current RAG call are accepted by the
deployable scorer.  Dataset, family, membership, retriever, generator, turn,
and budget metadata are deliberately absent from the inference API.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


FORBIDDEN_INFERENCE_FIELDS = frozenset({
    "attack_family", "member_label", "domain", "domain_id", "retriever",
    "retriever_id", "generator", "generator_id", "query_budget",
    "query_index", "experiment_id", "template_id", "attack_label",
})


def stable_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def strict_alarm(score: float | np.ndarray, threshold: float) -> bool | np.ndarray:
    """The frozen decision policy is strictly greater-than."""
    return np.asarray(score) > float(threshold) if isinstance(score, np.ndarray) else float(score) > float(threshold)


def empirical_upper_p(reference: Sequence[float], value: float) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    if not len(ref) or not np.isfinite(ref).all() or not np.isfinite(value):
        raise ValueError("finite nonempty reference and finite value required")
    return float((1 + np.count_nonzero(ref >= float(value))) / (len(ref) + 1))


def minp(pvalues: Sequence[float]) -> float:
    p = np.asarray(pvalues, dtype=np.float64)
    if not len(p) or not np.isfinite(p).all() or np.any((p <= 0) | (p > 1)):
        raise ValueError("p-values must be finite in (0,1]")
    return float(np.min(p))


def cct(pvalues: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Equal/custom-weight Cauchy combination with stable clipping."""
    p = np.asarray(pvalues, dtype=np.float64)
    if not len(p) or not np.isfinite(p).all() or np.any((p <= 0) | (p > 1)):
        raise ValueError("p-values must be finite in (0,1]")
    w = np.ones(len(p), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != p.shape or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid CCT weights")
    w /= w.sum()
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    statistic = float(np.sum(w * np.tan((.5 - clipped) * math.pi)))
    return float(np.clip(.5 - math.atan(statistic) / math.pi, 0.0, 1.0))


def symmetric_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Order-symmetric response-pair representation [|a-b|, a*b]."""
    left, right = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired embeddings must have equal finite shapes")
    return np.concatenate((np.abs(left - right), left * right), axis=-1)


def masked_deepset(values: Sequence[np.ndarray | None]) -> np.ndarray:
    """Permutation-invariant mean/max pooling with an explicit missing mask."""
    present = [np.asarray(value, dtype=np.float32) for value in values if value is not None]
    if not present:
        raise ValueError("at least one counterfactual must be present")
    if any(value.shape != present[0].shape or not np.isfinite(value).all() for value in present):
        raise ValueError("counterfactual representations must be equal finite shapes")
    matrix = np.stack(present)
    masks = np.asarray([value is not None for value in values], dtype=np.float32)
    return np.concatenate((matrix.mean(0), matrix.max(0), masks), axis=-1)


def exclude_top1(document_ids: Sequence[str], scores: Sequence[float], fill_id: str,
                 fill_score: float, *, top_k: int = 5) -> tuple[list[str], list[float]]:
    ids, vals = list(map(str, document_ids)), list(map(float, scores))
    if len(ids) < top_k or len(vals) < top_k or fill_id in ids[1:top_k]:
        raise ValueError("valid top-k and distinct fill document required")
    output_ids = ids[1:top_k] + [str(fill_id)]
    output_scores = vals[1:top_k] + [float(fill_score)]
    if ids[0] in output_ids or len(output_ids) != top_k:
        raise AssertionError("top-1 exclusion failed")
    return output_ids, output_scores


def matched_replacement(document_ids: Sequence[str], scores: Sequence[float], replacement_id: str,
                        replacement_score: float, *, top_k: int = 5) -> tuple[list[str], list[float]]:
    """Replace top-1 without accepting any attack/member/family label."""
    ids, vals = list(map(str, document_ids)), list(map(float, scores))
    if len(ids) < top_k or replacement_id in ids[:top_k]:
        raise ValueError("replacement must be outside original top-k")
    result_ids = [str(replacement_id)] + ids[1:top_k]
    result_scores = [float(replacement_score)] + vals[1:top_k]
    return result_ids, result_scores


def semantic_paraphrase_valid(original: str, paraphrase: str, cosine: float,
                              threshold: float, protected_entities: Iterable[str] = ()) -> bool:
    a, b = " ".join(str(original).split()), " ".join(str(paraphrase).split())
    if not a or not b or a.casefold() == b.casefold() or not np.isfinite(cosine) or cosine < threshold:
        return False
    lowered = b.casefold()
    return all(str(entity).casefold() in lowered for entity in protected_entities if str(entity).strip())


def family_macro_adr(labels: Sequence[int], predictions: Sequence[bool],
                     families: Sequence[str], members: Sequence[int]) -> dict[str, object]:
    y, pred = np.asarray(labels, int), np.asarray(predictions, bool)
    fam, mem = np.asarray(families, object), np.asarray(members, int)
    if not (len(y) == len(pred) == len(fam) == len(mem)):
        raise ValueError("metric arrays differ in length")
    family_rows: dict[str, dict[str, float]] = {}
    for value in sorted(set(map(str, fam[y == 1]))):
        rates = []
        row: dict[str, float] = {}
        for member in (0, 1):
            mask = (y == 1) & (fam.astype(str) == value) & (mem == member)
            rate = float(pred[mask].mean()) if mask.any() else math.nan
            row[f"tpr_{'member' if member else 'nonmember'}"] = rate
            if np.isfinite(rate): rates.append(rate)
        row["adr"] = float(np.mean(rates)) if rates else math.nan
        row["gap"] = abs(row["tpr_member"] - row["tpr_nonmember"]) if np.isfinite(row["tpr_member"]) and np.isfinite(row["tpr_nonmember"]) else math.nan
        family_rows[value] = row
    adr = [row["adr"] for row in family_rows.values() if np.isfinite(row["adr"])]
    member_rates = [row["tpr_member"] for row in family_rows.values() if np.isfinite(row["tpr_member"])]
    nonmember_rates = [row["tpr_nonmember"] for row in family_rows.values() if np.isfinite(row["tpr_nonmember"])]
    return {"fm_adr": float(np.mean(adr)) if adr else math.nan,
            "member_tpr": float(np.mean(member_rates)) if member_rates else math.nan,
            "nonmember_tpr": float(np.mean(nonmember_rates)) if nonmember_rates else math.nan,
            "member_nonmember_gap": abs(float(np.mean(member_rates)) - float(np.mean(nonmember_rates))) if member_rates and nonmember_rates else math.nan,
            "families": family_rows}


def effective_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    if len(np.unique(y)) < 2: return math.nan
    auc = float(roc_auc_score(y, s))
    return max(auc, 1.0 - auc)


def validate_inference_record(record: Mapping[str, object]) -> None:
    overlap = FORBIDDEN_INFERENCE_FIELDS.intersection(record)
    if overlap:
        raise ValueError(f"forbidden inference metadata: {sorted(overlap)}")
    arrays = [np.asarray(value) for value in record.values() if isinstance(value, (list, tuple, np.ndarray))]
    if any(not np.isfinite(array.astype(float)).all() for array in arrays):
        raise ValueError("NaN/Inf inference input")

