"""Leakage audits and statistical utilities registered for Exp49."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re
from typing import Callable, Mapping, Sequence

import numpy as np


BOOTSTRAP_SEED = 20260815


def stable_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def identifier_hash(values: Sequence[object]) -> str:
    payload = "\n".join(sorted(map(str, values))) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def wilson_interval(positives: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= int(positives) <= int(total):
        raise ValueError("require 0 <= positives <= total and total > 0")
    z = 1.959963984540054
    n, count = int(total), int(positives)
    p = count / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def fpr_gate(positives: int, total: int, *, tpr: float) -> dict[str, object]:
    observed = int(positives) / int(total)
    lower, upper = wilson_interval(int(positives), int(total))
    finite = all(math.isfinite(float(value)) for value in (observed, upper, tpr))
    degenerate = observed == 0.0 and float(tpr) == 0.0
    passed = finite and observed <= 0.02 and upper <= 0.03 and not degenerate
    return {"observed_fpr": observed, "ci_low": lower, "ci_high": upper,
            "tpr": float(tpr), "finite": finite, "degenerate": degenerate,
            "passed": bool(passed)}


def family_macro_rate(families: Sequence[object], decisions: Sequence[bool]) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for family, decision in zip(families, decisions):
        groups[str(family)].append(bool(decision))
    if not groups:
        return math.nan
    return float(np.mean([np.mean(values) for values in groups.values()]))


def first_alarm(decisions: Sequence[bool]) -> int | None:
    for index, value in enumerate(decisions, start=1):
        if bool(value):
            return index
    return None


def shortcut_ratio(shallow_auc: float, full_auc: float, epsilon: float = 1e-8) -> float:
    return float((float(shallow_auc) - 0.5) / (float(full_auc) - 0.5 + float(epsilon)))


def shortcut_verdict(ratio: float) -> str:
    if not math.isfinite(float(ratio)):
        return "SEVERE_SHORTCUT_RISK"
    if float(ratio) >= 0.8:
        return "SEVERE_SHORTCUT_RISK"
    if float(ratio) >= 0.5:
        return "MODERATE_SHORTCUT_RISK"
    return "LOW_SHORTCUT_RISK"


def cluster_bootstrap_paired_difference(
    left: Sequence[bool | float],
    right: Sequence[bool | float],
    clusters: Sequence[object],
    *,
    strata: Sequence[object] | None = None,
    iterations: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Paired source-document cluster bootstrap, optionally family stratified."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if len(a) != len(b) or len(a) != len(clusters) or not len(a):
        raise ValueError("paired values and cluster IDs must align")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or iterations < 2_000:
        raise ValueError("finite values and at least 2,000 iterations are required")
    strata_values = np.asarray(["all"] * len(a) if strata is None else list(map(str, strata)), dtype=object)
    cluster_values = np.asarray(list(map(str, clusters)), dtype=object)
    by_stratum: dict[str, dict[str, np.ndarray]] = {}
    for stratum in np.unique(strata_values):
        selected = np.flatnonzero(strata_values == stratum)
        groups: dict[str, list[int]] = defaultdict(list)
        for index in selected:
            groups[str(cluster_values[index])].append(int(index))
        by_stratum[str(stratum)] = {key: np.asarray(value, dtype=int) for key, value in groups.items()}
    rng = np.random.default_rng(int(seed)); samples = np.empty(int(iterations), dtype=float)
    for iteration in range(int(iterations)):
        pieces = []
        for groups in by_stratum.values():
            keys = np.asarray(list(groups), dtype=object)
            chosen = rng.choice(keys, size=len(keys), replace=True)
            pieces.extend(groups[str(key)] for key in chosen)
        indices = np.concatenate(pieces)
        samples[iteration] = float(np.mean(a[indices] - b[indices]))
    return {
        "point": float(np.mean(a - b)), "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)), "iterations": int(iterations),
        "seed": int(seed), "clusters": int(len(set(cluster_values))),
    }


def minhash_signature(text: str, *, permutations: int = 64) -> np.ndarray:
    tokens = normalized_text(text).split()
    shingles = {" ".join(tokens[index:index + 5]) for index in range(max(1, len(tokens) - 4))}
    if not shingles:
        shingles = {normalized_text(text)}
    # Keep the audit bounded for corpora with tens of thousands of documents.
    # The deterministic sample preserves exact/near-duplicate sensitivity while
    # avoiding an O(document_length × permutations) Python loop for every
    # shingle.  Exact normalized hashes are checked separately.
    if len(shingles) > 32:
        shingles = set(sorted(shingles, key=lambda value: hashlib.sha256(value.encode()).digest())[:32])
    result = np.full(permutations, np.uint64(2**64 - 1), dtype=np.uint64)
    for shingle in shingles:
        digest = hashlib.sha256(shingle.encode()).digest()
        first = int.from_bytes(digest[:8], "little")
        second = int.from_bytes(digest[8:16], "little") | 1
        for index in range(permutations):
            value = np.uint64((first + index * second) % (2**64 - 59))
            if value < result[index]:
                result[index] = value
    return result


def minhash_lsh_overlap(
    left: Mapping[str, str], right: Mapping[str, str], *, threshold: float = 0.8,
    permutations: int = 64, bands: int = 16,
) -> dict[str, object]:
    if permutations % bands:
        raise ValueError("permutations must be divisible by bands")
    rows = permutations // bands
    right_signatures = {key: minhash_signature(text, permutations=permutations) for key, text in right.items()}
    buckets: dict[tuple[int, bytes], set[str]] = defaultdict(set)
    for key, signature in right_signatures.items():
        for band in range(bands):
            buckets[(band, signature[band * rows:(band + 1) * rows].tobytes())].add(key)
    matches, maximum = [], 0.0
    for left_key, text in left.items():
        signature = minhash_signature(text, permutations=permutations)
        candidates: set[str] = set()
        for band in range(bands):
            candidates.update(buckets.get((band, signature[band * rows:(band + 1) * rows].tobytes()), set()))
        for right_key in candidates:
            similarity = float(np.mean(signature == right_signatures[right_key]))
            maximum = max(maximum, similarity)
            if similarity >= threshold:
                matches.append({"left": left_key, "right": right_key, "estimated_jaccard": similarity})
    return {"threshold": float(threshold), "matches": matches, "match_count": len(matches),
            "maximum_candidate_similarity": maximum, "permutations": permutations, "bands": bands}


def generalization_labels(
    final_fm_mdr: float,
    current_ldf: float,
    early_ci_low: float,
    normal_gate: bool,
    lofo_rates: Mapping[str, float],
    current_lofo_median: float,
) -> dict[str, str]:
    rates = np.asarray(list(lofo_rates.values()), dtype=float)
    median = float(np.median(rates)) if len(rates) else math.nan
    worst = float(np.min(rates)) if len(rates) else math.nan
    improved_domain = (float(final_fm_mdr) - float(current_ldf) >= 0.10 and
                       float(early_ci_low) > 0.0 and bool(normal_gate))
    strong_domain = (float(final_fm_mdr) >= 0.50 and float(early_ci_low) > 0.0 and bool(normal_gate))
    if strong_domain:
        domain = "CROSS_DOMAIN_GENERALIZATION_STRONG"
    elif improved_domain:
        domain = "CROSS_DOMAIN_GENERALIZATION_IMPROVED"
    else:
        domain = "CROSS_DOMAIN_GENERALIZATION_FAILED"
    improved_family = (median >= 0.30 and int(np.sum(rates >= 0.20)) >= 4 and
                       median > float(current_lofo_median))
    strong_family = median >= 0.50 and int(np.sum(rates >= 0.50)) >= 4 and worst >= 0.20
    if strong_family:
        family = "ATTACK_FAMILY_GENERALIZATION_STRONG"
    elif improved_family:
        family = "ATTACK_FAMILY_GENERALIZATION_IMPROVED"
    else:
        family = "ATTACK_FAMILY_GENERALIZATION_FAILED"
    return {"cross_domain": domain, "attack_family": family}
