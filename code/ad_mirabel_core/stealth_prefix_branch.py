"""Compact top-k consistency features for an offline stealth-prefix branch.

The module contains no learned attack model.  It derives explainable prefix
statistics from retrieval rows and converts them to benign empirical p-values
outside this module.  All concentration values are normalized by prefix length
so Q1, Q2, and Q3 can receive separate benign nulls without a raw-size bias.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Mapping, Sequence

import numpy as np


S1_DOCUMENT = ("soft_document_concentration",)
S2_CLUSTER = ("soft_cluster_concentration",)
S3_OVERLAP = (
    "weighted_jaccard_mean",
    "rank_biased_overlap_mean",
    "reciprocal_rank_consistency",
    "shared_topk_mass",
)
S4_COHERENCE = (
    "query_embedding_coherence",
    "document_centroid_coherence",
    "dominant_direction_concentration",
)


def soft_retrieval_weights(scores: Sequence[float], tau: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be one finite vector")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive")
    logits = (values - values.max()) / float(tau)
    weights = np.exp(logits)
    return weights / weights.sum()


def weighted_jaccard(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = sum(min(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys)
    denominator = sum(max(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys)
    return float(numerator / denominator) if denominator else 0.0


def shared_mass(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return float(sum(min(float(left[key]), float(right[key])) for key in set(left) & set(right)))


def reciprocal_rank_consistency(left: Sequence[str], right: Sequence[str]) -> float:
    left_rank = {str(doc): index + 1 for index, doc in enumerate(left)}
    right_rank = {str(doc): index + 1 for index, doc in enumerate(right)}
    shared = set(left_rank) & set(right_rank)
    if not shared:
        return 0.0
    numerator = sum(2.0 / (left_rank[doc] + right_rank[doc]) for doc in shared)
    normalizer = sum(1.0 / rank for rank in range(1, min(len(left), len(right)) + 1))
    return float(numerator / normalizer) if normalizer else 0.0


def rank_biased_overlap(left: Sequence[str], right: Sequence[str], persistence: float = 0.9) -> float:
    if not 0.0 < persistence < 1.0:
        raise ValueError("persistence must lie in (0, 1)")
    depth = min(len(left), len(right))
    if depth == 0:
        return 0.0
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    score = 0.0
    for index in range(depth):
        left_seen.add(str(left[index]))
        right_seen.add(str(right[index]))
        agreement = len(left_seen & right_seen) / (index + 1)
        score += (1.0 - persistence) * (persistence**index) * agreement
    score += (persistence**depth) * len(left_seen & right_seen) / depth
    return float(score)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0 else 0.0


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("vectors must be a nonempty matrix")
    if len(values) == 1:
        return 0.0
    scores = [_cosine(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    return float(np.mean(scores))


def _row_distribution(row: Mapping[str, object], tau: float, top_k: int) -> tuple[list[str], np.ndarray]:
    scores = np.asarray(row["scores"], dtype=float)[:top_k]
    doc_ids = [str(value) for value in row["doc_ids"]][: len(scores)]
    if len(scores) != len(doc_ids) or len(scores) < 2:
        raise ValueError("each row requires aligned top-k scores and document IDs")
    return doc_ids, soft_retrieval_weights(scores, tau)


def prefix_features(
    rows: Sequence[Mapping[str, object]],
    *,
    tau: float,
    top_k: int,
    document_clusters: Mapping[str, int] | None = None,
    document_embeddings: Mapping[str, np.ndarray] | None = None,
    query_embeddings: Sequence[np.ndarray] | None = None,
) -> dict[str, float]:
    """Calculate Q1--Q3 top-k concentration and consistency statistics."""
    if not 1 <= len(rows) <= 3:
        raise ValueError("stealth-prefix features support Q1 through Q3")
    distributions = [_row_distribution(row, tau, top_k) for row in rows]
    document_mass: defaultdict[str, float] = defaultdict(float)
    cluster_mass: defaultdict[int, float] = defaultdict(float)
    entropies = []
    slopes = []
    centroids = []
    weight_maps = []
    rankings = []
    for doc_ids, weights in distributions:
        rankings.append(doc_ids)
        mapping = {doc: float(weight) for doc, weight in zip(doc_ids, weights)}
        weight_maps.append(mapping)
        for doc, weight in mapping.items():
            document_mass[doc] += weight
            if document_clusters is not None and doc in document_clusters:
                cluster_mass[int(document_clusters[doc])] += weight
        entropy = -float(np.sum(weights * np.log(np.clip(weights, 1e-15, 1.0))))
        entropies.append(entropy / math.log(len(weights)))
        scores = np.asarray([float(value) for value in rows[len(entropies) - 1]["scores"]], dtype=float)[: len(weights)]
        slopes.append(float(-np.polyfit(np.arange(1, len(scores) + 1), scores, 1)[0]))
        if document_embeddings is not None:
            available = [(document_embeddings[doc], weight) for doc, weight in mapping.items() if doc in document_embeddings]
            if available:
                matrix = np.asarray([item[0] for item in available], dtype=float)
                mass = np.asarray([item[1] for item in available], dtype=float)
                mass /= mass.sum()
                centroids.append(mass @ matrix)

    pair_jaccard = []
    pair_rbo = []
    pair_rr = []
    pair_mass = []
    for index in range(1, len(rows)):
        pair_jaccard.append(weighted_jaccard(weight_maps[index - 1], weight_maps[index]))
        pair_rbo.append(rank_biased_overlap(rankings[index - 1], rankings[index]))
        pair_rr.append(reciprocal_rank_consistency(rankings[index - 1], rankings[index]))
        pair_mass.append(shared_mass(weight_maps[index - 1], weight_maps[index]))

    result = {
        "soft_document_concentration": max(document_mass.values()) / len(rows),
        "hard_document_concentration": max(sum(doc in ranking for ranking in rankings) for doc in document_mass) / len(rows),
        "soft_cluster_concentration": (max(cluster_mass.values()) / len(rows)) if cluster_mass else 0.0,
        "negative_topk_entropy": -float(np.mean(entropies)),
        "score_slope": float(np.mean(slopes)),
        "weighted_jaccard_mean": float(np.mean(pair_jaccard)) if pair_jaccard else 0.0,
        "rank_biased_overlap_mean": float(np.mean(pair_rbo)) if pair_rbo else 0.0,
        "reciprocal_rank_consistency": float(np.mean(pair_rr)) if pair_rr else 0.0,
        "shared_topk_mass": float(np.mean(pair_mass)) if pair_mass else 0.0,
        "query_embedding_coherence": 0.0,
        "document_centroid_coherence": 0.0,
        "dominant_direction_concentration": 0.0,
    }
    if query_embeddings is not None:
        values = np.asarray(query_embeddings, dtype=float)
        if len(values) != len(rows):
            raise ValueError("query embedding count must equal prefix length")
        result["query_embedding_coherence"] = mean_pairwise_cosine(values)
    if centroids:
        values = np.asarray(centroids, dtype=float)
        result["document_centroid_coherence"] = mean_pairwise_cosine(values)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized = values / np.maximum(norms, np.finfo(float).eps)
        result["dominant_direction_concentration"] = float(np.linalg.norm(normalized.mean(axis=0)))
    return result

