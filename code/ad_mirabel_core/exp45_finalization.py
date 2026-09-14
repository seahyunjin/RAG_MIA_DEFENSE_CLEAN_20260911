"""Shared, frozen helpers for Exp45 publication finalization.

This module contains no training path and no network/API integration.  It is
deliberately small so the public names, detector semantics, palette, and
calibration rules can be tested independently of the long-running driver.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta


PUBLIC_DETECTORS = ("Mirabel", "Early-Mirabel", "MPNet-LDF")
ATTACK_ORDER = ("RAG-MIA", "S²-MIA", "MBA", "IA", "DCMI", "MEntA")
INTERNAL_TO_PUBLIC_ATTACK = {"IA-MIA": "IA", "S2-MIA": "S²-MIA"}
TARGET_FPRS = (0.01, 0.03, 0.05)
BLOCKED_RESPONSE = "REQUEST_BLOCKED"

PALETTE = {
    "No Defense": "#F2BE9B",
    "Mirabel": "#B8BDC7",
    "Early-Mirabel": "#C8B5E3",
    "MPNet-LDF": "#84BCE0",
    "Legacy-V16": "#8F99A8",
    "GTE": "#A6D4C2",
    "Adaptive": "#E7ADBE",
    "Reference": "#E9D8A6",
    "Text": "#2F3B47",
    "Grid": "#D9DDE3",
}


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def public_attack_name(value: str) -> str:
    return INTERNAL_TO_PUBLIC_ATTACK.get(str(value), str(value))


def empirical_upper_p(reference: Sequence[float], value: float) -> float:
    array = np.asarray(reference, dtype=float)
    return float((1 + np.count_nonzero(array >= float(value))) / (len(array) + 1))


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def stable_softmax(values: Sequence[float], temperature: float = 0.08) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    scaled = (array - np.max(array)) / max(float(temperature), 1e-6)
    weights = np.exp(np.clip(scaled, -60.0, 0.0))
    return weights / max(float(weights.sum()), 1e-12)


def mirabel_session_score(rows: Sequence[Mapping[str, object]], budget: int) -> float:
    """Anytime score from the canonical single-query Mirabel margin.

    Mirabel remains a single-query detector: each query receives its canonical
    margin independently.  The session action is sticky, hence the observable
    session score is the maximum query margin seen so far.
    """

    selected = list(rows[: int(budget)])
    if not selected:
        raise ValueError("Mirabel requires at least one observed query")
    return float(max(float(row["mirabel_margin"]) for row in selected))


def early_mirabel_score(rows: Sequence[Mapping[str, object]], budget: int) -> float:
    """Frozen six-feature Early-Mirabel score.

    The historical branch supports query prefixes Q1--Q4.  For longer native
    attacks the Q4 score is retained; no query-count-specific extension is
    invented in Exp45.  Canonical cached margins are used directly.
    """

    selected = list(rows[: min(int(budget), 4)])
    if not selected:
        raise ValueError("Early-Mirabel requires at least one observed query")
    top1 = np.asarray([float(row["top1"]) for row in selected], dtype=float)
    gaps, concentrations, margins, docs = [], [], [], []
    for row in selected:
        scores = np.asarray(row["scores"], dtype=float)
        top2 = float(scores[1]) if len(scores) > 1 else float(scores[0])
        gaps.append(max(0.0, float(scores[0]) - top2))
        concentrations.append(float(stable_softmax(scores[:5])[0]))
        margins.append(float(row["mirabel_margin"]))
        docs.append(str(row["doc_ids"][0]))
    aggregate = {
        "top1": float(top1.mean()),
        "gap": float(np.mean(gaps)),
        "softmax_concentration": float(np.mean(concentrations)),
        "gumbel_margin": float(np.mean(margins)),
        "tail_fraction": float(np.mean(top1 > 0.80)),
        "doc_concentration": float(max(Counter(docs).values()) / len(docs)),
    }
    clipped = {
        "top1": float(np.clip(aggregate["top1"], 0.0, 1.0)),
        "gap": float(np.clip(aggregate["gap"] / 0.35, 0.0, 1.0)),
        "softmax_concentration": float(np.clip(aggregate["softmax_concentration"], 0.0, 1.0)),
        "gumbel_margin": float(np.clip(0.5 + aggregate["gumbel_margin"] / 0.30, 0.0, 1.0)),
        "tail_fraction": float(np.clip(aggregate["tail_fraction"], 0.0, 1.0)),
        "doc_concentration": float(np.clip(aggregate["doc_concentration"], 0.0, 1.0)),
    }
    weights = {
        "top1": 0.28,
        "gap": 0.15,
        "softmax_concentration": 0.12,
        "gumbel_margin": 0.12,
        "tail_fraction": 0.18,
        "doc_concentration": 0.15,
    }
    return float(sum(weights[name] * clipped[name] for name in weights))


def deterministic_generator_subset(sessions: Sequence[dict], per_label: int = 200) -> list[dict]:
    """Select a fixed split-aware subset without reading attack outcomes."""

    output: list[dict] = []
    for family in sorted({str(row["attack_family"]) for row in sessions}):
        for label in (0, 1):
            values = [row for row in sessions if str(row["attack_family"]) == family and int(row["member_label"]) == label]
            values.sort(key=lambda row: sha256_text(str(row["session_id"]) + "|exp45-local-generator"))
            output.extend(values[: min(per_label, len(values))])
    return output


def validate_public_names(text: str, *, allow_provenance: bool = False) -> None:
    if allow_provenance:
        return
    forbidden = ("Exp39", "Session-DRO-LDF", "Gate F", "V16-M", "Round 2")
    hits = [token for token in forbidden if token in text]
    if hits:
        raise ValueError(f"internal detector labels in public artifact: {hits}")


def json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
