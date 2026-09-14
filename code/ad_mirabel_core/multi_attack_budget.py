"""Utilities for attack-family and query-budget defense evaluation.

The module deliberately keeps the two false-positive notions separate:

* ``FPR_NORMAL`` is measured on ordinary BEIR queries/streams and is used to
  select deployment thresholds.
* ``FPR_MIA`` is measured on nonmember-target attack queries and is reported
  only as a membership-inference ranking metric.

No attack label is used when fitting retrieval-feature normalization, CCT, or
deployment thresholds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from .early_mirabel_detector import stable_softmax


POLICIES = (
    "Original Mirabel",
    "Early-Mirabel (six features)",
    "Mirabel + Early-Mirabel",
    "Mirabel + rolling CCT (six features)",
    "Sequential top1-Early + rolling CCT",
    "Sequential six-feature Early + rolling CCT",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_ids(values: Iterable[str], seed: int) -> dict[str, str]:
    """Deterministic 40/30/30 source-disjoint split."""
    unique = np.asarray(sorted(set(map(str, values))), dtype=object)
    np.random.default_rng(seed).shuffle(unique)
    calibration_end = int(0.4 * len(unique))
    validation_end = int(0.7 * len(unique))
    return {
        str(value): (
            "calibration"
            if index < calibration_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, value in enumerate(unique)
    }


def threshold_at_fpr(scores: Sequence[float], target: float) -> float:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return float("inf")
    return float(np.quantile(values, 1.0 - float(target), method="higher"))


def empirical_cdf(values: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    if ordered.size == 0:
        raise ValueError("Empirical reference cannot be empty")
    array = np.asarray(values, dtype=float)
    return np.searchsorted(ordered, array, side="right") / (len(ordered) + 1.0)


def empirical_upper_p(values: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    if ordered.size == 0:
        raise ValueError("Empirical reference cannot be empty")
    array = np.asarray(values, dtype=float)
    ge = len(ordered) - np.searchsorted(ordered, array, side="left")
    return (ge + 1.0) / (len(ordered) + 1.0)


def _cdf_from_sorted(value: float, ordered: np.ndarray) -> float:
    return float(np.searchsorted(ordered, value, side="right") / (len(ordered) + 1.0))


def _upper_p_from_sorted(value: float, ordered: np.ndarray) -> float:
    ge = len(ordered) - int(np.searchsorted(ordered, value, side="left"))
    return float((ge + 1.0) / (len(ordered) + 1.0))


def cct_score(p_values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(p_values, dtype=float), 1e-12, 1.0 - 1e-12)
    statistic = np.mean(np.tan((0.5 - values) * np.pi), axis=1)
    p_value = 0.5 - np.arctan(statistic) / np.pi
    return -np.log10(np.maximum(p_value, 1e-15))


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    label_array = np.asarray(labels, dtype=int)
    if len(np.unique(label_array)) != 2:
        return float("nan")
    return float(roc_auc_score(label_array, np.asarray(scores, dtype=float)))


def current_early_score(rows: Sequence[Mapping[str, object]]) -> float:
    """Run the shipped six-feature Early-Mirabel on a 1--4 query prefix."""
    if not 1 <= len(rows) <= 4:
        raise ValueError("Early-Mirabel requires a prefix of length 1--4")
    # Algebraically identical to EarlyMirabelDetector.update, but avoids
    # constructing a stateful detector for every rolling window.
    top1_values = []
    gaps = []
    concentrations = []
    margins = []
    docs = []
    for row in rows:
        scores = np.asarray(row["scores"], dtype=float)
        top1 = float(scores[0])
        top2 = float(scores[1]) if len(scores) > 1 else top1
        background = scores[1:]
        if len(background) >= 3:
            mean = float(background.mean())
            std = float(background.std()) + 1e-12
            root = math.sqrt(2.0 * math.log(len(background)))
            threshold = (
                mean
                + std * root
                + (-math.log(-math.log(0.95))) * std / root
            )
            margin = top1 - threshold
        else:
            margin = 0.0
        top1_values.append(top1)
        gaps.append(max(0.0, top1 - top2))
        concentrations.append(float(stable_softmax(scores[:5], 0.08)[0]))
        margins.append(margin)
        docs.append(str(row["doc_ids"][0]))
    top1_array = np.asarray(top1_values, dtype=float)
    aggregate = {
        "top1": float(top1_array.mean()),
        "gap": float(np.mean(gaps)),
        "softmax_concentration": float(np.mean(concentrations)),
        "gumbel_margin": float(np.mean(margins)),
        "tail_fraction": float(np.mean(top1_array > 0.80)),
        "doc_concentration": float(max(Counter(docs).values()) / len(docs)),
    }
    weights = {
        "top1": 0.28,
        "gap": 0.15,
        "softmax_concentration": 0.12,
        "gumbel_margin": 0.12,
        "tail_fraction": 0.18,
        "doc_concentration": 0.15,
    }
    clipped = {
        "top1": float(np.clip(aggregate["top1"], 0.0, 1.0)),
        "gap": float(np.clip(aggregate["gap"] / 0.35, 0.0, 1.0)),
        "softmax_concentration": float(
            np.clip(aggregate["softmax_concentration"], 0.0, 1.0)
        ),
        "gumbel_margin": float(
            np.clip(0.5 + aggregate["gumbel_margin"] / 0.30, 0.0, 1.0)
        ),
        "tail_fraction": float(np.clip(aggregate["tail_fraction"], 0.0, 1.0)),
        "doc_concentration": float(
            np.clip(aggregate["doc_concentration"], 0.0, 1.0)
        ),
    }
    return float(sum(weights[name] * clipped[name] for name in weights))


def top1_prefix_score(rows: Sequence[Mapping[str, object]]) -> float:
    return float(np.mean([float(row["top1"]) for row in rows]))


def _windows(
    sessions: Sequence[Sequence[Mapping[str, object]]], width: int = 5
) -> list[Sequence[Mapping[str, object]]]:
    output: list[Sequence[Mapping[str, object]]] = []
    for session in sessions:
        for end in range(width, len(session) + 1):
            output.append(session[end - width : end])
    return output


@dataclass
class BenignCalibration:
    margin_reference: np.ndarray
    early6_reference: dict[int, np.ndarray]
    top1_reference: dict[int, np.ndarray]
    ordered_location: np.ndarray | None
    ordered_precision: np.ndarray | None
    mahalanobis_reference: np.ndarray | None
    cct6_reference: np.ndarray | None
    cct_top1_reference: np.ndarray | None


def fit_benign_calibration(
    sessions: Sequence[Sequence[Mapping[str, object]]],
) -> BenignCalibration:
    """Fit all branch distributions on normal calibration streams only."""
    if not sessions:
        raise ValueError("Normal calibration streams are empty")
    margin_reference = np.sort(np.asarray(
        [float(row["mirabel_margin"]) for session in sessions for row in session],
        dtype=float,
    ))
    early6_reference: dict[int, np.ndarray] = {}
    top1_reference: dict[int, np.ndarray] = {}
    for prefix in range(1, 5):
        eligible = [session for session in sessions if len(session) >= prefix]
        early6_reference[prefix] = np.sort(np.asarray(
            [current_early_score(session[:prefix]) for session in eligible], dtype=float
        ))
        top1_reference[prefix] = np.sort(np.asarray(
            [top1_prefix_score(session[:prefix]) for session in eligible], dtype=float
        ))

    rolling = _windows(sessions, 5)
    if not rolling:
        return BenignCalibration(
            margin_reference,
            early6_reference,
            top1_reference,
            None,
            None,
            None,
            None,
            None,
        )
    ordered = np.asarray(
        [[float(row["top1"]) for row in window] for window in rolling], dtype=float
    )
    covariance = LedoitWolf().fit(ordered)
    distances = covariance.mahalanobis(ordered)
    early6 = np.asarray(
        [current_early_score(window[:4]) for window in rolling], dtype=float
    )
    top1 = np.asarray(
        [top1_prefix_score(window[:4]) for window in rolling], dtype=float
    )
    sorted_distances = np.sort(distances)
    p_m = np.asarray([_upper_p_from_sorted(value, sorted_distances) for value in distances])
    p_e6 = empirical_upper_p(early6, early6_reference[4])
    p_etop = empirical_upper_p(top1, top1_reference[4])
    cct6 = cct_score(np.column_stack([p_e6, p_m]))
    cct_top1 = cct_score(np.column_stack([p_etop, p_m]))
    return BenignCalibration(
        margin_reference,
        early6_reference,
        top1_reference,
        np.asarray(covariance.location_, dtype=float),
        np.asarray(covariance.precision_, dtype=float),
        sorted_distances,
        np.sort(np.asarray(cct6, dtype=float)),
        np.sort(np.asarray(cct_top1, dtype=float)),
    )


def _cct_risks(
    rows: Sequence[Mapping[str, object]], calibration: BenignCalibration
) -> tuple[float, float]:
    if calibration.ordered_location is None or calibration.ordered_precision is None:
        raise ValueError("CCT requested without five-query benign calibration")
    vector = np.asarray([float(row["top1"]) for row in rows], dtype=float)
    delta = vector - calibration.ordered_location
    distance = float(delta @ calibration.ordered_precision @ delta)
    p_m = _upper_p_from_sorted(distance, calibration.mahalanobis_reference)
    early6 = current_early_score(rows[:4])
    top1 = top1_prefix_score(rows[:4])
    p_e6 = _upper_p_from_sorted(early6, calibration.early6_reference[4])
    p_etop = _upper_p_from_sorted(top1, calibration.top1_reference[4])
    raw6 = float(cct_score(np.asarray([[p_e6, p_m]]))[0])
    raw_top1 = float(cct_score(np.asarray([[p_etop, p_m]]))[0])
    risk6 = _cdf_from_sorted(raw6, calibration.cct6_reference)
    risk_top1 = _cdf_from_sorted(raw_top1, calibration.cct_top1_reference)
    return risk6, risk_top1


def policy_risk_curves(
    session: Sequence[Mapping[str, object]],
    calibration: BenignCalibration,
    *,
    reset_every: int | None = None,
) -> dict[str, np.ndarray]:
    """Return cumulative high-is-risky scores at every global query turn.

    ``reset_every`` clears Early/CCT state at each chunk boundary. Mirabel still
    inspects every query independently. The returned curve takes the maximum
    over all chunks, corresponding to detection by any submitted query.
    """
    horizon = len(session)
    if horizon == 0:
        raise ValueError("Cannot score an empty session")
    chunk_size = horizon if reset_every is None else max(1, int(reset_every))
    mirabel = np.zeros(horizon, dtype=float)
    early6 = np.zeros(horizon, dtype=float)
    top1 = np.zeros(horizon, dtype=float)
    cct6 = np.zeros(horizon, dtype=float)
    cct_top1 = np.zeros(horizon, dtype=float)
    for start in range(0, horizon, chunk_size):
        chunk = session[start : start + chunk_size]
        for local_index, row in enumerate(chunk, 1):
            global_index = start + local_index - 1
            mirabel[global_index] = _cdf_from_sorted(
                float(row["mirabel_margin"]), calibration.margin_reference
            )
            if local_index <= 4:
                prefix = chunk[:local_index]
                early6[global_index] = _cdf_from_sorted(
                    current_early_score(prefix),
                    calibration.early6_reference[local_index],
                )
                top1[global_index] = _cdf_from_sorted(
                    top1_prefix_score(prefix),
                    calibration.top1_reference[local_index],
                )
            if local_index >= 5:
                window = chunk[local_index - 5 : local_index]
                cct6[global_index], cct_top1[global_index] = _cct_risks(
                    window, calibration
                )

    instantaneous = {
        "Original Mirabel": mirabel,
        "Early-Mirabel (six features)": early6,
        "Mirabel + Early-Mirabel": np.maximum(mirabel, early6),
        "Mirabel + rolling CCT (six features)": np.maximum(mirabel, cct6),
        "Sequential top1-Early + rolling CCT": np.maximum(
            np.maximum(mirabel, top1), cct_top1
        ),
        "Sequential six-feature Early + rolling CCT": np.maximum(
            np.maximum(mirabel, early6), cct6
        ),
    }
    return {
        policy: np.maximum.accumulate(values)
        for policy, values in instantaneous.items()
    }


def build_cyclic_streams(
    rows: Sequence[Mapping[str, object]],
    horizon: int,
    *,
    seed: int,
) -> list[list[Mapping[str, object]]]:
    """Build deterministic normal stress streams without cross-role reuse.

    Every base query remains within its pre-assigned split role. Queries are
    cyclically reused *within* that role so cumulative FPR can be estimated for
    long horizons; this is a stress stream, not a claim of human conversation.
    """
    if not rows:
        raise ValueError("No normal rows available")
    order = np.arange(len(rows))
    np.random.default_rng(seed).shuffle(order)
    ordered = [rows[index] for index in order]
    return [
        [ordered[(start + offset) % len(ordered)] for offset in range(horizon)]
        for start in range(len(ordered))
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
