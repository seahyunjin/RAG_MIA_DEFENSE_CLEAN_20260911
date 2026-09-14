"""Heuristic-free Early-Mirabel feature calibration and rolling CCT.

The original Early-Mirabel used hand-set clipping constants, a fixed 0.80
high-score indicator, a softmax temperature, and six hand-set weights.  This
module keeps six interpretable retrieval signals but removes those constants:

* raw prefix statistics are retained without manual clipping;
* every signal is converted to a prefix-specific empirical upper-tail p-value
  using benign calibration sessions only;
* the primary Early score is an equal-weight Fisher combination of those
  p-values, so a tied p-value of one contributes zero evidence instead of
  cancelling another feature's small p-value;
* optional nonnegative weights are accepted only for explicitly labelled
  supervised diagnostic baselines.

No model/API calls occur here.  Inputs are cached retrieval rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf


FEATURES = (
    "mean_top1_similarity",
    "mean_top1_top2_gap",
    "mean_standardized_retrieval_concentration",
    "mean_gumbel_margin",
    "max_top1_similarity",
    "same_document_concentration",
)


def _scores(row: Mapping[str, object]) -> np.ndarray:
    value = row.get("scores", row.get("top_k_similarities"))
    if value is None:
        raise KeyError("retrieval row is missing top-k similarities")
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or len(result) < 2 or not np.all(np.isfinite(result)):
        raise ValueError("top-k similarities must be a finite one-dimensional array")
    return result


def _top1(row: Mapping[str, object]) -> float:
    for key in ("top1", "top1_similarity", "top1_score"):
        if key in row:
            return float(row[key])
    return float(_scores(row)[0])


def _margin(row: Mapping[str, object]) -> float:
    for key in ("mirabel_margin", "gumbel_margin"):
        if key in row:
            return float(row[key])
    raise KeyError("retrieval row is missing the Mirabel/Gumbel margin")


def _top_document(row: Mapping[str, object]) -> str:
    for key in ("doc_ids", "top_k_document_ids"):
        if key in row:
            values = row[key]
            if isinstance(values, Sequence) and len(values):
                return str(values[0])
    if "top1_document_id" in row:
        return str(row["top1_document_id"])
    raise KeyError("retrieval row is missing top document IDs")


def aggregate_prefix(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Compute six raw, unclipped prefix statistics with no tuned constants."""
    if not rows:
        raise ValueError("A prefix requires at least one retrieval row")
    top1_values = []
    gaps = []
    concentrations = []
    margins = []
    documents = []
    for row in rows:
        scores = _scores(row)
        top1 = _top1(row)
        background = scores[1:]
        # Scale-free concentration: how far top-1 lies above the retrieved
        # background in units of that query's own background spread.  The only
        # constant is floating-point protection, not a tuned temperature.
        spread = float(background.std(ddof=0))
        concentration = (top1 - float(background.mean())) / max(
            spread, np.finfo(float).eps
        )
        top1_values.append(top1)
        gaps.append(top1 - float(scores[1]))
        concentrations.append(concentration)
        margins.append(_margin(row))
        documents.append(_top_document(row))
    top1_array = np.asarray(top1_values, dtype=float)
    return {
        "mean_top1_similarity": float(top1_array.mean()),
        "mean_top1_top2_gap": float(np.mean(gaps)),
        "mean_standardized_retrieval_concentration": float(
            np.mean(concentrations)
        ),
        "mean_gumbel_margin": float(np.mean(margins)),
        # Replaces the hand-set fraction(top1 > 0.80) with an unthresholded
        # order statistic. Benign empirical calibration supplies its scale.
        "max_top1_similarity": float(top1_array.max()),
        "same_document_concentration": float(
            max(Counter(documents).values()) / len(documents)
        ),
    }


def empirical_upper_p(value: float, ordered_reference: np.ndarray) -> float:
    reference = np.asarray(ordered_reference, dtype=float)
    if reference.ndim != 1 or not len(reference):
        raise ValueError("empirical p-value reference cannot be empty")
    ge = len(reference) - int(np.searchsorted(reference, value, side="left"))
    return float((ge + 1.0) / (len(reference) + 1.0))


def empirical_risk_percentile(value: float, ordered_reference: np.ndarray) -> float:
    reference = np.asarray(ordered_reference, dtype=float)
    if reference.ndim != 1 or not len(reference):
        raise ValueError("empirical risk reference cannot be empty")
    return float(
        np.searchsorted(reference, value, side="right") / (len(reference) + 1.0)
    )


def weighted_cct_score(
    p_values: Sequence[float], weights: Sequence[float] | None = None
) -> float:
    p = np.clip(np.asarray(p_values, dtype=float), 1e-12, 1.0 - 1e-12)
    if not len(p):
        raise ValueError("CCT requires at least one p-value")
    if weights is None:
        w = np.full(len(p), 1.0 / len(p), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != p.shape or np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("CCT weights must be finite, nonnegative, and aligned")
        if float(w.sum()) <= 0:
            raise ValueError("CCT weights must have positive total mass")
        w = w / w.sum()
    statistic = float(np.sum(w * np.tan((0.5 - p) * np.pi)))
    combined_p = 0.5 - math.atan(statistic) / math.pi
    return float(-math.log10(max(combined_p, 1e-15)))


def weighted_fisher_score(
    p_values: Sequence[float], weights: Sequence[float] | None = None
) -> float:
    """Equal/data-weighted Fisher evidence, empirically calibrated downstream."""
    p = np.clip(np.asarray(p_values, dtype=float), 1e-12, 1.0)
    if not len(p):
        raise ValueError("Fisher combination requires at least one p-value")
    if weights is None:
        w = np.full(len(p), 1.0 / len(p), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != p.shape or np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError("Fisher weights must be finite, nonnegative, and aligned")
        if float(w.sum()) <= 0:
            raise ValueError("Fisher weights must have positive total mass")
        w = w / w.sum()
    return float(-2.0 * np.sum(w * np.log(p)))


@dataclass
class HeuristicFreeCalibration:
    features: tuple[str, ...]
    weights: np.ndarray
    feature_references: dict[int, dict[str, np.ndarray]]
    early_score_references: dict[int, np.ndarray]
    mirabel_margin_reference: np.ndarray
    ordered_location: np.ndarray | None
    ordered_precision: np.ndarray | None
    ordered_distance_reference: np.ndarray | None
    rolling_cct_reference: np.ndarray | None
    early_combiner: str = "fisher"
    # Prefix-specific, permutation-invariant CCT calibration.  The singular
    # fields above are retained as the Q5 compatibility view for earlier
    # experiment readers.
    set_locations: dict[int, np.ndarray] = field(default_factory=dict)
    set_precisions: dict[int, np.ndarray] = field(default_factory=dict)
    set_distance_references: dict[int, np.ndarray] = field(default_factory=dict)
    set_cct_references: dict[int, np.ndarray] = field(default_factory=dict)


def _early_raw_score(
    rows: Sequence[Mapping[str, object]],
    calibration: HeuristicFreeCalibration,
    prefix: int,
) -> float:
    aggregate = aggregate_prefix(rows)
    p_values = [
        empirical_upper_p(
            aggregate[feature], calibration.feature_references[prefix][feature]
        )
        for feature in calibration.features
    ]
    if calibration.early_combiner == "fisher":
        return weighted_fisher_score(p_values, calibration.weights)
    if calibration.early_combiner == "cct":
        return weighted_cct_score(p_values, calibration.weights)
    raise KeyError(calibration.early_combiner)


def feature_p_values(
    rows: Sequence[Mapping[str, object]],
    calibration: HeuristicFreeCalibration,
) -> np.ndarray:
    """Return the benign-calibrated p-value vector for one 1--4 prefix."""
    prefix = len(rows)
    if prefix not in calibration.feature_references:
        raise ValueError("feature p-values support prefix lengths 1 through 4")
    aggregate = aggregate_prefix(rows)
    return np.asarray(
        [
            empirical_upper_p(
                aggregate[feature],
                calibration.feature_references[prefix][feature],
            )
            for feature in calibration.features
        ],
        dtype=float,
    )


def _windows(
    sessions: Sequence[Sequence[Mapping[str, object]]], width: int = 5
) -> list[Sequence[Mapping[str, object]]]:
    return [
        session[end - width : end]
        for session in sessions
        for end in range(width, len(session) + 1)
    ]


def fit_calibration(
    sessions: Sequence[Sequence[Mapping[str, object]]],
    *,
    features: Sequence[str] = FEATURES,
    weights: Sequence[float] | None = None,
    early_combiner: str = "fisher",
) -> HeuristicFreeCalibration:
    """Fit all scales and null distributions from benign sessions only."""
    selected = tuple(features)
    if not selected or any(feature not in FEATURES for feature in selected):
        raise ValueError("features must be a non-empty subset of FEATURES")
    if weights is None:
        normalized_weights = np.full(len(selected), 1.0 / len(selected))
    else:
        normalized_weights = np.asarray(weights, dtype=float)
        if normalized_weights.shape != (len(selected),) or np.any(
            normalized_weights < 0
        ):
            raise ValueError("weights must align with selected features")
        normalized_weights = normalized_weights / normalized_weights.sum()
    if not sessions:
        raise ValueError("benign calibration sessions cannot be empty")
    if early_combiner not in {"fisher", "cct"}:
        raise KeyError(early_combiner)

    feature_references: dict[int, dict[str, np.ndarray]] = {}
    aggregates_by_prefix: dict[int, list[dict[str, float]]] = {}
    max_prefix = max(len(session) for session in sessions)
    for prefix in range(1, max_prefix + 1):
        aggregates = [
            aggregate_prefix(session[:prefix])
            for session in sessions
            if len(session) >= prefix
        ]
        if not aggregates:
            raise ValueError(f"no benign sessions reach prefix {prefix}")
        aggregates_by_prefix[prefix] = aggregates
        feature_references[prefix] = {
            feature: np.sort(
                np.asarray([row[feature] for row in aggregates], dtype=float)
            )
            for feature in selected
        }

    placeholder = HeuristicFreeCalibration(
        features=selected,
        weights=normalized_weights,
        feature_references=feature_references,
        early_score_references={},
        mirabel_margin_reference=np.sort(
            np.asarray(
                [_margin(row) for session in sessions for row in session], dtype=float
            )
        ),
        ordered_location=None,
        ordered_precision=None,
        ordered_distance_reference=None,
        rolling_cct_reference=None,
        early_combiner=early_combiner,
    )
    early_references = {}
    for prefix, aggregates in aggregates_by_prefix.items():
        values = []
        for aggregate in aggregates:
            p_values = [
                empirical_upper_p(
                    aggregate[feature], feature_references[prefix][feature]
                )
                for feature in selected
            ]
            values.append(
                weighted_fisher_score(p_values, normalized_weights)
                if early_combiner == "fisher"
                else weighted_cct_score(p_values, normalized_weights)
            )
        early_references[prefix] = np.sort(np.asarray(values, dtype=float))
    placeholder.early_score_references = early_references

    if max_prefix < 5:
        return placeholder
    for prefix in range(5, max_prefix + 1):
        eligible = [session[:prefix] for session in sessions if len(session) >= prefix]
        if not eligible:
            continue
        # Sorting turns the top-1 sequence into order statistics.  The vector,
        # the all-prefix Early aggregate, and hence the CCT score are exactly
        # invariant to any permutation of the same prefix queries.
        order_statistics = np.asarray(
            [np.sort([_top1(row) for row in window]) for window in eligible],
            dtype=float,
        )
        covariance = LedoitWolf().fit(order_statistics)
        raw_distances = covariance.mahalanobis(order_statistics)
        distances = np.sort(raw_distances)
        cct_raw = []
        for window, distance in zip(eligible, raw_distances):
            early_raw = _early_raw_score(window, placeholder, prefix)
            p_early = empirical_upper_p(early_raw, early_references[prefix])
            p_set_distance = empirical_upper_p(float(distance), distances)
            cct_raw.append(weighted_cct_score((p_early, p_set_distance)))
        placeholder.set_locations[prefix] = np.asarray(
            covariance.location_, dtype=float
        )
        placeholder.set_precisions[prefix] = np.asarray(
            covariance.precision_, dtype=float
        )
        placeholder.set_distance_references[prefix] = distances
        placeholder.set_cct_references[prefix] = np.sort(
            np.asarray(cct_raw, dtype=float)
        )
    # Compatibility view for consumers that only know the historical Q5
    # rolling fields.  New scoring below always uses the prefix dictionaries.
    placeholder.ordered_location = placeholder.set_locations.get(5)
    placeholder.ordered_precision = placeholder.set_precisions.get(5)
    placeholder.ordered_distance_reference = placeholder.set_distance_references.get(5)
    placeholder.rolling_cct_reference = placeholder.set_cct_references.get(5)
    return placeholder


def early_risk(
    rows: Sequence[Mapping[str, object]], calibration: HeuristicFreeCalibration
) -> float:
    prefix = len(rows)
    if prefix not in calibration.early_score_references:
        raise ValueError("Early risk supports prefix lengths 1 through 4")
    raw = _early_raw_score(rows, calibration, prefix)
    return empirical_risk_percentile(raw, calibration.early_score_references[prefix])


def rolling_cct_risk(
    window: Sequence[Mapping[str, object]], calibration: HeuristicFreeCalibration
) -> float:
    prefix = len(window)
    if prefix < 5:
        raise ValueError("CCT requires at least five queries")
    if prefix not in calibration.set_locations:
        raise ValueError(
            f"CCT prefix Q{prefix} was not calibrated; fit benign sessions to that horizon"
        )
    order_statistics = np.sort(
        np.asarray([_top1(row) for row in window], dtype=float)
    )
    delta = order_statistics - calibration.set_locations[prefix]
    distance = float(delta @ calibration.set_precisions[prefix] @ delta)
    p_ordered = empirical_upper_p(
        distance, calibration.set_distance_references[prefix]
    )
    early_raw = _early_raw_score(window, calibration, prefix)
    p_early = empirical_upper_p(
        early_raw, calibration.early_score_references[prefix]
    )
    raw = weighted_cct_score((p_early, p_ordered))
    return empirical_risk_percentile(raw, calibration.set_cct_references[prefix])


def policy_risk_curves(
    session: Sequence[Mapping[str, object]],
    calibration: HeuristicFreeCalibration,
    *,
    reset_every: int | None = None,
) -> dict[str, np.ndarray]:
    """Return cumulative risk curves for heuristic-free deployment policies."""
    branches = branch_risk_curves(
        session, calibration, reset_every=reset_every
    )
    mirabel = branches["Mirabel"]
    early = branches["Early"]
    rolling = branches["Rolling CCT"]
    instantaneous = {
        "Original Mirabel": mirabel,
        "Heuristic-free Early (equal Fisher6)": early,
        "Mirabel + heuristic-free Early": np.maximum(mirabel, early),
        "Mirabel + heuristic-free rolling CCT": np.maximum(mirabel, rolling),
        "Heuristic-free Sequential Early-to-CCT": np.maximum(
            np.maximum(mirabel, early), rolling
        ),
    }
    return {
        policy: np.maximum.accumulate(values)
        for policy, values in instantaneous.items()
    }


def branch_risk_curves(
    session: Sequence[Mapping[str, object]],
    calibration: HeuristicFreeCalibration,
    *,
    reset_every: int | None = None,
) -> dict[str, np.ndarray]:
    """Return separately calibrated Mirabel, Early, and rolling-CCT risks.

    Keeping these branches separate is necessary for auditable branch-specific
    false-positive budgets.  A client-side reset can be represented by passing
    ``reset_every``; a persistent server policy instead calls this function on
    the complete linked history with ``reset_every=None``.
    """
    if not session:
        raise ValueError("session cannot be empty")
    horizon = len(session)
    chunk_size = horizon if reset_every is None else max(1, int(reset_every))
    mirabel = np.zeros(horizon, dtype=float)
    early = np.zeros(horizon, dtype=float)
    rolling = np.zeros(horizon, dtype=float)
    for start in range(0, horizon, chunk_size):
        chunk = session[start : start + chunk_size]
        for local_turn, row in enumerate(chunk, 1):
            global_index = start + local_turn - 1
            mirabel[global_index] = empirical_risk_percentile(
                _margin(row), calibration.mirabel_margin_reference
            )
            if local_turn <= 4:
                early[global_index] = early_risk(
                    chunk[:local_turn], calibration
                )
            if local_turn >= 5:
                rolling[global_index] = rolling_cct_risk(
                    chunk[:local_turn], calibration
                )
    return {
        "Mirabel": np.maximum.accumulate(mirabel),
        "Early": np.maximum.accumulate(early),
        "Rolling CCT": np.maximum.accumulate(rolling),
    }
