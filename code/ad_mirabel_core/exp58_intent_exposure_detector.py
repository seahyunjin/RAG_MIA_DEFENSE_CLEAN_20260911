"""Pure components for the Exp58 two-head intent/exposure detector.

The public inference object deliberately returns two independent scalar outputs.
It contains no learned or manually weighted fusion policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


FORBIDDEN_INFERENCE_FIELDS = frozenset({
    "attack_family", "member_label", "domain", "domain_id", "retriever",
    "retriever_id", "query_budget", "native_budget", "session_id",
})


@dataclass(frozen=True)
class FrozenQueryEncoderSpec:
    model_id: str
    revision: str
    tokenizer_revision: str
    embedding_dimension: int
    pooling: str = "mean"
    normalization: str = "l2"
    trainable: bool = False


@dataclass(frozen=True)
class LinearHead:
    coefficient: np.ndarray
    intercept: float = 0.0

    def __post_init__(self) -> None:
        coefficient = np.asarray(self.coefficient, dtype=float)
        if coefficient.ndim != 1 or not np.isfinite(coefficient).all():
            raise ValueError("coefficient must be a finite vector")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        object.__setattr__(self, "coefficient", coefficient)

    @property
    def hidden_layers(self) -> int:
        return 0

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.shape[-1] != len(self.coefficient) or not np.isfinite(values).all():
            raise ValueError("finite features with the trained dimension are required")
        return values @ self.coefficient + self.intercept


@dataclass(frozen=True)
class TwoHeadOutput:
    intent_score: np.ndarray
    exposure_score: np.ndarray


class TwoHeadDetector:
    """Two separate linear heads; policy/fusion is intentionally absent."""

    def __init__(self, intent: LinearHead, exposure: LinearHead) -> None:
        self.intent = intent
        self.exposure = exposure

    def score(self, query_embedding: np.ndarray, retrieval_features: np.ndarray,
              metadata: Mapping[str, object] | None = None) -> TwoHeadOutput:
        validate_inference_metadata(metadata or {})
        return TwoHeadOutput(
            intent_score=self.intent.score(query_embedding),
            exposure_score=self.exposure.score(retrieval_features),
        )


def validate_inference_metadata(metadata: Mapping[str, object]) -> None:
    present = FORBIDDEN_INFERENCE_FIELDS.intersection(metadata)
    if present:
        raise ValueError(f"forbidden inference metadata: {sorted(present)}")


def sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _balanced_weights(label: np.ndarray) -> np.ndarray:
    y = np.asarray(label, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both binary classes are required")
    counts = np.bincount(y, minlength=2).astype(float)
    return np.asarray([len(y) / (2.0 * counts[item]) for item in y], dtype=float)


def fit_erm(features: np.ndarray, label: np.ndarray, regularization: float = 1.0,
            sample_weight: np.ndarray | None = None, offset: np.ndarray | None = None,
            max_iter: int = 100) -> LinearHead:
    """L2-regularized linear BCE, optionally as a residual around an offset."""
    x, y = np.asarray(features, float), np.asarray(label, float)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("invalid training arrays")
    weights = _balanced_weights(y.astype(int)) if sample_weight is None else np.asarray(sample_weight, float)
    weights = weights / max(weights.mean(), 1e-12)
    base = np.zeros(len(y), float) if offset is None else np.asarray(offset, float)
    if base.shape != y.shape or not np.isfinite(base).all():
        raise ValueError("invalid offset")

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        coefficient, intercept = parameter[:-1], parameter[-1]
        score = base + x @ coefficient + intercept
        loss = np.logaddexp(0.0, score) - y * score
        probability = sigmoid(score)
        value = float(np.mean(weights * loss) + .5 * regularization * (coefficient @ coefficient))
        gradient = np.r_[x.T @ (weights * (probability - y)) / len(y) + regularization * coefficient,
                         np.mean(weights * (probability - y))]
        return value, gradient

    result = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B",
                      options={"maxiter": int(max_iter), "ftol": 1e-9, "gtol": 1e-6})
    if not result.success and not np.isfinite(result.x).all():
        raise RuntimeError(f"linear BCE failed: {result.message}")
    return LinearHead(result.x[:-1], float(result.x[-1]))


def fit_group_dro(features: np.ndarray, label: np.ndarray, groups: Sequence[str],
                  regularization: float = 1.0, rounds: int = 8,
                  eta: float = .25, offset: np.ndarray | None = None) -> tuple[LinearHead, dict[str, float]]:
    """Deterministic iteratively reweighted GroupDRO linear BCE."""
    x, y, g = np.asarray(features, float), np.asarray(label, int), np.asarray(groups, str)
    names = sorted(np.unique(g).tolist())
    if len(g) != len(y) or not names:
        raise ValueError("groups must align with labels")
    group_weight = np.ones(len(names), float) / len(names)
    head = LinearHead(np.zeros(x.shape[1]), 0.0)
    base = np.zeros(len(y), float) if offset is None else np.asarray(offset, float)
    for _ in range(rounds):
        sample = np.zeros(len(y), float)
        for index, name in enumerate(names):
            mask = g == name
            sample[mask] = group_weight[index] / max(mask.sum(), 1)
        sample *= len(sample) / max(sample.sum(), 1e-12)
        head = fit_erm(x, y, regularization, sample_weight=sample, offset=base, max_iter=60)
        losses = []
        score = base + head.score(x)
        raw = np.logaddexp(0.0, score) - y * score
        for name in names:
            losses.append(float(raw[g == name].mean()))
        group_weight *= np.exp(eta * np.asarray(losses))
        group_weight /= group_weight.sum()
    return head, {name: float(group_weight[index]) for index, name in enumerate(names)}


def fit_pairwise(attack_features: np.ndarray, hard_normal_features: np.ndarray,
                 regularization: float = 1.0, offset_difference: np.ndarray | None = None,
                 max_iter: int = 100) -> LinearHead:
    """Linear pairwise logistic ranking on pre-matched attack/normal pairs."""
    attack, normal = np.asarray(attack_features, float), np.asarray(hard_normal_features, float)
    if attack.shape != normal.shape or attack.ndim != 2 or not len(attack):
        raise ValueError("nonempty aligned pairs are required")
    difference = attack - normal
    x = np.vstack([difference, -difference])
    y = np.r_[np.ones(len(difference), int), np.zeros(len(difference), int)]
    if offset_difference is None:
        offset = None
    else:
        delta = np.asarray(offset_difference, float)
        if delta.shape != (len(difference),):
            raise ValueError("offset differences must align with pairs")
        offset = np.r_[delta, -delta]
    head = fit_erm(x, y, regularization, offset=offset, max_iter=max_iter)
    return LinearHead(head.coefficient, 0.0)


def empirical_upper_tail(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x, ref = np.asarray(values, float), np.sort(np.asarray(reference, float))
    if not len(ref) or not np.isfinite(x).all() or not np.isfinite(ref).all():
        raise ValueError("finite nonempty reference required")
    return (len(ref) - np.searchsorted(ref, x, side="left") + 1.0) / (len(ref) + 1.0)


def threshold_higher(values: np.ndarray, target_fpr: float = .01) -> float:
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("finite calibration scores required")
    return float(np.quantile(x, 1.0 - target_fpr, method="higher"))


def family_macro_metrics(family: Sequence[str], member: Sequence[int], detected: Sequence[bool]) -> dict[str, float]:
    f, m, d = np.asarray(family, str), np.asarray(member, int), np.asarray(detected, bool)
    family_adr, gaps, member_rates, nonmember_rates = [], [], [], []
    for name in sorted(np.unique(f)):
        rates = {}
        for label in (0, 1):
            mask = (f == name) & (m == label)
            if mask.any(): rates[label] = float(d[mask].mean())
        if rates:
            family_adr.append(float(np.mean(list(rates.values()))))
        if 0 in rates and 1 in rates:
            gaps.append(abs(rates[1] - rates[0])); member_rates.append(rates[1]); nonmember_rates.append(rates[0])
    return {
        "fm_adr": float(np.mean(family_adr)) if family_adr else float("nan"),
        "member_tpr": float(np.mean(member_rates)) if member_rates else float("nan"),
        "nonmember_tpr": float(np.mean(nonmember_rates)) if nonmember_rates else float("nan"),
        "member_nonmember_gap": float(np.mean(gaps)) if gaps else float("nan"),
    }


def quadrant(intent_high: np.ndarray, exposure_high: np.ndarray) -> np.ndarray:
    intent, exposure = np.asarray(intent_high, bool), np.asarray(exposure_high, bool)
    if intent.shape != exposure.shape:
        raise ValueError("head decisions must align")
    return np.select(
        [~intent & ~exposure, intent & ~exposure, ~intent & exposure, intent & exposure],
        ["intent_low_exposure_low", "intent_high_exposure_low",
         "intent_low_exposure_high", "intent_high_exposure_high"],
        default="invalid",
    )


def minp_curve(pvalues: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(pvalues, float), 1e-15, 1.0)
    return np.minimum.accumulate(p)


def cct_curve(pvalues: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(pvalues, float), 1e-15, 1.0 - 1e-15)
    transformed = np.tan(np.pi * (.5 - p))
    mean = np.cumsum(transformed) / np.arange(1, len(p) + 1)
    return np.clip(.5 - np.arctan(mean) / np.pi, 0.0, 1.0)

