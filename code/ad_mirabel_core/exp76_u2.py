"""Exact query-only linear scorer used by the Exp75 U2 baseline and Exp76."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class U2Artifact:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    seed: int

    @property
    def trainable_parameters(self) -> int:
        return int(self.coefficient.size + 1)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        # Exp75 supplied float32 embeddings to StandardScaler.  Reproduce its
        # transform dtype exactly instead of silently promoting to float64.
        values = np.asarray(embeddings, dtype=np.float32)
        standardized = (values - self.mean.astype(np.float32)) / self.scale.astype(np.float32)
        return standardized @ self.coefficient + self.intercept

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, mean=self.mean, scale=self.scale,
                            coefficient=self.coefficient, intercept=np.asarray([self.intercept]),
                            seed=np.asarray([self.seed], dtype=np.int64))

    @classmethod
    def load(cls, path: str | Path) -> "U2Artifact":
        value = np.load(path)
        return cls(value["mean"], value["scale"], value["coefficient"],
                   float(value["intercept"][0]), int(value["seed"][0]))


def fit_u2(embeddings: np.ndarray, labels: Sequence[int], seed: int) -> U2Artifact:
    values = np.asarray(embeddings, dtype=np.float32)
    target = np.asarray(labels, dtype=int)
    scaler = StandardScaler().fit(values)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(seed), C=1.0)
    model.fit(scaler.transform(values), target)
    return U2Artifact(scaler.mean_.copy(), scaler.scale_.copy(), model.coef_[0].copy(),
                      float(model.intercept_[0]), int(seed))


def threshold_higher(scores: Sequence[float], target_fpr: float = 0.02) -> float:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional scores required")
    return float(np.quantile(values, 1 - float(target_fpr), method="higher"))


def empirical_upper_tail_p(scores: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    ref = np.sort(np.asarray(reference, dtype=float))
    if not len(ref):
        raise ValueError("reference scores are empty")
    return (len(ref) - np.searchsorted(ref, values, side="left") + 1.0) / (len(ref) + 1.0)


def artifact_sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def artifact_spec(artifact: U2Artifact, path: str | Path) -> dict[str, object]:
    return {
        "artifact": str(path),
        "sha256": artifact_sha256(path),
        "embedding_dimension": int(artifact.coefficient.size),
        "linear_trainable_parameters": artifact.trainable_parameters,
        "seed": artifact.seed,
        "preprocessing": "StandardScaler fitted on development embeddings",
        "head": "sklearn LogisticRegression, L2, C=1.0, class_weight=balanced, lbfgs, max_iter=2000",
    }
