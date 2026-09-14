"""Frozen semantic response representation without hand-written security rules."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def relation_features(query: np.ndarray, response: np.ndarray, context: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=float); r = np.asarray(response, dtype=float); c = np.asarray(context, dtype=float)
    if q.shape != r.shape or r.shape != c.shape or q.ndim != 2:
        raise ValueError("query, response, and context embeddings must be aligned N x D matrices")
    output = np.concatenate((q, r, c, q * r, np.abs(q - r), r * c, np.abs(r - c)), axis=1)
    if not np.isfinite(output).all():
        raise ValueError("response relation features must be finite")
    return output


def permutation_invariant_pool(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("pool input must be a finite nonempty N x D matrix")
    return np.concatenate((values.mean(0), values.min(0), values.max(0), values.std(0, ddof=0)))


class FrozenTextEncoder:
    """Lazy local-only SentenceTransformer wrapper used by cohort scripts."""

    def __init__(self, model_path: str, *, device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer
        self.model_path = str(model_path)
        self.model = SentenceTransformer(self.model_path, device=device, local_files_only=True)

    def encode(self, texts: Sequence[str], *, batch_size: int = 128) -> np.ndarray:
        return np.asarray(self.model.encode(
            list(texts), batch_size=int(batch_size), convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float32)
