"""Order-invariant session surrogates used by Experiment 39.

The production detector remains the calibrated Adaptive Fisher detector.  The
functions in this module define only the differentiable training objective.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional


def softplus_evidence(logits: np.ndarray) -> np.ndarray:
    """Convert signed query logits to finite non-negative evidence."""

    values = np.asarray(logits, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("query logits must be finite")
    return np.logaddexp(0.0, values)


def session_surrogate(logits: Sequence[float]) -> float:
    """G_theta(S) = sum softplus(z_q), invariant to query order."""

    values = np.asarray(logits, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("a session subset must contain at least one query")
    return float(softplus_evidence(values).sum())


def grouped_session_surrogate(
    logits: torch.Tensor, group_index: torch.Tensor, group_count: int
) -> torch.Tensor:
    """Vectorized session surrogate for flattened query logits."""

    if logits.ndim != 1 or group_index.ndim != 1 or len(logits) != len(group_index):
        raise ValueError("logits and group_index must be aligned vectors")
    if group_count < 1:
        raise ValueError("group_count must be positive")
    output = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    output.scatter_add_(0, group_index, functional.softplus(logits))
    return output


def session_pair_loss(
    positive_logits: torch.Tensor,
    positive_group: torch.Tensor,
    negative_logits: torch.Tensor,
    negative_group: torch.Tensor,
    pair_count: int,
) -> torch.Tensor:
    """Mean log(1 + exp(G(S-) - G(S+))) over matched subset pairs."""

    positive = grouped_session_surrogate(positive_logits, positive_group, pair_count)
    negative = grouped_session_surrogate(negative_logits, negative_group, pair_count)
    return functional.softplus(negative - positive).mean()


def permutation_invariant_pool(matrix: np.ndarray) -> np.ndarray:
    """Mean/min/max/std pooling used only by response diagnostics."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("pool input must be a finite nonempty matrix")
    return np.concatenate(
        (values.mean(axis=0), values.min(axis=0), values.max(axis=0), values.std(axis=0))
    )
