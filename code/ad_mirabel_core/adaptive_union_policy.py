"""Attack-agnostic union of sparse-branch and dense-branch evidence.

An equal Bonferroni/min-p component is sensitive when one detector branch is
extreme.  V15's equal Cauchy component gains power when several branches carry
moderate evidence.  This module converts both component scores back to benign
empirical p-values and applies an outer Bonferroni union.

The construction contains no attack-specific weight or threshold.  Its final
score is calibrated on an independent benign Q30 split downstream.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .persistent_branch_policy import BRANCHES
from .robust_fpr_control import empirical_upper_p_matrix, fuse_branch_pvalues


def component_scores(
    matrices: Mapping[str, np.ndarray],
    benign_branch_reference: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return weight-free sparse and dense evidence on aligned horizons."""

    horizon = np.asarray(matrices["Mirabel"], dtype=float).shape[1]
    aligned_reference = {
        branch: np.asarray(benign_branch_reference[branch], dtype=float)[:, :horizon]
        for branch in BRANCHES
    }
    return {
        "sparse_equal_minp": fuse_branch_pvalues(
            matrices,
            aligned_reference,
            method="equal-bonferroni",
            sticky=True,
        ),
        "dense_V15": fuse_branch_pvalues(
            matrices, aligned_reference, method="cct", sticky=True
        ),
    }


def adaptive_union_scores(
    matrices: Mapping[str, np.ndarray],
    benign_branch_reference: Mapping[str, np.ndarray],
    *,
    sticky: bool = True,
) -> np.ndarray:
    """Return outer equal-Bonferroni union risk for sparse and dense evidence.

    One component may be sufficient to raise the score.  The union's actual
    cumulative false-positive rate is not taken from the analytic Bonferroni
    bound; it is fitted and certified with independent benign sessions.
    """

    values = component_scores(
        matrices,
        benign_branch_reference,
    )
    reference = component_scores(
        benign_branch_reference,
        benign_branch_reference,
    )
    p_values = []
    for name in ("sparse_equal_minp", "dense_V15"):
        horizon = values[name].shape[1]
        # The deployment null covers Q30.  A Q1--Q29 request uses the aligned
        # prefix columns without changing the Q30 terminal null or threshold.
        aligned_component_reference = reference[name][:, :horizon]
        p_values.append(
            empirical_upper_p_matrix(values[name], aligned_component_reference)
        )
    combined_p = np.minimum(1.0, 2.0 * np.minimum(p_values[0], p_values[1]))
    risk = -np.log10(np.clip(combined_p, 1e-15, 1.0))
    return np.maximum.accumulate(risk, axis=1) if sticky else risk
