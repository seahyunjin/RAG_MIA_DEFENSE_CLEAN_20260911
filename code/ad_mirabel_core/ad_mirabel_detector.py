"""Session-level Anderson-Darling extension of Mirabel."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Optional
import warnings

import numpy as np
from scipy.stats import anderson_ksamp


class ADMirabelDetector:
    """Delay hard detection until distribution, tail, and focus all look abnormal.

    Original Mirabel is query-level and can overreact to one legitimate similarity
    spike. AD-Mirabel accumulates a rolling session before making a hard decision.
    ``anderson_ksamp`` is a k-sample distribution-equality test, not an ADF/KPSS
    stationarity or normality test. It is useful here because Anderson-Darling
    empirical-distribution tests place relatively strong weight on distribution
    tails, where repeated high-similarity probing appears.
    """

    def __init__(
        self,
        benign_reference_scores: np.ndarray,
        min_queries: int = 5,
        alpha: float = 0.05,
        tail_threshold: float = 0.8,
        tail_mass_threshold: float = 0.6,
        concentration_threshold: float = 0.6,
        window_size: int = 10,
        use_entailment: bool = False,
        lambda_sim: float = 1.0,
        lambda_ent: float = 0.0,
    ) -> None:
        reference = np.asarray(benign_reference_scores, dtype=float)
        if reference.size < 2:
            raise ValueError("benign_reference_scores needs at least two values")
        if min_queries > window_size:
            raise ValueError("min_queries cannot exceed window_size")
        self.benign_reference_scores = reference
        self.min_queries = min_queries
        self.alpha = alpha
        self.tail_threshold = tail_threshold
        self.tail_mass_threshold = tail_mass_threshold
        self.concentration_threshold = concentration_threshold
        self.window_size = window_size
        self.use_entailment = use_entailment
        self.lambda_sim = lambda_sim
        self.lambda_ent = lambda_ent
        self.sessions: dict[str, dict[str, list]] = defaultdict(
            lambda: {
                "scores": [], "top_similarities": [], "top_doc_ids": [],
                "timestamps": [], "queries": [],
            }
        )

    def _ad_test(self, scores: list[float]) -> tuple[float, float]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = anderson_ksamp([np.asarray(scores), self.benign_reference_scores])
            statistic = float(result.statistic)
            p_value = float(getattr(result, "pvalue", result.significance_level))
            # Older SciPy reported significance_level as percent.
            if not hasattr(result, "pvalue") and p_value > 1.0:
                p_value /= 100.0
            return statistic, p_value
        except Exception:
            return 0.0, 1.0

    def update(
        self,
        session_id: str,
        query: str,
        sorted_scores: Sequence[float],
        sorted_doc_ids: Sequence[str],
        timestamp: Optional[Any] = None,
        entailment_score: Optional[float] = None,
    ) -> dict:
        if not sorted_scores or not sorted_doc_ids:
            raise ValueError("At least one score and document ID are required.")
        top_similarity = float(sorted_scores[0])
        top_doc_id = str(sorted_doc_ids[0])
        if self.use_entailment and entailment_score is not None:
            risk_score = self.lambda_sim * top_similarity + self.lambda_ent * entailment_score
        else:
            risk_score = top_similarity

        buffer = self.sessions[session_id]
        for key, value in (
            ("scores", risk_score), ("top_similarities", top_similarity),
            ("top_doc_ids", top_doc_id), ("timestamps", timestamp), ("queries", query),
        ):
            buffer[key].append(value)
            if len(buffer[key]) > self.window_size:
                del buffer[key][0]

        scores = buffer["scores"]
        doc_counts = Counter(buffer["top_doc_ids"])
        mode_doc, mode_count = doc_counts.most_common(1)[0]
        tail_mass = float(np.mean(np.asarray(scores) > self.tail_threshold))
        concentration = mode_count / len(scores)
        ad_statistic, p_value = (0.0, 1.0)
        decision = "watch"
        if len(scores) >= self.min_queries:
            ad_statistic, p_value = self._ad_test(scores)
            decision = (
                "suspicious"
                if p_value < self.alpha
                and tail_mass > self.tail_mass_threshold
                and concentration > self.concentration_threshold
                else "benign"
            )
        return {
            "decision": decision,
            "ad_statistic": ad_statistic,
            "p_value": p_value,
            "tail_mass": tail_mass,
            "max_doc_concentration": float(concentration),
            "num_queries": len(scores),
            "top_doc_id_mode": mode_doc,
            "latest_top_similarity": top_similarity,
            "latest_top_doc_id": top_doc_id,
            "latest_risk_score": float(risk_score),
            "session_id": session_id,
        }

    def get_session_buffer(self, session_id: str) -> dict[str, list]:
        """Return a copy for diagnostics and unit tests."""
        return {key: list(value) for key, value in self.sessions[session_id].items()}
