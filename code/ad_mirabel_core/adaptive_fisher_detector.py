"""Streaming, order-invariant Adaptive Fisher detector.

The detector consumes retrieval scores and full-corpus moments, never raw query,
document, or response text.  One calibrated path handles both a single strong
query and several weaker queries through the all-k adaptive Fisher maximum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import math
from typing import Any

import numpy as np

from .canonical_mirabel import (
    MIRABEL_FORMULA_VERSION,
    canonical_cache_fields,
    canonical_mirabel_from_moments,
    require_canonical_cache_row,
)
from .detector_calibration import (
    AdaptiveFisherCalibration,
    adaptive_score,
    empirical_risk,
    score_query_set,
)
from .detector_state_store import InMemoryStateStore, SessionState


@dataclass(frozen=True)
class AdaptiveFisherObservation:
    session_id: str
    query_count: int
    horizon: int
    query_p_value: float
    selected_k: int
    partial_fisher_statistic: float
    adaptive_score: float
    calibrated_risk: float
    threshold: float
    blocked: bool
    first_detection_turn: int | None
    model_version: str
    threshold_version: str
    mirabel_formula_version: str
    threshold_exceeded: bool = False
    action: str = "shadow"
    duplicate_request: bool = False


class AdaptiveFisherDetector:
    MODES = {"shadow", "alert", "block"}

    def __init__(
        self,
        calibration: AdaptiveFisherCalibration,
        *,
        horizon: int = 5,
        state_store: object | None = None,
        mode: str = "shadow",
        request_id_retention: int = 128,
        log_salt: str = "change-this-log-salt",
        reject_unsorted_scores: bool = True,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported detector mode: {mode}")
        if int(request_id_retention) < 1:
            raise ValueError("request_id_retention must be positive")
        calibration.for_horizon(int(horizon))
        self.calibration = calibration
        self.horizon = int(horizon)
        self.state_store = state_store or InMemoryStateStore()
        self.mode = mode
        self.request_id_retention = int(request_id_retention)
        self.log_salt = str(log_salt).encode("utf-8")
        self.reject_unsorted_scores = bool(reject_unsorted_scores)

    def _canonical_row(self, retrieval_result: Mapping[str, object]) -> dict[str, object]:
        source = dict(retrieval_result)
        self._validate_versions(source)
        if "top_scores" in source:
            scores = np.asarray(source["top_scores"], dtype=float)
        else:
            scores = np.asarray(source.get("scores", ()), dtype=float)
        if scores.ndim != 1 or len(scores) < self.calibration.query_model.common_top_k:
            raise ValueError("at least the calibrated number of top scores is required")
        if not np.isfinite(scores).all():
            raise ValueError("top scores must be finite")
        if np.any(scores[:-1] < scores[1:]):
            if self.reject_unsorted_scores:
                raise ValueError("top scores must be sorted in nonincreasing order")
            scores = np.sort(scores)[::-1]

        has_moments = all(
            name in source for name in ("corpus_size", "sum_all_scores", "sumsq_all_scores")
        )
        has_canonical = all(
            name in source
            for name in (
                "mirabel_formula_version", "mirabel_confidence", "mirabel_corpus_n",
                "mirabel_background_mean", "mirabel_background_std", "mirabel_threshold",
                "mirabel_margin",
            )
        )
        row: dict[str, object] = {"scores": scores.tolist(), "top1": float(scores[0])}
        if has_moments:
            stats = canonical_mirabel_from_moments(
                top1=float(scores[0]),
                sum_all=float(source["sum_all_scores"]),
                sumsq_all=float(source["sumsq_all_scores"]),
                corpus_size=int(source["corpus_size"]),
            )
            row.update(canonical_cache_fields(stats))
            if has_canonical:
                require_canonical_cache_row(source)
                checks = {
                    "mirabel_corpus_n": (int(source["mirabel_corpus_n"]), stats.corpus_size),
                    "mirabel_margin": (float(source["mirabel_margin"]), stats.margin),
                    "mirabel_threshold": (float(source["mirabel_threshold"]), stats.threshold),
                    "mirabel_background_mean": (
                        float(source["mirabel_background_mean"]), stats.background_mean
                    ),
                    "mirabel_background_std": (
                        float(source["mirabel_background_std"]), stats.background_std
                    ),
                }
                for name, (provided, calculated) in checks.items():
                    if not math.isclose(provided, calculated, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(f"raw moments and canonical statistic disagree: {name}")
        elif has_canonical:
            require_canonical_cache_row(source)
            row.update({
                name: source[name]
                for name in (
                    "mirabel_formula_version", "mirabel_confidence", "mirabel_corpus_n",
                    "mirabel_background_mean", "mirabel_background_std", "mirabel_threshold",
                    "mirabel_margin",
                )
            })
        else:
            raise ValueError("full-corpus moments or versioned canonical MIRABEL statistics are required")
        require_canonical_cache_row(row)
        return row

    def _validate_versions(self, source: Mapping[str, object]) -> None:
        expected = {
            "model_version": self.calibration.model_version,
            "threshold_version": self.calibration.threshold_version,
            "feature_schema_version": self.calibration.feature_schema_version,
            "retriever_schema_version": self.calibration.retriever_schema_version,
            "mirabel_formula_version": self.calibration.mirabel_formula_version,
        }
        for name, value in expected.items():
            if name in source and str(source[name]) != str(value):
                raise ValueError(f"{name} mismatch: expected {value!r}")
        if "retriever_id" in source and str(source["retriever_id"]) != self.calibration.retriever_id:
            raise ValueError("retriever_id mismatch; benign recalibration is required")

    def _new_state(self, session_id: str) -> SessionState:
        return SessionState(
            session_id=str(session_id),
            horizon=self.horizon,
            model_version=self.calibration.model_version,
            threshold_version=self.calibration.threshold_version,
            feature_schema_version=self.calibration.feature_schema_version,
        )

    def _observation_from_dict(self, value: Mapping[str, object], *, duplicate: bool) -> AdaptiveFisherObservation:
        allowed = AdaptiveFisherObservation.__dataclass_fields__
        payload = {name: value[name] for name in allowed if name in value}
        payload["duplicate_request"] = bool(duplicate)
        return AdaptiveFisherObservation(**payload)

    def observe(
        self,
        session_id: str,
        retrieval_result: Mapping[str, object],
        *,
        request_id: str | None = None,
    ) -> AdaptiveFisherObservation:
        session_id = str(session_id)
        request = request_id if request_id is not None else retrieval_result.get("request_id")
        lock_factory = getattr(self.state_store, "session_lock", None)
        context = lock_factory(session_id) if lock_factory else nullcontext()
        with context:
            state = self.state_store.load(session_id) or self._new_state(session_id)
            if (
                state.model_version != self.calibration.model_version
                or state.threshold_version != self.calibration.threshold_version
                or state.horizon != self.horizon
            ):
                raise ValueError("stored session state uses a different model, threshold, or horizon")
            if request is not None and str(request) in state.recent_requests:
                return self._observation_from_dict(
                    state.recent_requests[str(request)], duplicate=True
                )
            if len(state.query_p_values) >= self.horizon:
                raise ValueError("session has reached the calibrated horizon")
            row = self._canonical_row(retrieval_result)
            query_p = self.calibration.query_model.p_value(row)
            state.query_p_values.append(float(query_p))
            horizon_calibration = self.calibration.for_horizon(self.horizon)
            selected_k, statistic, raw_adaptive = adaptive_score(
                state.query_p_values, horizon_calibration.partial_fisher_references
            )
            risk = empirical_risk(raw_adaptive, horizon_calibration.adaptive_score_reference)
            exceeded = bool(risk > horizon_calibration.threshold)
            if exceeded and state.first_detection_turn is None:
                state.first_detection_turn = len(state.query_p_values)
            if exceeded:
                state.blocked = True
            actual_block = bool(state.blocked and self.mode == "block")
            action = "block" if actual_block else ("alert" if state.blocked and self.mode == "alert" else "shadow")
            observation = AdaptiveFisherObservation(
                session_id=session_id,
                query_count=len(state.query_p_values),
                horizon=self.horizon,
                query_p_value=float(query_p),
                selected_k=selected_k,
                partial_fisher_statistic=statistic,
                adaptive_score=raw_adaptive,
                calibrated_risk=risk,
                threshold=float(horizon_calibration.threshold),
                blocked=actual_block,
                first_detection_turn=state.first_detection_turn,
                model_version=self.calibration.model_version,
                threshold_version=self.calibration.threshold_version,
                mirabel_formula_version=self.calibration.mirabel_formula_version,
                threshold_exceeded=bool(state.blocked),
                action=action,
            )
            if request is not None:
                state.recent_requests[str(request)] = asdict(observation)
                while len(state.recent_requests) > self.request_id_retention:
                    state.recent_requests.popitem(last=False)
            state.updated_at = datetime.now(timezone.utc).isoformat()
            self.state_store.save(session_id, state)
            return observation

    def observe_many(
        self,
        session_id: str,
        retrieval_results: Sequence[Mapping[str, object]],
        *,
        request_ids: Sequence[str] | None = None,
    ) -> list[AdaptiveFisherObservation]:
        if request_ids is not None and len(request_ids) != len(retrieval_results):
            raise ValueError("request_ids and retrieval_results must have equal lengths")
        return [
            self.observe(
                session_id,
                row,
                request_id=None if request_ids is None else request_ids[index],
            )
            for index, row in enumerate(retrieval_results)
        ]

    def score_set(self, rows: Sequence[Mapping[str, object]]) -> dict[str, float | int | bool]:
        canonical = [self._canonical_row(row) for row in rows]
        return score_query_set(canonical, self.calibration, horizon=self.horizon)

    def get_state(self, session_id: str) -> Mapping[str, object]:
        state = self.state_store.load(str(session_id))
        return {} if state is None else state.to_dict()

    def reset(self, session_id: str) -> None:
        """Explicit security reset; a UI-only reset should not call this method."""

        self.state_store.delete(str(session_id))

    def structured_log(self, observation: AdaptiveFisherObservation, *, request_id: str | None) -> dict[str, object]:
        digest = hmac.new(self.log_salt, observation.session_id.encode("utf-8"), hashlib.sha256).hexdigest()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_hash": digest,
            "request_id": request_id,
            "model_version": observation.model_version,
            "query_count": observation.query_count,
            "retriever_id": self.calibration.retriever_id,
            "risk": observation.calibrated_risk,
            "threshold": observation.threshold,
            "blocked": observation.blocked,
            "error_code": None,
        }
