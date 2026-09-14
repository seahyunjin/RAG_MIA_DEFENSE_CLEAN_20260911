"""Unified runtime facade over retriever-specific frozen LDF baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .adaptive_fisher_detector import AdaptiveFisherDetector
from .retriever_model_registry import RetrieverModelRegistry


class RetrieverSpecificLDF:
    def __init__(self, registry: RetrieverModelRegistry, *, horizon: int = 5, mode: str = "shadow") -> None:
        self.registry = registry
        self.horizon = int(horizon)
        self.mode = str(mode)
        self._detectors: dict[str, AdaptiveFisherDetector] = {}
        self._session_retriever: dict[str, str] = {}

    def _detector(self, retriever_id: str) -> AdaptiveFisherDetector:
        retriever_id = str(retriever_id)
        self.registry.entry(retriever_id)
        if retriever_id not in self._detectors:
            self._detectors[retriever_id] = AdaptiveFisherDetector(
                self.registry.calibration(retriever_id), horizon=self.horizon, mode=self.mode
            )
        return self._detectors[retriever_id]

    @staticmethod
    def _retriever(row: Mapping[str, object]) -> str:
        if "retriever_id" not in row:
            raise ValueError("retriever_id is required")
        return str(row["retriever_id"])

    def observe(self, session_id: str, retrieval_result: Mapping[str, object], *, request_id: str | None = None):
        retriever = self._retriever(retrieval_result)
        previous = self._session_retriever.setdefault(str(session_id), retriever)
        if previous != retriever:
            raise ValueError("a session cannot change retriever calibration")
        return self._detector(retriever).observe(session_id, retrieval_result, request_id=request_id)

    def observe_many(self, session_id: str, retrieval_results: Sequence[Mapping[str, object]], *, request_ids: Sequence[str] | None = None):
        if not retrieval_results:
            return []
        retrievers = {self._retriever(row) for row in retrieval_results}
        if len(retrievers) != 1:
            raise ValueError("observe_many requires one retriever per session")
        retriever = next(iter(retrievers))
        previous = self._session_retriever.setdefault(str(session_id), retriever)
        if previous != retriever:
            raise ValueError("a session cannot change retriever calibration")
        return self._detector(retriever).observe_many(session_id, retrieval_results, request_ids=request_ids)

    def get_state(self, session_id: str) -> Mapping[str, object]:
        retriever = self._session_retriever.get(str(session_id))
        return {} if retriever is None else self._detector(retriever).get_state(session_id)

    def reset(self, session_id: str) -> None:
        retriever = self._session_retriever.pop(str(session_id), None)
        if retriever is not None:
            self._detector(retriever).reset(session_id)
