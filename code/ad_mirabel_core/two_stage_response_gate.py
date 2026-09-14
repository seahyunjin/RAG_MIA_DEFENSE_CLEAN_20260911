"""Pre-release two-stage response gate with idempotent request handling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock

from .response_augmented_ldf import ResponseAugmentedCalibration, score_response_set


STATES = {
    "RETRIEVED", "GENERATION_IN_PROGRESS", "RESPONSE_READY", "RELEASED",
    "BLOCKED", "GENERATION_FAILED", "ABORTED",
}


@dataclass(frozen=True)
class PreGenerationObservation:
    session_id: str
    request_id: str
    state: str
    duplicate_request: bool


@dataclass(frozen=True)
class FinalObservation:
    session_id: str
    request_id: str
    state: str
    query_count: int
    calibrated_risk: float | None
    threshold: float | None
    blocked: bool
    duplicate_request: bool


class ResponseAugmentedDetector:
    def __init__(self, calibration: ResponseAugmentedCalibration, *, horizon: int = 5) -> None:
        calibration.for_horizon(horizon)
        self.calibration = calibration
        self.horizon = int(horizon)
        self._sessions: dict[str, dict[str, object]] = {}
        self._lock = RLock()

    def _session(self, session_id: str) -> dict[str, object]:
        return self._sessions.setdefault(str(session_id), {"completed": [], "requests": {}, "first_detection_turn": None})

    def begin_query(self, session_id: str, retrieval_result: Mapping[str, object], *, request_id: str) -> PreGenerationObservation:
        with self._lock:
            session = self._session(session_id); requests = session["requests"]
            if request_id in requests:
                return PreGenerationObservation(str(session_id), str(request_id), str(requests[request_id]["state"]), True)
            if len(session["completed"]) >= self.horizon:
                raise ValueError("session has reached the calibrated horizon")
            if str(retrieval_result.get("retriever_id", self.calibration.retriever_id)) != self.calibration.retriever_id:
                raise ValueError("retriever calibration mismatch")
            requests[str(request_id)] = {"state": "GENERATION_IN_PROGRESS", "retrieval": dict(retrieval_result), "final": None}
            return PreGenerationObservation(str(session_id), str(request_id), "GENERATION_IN_PROGRESS", False)

    def complete_query(self, session_id: str, request_id: str, response_record: Mapping[str, object]) -> FinalObservation:
        with self._lock:
            session = self._session(session_id); requests = session["requests"]
            if str(request_id) not in requests:
                raise KeyError("begin_query must precede complete_query")
            item = requests[str(request_id)]
            if item["final"] is not None:
                value = item["final"]
                return FinalObservation(**{**value.__dict__, "duplicate_request": True})
            if bool(response_record.get("generation_failed", False)):
                item["state"] = "GENERATION_FAILED"
                final = FinalObservation(str(session_id), str(request_id), "GENERATION_FAILED", len(session["completed"]), None, None, False, False)
                item["final"] = final
                return final
            retrieval_hash = item["retrieval"].get("query_hash")
            if retrieval_hash is not None and str(response_record.get("query_hash")) != str(retrieval_hash):
                raise ValueError("query-response pair mismatch")
            feature = response_record.get("response_feature")
            if feature is None:
                raise ValueError("response_feature is required after generation")
            item["state"] = "RESPONSE_READY"
            session["completed"].append((item["retrieval"], feature))
            result = score_response_set(session["completed"], self.calibration, horizon=self.horizon)
            blocked = bool(result["blocked"])
            state = "BLOCKED" if blocked else "RELEASED"
            if blocked and session["first_detection_turn"] is None:
                session["first_detection_turn"] = len(session["completed"])
            item["state"] = state
            final = FinalObservation(str(session_id), str(request_id), state, len(session["completed"]), float(result["score"]), float(result["threshold"]), blocked, False)
            item["final"] = final
            return final

    def abort_query(self, session_id: str, request_id: str, reason: str) -> None:
        with self._lock:
            session = self._session(session_id); requests = session["requests"]
            if str(request_id) not in requests:
                raise KeyError("unknown request_id")
            item = requests[str(request_id)]
            if item["final"] is None:
                item["state"] = "ABORTED"; item["abort_reason"] = str(reason)

    def get_state(self, session_id: str) -> Mapping[str, object]:
        with self._lock:
            session = self._sessions.get(str(session_id))
            if session is None:
                return {}
            return {
                "query_count": len(session["completed"]),
                "first_detection_turn": session["first_detection_turn"],
                "requests": {key: value["state"] for key, value in session["requests"].items()},
            }

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(str(session_id), None)
