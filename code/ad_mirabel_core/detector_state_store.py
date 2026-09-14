"""Bounded in-memory and durable SQLite state stores for streaming detection."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Protocol


@dataclass
class SessionState:
    session_id: str
    horizon: int
    query_p_values: list[float] = field(default_factory=list)
    first_detection_turn: int | None = None
    blocked: bool = False
    recent_requests: OrderedDict[str, dict[str, Any]] = field(default_factory=OrderedDict)
    model_version: str = ""
    threshold_version: str = ""
    feature_schema_version: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["recent_requests"] = list(self.recent_requests.items())
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SessionState":
        copied = dict(value)
        copied["recent_requests"] = OrderedDict(copied.get("recent_requests", []))
        return cls(**copied)


class DetectorStateStore(Protocol):
    def load(self, session_id: str) -> SessionState | None: ...
    def save(self, session_id: str, state: SessionState) -> None: ...
    def delete(self, session_id: str) -> None: ...
    def cleanup_expired(self, now: datetime) -> int: ...


class InMemoryStateStore:
    def __init__(self, *, ttl_seconds: float = 86400.0) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self._states: dict[str, SessionState] = {}
        self._locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)

    @contextmanager
    def session_lock(self, session_id: str) -> Iterator[None]:
        with self._locks[str(session_id)]:
            yield

    def load(self, session_id: str) -> SessionState | None:
        value = self._states.get(str(session_id))
        return None if value is None else SessionState.from_dict(value.to_dict())

    def save(self, session_id: str, state: SessionState) -> None:
        self._states[str(session_id)] = SessionState.from_dict(state.to_dict())

    def delete(self, session_id: str) -> None:
        self._states.pop(str(session_id), None)

    def cleanup_expired(self, now: datetime) -> int:
        cutoff = now.timestamp() - self.ttl_seconds
        expired = [
            key for key, state in self._states.items()
            if datetime.fromisoformat(state.updated_at).timestamp() < cutoff
        ]
        for key in expired:
            self.delete(key)
        return len(expired)

    def count(self) -> int:
        return len(self._states)


class SQLiteStateStore:
    """SQLite/WAL backend with serialized transactional updates per process."""

    def __init__(self, path: str | Path, *, ttl_seconds: float = 86400.0) -> None:
        self.path = Path(path)
        self.ttl_seconds = float(ttl_seconds)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS adaptive_fisher_state ("
            "session_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, updated_epoch REAL NOT NULL)"
        )
        self._connection.commit()
        self._transaction_lock = threading.RLock()
        self._session_locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)

    @contextmanager
    def session_lock(self, session_id: str) -> Iterator[None]:
        # The per-session lock prevents lost updates in threaded services.  The
        # transaction lock makes the shared sqlite connection safe as well.
        with self._session_locks[str(session_id)], self._transaction_lock:
            yield

    def load(self, session_id: str) -> SessionState | None:
        row = self._connection.execute(
            "SELECT state_json FROM adaptive_fisher_state WHERE session_id=?",
            (str(session_id),),
        ).fetchone()
        return None if row is None else SessionState.from_dict(json.loads(row[0]))

    def save(self, session_id: str, state: SessionState) -> None:
        payload = json.dumps(state.to_dict(), ensure_ascii=False, separators=(",", ":"))
        updated = datetime.fromisoformat(state.updated_at).timestamp()
        with self._connection:
            self._connection.execute(
                "INSERT INTO adaptive_fisher_state(session_id,state_json,updated_epoch) VALUES(?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET state_json=excluded.state_json, "
                "updated_epoch=excluded.updated_epoch",
                (str(session_id), payload, float(updated)),
            )

    def delete(self, session_id: str) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM adaptive_fisher_state WHERE session_id=?", (str(session_id),)
            )

    def cleanup_expired(self, now: datetime) -> int:
        cutoff = now.timestamp() - self.ttl_seconds
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM adaptive_fisher_state WHERE updated_epoch < ?", (float(cutoff),)
            )
        return int(cursor.rowcount)

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM adaptive_fisher_state").fetchone()[0])

    def serialized_bytes(self) -> int:
        value = self._connection.execute(
            "SELECT COALESCE(SUM(length(session_id)+length(state_json)),0) "
            "FROM adaptive_fisher_state"
        ).fetchone()[0]
        return int(value)

    def close(self) -> None:
        self._connection.close()
