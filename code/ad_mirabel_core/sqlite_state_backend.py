"""Minimal durable state backend for the V16 deployment prototype.

The backend stores pseudonymous linkage keys, the retrieval feature rows needed
to recompute the frozen detector, a sticky block bit, and the last update time.
It does not provide identity proof or Sybil resistance.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence


class SQLiteStateBackend:
    def __init__(self, path: str | Path, *, max_history: int = 30) -> None:
        self.path = Path(path)
        self.max_history = int(max_history)
        if self.max_history < 1:
            raise ValueError("max_history must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS detector_state ("
            "linkage_key TEXT PRIMARY KEY, history_json TEXT NOT NULL, "
            "blocked INTEGER NOT NULL, updated_at REAL NOT NULL)"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def load(self, linkage_key: str, *, now: float, ttl_seconds: float) -> tuple[list[dict], bool, bool]:
        key = str(linkage_key)
        row = self.connection.execute(
            "SELECT history_json, blocked, updated_at FROM detector_state WHERE linkage_key=?",
            (key,),
        ).fetchone()
        if row is None:
            return [], False, False
        expired = float(now) - float(row[2]) > float(ttl_seconds)
        if expired:
            self.connection.execute("DELETE FROM detector_state WHERE linkage_key=?", (key,))
            self.connection.commit()
            return [], False, True
        return list(json.loads(row[0])), bool(row[1]), False

    def save(
        self,
        linkage_key: str,
        history: Sequence[Mapping[str, object]],
        *,
        blocked: bool,
        now: float,
    ) -> None:
        compact = list(history)[-self.max_history :]
        payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        self.connection.execute(
            "INSERT INTO detector_state(linkage_key,history_json,blocked,updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(linkage_key) DO UPDATE SET history_json=excluded.history_json, "
            "blocked=excluded.blocked, updated_at=excluded.updated_at",
            (str(linkage_key), payload, int(bool(blocked)), float(now)),
        )
        self.connection.commit()

    def clear(self, linkage_key: str) -> None:
        self.connection.execute("DELETE FROM detector_state WHERE linkage_key=?", (str(linkage_key),))
        self.connection.commit()

    def active_users(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM detector_state").fetchone()[0])

    def serialized_bytes(self) -> list[int]:
        return [int(row[0]) for row in self.connection.execute(
            "SELECT length(linkage_key)+length(history_json)+9 FROM detector_state"
        )]

