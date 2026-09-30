"""Persistent B=2 composition safeguard around a separately deployed RIF."""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HORIZON = 2
ABSTENTION = "I don't know."


@dataclass(frozen=True)
class HorizonDecision:
    allowed: bool
    request_index: int
    horizon: int = HORIZON


class FixedEvidenceHorizon:
    """Atomic, account-bound ledger that never reads membership or attack type."""

    def __init__(self, database: str | Path, server_secret: bytes):
        if not server_secret:
            raise ValueError("A non-empty server-side HMAC secret is required")
        self.database = str(database)
        self.server_secret = server_secret
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS evidence_ledger ("
                "account_key TEXT PRIMARY KEY, protected_requests INTEGER NOT NULL)"
            )

    def _account_key(self, credential: str) -> str:
        if not credential:
            raise ValueError("An authenticated account credential is required")
        return hmac.new(
            self.server_secret,
            credential.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def check_and_consume(self, credential: str) -> HorizonDecision:
        key = self._account_key(credential)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT protected_requests FROM evidence_ledger WHERE account_key=?",
                (key,),
            ).fetchone()
            previous = int(row[0]) if row else 0
            current = previous + 1
            connection.execute(
                "INSERT INTO evidence_ledger(account_key, protected_requests) VALUES(?, ?) "
                "ON CONFLICT(account_key) DO UPDATE SET protected_requests=excluded.protected_requests",
                (key, current),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return HorizonDecision(allowed=current <= HORIZON, request_index=current)

    def serve(self, credential: str, protected_rif_call: Callable[[], Any]) -> dict[str, Any]:
        """Authorize before retrieval; call RIF only for the first two requests."""
        decision = self.check_and_consume(credential)
        if not decision.allowed:
            return {
                "answer": ABSTENTION,
                "horizon_exhausted": True,
                "protected_retrieval_executed": False,
                "request_index": decision.request_index,
                "horizon": HORIZON,
            }
        value = protected_rif_call()
        return {
            "answer": value,
            "horizon_exhausted": False,
            "protected_retrieval_executed": True,
            "request_index": decision.request_index,
            "horizon": HORIZON,
        }

