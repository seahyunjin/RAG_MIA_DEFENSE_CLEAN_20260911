#!/usr/bin/env python3
"""Correct the over-broad initial content gate without regenerating questions."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ia_st1_protocol import validate_questions


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1"
DB = EXP / "runtime" / "ia_st1.sqlite3"
TARGETS = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1" / "inputs" / "LARGE_SHARED_TARGETS.csv"


def main() -> None:
    import csv

    targets = {r["document_id"]: r for r in csv.DictReader(TARGETS.open(encoding="utf-8"))}
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT target_id,status,parsed_json,validation_json FROM calls WHERE stage='questions' ORDER BY started_utc"
    ).fetchall()
    if len(rows) != 100:
        raise SystemExit(f"expected 100 preserved question calls, got {len(rows)}")
    old_status = Counter(row[1] for row in rows)
    old_reasons: Counter[str] = Counter()
    new_status: Counter[str] = Counter()
    new_reasons: Counter[str] = Counter()
    meta_sessions = 0
    changed = 0
    for target_id, status, parsed_json, validation_json in rows:
        if validation_json:
            for reason in json.loads(validation_json).get("reasons", []):
                old_reasons[reason] += 1
        if not parsed_json:
            validation = {"valid": False, "reasons": ["MISSING_STRUCTURED_OUTPUT"]}
        else:
            validation = validate_questions(json.loads(parsed_json), targets[target_id]["source_text"])
        corrected_status = "VALID" if validation["valid"] else "INVALID"
        new_status[corrected_status] += 1
        for reason in validation.get("reasons", []):
            new_reasons[reason] += 1
        if validation.get("meta_reference_slots_diagnostic"):
            meta_sessions += 1
        if corrected_status != status:
            changed += 1
        con.execute(
            "UPDATE calls SET status=?,validation_json=?,error_type=NULL,error_message=NULL WHERE target_id=? AND stage='questions'",
            (corrected_status, json.dumps(validation, ensure_ascii=False), target_id),
        )
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    audit = {
        "campaign": "IA_STEALTH_CONFIRMATION_V1",
        "correction": "INVALID_OVERRESTRICTIVE_VALIDATOR",
        "corrected_utc": datetime.now(timezone.utc).isoformat(),
        "question_api_calls_reused": 100,
        "question_regeneration": 0,
        "performance_metrics_computed": False,
        "old_status": dict(old_status),
        "old_reasons": dict(old_reasons),
        "new_status": dict(new_status),
        "new_reasons": dict(new_reasons),
        "status_rows_changed": changed,
        "meta_reference_sessions_diagnostic": meta_sessions,
        "sqlite_integrity": integrity,
    }
    out = EXP / "audits" / "SPEC_CORRECTION_AUDIT.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

