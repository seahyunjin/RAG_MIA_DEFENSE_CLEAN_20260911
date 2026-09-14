#!/usr/bin/env python3
"""Administrative finalizer for the frozen, format-failed IA-v6 run.

This script does not parse, repair, score, or generate. It only checkpoints the
already committed SQLite rows after the authorized graceful stop and records the
two required terminal labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments/FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
V6 = PARENT / "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    database = V6 / "runtime/ia_v6.sqlite3"
    connection = sqlite3.connect(database)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.execute("PRAGMA wal_checkpoint(FULL)")
    questions = list(connection.execute(
        "SELECT target_id,membership,slot,valid,reason FROM question_slot ORDER BY target_id,slot"))
    judgments = list(connection.execute(
        "SELECT target_id,membership,slot,valid,reason FROM judgment_slot ORDER BY target_id,slot"))
    connection.close()

    by_target: dict[str, list[tuple]] = defaultdict(list)
    for row in questions:
        by_target[row[0]].append(row)
    invalid_targets = {
        target for target, rows in by_target.items()
        if any(not bool(row[3]) for row in rows)
    }
    failure_reasons = Counter(row[4] for row in questions if not bool(row[3]))
    complete_question_sessions = sum(
        len(rows) == 15 and [row[2] for row in rows] == list(range(1, 16))
        for rows in by_target.values()
    )
    result = {
        "campaign": "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED",
        "verdict": "IA_STD_Q15_V6_FAILED_FINAL",
        "scientific_status": "STANDARDIZED_ATTACK_UNAVAILABLE",
        "completed_utc": now(),
        "termination": "AUTHORIZED_GRACEFUL_CLOSE_AFTER_MATHEMATICAL_FORMAT_GATE_FAILURE",
        "partial_artifact": {
            "database": str(database),
            "sqlite_integrity": integrity,
            "database_sha256": sha_file(database),
            "question_rows": len(questions),
            "judgment_rows": len(judgments),
            "targets_started": len(by_target),
            "complete_question_sessions": complete_question_sessions,
            "known_invalid_sessions": len(invalid_targets),
            "failure_reasons": dict(failure_reasons),
        },
        "preflight_required_valid": 98,
        "preflight_max_invalid": 2,
        "mathematically_possible": len(invalid_targets) <= 2,
        "performance_metrics_computed": False,
        "detection_run": False,
        "e2e_run": False,
        "v7_created": False,
        "additional_recovery_allowed": False,
        "final_lc_modified": False,
    }
    atomic_json(V6 / "FINAL_RESULT.json", result)
    atomic_json(V6 / "checkpoints/IA_STD_Q15_V6_FAILED_FINAL.json", result)
    report = [
        "# IA-Std-Q15-v6 Final Status", "",
        "- Verdict: `IA_STD_Q15_V6_FAILED_FINAL`",
        "- Scientific status: `STANDARDIZED_ATTACK_UNAVAILABLE`",
        f"- Committed question/judgment rows: `{len(questions)}` / `{len(judgments)}`",
        f"- Started/complete question sessions: `{len(by_target)}` / `{complete_question_sessions}`",
        f"- Known-invalid sessions: `{len(invalid_targets)}` (allowed: 2)",
        f"- SQLite integrity: `{integrity}`",
        f"- SQLite SHA-256: `{result['partial_artifact']['database_sha256']}`", "",
        "The 98/100 format gate was mathematically impossible. No detection, attack-success,",
        "AUC/E-AUC, or privacy result was computed. No v7 or semantic/parser relaxation is allowed.",
    ]
    atomic_text(V6 / "STATUS.md", "\n".join(report) + "\n")
    with (V6 / "logs/IA_V6.log").open("a", encoding="utf-8") as handle:
        handle.write(f"\n{now()} IA_STD_Q15_V6_FAILED_FINAL STANDARDIZED_ATTACK_UNAVAILABLE "
                     f"questions={len(questions)} invalid_sessions={len(invalid_targets)}\n")

    parent_status = PARENT / "STATUS.md"
    with parent_status.open("a", encoding="utf-8") as handle:
        handle.write("\n## IA-v6 terminal closure\n\n"
                     "- `IA_STD_Q15_V6_FAILED_FINAL`\n"
                     "- `STANDARDIZED_ATTACK_UNAVAILABLE`\n"
                     "- IA is excluded from the main paper result; Final LC/Core5/Gold remain unchanged.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
