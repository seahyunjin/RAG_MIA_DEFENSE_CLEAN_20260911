#!/usr/bin/env python3
"""Monitor frozen IA-v3 and fail-close at the 101st invalid session.

This process never edits the v3 database.  At the mathematical stop boundary it
stops the v3 worker, creates an immutable SQLite backup and audit, then waits for
the separately precommitted v4 runner and launches it once.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import EXP, atomic_json, checkpoint, now, sha_file

DB = EXP / "runtime" / "ia_v3.sqlite3"
SESSION_V3 = "final-lc-ia-v3"
SESSION_V4 = "final-lc-ia-v4"


def tmux_exists(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def process_rows() -> list[dict]:
    con = sqlite3.connect(DB, timeout=120)
    columns = [row[1] for row in con.execute("PRAGMA table_info(query_generation)")]
    rows = [dict(zip(columns, row)) for row in con.execute("SELECT * FROM query_generation ORDER BY target_id")]
    con.close()
    return rows


def stop_v3() -> None:
    if tmux_exists(SESSION_V3):
        subprocess.run(["tmux", "send-keys", "-t", SESSION_V3, "C-c"], check=False)
    for _ in range(30):
        if not tmux_exists(SESSION_V3):
            return
        time.sleep(1)
    subprocess.run(["tmux", "kill-session", "-t", SESSION_V3], check=False)


def snapshot(rows: list[dict], monitor_started: float) -> None:
    history = EXP / "history" / "IA_V3_MATHEMATICAL_EARLY_STOP"
    history.mkdir(parents=True, exist_ok=True)
    backup = history / "ia_v3_early_stop.sqlite3"
    src = sqlite3.connect(DB, timeout=120)
    dst = sqlite3.connect(backup)
    src.backup(dst)
    dst.close(); src.close()
    with (history / "IA_V3_EARLY_STOP_VALIDITY_AUDIT.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["target_id", "membership", "valid", "reason", "selected_attempt", "prompt_sha256", "wall_seconds", "completed_utc"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: row.get(key) for key in fields})
    valid = [row for row in rows if int(row["valid"])]
    invalid = [row for row in rows if not int(row["valid"])]
    by_membership = {}
    for membership in ("member", "nonmember"):
        group = [row for row in rows if row["membership"] == membership]
        n_valid = sum(int(row["valid"]) for row in group)
        by_membership[membership] = {"generated": len(group), "valid": n_valid,
                                     "valid_rate": n_valid / len(group) if group else None}
    result = {
        "campaign": EXP.name,
        "variant": "IA-Std-Q15-v3",
        "verdict": "IA_STD_Q15_V3_VALIDITY_GATE_MATHEMATICALLY_IMPOSSIBLE",
        "completed_utc": now(),
        "generated": len(rows),
        "valid": len(valid),
        "invalid": len(invalid),
        "maximum_invalid_allowed": 100,
        "invalid_category_histogram": dict(Counter(row["reason"] for row in invalid)),
        "by_membership": by_membership,
        "member_nonmember_validity_gap": (
            abs(by_membership["member"]["valid_rate"] - by_membership["nonmember"]["valid_rate"])
            if by_membership["member"]["valid_rate"] is not None and by_membership["nonmember"]["valid_rate"] is not None else None
        ),
        "recorded_generation_wall_seconds": sum(float(row.get("wall_seconds") or 0) for row in rows),
        "transition_monitor_elapsed_seconds": time.monotonic() - monitor_started,
        "gpu_time_note": "CUDA kernel time was not instrumented; recorded_generation_wall_seconds is the preserved generation-time proxy.",
        "raw_cache": {"path": str(backup), "sha256": sha_file(backup)},
        "live_database_preserved": {"path": str(DB), "sha256_after_checkpoint": sha_file(DB)},
        "v3_precommit_sha256": sha_file(EXP / "configs" / "IA_STD_Q15_V3_PRECOMMIT.json"),
        "next_stage": "IA_STD_Q15_V4_TWO_STAGE_PRECOMMIT",
    }
    atomic_json(EXP / "IA_V3_EARLY_STOP_RESULT.json", result)
    shutil.copy2(EXP / "configs" / "IA_STD_Q15_V3_PRECOMMIT.json", history / "IA_STD_Q15_V3_PRECOMMIT.json")
    if (EXP / "logs" / "IA_V3.log").exists():
        shutil.copy2(EXP / "logs" / "IA_V3.log", history / "IA_V3.log")
    checkpoint(result["verdict"], generated=len(rows), valid=len(valid), invalid=len(invalid),
               next_stage=result["next_stage"])


def launch_v4_when_ready() -> None:
    prepare = EXP / "code" / "prepare_ia_v4.py"
    runner = EXP / "code" / "run_ia_v4.py"
    while not prepare.is_file() or not runner.is_file():
        checkpoint("IA_V4_CODE_PREPARATION_WAIT", next_stage="IA_STD_Q15_V4_TWO_STAGE_PRECOMMIT")
        time.sleep(2)
    python = "/home/traffic_3/workspace/miniconda3/envs/torch/bin/python"
    subprocess.run([python, str(prepare)], cwd=EXP, check=True,
                   stdout=(EXP / "logs" / "IA_V4_PREPARE.log").open("a"), stderr=subprocess.STDOUT)
    if not tmux_exists(SESSION_V4):
        command = f"cd '{EXP}' && '{python}' code/run_ia_v4.py >> logs/IA_V4.log 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", SESSION_V4, command], check=True)
    checkpoint("IA_V4_PREFLIGHT_LAUNCHED", tmux_session=SESSION_V4,
               next_stage="IA_V4_FORMAT_PREFLIGHT")


def main() -> None:
    monitor_started = time.monotonic()
    while True:
        rows = process_rows()
        invalid = sum(not int(row["valid"]) for row in rows)
        checkpoint("IA_V3_EARLY_STOP_MONITOR", generated=len(rows), valid=len(rows) - invalid,
                   invalid=invalid, invalid_limit=101)
        if invalid >= 101:
            stop_v3()
            # Re-read only after the writer has stopped so the boundary audit is consistent.
            rows = process_rows()
            snapshot(rows, monitor_started)
            launch_v4_when_ready()
            return
        if not tmux_exists(SESSION_V3):
            raise RuntimeError("IA-v3 stopped before the mathematical boundary")
        time.sleep(2)


if __name__ == "__main__":
    main()
