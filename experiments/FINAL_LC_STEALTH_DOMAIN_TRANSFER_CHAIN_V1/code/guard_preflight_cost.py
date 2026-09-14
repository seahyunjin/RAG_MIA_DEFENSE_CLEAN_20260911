#!/usr/bin/env python3
"""Stop before full IA generation if observed preflight cost projects above USD 8."""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
IA = EXP / "IA_STD_Q15_API1"
PREFLIGHT = IA / "PREFLIGHT_RESULT.json"
HEARTBEAT = IA / "HEARTBEAT.json"
BLOCK = IA / "COST_PROJECTION_BLOCK.json"
CAP = 8.0


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    while not PREFLIGHT.is_file():
        time.sleep(0.1)
    result = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if result.get("verdict") != "IA_API1_PROTOCOL_READY":
        return
    observed = float(result["estimated_campaign_spend_usd"])
    projected = observed * 20.0
    if projected <= CAP:
        return
    heartbeat = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    pid = int(heartbeat["pid"])
    payload = {
        "verdict": "IA_API_COST_CAP_INSUFFICIENT_FOR_FULL_2000",
        "preflight_sessions": 100,
        "preflight_spend_usd": observed,
        "projected_full_2000_spend_usd": projected,
        "hard_cap_usd": CAP,
        "full_performance_metrics_computed": False,
        "full_responses_received_after_preflight": 0,
        "full_requests_started_before_termination": "AUDIT_REQUIRED",
        "target_document_transmission_after_preflight": "NOT_ASSUMED_ZERO",
        "action": "stopped before full-scale requests as precommitted",
        "runner_pid": pid,
    }
    atomic_json(BLOCK, payload)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(1.0)
    atomic_json(IA / "FINAL_RESULT.json", payload)


if __name__ == "__main__":
    main()
