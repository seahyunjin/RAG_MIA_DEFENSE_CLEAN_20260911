#!/usr/bin/env python3
"""Availability audit and precommit for IA-Std-Q15-API1.

This stage is intentionally gated on completed Phase A and Phase B artifacts.
It performs no detector/privacy evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from common import EXP, LC, ROOT, atomic_json, checkpoint, freeze_json, now, sha_file
from ia_api_protocol import (
    GT_MODEL, GT_PROMPT, GT_SCHEMA, GT_SEED,
    QUESTION_MODEL, QUESTION_PROMPT, QUESTION_SCHEMA, QUESTION_SEED,
)


KEY_PATH = Path("/home/traffic_3/workspace/workspace/SH/.secrets/openai_api_key")
TARGETS = LC / "inputs" / "LARGE_SHARED_TARGETS.csv"
IA = EXP / "IA_STD_Q15_API1"


def model_available(model: str, key: str) -> tuple[bool, str | None]:
    request = urllib.request.Request(
        "https://api.openai.com/v1/models/" + urllib.parse.quote(model, safe=""),
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60, context=ssl.create_default_context()) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status == 200 and body.get("id") == model, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = (json.loads(raw).get("error") or {}).get("code") or f"HTTP_{exc.code}"
        except json.JSONDecodeError:
            detail = f"HTTP_{exc.code}"
        return False, detail
    except Exception as exc:
        return False, type(exc).__name__


def item(path: Path, **extra) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing Phase-C dependency: {path}")
    return {"path": str(path), "sha256": sha_file(path), **extra}


def stop_unavailable(availability: dict) -> None:
    result = {
        "campaign": EXP.name,
        "attack": "IA-Std-Q15-API1",
        "completed_utc": now(),
        "verdict": "IA_API_MODEL_UNAVAILABLE",
        "availability": availability,
        "api_key_recorded": False,
        "performance_metrics_computed": False,
        "final_lc_modified": False,
    }
    atomic_json(IA / "FINAL_RESULT.json", result)
    checkpoint("IA_API_MODEL_UNAVAILABLE", ia_result=str(IA / "FINAL_RESULT.json"))
    print(json.dumps(result, indent=2))


def main() -> None:
    for name in ("audits", "checkpoints", "configs", "inputs", "logs", "outputs", "post_ready", "runtime", "tables", "tests"):
        (IA / name).mkdir(parents=True, exist_ok=True)
    # Phase A/B must end before any model availability or API generation stage.
    phase_a = item(EXP / "PHASE_A2_RESULT.json")
    phase_b = item(EXP / "PHASE_B_RESULT.json")
    if not KEY_PATH.is_file() or not KEY_PATH.read_text(encoding="utf-8").strip():
        stop_unavailable({QUESTION_MODEL: "KEY_UNAVAILABLE", GT_MODEL: "KEY_UNAVAILABLE"})
        return
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    availability = {}
    for model in (QUESTION_MODEL, GT_MODEL):
        ok, detail = model_available(model, key)
        availability[model] = {"available": ok, "failure_code": detail}
    if not all(row["available"] for row in availability.values()):
        stop_unavailable(availability)
        return

    rows = list(csv.DictReader(TARGETS.open(encoding="utf-8")))
    if len(rows) != 2000:
        raise RuntimeError(f"IA target cohort drift: {len(rows)}")
    counts = {label: sum(row["membership"] == label for row in rows) for label in ("member", "nonmember")}
    if counts != {"member": 1000, "nonmember": 1000}:
        raise RuntimeError(f"IA membership drift: {counts}")
    selected = (
        sorted((row for row in rows if row["membership"] == "member"), key=lambda row: row["document_id"])[:50]
        + sorted((row for row in rows if row["membership"] == "nonmember"), key=lambda row: row["document_id"])[:50]
    )
    ids = [row["document_id"] for row in selected]
    id_hash = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    input_path = IA / "inputs" / "PREFLIGHT_TARGETS.jsonl"
    input_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected), encoding="utf-8")

    code_paths = [
        EXP / "code" / "common.py",
        EXP / "code" / "ia_api_protocol.py",
        EXP / "code" / "prepare_phase_c.py",
        EXP / "code" / "run_phase_c_api.py",
    ]
    precommit = {
        "campaign": EXP.name,
        "phase": "C_IA_STEALTH_RECOVERY",
        "attack_name": "IA-Std-Q15-API1",
        "created_utc": now(),
        "claim_boundary": "standardized IA Q15 stress test; not Original IA and not paper-exact IA",
        "dependencies": {"phase_a2": phase_a, "phase_b": phase_b},
        "final_lc": {"modified": False, "operating_benign_budget": 0.025},
        "prior_ia_v1_v6": "preserved failures; no Qwen prompt/parser tuning reopened",
        "model_availability": availability,
        "cohort": {
            "path": str(TARGETS), "sha256": sha_file(TARGETS), "sessions": 2000,
            "member": 1000, "nonmember": 1000,
            "preflight_ids": ids, "preflight_id_sha256": id_hash,
            "preflight_member": 50, "preflight_nonmember": 50,
            "preflight_included_in_full": True,
        },
        "question_generation": {
            "endpoint": "/v1/chat/completions", "model": QUESTION_MODEL,
            "temperature": 0.2, "seed": QUESTION_SEED, "max_tokens": 1400,
            "prompt": QUESTION_PROMPT, "schema": QUESTION_SCHEMA,
            "structured_output": {"type": "json_schema", "strict": True},
        },
        "gt_generation": {
            "endpoint": "/v1/chat/completions", "model": GT_MODEL,
            "temperature": 0.0, "seed": GT_SEED, "max_tokens": 500,
            "prompt": GT_PROMPT, "schema": GT_SCHEMA,
            "labels": ["Yes", "No", "Unknown"], "unknown_penalty_lambda": 5,
        },
        "retry_policy": {
            "schema_validation_failure": "at most one identical prompt/config retry",
            "duplicate_or_semantic_quality_failure": "no retry",
            "http_429": "bounded identical transport retry; no prompt/config change",
            "orphaned_inflight": "invalid; do not regenerate automatically",
        },
        "preflight_gate": {
            "sessions": 100, "valid_min": 98, "validity_gap_max": 0.02,
            "exact_q15_and_gt15": True, "duplicates": 0, "inferred_fields": 0,
            "performance_metrics_forbidden": True,
            "pass": "IA_STD_Q15_API1_PREFLIGHT_PASS",
            "fail": "IA_STD_Q15_API1_UNAVAILABLE",
        },
        "full_gate": {
            "sessions": 2000, "valid_min": 1900, "validity_gap_max": 0.02,
            "pass": "IA_STD_Q15_API1_READY", "additional_recovery": False,
        },
        "post_ready": {
            "detection": "MIRABEL vs frozen Final LC at same benign FPR 2.5%; query TPR primary; session bootstrap 2000",
            "e2e": ["No Defense", "MIRABEL same-budget", "Final LC same-budget"],
            "pass": "Final LC privacy improves vs No Defense and <= MIRABEL+0.03; preferred <=0.60; strong <=0.55",
        },
        "spending_safety": {
            "campaign_estimated_cost_ceiling_usd": 8.0,
            "reason": "preserve the user's previously stated approximately USD 10 API budget including prior calls",
            "pricing_estimate_only": True,
            "stop_before_request_if_estimate_would_exceed": True,
        },
        "api_key": {"path_recorded": False, "value_recorded": False, "git_artifact": False},
        "forbidden": ["performance during format preflight", "model substitution", "prompt search", "semantic retry", "Final LC changes"],
        "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code_paths},
    }
    digest = freeze_json(IA / "configs" / "IA_STD_Q15_API1_PRECOMMIT.json", precommit)
    atomic_json(IA / "checkpoints" / "PRECOMMITTED.json", {"created_utc": now(), "sha256": digest})
    checkpoint("PHASE_C_PRECOMMITTED", precommit_sha256=digest, preflight_sessions=100,
               question_model=QUESTION_MODEL, gt_model=GT_MODEL)
    print(json.dumps({"precommit": str(IA / 'configs' / 'IA_STD_Q15_API1_PRECOMMIT.json'), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
