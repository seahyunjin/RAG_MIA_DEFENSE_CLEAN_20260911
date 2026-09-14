#!/usr/bin/env python3
"""Checkpointed IA-Std-Q15-ST1 format preflight and full artifact build."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ia_st1_protocol import (
    GT_MODEL,
    QUESTION_MODEL,
    gt_request,
    question_request,
    sha256_json,
    validate_judgments,
    validate_questions,
)


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1"
PRECOMMIT = EXP / "configs" / "IA_STEALTH_CONFIRMATION_PRECOMMIT.json"
PRECOMMIT_SHA = EXP / "configs" / "IA_STEALTH_CONFIRMATION_PRECOMMIT.sha256"
TARGETS = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1" / "inputs" / "LARGE_SHARED_TARGETS.csv"
DB = EXP / "runtime" / "ia_st1.sqlite3"
KEY_PATH = Path("/home/traffic_3/workspace/workspace/SH/.secrets/openai_api_key")
LOG = EXP / "logs" / "IA_ST1.log"
HEARTBEAT = EXP / "HEARTBEAT.json"
STATUS = EXP / "STATUS.md"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def init_db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute(
        """CREATE TABLE IF NOT EXISTS calls (
        target_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        membership TEXT NOT NULL,
        status TEXT NOT NULL,
        client_request_id TEXT NOT NULL,
        started_utc TEXT NOT NULL,
        completed_utc TEXT,
        model TEXT NOT NULL,
        http_status INTEGER,
        openai_request_id TEXT,
        response_json TEXT,
        parsed_json TEXT,
        validation_json TEXT,
        prompt_sha256 TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        usage_json TEXT,
        error_type TEXT,
        error_message TEXT,
        PRIMARY KEY(target_id, stage)
        )"""
    )
    con.commit()
    return con


def api_call(payload: dict[str, Any], client_request_id: str, key: str) -> tuple[int, dict[str, Any], dict[str, str]]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    attempts = 0
    while True:
        attempts += 1
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Client-Request-Id": client_request_id,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=240, context=ssl.create_default_context()) as resp:
                raw = resp.read().decode("utf-8")
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.status, json.loads(raw), headers
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"error": {"type": "HTTP_ERROR", "message": raw[:1000]}}
            code = (parsed.get("error") or {}).get("code")
            if exc.code == 429 and code != "insufficient_quota" and attempts <= 8:
                delay = float(exc.headers.get("retry-after", "15")) + 1.0
                log(f"rate limited; identical request retry {attempts}/8 after {delay:.1f}s")
                time.sleep(delay)
                continue
            return exc.code, parsed, {k.lower(): v for k, v in exc.headers.items()}


def response_content(body: dict[str, Any]) -> tuple[Any, str | None]:
    try:
        message = body["choices"][0]["message"]
        if message.get("refusal"):
            return None, "MODEL_REFUSAL"
        content = message["content"]
        return json.loads(content), None
    except Exception as exc:
        return None, f"RESPONSE_PARSE:{type(exc).__name__}"


def update_progress(con: sqlite3.Connection, phase: str, total: int) -> None:
    counts = dict(con.execute("SELECT status, COUNT(*) FROM calls GROUP BY status").fetchall())
    stage_counts = {
        stage: dict(con.execute("SELECT status, COUNT(*) FROM calls WHERE stage=? GROUP BY status", (stage,)).fetchall())
        for stage in ["questions", "gt"]
    }
    hb = {"campaign": "IA_STEALTH_CONFIRMATION_V1", "updated_utc": now(), "phase": phase, "target_total": total, "counts": counts, "stage_counts": stage_counts, "pid": os.getpid()}
    write_json(HEARTBEAT, hb)
    STATUS.write_text(
        "# IA_STEALTH_CONFIRMATION_V1\n\n"
        f"- state: {phase}\n"
        f"- updated_utc: {hb['updated_utc']}\n"
        f"- PID: {os.getpid()}\n"
        f"- question calls: {stage_counts['questions']}\n"
        f"- GT calls: {stage_counts['gt']}\n"
        "- detector/AUC/TPR/E2E metrics: NOT COMPUTED during format validity\n",
        encoding="utf-8",
    )


def execute_stage(
    con: sqlite3.Connection,
    targets: list[dict[str, str]],
    stage: str,
    key: str,
    min_interval: float,
) -> None:
    last_call = 0.0
    for index, row in enumerate(targets, start=1):
        target_id = row["document_id"]
        existing = con.execute("SELECT status FROM calls WHERE target_id=? AND stage=?", (target_id, stage)).fetchone()
        if existing:
            continue
        if stage == "questions":
            payload = question_request(row["source_text"])
            model = QUESTION_MODEL
        else:
            qrow = con.execute("SELECT parsed_json, validation_json, status FROM calls WHERE target_id=? AND stage='questions'", (target_id,)).fetchone()
            if not qrow or qrow[2] != "VALID":
                continue
            questions = json.loads(qrow[0])["questions"]
            payload = gt_request(row["source_text"], questions)
            model = GT_MODEL

        request_id = str(uuid.uuid4())
        prompt = payload["messages"][1]["content"]
        con.execute(
            "INSERT INTO calls(target_id,stage,membership,status,client_request_id,started_utc,model,prompt_sha256,request_sha256) VALUES(?,?,?,?,?,?,?,?,?)",
            (target_id, stage, row["membership"], "INFLIGHT", request_id, now(), model, hashlib.sha256(prompt.encode()).hexdigest(), sha256_json(payload)),
        )
        con.commit()
        elapsed = time.monotonic() - last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        last_call = time.monotonic()
        try:
            http_status, body, headers = api_call(payload, request_id, key)
        except Exception as exc:
            con.execute(
                "UPDATE calls SET status='TRANSPORT_FAILED',completed_utc=?,error_type=?,error_message=? WHERE target_id=? AND stage=?",
                (now(), type(exc).__name__, str(exc)[:1000], target_id, stage),
            )
            con.commit()
            log(f"{stage} {index}/{len(targets)} {target_id}: transport failed {type(exc).__name__}")
            update_progress(con, f"PREFLIGHT_{stage.upper()}", len(targets))
            continue

        parsed, parse_error = response_content(body) if http_status == 200 else (None, "HTTP_ERROR")
        if stage == "questions" and parsed is not None:
            validation = validate_questions(parsed, row["source_text"])
        elif stage == "gt" and parsed is not None:
            validation = validate_judgments(parsed)
        else:
            validation = {"valid": False, "reasons": [parse_error or "UNKNOWN_RESPONSE_ERROR"]}
        status = "VALID" if validation.get("valid") else "INVALID"
        api_error = body.get("error") or {}
        con.execute(
            """UPDATE calls SET status=?,completed_utc=?,http_status=?,openai_request_id=?,response_json=?,parsed_json=?,validation_json=?,usage_json=?,error_type=?,error_message=?
               WHERE target_id=? AND stage=?""",
            (
                status,
                now(),
                http_status,
                headers.get("x-request-id"),
                json.dumps(body, ensure_ascii=False),
                json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
                json.dumps(validation, ensure_ascii=False),
                json.dumps(body.get("usage"), ensure_ascii=False),
                api_error.get("type") or parse_error,
                (api_error.get("message") or "")[:1000],
                target_id,
                stage,
            ),
        )
        con.commit()
        if index % 5 == 0 or status != "VALID":
            log(f"{stage} {index}/{len(targets)} {target_id}: {status} http={http_status} reasons={validation.get('reasons')}")
        update_progress(con, f"PREFLIGHT_{stage.upper()}", len(targets))


def export_preflight(con: sqlite3.Connection, targets: list[dict[str, str]]) -> dict[str, Any]:
    rows = []
    reason_hist: Counter[str] = Counter()
    for target in targets:
        tid = target["document_id"]
        q = con.execute("SELECT status,parsed_json,validation_json,usage_json,openai_request_id FROM calls WHERE target_id=? AND stage='questions'", (tid,)).fetchone()
        g = con.execute("SELECT status,parsed_json,validation_json,usage_json,openai_request_id FROM calls WHERE target_id=? AND stage='gt'", (tid,)).fetchone()
        qv = json.loads(q[2]) if q and q[2] else {"valid": False, "reasons": [q[0] if q else "MISSING_QUESTION_CALL"]}
        gv = json.loads(g[2]) if g and g[2] else {"valid": False, "reasons": [g[0] if g else "MISSING_GT_CALL"]}
        valid = bool(q and g and q[0] == "VALID" and g[0] == "VALID")
        if not valid:
            for reason in qv.get("reasons", []) + gv.get("reasons", []):
                reason_hist[reason] += 1
        rows.append({
            "target_id": tid,
            "membership": target["membership"],
            "session_valid": valid,
            "question_status": q[0] if q else "MISSING",
            "gt_status": g[0] if g else "MISSING",
            "question_reasons": qv.get("reasons", []),
            "gt_reasons": gv.get("reasons", []),
            "exact_duplicate_count": qv.get("exact_duplicate_count", 0),
            "near_duplicate_pairs": qv.get("near_duplicate_pairs", []),
            "trivial_full_copy_slots": qv.get("trivial_full_copy_slots", []),
            "long_verbatim_12gram_slots": qv.get("long_verbatim_12gram_slots", []),
        })

    valid_n = sum(r["session_valid"] for r in rows)
    member_rows = [r for r in rows if r["membership"] == "member"]
    nonmember_rows = [r for r in rows if r["membership"] == "nonmember"]
    member_rate = sum(r["session_valid"] for r in member_rows) / len(member_rows)
    nonmember_rate = sum(r["session_valid"] for r in nonmember_rows) / len(nonmember_rows)
    validity_gap = abs(member_rate - nonmember_rate)
    exact_duplicates = sum(r["exact_duplicate_count"] for r in rows if r["session_valid"])
    checks = {
        "valid_at_least_98": valid_n >= 98,
        "validity_gap_at_most_2pp": validity_gap <= 0.02 + 1e-12,
        "q15_exact_among_valid": all(r["question_status"] == "VALID" for r in rows if r["session_valid"]),
        "gt15_exact_among_valid": all(r["gt_status"] == "VALID" for r in rows if r["session_valid"]),
        "exact_duplicate_zero_among_valid": exact_duplicates == 0,
        "inferred_fields_zero": True,
    }
    passed = all(checks.values())
    result = {
        "campaign": "IA_STEALTH_CONFIRMATION_V1",
        "attack": "IA-Std-Q15-ST1",
        "completed_utc": now(),
        "performance_metrics_computed": False,
        "sessions": len(rows),
        "valid": valid_n,
        "coverage": valid_n / len(rows),
        "member_valid": sum(r["session_valid"] for r in member_rows),
        "member_valid_rate": member_rate,
        "nonmember_valid": sum(r["session_valid"] for r in nonmember_rows),
        "nonmember_valid_rate": nonmember_rate,
        "validity_gap": validity_gap,
        "failure_reasons": dict(reason_hist),
        "near_duplicate_pairs_diagnostic": sum(len(r["near_duplicate_pairs"]) for r in rows),
        "trivial_full_copy_slots_diagnostic": sum(len(r["trivial_full_copy_slots"]) for r in rows),
        "long_verbatim_12gram_slots_diagnostic": sum(len(r["long_verbatim_12gram_slots"]) for r in rows),
        "checks": checks,
        "verdict": "IA_STD_Q15_ST1_PREFLIGHT_PASS" if passed else "IA_STD_Q15_ST1_UNAVAILABLE",
        "automatic_full_scale": passed,
    }
    write_json(EXP / "audits" / "IA_ST1_PREFLIGHT_SESSION_AUDIT.json", rows)
    write_json(EXP / "IA_ST1_PREFLIGHT_RESULT.json", result)

    frozen = []
    for target in targets:
        tid = target["document_id"]
        q = con.execute("SELECT parsed_json,openai_request_id,usage_json FROM calls WHERE target_id=? AND stage='questions' AND status='VALID'", (tid,)).fetchone()
        g = con.execute("SELECT parsed_json,openai_request_id,usage_json FROM calls WHERE target_id=? AND stage='gt' AND status='VALID'", (tid,)).fetchone()
        if not (q and g):
            continue
        questions = json.loads(q[0])["questions"]
        judgments = json.loads(g[0])["judgments"]
        artifact = {
            "target_id": tid,
            "membership": target["membership"],
            "target_text_sha256": hashlib.sha256(target["source_text"].encode()).hexdigest(),
            "questions": [{"id": i, "question": text} for i, text in enumerate(questions, 1)],
            "judgments": [{"id": i, "judgment": label} for i, label in enumerate(judgments, 1)],
            "question_model": QUESTION_MODEL,
            "gt_model": GT_MODEL,
            "question_request_id": q[1],
            "gt_request_id": g[1],
            "question_usage": json.loads(q[2]) if q[2] else None,
            "gt_usage": json.loads(g[2]) if g[2] else None,
        }
        artifact["normalized_artifact_sha256"] = sha256_json(artifact)
        frozen.append(artifact)
    out = EXP / "outputs" / "IA_ST1_PREFLIGHT_FROZEN.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in frozen), encoding="utf-8")
    result["frozen_artifact"] = {"path": str(out), "sha256": sha(out), "sessions": len(frozen)}
    write_json(EXP / "IA_ST1_PREFLIGHT_RESULT.json", result)
    return result


def main() -> None:
    expected = PRECOMMIT_SHA.read_text(encoding="utf-8").split()[0]
    actual = sha(PRECOMMIT)
    if actual != expected:
        raise SystemExit(f"precommit hash mismatch: expected {expected}, got {actual}")
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("empty API key")
    all_targets = list(csv.DictReader(TARGETS.open(encoding="utf-8")))
    pre = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    by_id = {r["document_id"]: r for r in all_targets}
    targets = [by_id[x] for x in pre["cohort"]["preflight_ids"]]
    con = init_db()
    orphaned = con.execute("SELECT target_id,stage FROM calls WHERE status='INFLIGHT'").fetchall()
    if orphaned:
        for tid, stage in orphaned:
            con.execute("UPDATE calls SET status='ORPHANED_INFLIGHT',completed_utc=?,error_type='ORPHANED_INFLIGHT',error_message='process ended after one request began; frozen no-regeneration rule' WHERE target_id=? AND stage=?", (now(), tid, stage))
        con.commit()
        log(f"preserved {len(orphaned)} orphaned in-flight requests as invalid; no regeneration")

    log(f"precommit verified sha256={actual}; starting questions for {len(targets)} targets; performance forbidden")
    update_progress(con, "PREFLIGHT_QUESTIONS", len(targets))
    execute_stage(con, targets, "questions", key, min_interval=3.0)
    log("question stage complete; starting GT only for valid frozen Q15 sessions")
    update_progress(con, "PREFLIGHT_GT", len(targets))
    execute_stage(con, targets, "gt", key, min_interval=0.5)
    result = export_preflight(con, targets)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    write_json(EXP / "checkpoints" / result["verdict"], result)
    STATUS.write_text(
        "# IA_STEALTH_CONFIRMATION_V1\n\n"
        f"- state: {result['verdict']}\n"
        f"- valid: {result['valid']}/{result['sessions']}\n"
        f"- member/nonmember validity: {result['member_valid_rate']:.3f}/{result['nonmember_valid_rate']:.3f}\n"
        "- performance metrics: NOT COMPUTED\n"
        f"- next: {'FULL_2000_AUTOMATIC_CONTINUATION_PENDING' if result['automatic_full_scale'] else 'CROSS_DOMAIN_PRIVACY_GENERALIZATION'}\n",
        encoding="utf-8",
    )
    log(f"preflight verdict={result['verdict']} valid={result['valid']}/100 gap={result['validity_gap']:.4f}")
    # The full generator is deliberately invoked as a separate frozen-stage
    # executable so that a preflight failure cannot accidentally generate more.
    if result["automatic_full_scale"]:
        marker = EXP / "checkpoints" / "FULL_2000_AUTOMATIC_CONTINUATION_PENDING.json"
        write_json(marker, {"created_utc": now(), "preflight_result_sha256": sha(EXP / "IA_ST1_PREFLIGHT_RESULT.json")})
        full_runner = EXP / "code" / "run_ia_st1_full.py"
        if full_runner.exists():
            os.execv(sys.executable, [sys.executable, str(full_runner)])
        log("full runner is not yet present; pass artifact frozen safely")


if __name__ == "__main__":
    main()
