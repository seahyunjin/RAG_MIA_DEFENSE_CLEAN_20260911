#!/usr/bin/env python3
"""Checkpointed IA-Std-Q15-API1 preflight and full artifact generation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import ssl
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from common import EXP, LC, ROOT, atomic_json, atomic_text, checkpoint, now, sha_file, verify_hashed_json, write_jsonl
from ia_api_protocol import (
    GT_MODEL, QUESTION_MODEL, gt_request, question_request, sha256_json,
    validate_judgments, validate_questions,
)


IA = EXP / "IA_STD_Q15_API1"
PRECOMMIT = IA / "configs" / "IA_STD_Q15_API1_PRECOMMIT.json"
TARGETS = LC / "inputs" / "LARGE_SHARED_TARGETS.csv"
KEY_PATH = Path("/home/traffic_3/workspace/workspace/SH/.secrets/openai_api_key")
DB = IA / "runtime" / "ia_api1.sqlite3"
LOG = IA / "logs" / "IA_API1.log"
STATUS = IA / "STATUS.md"
HEARTBEAT = IA / "HEARTBEAT.json"
PRICE_PER_MILLION = {
    QUESTION_MODEL: {"input": 2.50, "output": 10.00},
    GT_MODEL: {"input": 0.15, "output": 0.60},
}


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def init_db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS call (
      target_id TEXT NOT NULL, membership TEXT NOT NULL, stage TEXT NOT NULL,
      attempt INTEGER NOT NULL, status TEXT NOT NULL, client_request_id TEXT NOT NULL,
      started_utc TEXT NOT NULL, completed_utc TEXT, model TEXT NOT NULL,
      http_status INTEGER, openai_request_id TEXT, response_json TEXT, parsed_json TEXT,
      validation_json TEXT, prompt_sha256 TEXT NOT NULL, request_sha256 TEXT NOT NULL,
      usage_json TEXT, estimated_cost_usd REAL NOT NULL DEFAULT 0,
      error_type TEXT, error_message TEXT,
      PRIMARY KEY(target_id,stage,attempt))""")
    con.commit()
    return con


def estimated_spend(con: sqlite3.Connection) -> float:
    return float(con.execute("SELECT COALESCE(SUM(estimated_cost_usd),0) FROM call").fetchone()[0])


def usage_cost(model: str, usage: dict[str, Any] | None) -> float:
    if not usage:
        return 0.0
    price = PRICE_PER_MILLION[model]
    return (float(usage.get("prompt_tokens", 0)) * price["input"]
            + float(usage.get("completion_tokens", 0)) * price["output"]) / 1_000_000


def api_call(payload: dict[str, Any], request_id: str, key: str) -> tuple[int, dict[str, Any], dict[str, str]]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    for transport_attempt in range(1, 6):
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Client-Request-Id": request_id},
        )
        try:
            with urllib.request.urlopen(request, timeout=240, context=ssl.create_default_context()) as response:
                return response.status, json.loads(response.read().decode("utf-8")), {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"error": {"type": "HTTP_ERROR", "message": raw[:1000]}}
            code = (parsed.get("error") or {}).get("code")
            if exc.code == 429 and code != "insufficient_quota" and transport_attempt < 5:
                delay = min(60.0, float(exc.headers.get("retry-after", "15")) + 1.0)
                log(f"rate-limited identical transport retry {transport_attempt}/4 after {delay:.1f}s")
                time.sleep(delay)
                continue
            return exc.code, parsed, {k.lower(): v for k, v in exc.headers.items()}
    raise RuntimeError("unreachable transport retry state")


def parse_response(body: dict[str, Any]) -> tuple[Any, str | None]:
    try:
        message = body["choices"][0]["message"]
        if message.get("refusal"):
            return None, "MODEL_REFUSAL"
        return json.loads(message["content"]), None
    except Exception as exc:
        return None, f"RESPONSE_PARSE_{type(exc).__name__}"


def latest(con: sqlite3.Connection, target_id: str, stage: str):
    return con.execute(
        "SELECT attempt,status,parsed_json,validation_json FROM call WHERE target_id=? AND stage=? ORDER BY attempt DESC LIMIT 1",
        (target_id, stage),
    ).fetchone()


def valid_payload(con: sqlite3.Connection, target_id: str, stage: str) -> dict[str, Any] | None:
    row = con.execute(
        "SELECT parsed_json FROM call WHERE target_id=? AND stage=? AND status='VALID' ORDER BY attempt DESC LIMIT 1",
        (target_id, stage),
    ).fetchone()
    return json.loads(row[0]) if row else None


def update_progress(con: sqlite3.Connection, scope: str, stage: str, total: int) -> None:
    counts = dict(con.execute("SELECT status,COUNT(*) FROM call GROUP BY status").fetchall())
    terminal = con.execute(
        "SELECT COUNT(DISTINCT target_id) FROM call WHERE stage=? AND status IN ('VALID','INVALID','HTTP_FAILED','TRANSPORT_FAILED','ORPHANED_INFLIGHT','BUDGET_STOP')",
        (stage,),
    ).fetchone()[0]
    value = {
        "campaign": EXP.name, "attack": "IA-Std-Q15-API1", "scope": scope, "stage": stage,
        "updated_utc": now(), "target_total": total, "terminal_targets": terminal,
        "call_status_counts": counts, "estimated_campaign_spend_usd": estimated_spend(con), "pid": os.getpid(),
        "performance_metrics_computed": False,
    }
    atomic_json(HEARTBEAT, value)
    atomic_text(STATUS, "\n".join([
        "# IA-Std-Q15-API1", "", f"- scope: `{scope}`", f"- stage: `{stage}`",
        f"- terminal targets: `{terminal}/{total}`", f"- estimated campaign API spend: `${value['estimated_campaign_spend_usd']:.4f}`",
        "- detector/privacy metrics: `NOT COMPUTED` during validity generation", f"- updated UTC: `{value['updated_utc']}`",
    ]) + "\n")


def run_one_attempt(con: sqlite3.Connection, row: dict[str, str], stage: str, attempt: int,
                    key: str, ceiling: float) -> str:
    if estimated_spend(con) >= ceiling - 0.05:
        request_id = str(uuid.uuid4())
        con.execute(
            "INSERT INTO call(target_id,membership,stage,attempt,status,client_request_id,started_utc,completed_utc,model,prompt_sha256,request_sha256,error_type,error_message) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["document_id"], row["membership"], stage, attempt, "BUDGET_STOP", request_id, now(), now(),
             QUESTION_MODEL if stage == "questions" else GT_MODEL, "NOT_SENT", "NOT_SENT", "CAMPAIGN_COST_CEILING", f"estimated campaign spend reached USD {ceiling:.2f}"),
        )
        con.commit()
        return "BUDGET_STOP"
    if stage == "questions":
        payload = question_request(row["source_text"])
        model = QUESTION_MODEL
    else:
        qvalue = valid_payload(con, row["document_id"], "questions")
        if qvalue is None:
            return "QUESTION_NOT_VALID"
        payload = gt_request(row["source_text"], qvalue["questions"])
        model = GT_MODEL
    request_id = str(uuid.uuid4())
    prompt = payload["messages"][1]["content"]
    con.execute(
        "INSERT INTO call(target_id,membership,stage,attempt,status,client_request_id,started_utc,model,prompt_sha256,request_sha256) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (row["document_id"], row["membership"], stage, attempt, "INFLIGHT", request_id, now(), model,
         hashlib.sha256(prompt.encode()).hexdigest(), sha256_json(payload)),
    )
    con.commit()
    try:
        http_status, body, headers = api_call(payload, request_id, key)
    except Exception as exc:
        con.execute(
            "UPDATE call SET status='TRANSPORT_FAILED',completed_utc=?,error_type=?,error_message=? WHERE target_id=? AND stage=? AND attempt=?",
            (now(), type(exc).__name__, str(exc)[:1000], row["document_id"], stage, attempt),
        )
        con.commit()
        return "TRANSPORT_FAILED"
    parsed, parse_error = parse_response(body) if http_status == 200 else (None, "HTTP_ERROR")
    if parsed is None:
        validation = {"valid": False, "structural_valid": False, "reasons": [parse_error or "HTTP_ERROR"]}
    elif stage == "questions":
        validation = validate_questions(parsed)
    else:
        validation = validate_judgments(parsed)
    status = "VALID" if validation["valid"] else ("INVALID" if http_status == 200 else "HTTP_FAILED")
    usage = body.get("usage")
    error = body.get("error") or {}
    con.execute(
        """UPDATE call SET status=?,completed_utc=?,http_status=?,openai_request_id=?,response_json=?,parsed_json=?,validation_json=?,usage_json=?,estimated_cost_usd=?,error_type=?,error_message=?
           WHERE target_id=? AND stage=? AND attempt=?""",
        (status, now(), http_status, headers.get("x-request-id"), json.dumps(body, ensure_ascii=False),
         json.dumps(parsed, ensure_ascii=False) if parsed is not None else None, json.dumps(validation, ensure_ascii=False),
         json.dumps(usage, ensure_ascii=False), usage_cost(model, usage), error.get("type") or parse_error,
         (error.get("message") or "")[:1000], row["document_id"], stage, attempt),
    )
    con.commit()
    return status


def execute_stage(con: sqlite3.Connection, targets: list[dict[str, str]], stage: str, key: str,
                  ceiling: float, scope: str) -> None:
    for index, row in enumerate(targets, 1):
        existing_valid = valid_payload(con, row["document_id"], stage)
        if existing_valid is not None:
            continue
        current = latest(con, row["document_id"], stage)
        if current and current[1] in {"INVALID", "HTTP_FAILED", "TRANSPORT_FAILED", "ORPHANED_INFLIGHT", "BUDGET_STOP"}:
            # Exactly one second call is allowed only after a structural schema failure.
            validation = json.loads(current[3]) if current[3] else {}
            retry = current[1] == "INVALID" and validation.get("structural_valid") is False and current[0] == 1
            if not retry:
                continue
            attempt = 2
        elif current:
            continue
        else:
            attempt = 1
        status = run_one_attempt(con, row, stage, attempt, key, ceiling)
        if status == "INVALID" and attempt == 1:
            validation = json.loads(latest(con, row["document_id"], stage)[3])
            if validation.get("structural_valid") is False:
                status = run_one_attempt(con, row, stage, 2, key, ceiling)
        if index % 5 == 0 or status != "VALID":
            log(f"{scope} {stage} {index}/{len(targets)} target={row['document_id']} status={status}")
        update_progress(con, scope, stage, len(targets))


def session_audit(con: sqlite3.Connection, targets: list[dict[str, str]]) -> tuple[list[dict], dict]:
    audits, reason_hist = [], Counter()
    for target in targets:
        q = latest(con, target["document_id"], "questions")
        g = latest(con, target["document_id"], "gt")
        qvalid = valid_payload(con, target["document_id"], "questions")
        gvalid = valid_payload(con, target["document_id"], "gt")
        valid = qvalid is not None and gvalid is not None
        qvalidation = json.loads(q[3]) if q and q[3] else {"reasons": [q[1] if q else "MISSING_QUESTIONS"]}
        gvalidation = json.loads(g[3]) if g and g[3] else {"reasons": [g[1] if g else "MISSING_GT"]}
        if not valid:
            reason_hist.update(qvalidation.get("reasons", [])); reason_hist.update(gvalidation.get("reasons", []))
        audits.append({
            "target_id": target["document_id"], "membership": target["membership"], "valid": valid,
            "question_status": q[1] if q else "MISSING", "gt_status": g[1] if g else "MISSING",
            "question_attempts": q[0] if q else 0, "gt_attempts": g[0] if g else 0,
            "question_reasons": qvalidation.get("reasons", []), "gt_reasons": gvalidation.get("reasons", []),
            "exact_duplicate_count": qvalidation.get("exact_duplicate_count", 0), "inferred_fields": 0,
        })
    members = [row for row in audits if row["membership"] == "member"]
    nonmembers = [row for row in audits if row["membership"] == "nonmember"]
    member_rate = sum(row["valid"] for row in members) / len(members)
    nonmember_rate = sum(row["valid"] for row in nonmembers) / len(nonmembers)
    summary = {
        "sessions": len(audits), "valid": sum(row["valid"] for row in audits),
        "coverage": sum(row["valid"] for row in audits) / len(audits),
        "member_valid": sum(row["valid"] for row in members), "member_valid_rate": member_rate,
        "nonmember_valid": sum(row["valid"] for row in nonmembers), "nonmember_valid_rate": nonmember_rate,
        "validity_gap": abs(member_rate - nonmember_rate), "failure_reasons": dict(reason_hist),
        "exact_q15_gt15": all(row["question_status"] == "VALID" and row["gt_status"] == "VALID" for row in audits if row["valid"]),
        "duplicate_zero": all(row["exact_duplicate_count"] == 0 for row in audits if row["valid"]),
        "inferred_fields": 0,
    }
    return audits, summary


def export_full(con: sqlite3.Connection, targets: list[dict[str, str]], summary: dict) -> dict:
    sessions, queries, gt = [], [], []
    for target in targets:
        questions = valid_payload(con, target["document_id"], "questions")
        judgments = valid_payload(con, target["document_id"], "gt")
        if questions is None or judgments is None:
            continue
        session = {
            "target_id": target["document_id"], "local_target_id": target["local_document_id"],
            "membership": target["membership"], "domain": target["domain"],
            "target_text_sha256": hashlib.sha256(target["source_text"].encode()).hexdigest(),
            "questions": questions["questions"], "judgments": judgments["judgments"],
            "question_model": QUESTION_MODEL, "gt_model": GT_MODEL,
        }
        session["normalized_artifact_sha256"] = sha256_json(session)
        sessions.append(session)
        labels = {row["id"]: row["judgment"] for row in judgments["judgments"]}
        for item in questions["questions"]:
            qid = f"ia_api1::{target['document_id']}::q{item['id']:02d}"
            common = {
                "_id": qid, "text": item["question"], "target_doc_id": target["local_document_id"],
                "canonical_target_id": target["document_id"], "_membership": target["membership"],
                "variation_index": item["id"] - 1, "session_id": target["document_id"], "turn": item["id"],
                "attack": "IA-Std-Q15-API1",
            }
            queries.append(common)
            label = labels[item["id"]]
            gt.append({**common, "ground_truth_answer": "I don't know." if label == "Unknown" else label + ".", "ground_truth_label": label})
    paths = {
        "sessions": IA / "outputs" / "IA_API1_FULL_FROZEN.jsonl",
        "queries": IA / "inputs" / "IA_API1_FULL_QUERIES.jsonl",
        "gt": IA / "inputs" / "IA_API1_FULL_GT.jsonl",
    }
    write_jsonl(paths["sessions"], sessions); write_jsonl(paths["queries"], queries); write_jsonl(paths["gt"], gt)
    result = {
        "campaign": EXP.name, "attack": "IA-Std-Q15-API1", "completed_utc": now(),
        "verdict": "IA_STD_Q15_API1_READY", **summary, "queries": len(queries),
        "artifacts": {name: {"path": str(path), "sha256": sha_file(path)} for name, path in paths.items()},
        "sqlite": {"path": str(DB)}, "estimated_campaign_spend_usd": estimated_spend(con),
        "performance_metrics_computed": False, "final_lc_modified": False, "additional_recovery_allowed": False,
    }
    atomic_json(IA / "FINAL_RESULT.json", result)
    return result


def main() -> None:
    pre = verify_hashed_json(PRECOMMIT)
    for relative, digest in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != digest:
            raise RuntimeError(f"Phase-C code drift: {relative}")
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("OpenAI key unavailable")
    all_targets = list(csv.DictReader(TARGETS.open(encoding="utf-8")))
    by_id = {row["document_id"]: row for row in all_targets}
    preflight = [by_id[target_id] for target_id in pre["cohort"]["preflight_ids"]]
    ceiling = float(pre["spending_safety"]["campaign_estimated_cost_ceiling_usd"])
    con = init_db()
    orphaned = con.execute("SELECT target_id,stage,attempt FROM call WHERE status='INFLIGHT'").fetchall()
    for target_id, stage, attempt in orphaned:
        con.execute("UPDATE call SET status='ORPHANED_INFLIGHT',completed_utc=?,error_type='ORPHANED_INFLIGHT',error_message='request outcome ambiguous after interruption; no regeneration' WHERE target_id=? AND stage=? AND attempt=?", (now(), target_id, stage, attempt))
    con.commit()

    log("precommit verified; starting 100-session format-only preflight; performance metrics forbidden")
    execute_stage(con, preflight, "questions", key, ceiling, "PREFLIGHT")
    execute_stage(con, preflight, "gt", key, ceiling, "PREFLIGHT")
    audits, summary = session_audit(con, preflight)
    checks = {
        "valid_at_least_98": summary["valid"] >= 98,
        "validity_gap_at_most_2pp": summary["validity_gap"] <= 0.02 + 1e-12,
        "exact_q15_gt15": summary["exact_q15_gt15"], "duplicate_zero": summary["duplicate_zero"],
        "inferred_fields_zero": summary["inferred_fields"] == 0,
    }
    passed = all(checks.values())
    preflight_result = {
        "campaign": EXP.name, "attack": "IA-Std-Q15-API1", "completed_utc": now(),
        **summary, "checks": checks,
        "verdict": "IA_API1_PROTOCOL_READY" if passed else "IA_API1_PROTOCOL_FAILED",
        "performance_metrics_computed": False, "estimated_campaign_spend_usd": estimated_spend(con),
    }
    atomic_json(IA / "audits" / "PREFLIGHT_SESSION_AUDIT.json", audits)
    atomic_json(IA / "PREFLIGHT_RESULT.json", preflight_result)
    if not passed:
        atomic_json(IA / "FINAL_RESULT.json", preflight_result)
        checkpoint("IA_API1_PROTOCOL_FAILED", valid=f"{summary['valid']}/100", gap=summary["validity_gap"])
        con.close(); print(json.dumps(preflight_result, indent=2)); return

    checkpoint("IA_API1_PROTOCOL_READY", valid=f"{summary['valid']}/100", next="FULL_2000")
    log("preflight PASS; automatically continuing identical frozen protocol to full 2,000")
    execute_stage(con, all_targets, "questions", key, ceiling, "FULL_2000")
    execute_stage(con, all_targets, "gt", key, ceiling, "FULL_2000")
    full_audits, full = session_audit(con, all_targets)
    full_checks = {
        "valid_at_least_1900": full["valid"] >= 1900,
        "validity_gap_at_most_2pp": full["validity_gap"] <= 0.02 + 1e-12,
        "exact_q15_gt15": full["exact_q15_gt15"], "duplicate_zero": full["duplicate_zero"],
        "inferred_fields_zero": full["inferred_fields"] == 0,
    }
    atomic_json(IA / "audits" / "FULL_SESSION_AUDIT.json", full_audits)
    if all(full_checks.values()):
        result = export_full(con, all_targets, {**full, "checks": full_checks})
        checkpoint("IA_STD_Q15_API1_READY", valid=f"{full['valid']}/2000", queries=result["queries"], next="IA_DETECTION")
    else:
        result = {
            "campaign": EXP.name, "attack": "IA-Std-Q15-API1", "completed_utc": now(),
            "verdict": "IA_API1_PROTOCOL_FAILED", **full, "checks": full_checks,
            "estimated_campaign_spend_usd": estimated_spend(con), "performance_metrics_computed": False,
            "additional_recovery_allowed": False, "final_lc_modified": False,
        }
        atomic_json(IA / "FINAL_RESULT.json", result)
        checkpoint("IA_API1_PROTOCOL_FAILED", valid=f"{full['valid']}/2000", gap=full["validity_gap"])
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)"); con.close()
    if "sqlite" in result:
        result["sqlite"]["sha256"] = sha_file(DB)
    atomic_json(IA / "FINAL_RESULT.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
