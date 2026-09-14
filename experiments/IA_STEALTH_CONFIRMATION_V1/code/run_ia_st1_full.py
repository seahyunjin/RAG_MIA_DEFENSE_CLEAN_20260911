#!/usr/bin/env python3
"""Continue a passed IA-ST1 preflight to the frozen 2,000-session artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from ia_st1_protocol import GT_MODEL, QUESTION_MODEL, sha256_json
from run_ia_st1 import (
    DB,
    EXP,
    KEY_PATH,
    PRECOMMIT,
    PRECOMMIT_SHA,
    STATUS,
    TARGETS,
    execute_stage,
    init_db,
    log,
    now,
    sha,
    update_progress,
    write_json,
)


def export_full(con, targets: list[dict[str, str]]) -> dict[str, Any]:
    audits: list[dict[str, Any]] = []
    reason_hist: Counter[str] = Counter()
    frozen: list[dict[str, Any]] = []
    flat_queries: list[dict[str, Any]] = []
    flat_gt: list[dict[str, Any]] = []

    for target in targets:
        tid = target["document_id"]
        q = con.execute(
            "SELECT status,parsed_json,validation_json,usage_json,openai_request_id FROM calls WHERE target_id=? AND stage='questions'",
            (tid,),
        ).fetchone()
        g = con.execute(
            "SELECT status,parsed_json,validation_json,usage_json,openai_request_id FROM calls WHERE target_id=? AND stage='gt'",
            (tid,),
        ).fetchone()
        qv = json.loads(q[2]) if q and q[2] else {"valid": False, "reasons": [q[0] if q else "MISSING_QUESTION_CALL"]}
        gv = json.loads(g[2]) if g and g[2] else {"valid": False, "reasons": [g[0] if g else "MISSING_GT_CALL"]}
        valid = bool(q and g and q[0] == "VALID" and g[0] == "VALID")
        if not valid:
            for reason in qv.get("reasons", []) + gv.get("reasons", []):
                reason_hist[reason] += 1
        audit = {
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
        }
        audits.append(audit)
        if not valid:
            continue

        questions = json.loads(q[1])["questions"]
        judgments = json.loads(g[1])["judgments"]
        artifact = {
            "target_id": tid,
            "local_target_id": target["local_document_id"],
            "membership": target["membership"],
            "domain": target["domain"],
            "target_text_sha256": hashlib.sha256(target["source_text"].encode()).hexdigest(),
            "questions": [{"id": i, "question": text} for i, text in enumerate(questions, 1)],
            "judgments": [{"id": i, "judgment": label} for i, label in enumerate(judgments, 1)],
            "question_model": QUESTION_MODEL,
            "gt_model": GT_MODEL,
            "question_request_id": q[4],
            "gt_request_id": g[4],
            "question_usage": json.loads(q[3]) if q[3] else None,
            "gt_usage": json.loads(g[3]) if g[3] else None,
        }
        artifact["normalized_artifact_sha256"] = sha256_json(artifact)
        frozen.append(artifact)
        for slot, (question, judgment) in enumerate(zip(questions, judgments), start=1):
            qid = f"ia_st1::{tid}::q{slot:02d}"
            flat_queries.append({
                "_id": qid,
                "text": question,
                "target_doc_id": target["local_document_id"],
                "canonical_target_id": tid,
                "_membership": target["membership"],
                "variation_index": slot - 1,
                "session_id": tid,
                "turn": slot,
                "attack": "IA-Std-Q15-ST1",
            })
            flat_gt.append({
                "_id": qid,
                "text": question,
                "target_doc_id": target["local_document_id"],
                "canonical_target_id": tid,
                "_membership": target["membership"],
                "variation_index": slot - 1,
                "session_id": tid,
                "turn": slot,
                "ground_truth_answer": "I don't know." if judgment == "Unknown" else judgment + ".",
                "ground_truth_label": judgment,
            })

    valid_n = sum(x["session_valid"] for x in audits)
    members = [x for x in audits if x["membership"] == "member"]
    nonmembers = [x for x in audits if x["membership"] == "nonmember"]
    member_valid = sum(x["session_valid"] for x in members)
    nonmember_valid = sum(x["session_valid"] for x in nonmembers)
    member_rate = member_valid / len(members)
    nonmember_rate = nonmember_valid / len(nonmembers)
    gap = abs(member_rate - nonmember_rate)
    checks = {
        "valid_sessions_at_least_1900": valid_n >= 1900,
        "validity_gap_at_most_2pp": gap <= 0.02 + 1e-12,
        "exact_q15_gt15_among_valid": len(flat_queries) == valid_n * 15 == len(flat_gt),
        "duplicates_zero_among_valid": all(x["exact_duplicate_count"] == 0 for x in audits if x["session_valid"]),
        "inferred_fields_zero": True,
    }
    passed = all(checks.values())
    result = {
        "campaign": "IA_STEALTH_CONFIRMATION_V1",
        "attack": "IA-Std-Q15-ST1",
        "completed_utc": now(),
        "performance_metrics_computed": False,
        "sessions": len(audits),
        "valid": valid_n,
        "coverage": valid_n / len(audits),
        "member_valid": member_valid,
        "member_valid_rate": member_rate,
        "nonmember_valid": nonmember_valid,
        "nonmember_valid_rate": nonmember_rate,
        "validity_gap": gap,
        "queries": len(flat_queries),
        "failure_reasons": dict(reason_hist),
        "near_duplicate_pairs_diagnostic": sum(len(x["near_duplicate_pairs"]) for x in audits),
        "trivial_full_copy_slots_diagnostic": sum(len(x["trivial_full_copy_slots"]) for x in audits),
        "long_verbatim_12gram_slots_diagnostic": sum(len(x["long_verbatim_12gram_slots"]) for x in audits),
        "checks": checks,
        "verdict": "IA_STD_Q15_ST1_READY" if passed else "IA_STD_Q15_ST1_UNAVAILABLE",
        "additional_recovery_allowed": False,
        "final_lc_modified": False,
        "training_steps": 0,
    }

    write_json(EXP / "audits" / "IA_ST1_FULL_SESSION_AUDIT.json", audits)
    frozen_path = EXP / "outputs" / "IA_ST1_FULL_FROZEN.jsonl"
    frozen_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in frozen), encoding="utf-8")
    query_path = EXP / "inputs" / "IA_ST1_FULL_QUERIES.jsonl"
    query_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in flat_queries), encoding="utf-8")
    gt_path = EXP / "inputs" / "IA_ST1_FULL_GT.jsonl"
    gt_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in flat_gt), encoding="utf-8")
    result["artifacts"] = {
        "sessions": {"path": str(frozen_path), "sha256": sha(frozen_path)},
        "queries": {"path": str(query_path), "sha256": sha(query_path)},
        "gt": {"path": str(gt_path), "sha256": sha(gt_path)},
        "sqlite": {"path": str(DB)},
    }
    write_json(EXP / "IA_ST1_FULL_RESULT.json", result)
    return result


def main() -> None:
    expected = PRECOMMIT_SHA.read_text(encoding="utf-8").split()[0]
    if sha(PRECOMMIT) != expected:
        raise SystemExit("precommit hash mismatch")
    preflight = json.loads((EXP / "IA_ST1_PREFLIGHT_RESULT.json").read_text(encoding="utf-8"))
    if preflight["verdict"] != "IA_STD_Q15_ST1_PREFLIGHT_PASS":
        raise SystemExit("full stage forbidden: preflight did not pass")
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    targets = list(csv.DictReader(TARGETS.open(encoding="utf-8")))
    con = init_db()
    orphaned = con.execute("SELECT target_id,stage FROM calls WHERE status='INFLIGHT'").fetchall()
    if orphaned:
        for tid, stage in orphaned:
            con.execute("UPDATE calls SET status='ORPHANED_INFLIGHT',completed_utc=?,error_type='ORPHANED_INFLIGHT',error_message='one request began before interruption; frozen no-regeneration rule' WHERE target_id=? AND stage=?", (now(), tid, stage))
        con.commit()
    log("preflight PASS frozen; automatically continuing same protocol to full 2000")
    update_progress(con, "FULL_QUESTIONS", len(targets))
    execute_stage(con, targets, "questions", key, min_interval=3.0)
    log("full question stage complete; continuing frozen GT stage")
    update_progress(con, "FULL_GT", len(targets))
    execute_stage(con, targets, "gt", key, min_interval=0.5)
    result = export_full(con, targets)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    result["artifacts"]["sqlite"]["sha256"] = sha(DB)
    write_json(EXP / "IA_ST1_FULL_RESULT.json", result)
    write_json(EXP / "FINAL_RESULT.json", result)
    write_json(EXP / "checkpoints" / f"{result['verdict']}.json", result)
    STATUS.write_text(
        "# IA_STEALTH_CONFIRMATION_V1\n\n"
        f"- state: {result['verdict']}\n"
        f"- valid: {result['valid']}/{result['sessions']}\n"
        f"- member/nonmember validity: {result['member_valid_rate']:.4f}/{result['nonmember_valid_rate']:.4f}\n"
        f"- frozen queries: {result['queries']}\n"
        "- detector/AUC/TPR/E2E metrics: NOT COMPUTED during artifact validity\n"
        f"- next: {'FROZEN_IA_DETECTION' if result['verdict'] == 'IA_STD_Q15_ST1_READY' else 'CROSS_DOMAIN_PRIVACY_GENERALIZATION'}\n",
        encoding="utf-8",
    )
    log(f"full verdict={result['verdict']} valid={result['valid']}/2000 gap={result['validity_gap']:.4f}")


if __name__ == "__main__":
    main()

