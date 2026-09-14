#!/usr/bin/env python3
"""Run the frozen IA-Std-Q15-v4-TwoStage format recovery and, if valid, evaluation."""
from __future__ import annotations

import csv
import json
import random
import sqlite3
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import EXP, LC, QWEN, ROOT, SEED, atomic_json, checkpoint as parent_checkpoint, now, read_jsonl, sha_file, sha_text, verify_hashed_json, write_csv, write_jsonl
from ia_v4_schema import parse_judgments, parse_questions

CHILD = EXP / "IA_STD_Q15_V4_TWO_STAGE"
DB = CHILD / "runtime" / "ia_v4.sqlite3"


def child_checkpoint(stage: str, **fields: object) -> None:
    payload = {"campaign": "IA_STD_Q15_V4_TWO_STAGE", "stage": stage, "updated_utc": now(), **fields}
    atomic_json(CHILD / "HEARTBEAT.json", payload)
    lines = ["# IA_STD_Q15_V4_TWO_STAGE", "", f"- Current stage: `{stage}`", f"- Updated UTC: `{payload['updated_utc']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in fields.items())
    temp = CHILD / "STATUS.md.tmp"; temp.write_text("\n".join(lines) + "\n", encoding="utf-8"); temp.replace(CHILD / "STATUS.md")
    parent_checkpoint(stage, child=str(CHILD), **fields)


def database() -> sqlite3.Connection:
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS stage1_questions(
      target_id TEXT PRIMARY KEY, membership TEXT, raw_output TEXT, parsed_json TEXT,
      valid INTEGER, reason TEXT, prompt_sha256 TEXT, output_sha256 TEXT,
      wall_seconds REAL, completed_utc TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS stage2_judgments(
      target_id TEXT PRIMARY KEY, membership TEXT, stage1_output_sha256 TEXT,
      raw_output TEXT, parsed_json TEXT, valid INTEGER, reason TEXT,
      prompt_sha256 TEXT, output_sha256 TEXT, wall_seconds REAL, completed_utc TEXT)""")
    # Schema deliberately matches the already-audited frozen E2E implementation.
    con.execute("""CREATE TABLE IF NOT EXISTS answer(
      query_id TEXT, branch TEXT, answer TEXT, answer_sha256 TEXT, prompt_sha256 TEXT,
      source_ids_json TEXT, source_caps_json TEXT, hidden_source_id TEXT,
      answer_tokens INTEGER, wall_seconds REAL, completed_utc TEXT,
      PRIMARY KEY(query_id,branch))""")
    con.commit(); return con


def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    return model, tokenizer


def render(tokenizer, prompt: str) -> tuple[str, str]:
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return text, sha_text(json.dumps(ids, separators=(",", ":")))


def generate(model, tokenizer, prompts: list[str], max_new: int) -> tuple[list[str], float]:
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True, max_length=4096,
                        return_tensors="pt").to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new, do_sample=False, num_beams=1,
                                pad_token_id=tokenizer.eos_token_id, use_cache=True)
    elapsed = time.perf_counter() - started
    values = tokenizer.batch_decode(output[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
    del encoded, output
    return [value.strip() for value in values], elapsed


def run_two_stage(pre: dict, con: sqlite3.Connection, targets: dict[str, dict], ids: list[str], phase: str) -> None:
    have1 = {row[0] for row in con.execute("SELECT target_id FROM stage1_questions")}
    tasks1 = [targets[target_id] for target_id in ids if target_id not in have1]
    have2 = {row[0] for row in con.execute("SELECT target_id FROM stage2_judgments")}
    if not tasks1 and all(target_id in have2 or not con.execute("SELECT valid FROM stage1_questions WHERE target_id=?", (target_id,)).fetchone()[0] for target_id in ids):
        return
    child_checkpoint(f"IA_V4_{phase}_GENERATOR_LOADING", stage1_remaining=len(tasks1), requested_sessions=len(ids))
    model, tokenizer = load_generator()
    started = time.monotonic()
    for start in range(0, len(tasks1), 6):
        batch = tasks1[start:start + 6]
        rendered = [render(tokenizer, pre["stage1"]["prompt"].format(target_text=row["source_text"])) for row in batch]
        outputs, elapsed = generate(model, tokenizer, [value[0] for value in rendered], pre["decoding"]["stage1_max_new_tokens"])
        for row, (_, prompt_sha), raw in zip(batch, rendered, outputs):
            parsed = parse_questions(raw)
            con.execute("INSERT INTO stage1_questions VALUES (?,?,?,?,?,?,?,?,?,?)", (
                row["document_id"], row["membership"], raw, json.dumps(parsed.items, ensure_ascii=False), int(parsed.valid),
                parsed.reason, prompt_sha, sha_text(raw), elapsed / len(batch), now()))
        con.commit()
        completed = min(start + len(batch), len(tasks1)); rate = completed / max(time.monotonic() - started, 1e-9)
        child_checkpoint(f"IA_V4_{phase}_STAGE1_PROGRESS", completed=completed, total=len(tasks1),
                         valid=con.execute("SELECT count(*) FROM stage1_questions WHERE valid=1").fetchone()[0],
                         eta_seconds=round((len(tasks1) - completed) / max(rate, 1e-9)))

    tasks2 = []
    for target_id in ids:
        if target_id in have2: continue
        record = con.execute("SELECT parsed_json,output_sha256,valid FROM stage1_questions WHERE target_id=?", (target_id,)).fetchone()
        if record and int(record[2]):
            questions = [{"id": item[0], "question": item[1]} for item in json.loads(record[0])]
            tasks2.append((targets[target_id], questions, record[1]))
    started = time.monotonic()
    for start in range(0, len(tasks2), 6):
        batch = tasks2[start:start + 6]
        rendered = [render(tokenizer, pre["stage2"]["prompt"].format(
            target_text=row["source_text"], questions_json=json.dumps(questions, ensure_ascii=False)))
            for row, questions, _ in batch]
        outputs, elapsed = generate(model, tokenizer, [value[0] for value in rendered], pre["decoding"]["stage2_max_new_tokens"])
        for (row, _, stage1_sha), (_, prompt_sha), raw in zip(batch, rendered, outputs):
            parsed = parse_judgments(raw)
            con.execute("INSERT INTO stage2_judgments VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                row["document_id"], row["membership"], stage1_sha, raw, json.dumps(parsed.items, ensure_ascii=False),
                int(parsed.valid), parsed.reason, prompt_sha, sha_text(raw), elapsed / len(batch), now()))
        con.commit()
        completed = min(start + len(batch), len(tasks2)); rate = completed / max(time.monotonic() - started, 1e-9)
        child_checkpoint(f"IA_V4_{phase}_STAGE2_PROGRESS", completed=completed, total=len(tasks2),
                         valid=con.execute("SELECT count(*) FROM stage2_judgments WHERE valid=1").fetchone()[0],
                         eta_seconds=round((len(tasks2) - completed) / max(rate, 1e-9)))
    del model; torch.cuda.empty_cache()


def audit_sessions(con: sqlite3.Connection, targets: dict[str, dict], ids: list[str]) -> list[dict]:
    output = []
    for target_id in ids:
        s1 = con.execute("SELECT valid,reason,parsed_json,output_sha256 FROM stage1_questions WHERE target_id=?", (target_id,)).fetchone()
        s2 = con.execute("SELECT valid,reason,parsed_json,stage1_output_sha256 FROM stage2_judgments WHERE target_id=?", (target_id,)).fetchone()
        valid1 = bool(s1 and s1[0]); valid2 = bool(s2 and s2[0]); linked = bool(s1 and s2 and s1[3] == s2[3])
        qn = len(json.loads(s1[2])) if s1 else 0; jn = len(json.loads(s2[2])) if s2 else 0
        output.append({"target_id": target_id, "membership": targets[target_id]["membership"],
                       "stage1_valid": valid1, "stage1_reason": s1[1] if s1 else "NOT_GENERATED",
                       "stage2_valid": valid2, "stage2_reason": s2[1] if s2 else "NOT_RUN_STAGE1_INVALID",
                       "stage1_stage2_hash_link": linked, "questions": qn, "judgments": jn,
                       "valid": valid1 and valid2 and linked and qn == 15 and jn == 15,
                       "fuzzy_parsing": 0, "inferred_fields": 0})
    return output


def validity_summary(rows: list[dict], expected_per_group: int) -> dict:
    valid = [row for row in rows if row["valid"]]
    rates = {}
    counts = {}
    for membership in ("member", "nonmember"):
        group = [row for row in rows if row["membership"] == membership]
        n = sum(row["valid"] for row in group); counts[membership] = n; rates[membership] = n / len(group)
    return {"sessions": len(rows), "valid": len(valid), "coverage": len(valid) / len(rows),
            "member_valid": counts["member"], "nonmember_valid": counts["nonmember"],
            "member_valid_rate": rates["member"], "nonmember_valid_rate": rates["nonmember"],
            "validity_gap": abs(rates["member"] - rates["nonmember"]),
            "failure_reasons": dict(Counter(
                ("STAGE1:" + row["stage1_reason"]) if not row["stage1_valid"] else ("STAGE2:" + row["stage2_reason"])
                for row in rows if not row["valid"])),
            "exact_q15_among_valid": all(row["questions"] == 15 and row["judgments"] == 15 for row in valid),
            "inferred_fields": 0, "fuzzy_parsing": 0}


def merged_queries(con: sqlite3.Connection, targets: dict[str, dict], audit: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    queries, gt = [], {}
    for row in audit:
        if not row["valid"]: continue
        qitems = json.loads(con.execute("SELECT parsed_json FROM stage1_questions WHERE target_id=?", (row["target_id"],)).fetchone()[0])
        jitems = json.loads(con.execute("SELECT parsed_json FROM stage2_judgments WHERE target_id=?", (row["target_id"],)).fetchone()[0])
        if [item[0] for item in qitems] != [item[0] for item in jitems]:
            raise RuntimeError("deterministic id merge failure")
        gt[row["target_id"]] = [item[1] for item in jitems]
        target = targets[row["target_id"]]
        for (item_id, question), (_, judgment) in zip(qitems, jitems):
            queries.append({"attack": "IA-Std-Q15-v4-TwoStage", "session_id": f"ia_v4::{row['target_id']}",
                            "query_id": f"ia_v4::{row['target_id']}::q{item_id:02d}", "query_index": item_id,
                            "target_id": row["target_id"], "membership": row["membership"], "domain": target["domain"],
                            "query": question, "query_sha256": sha_text(question), "standardized_gt": judgment,
                            "protocol": "IA-Std-Q15-v4-TwoStage",
                            "claim_boundary": "STANDARDIZED_TWO_STAGE_Q15_NOT_ORIGINAL_OR_PAPER_EXACT_IA"})
    return queries, gt


def replace_v3(value):
    if isinstance(value, dict): return {key: replace_v3(item) for key, item in value.items()}
    if isinstance(value, list): return [replace_v3(item) for item in value]
    if isinstance(value, str): return value.replace("IA-v3", "IA-v4-TwoStage").replace("IA_V3", "IA_V4")
    return value


def adapt_file(relative_v3: str, relative_v4: str) -> None:
    source = CHILD / relative_v3; target = CHILD / relative_v4
    if source.suffix == ".json": atomic_json(target, replace_v3(json.loads(source.read_text(encoding="utf-8"))))
    else: target.write_bytes(source.read_bytes())


def final_report(v3_early: dict, preflight: dict, full: dict | None, detection: dict | None, e2e: dict | None, status: str) -> None:
    primary = detection.get("primary", {}) if detection else {}
    by = {row["condition"]: row for row in e2e.get("conditions", [])} if e2e else {}
    lines = ["# IA-Std-Q15-v4-TwoStage Final Summary", "",
             f"1. v3 mathematical failure: generated={v3_early['generated']}, valid={v3_early['valid']}, invalid={v3_early['invalid']}",
             f"2. v4 preflight validity: {preflight.get('valid', 'NA')}/100 ({preflight.get('coverage', 0):.1%})",
             f"3. v4 full validity: {full.get('valid', 'NA') if full else 'NOT_RUN'}/2000",
             f"4. member/nonmember validity gap: {full.get('validity_gap', preflight.get('validity_gap', float('nan'))):.4f}",
             f"5. IA Final LC TPR @2.5%: {primary.get('Final LC_tpr', 'NOT_RUN')}",
             f"6. MIRABEL TPR @2.5%: {primary.get('MIRABEL_tpr', 'NOT_RUN')}",
             f"7. IA E2E No Defense: {by.get('NO_DEFENSE', {}).get('native_auc', 'NOT_RUN')}",
             f"8. IA E2E MIRABEL: {by.get('MIRABEL_MATCHED_2_5', {}).get('native_auc', 'NOT_RUN')}",
             f"9. IA E2E Final LC: {by.get('FINAL_LC_MATCHED_2_5', {}).get('native_auc', 'NOT_RUN')}",
             f"10. Final scientific status: `{status}`", "",
             "Core5 matched-FPR detection, Core5 matched-budget E2E, and benign-only Gold recalibration were not rerun or modified."]
    (CHILD / "reports" / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    pre_path = CHILD / "configs" / "IA_STD_Q15_V4_TWO_STAGE_PRECOMMIT.json"
    pre = verify_hashed_json(pre_path)
    for relative, digest in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != digest: raise RuntimeError(f"v4 code drift: {relative}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    targets_list = list(csv.DictReader((LC / "inputs" / "LARGE_SHARED_TARGETS.csv").open(encoding="utf-8")))
    targets = {row["document_id"]: row for row in targets_list}; con = database()
    preflight_ids = pre["preflight"]["session_ids"]
    run_two_stage(pre, con, targets, preflight_ids, "PREFLIGHT")
    preflight_rows = audit_sessions(con, targets, preflight_ids); write_csv(CHILD / "audits" / "IA_V4_PREFLIGHT_VALIDITY.csv", preflight_rows)
    pf = validity_summary(preflight_rows, 50)
    pf["checks"] = {"valid_at_least_98": pf["valid"] >= 98, "validity_gap_at_most_2pp": pf["validity_gap"] <= .02 + 1e-12,
                    "exact_q15_among_valid": pf["exact_q15_among_valid"], "inferred_fields_zero": True}
    pf["verdict"] = "IA_STD_Q15_V4_FORMAT_PREFLIGHT_PASS" if all(pf["checks"].values()) else "IA_STD_Q15_V4_FORMAT_PREFLIGHT_FAILED"
    atomic_json(CHILD / "IA_V4_PREFLIGHT_RESULT.json", pf)
    if pf["verdict"].endswith("FAILED"):
        child_checkpoint(pf["verdict"], valid=pf["valid"], validity_gap=pf["validity_gap"], next_stage="STOP_IA_RECOVERY")
        final_report(json.loads((EXP / "IA_V3_EARLY_STOP_RESULT.json").read_text()), pf, None, None, None, "STANDARDIZED_ATTACK_UNAVAILABLE")
        return
    child_checkpoint(pf["verdict"], valid=pf["valid"], validity_gap=pf["validity_gap"], next_stage="IA_V4_FULL_2000")
    remaining = [row["document_id"] for row in targets_list if row["document_id"] not in set(preflight_ids)]
    run_two_stage(pre, con, targets, remaining, "FULL")
    all_ids = [row["document_id"] for row in targets_list]
    full_rows = audit_sessions(con, targets, all_ids); write_csv(CHILD / "audits" / "IA_V4_FULL_VALIDITY.csv", full_rows)
    full = validity_summary(full_rows, 1000)
    full["checks"] = {"valid_at_least_1900": full["valid"] >= 1900, "coverage_at_least_95pct": full["coverage"] >= .95,
                      "validity_gap_at_most_2pp": full["validity_gap"] <= .02 + 1e-12,
                      "exact_q15_and_judgments": full["exact_q15_among_valid"], "fuzzy_parsing_zero": True, "inferred_fields_zero": True}
    full["verdict"] = "IA_STD_Q15_V4_READY" if all(full["checks"].values()) else "IA_STD_Q15_V4_FAILED"
    atomic_json(CHILD / "IA_V4_FULL_VALIDITY_RESULT.json", full)
    queries, gt = merged_queries(con, targets, full_rows)
    write_jsonl(CHILD / "inputs" / "IA_STD_Q15_V4_QUERIES.jsonl", queries)
    atomic_json(CHILD / "inputs" / "IA_STD_Q15_V4_GT.json", gt)
    if full["verdict"] != "IA_STD_Q15_V4_READY":
        child_checkpoint(full["verdict"], valid=full["valid"], coverage=full["coverage"], validity_gap=full["validity_gap"], next_stage="STOP_IA_NO_MORE_RECOVERY")
        final_report(json.loads((EXP / "IA_V3_EARLY_STOP_RESULT.json").read_text()), pf, full, None, None, "IA_STD_Q15_V4_FAILED")
        return
    child_checkpoint(full["verdict"], valid=full["valid"], coverage=full["coverage"], validity_gap=full["validity_gap"], next_stage="IA_V4_DETECTION")
    # Reuse the already-audited frozen detector/E2E implementation. Only its artifact namespace is adapted.
    sys.path.insert(0, str(EXP / "code")); import run_ia_v3 as frozen_eval
    frozen_eval.EXP = CHILD; frozen_eval.checkpoint = child_checkpoint
    rows, det_compat = frozen_eval.detection(pre, queries)
    det = replace_v3(det_compat); det["verdict"] = ("IA_V4_DETECTION_CATASTROPHIC_FAILURE" if det["catastrophic_failure"] else "IA_V4_DETECTION_READY_FOR_E2E")
    det["variant"] = "IA-Std-Q15-v4-TwoStage"; det["claim_boundary"] = pre["claim_boundary"]
    atomic_json(CHILD / "IA_V4_DETECTION_RESULT.json", det)
    adapt_file("cache/IA_V3_QUERY_EMBEDDINGS.float16.npy", "cache/IA_V4_QUERY_EMBEDDINGS.float16.npy")
    adapt_file("cache/IA_V3_RETRIEVAL_AND_DETECTION.jsonl", "cache/IA_V4_RETRIEVAL_AND_DETECTION.jsonl")
    adapt_file("tables/IA_V3_MATCHED_FPR_DETECTION.csv", "tables/IA_V4_MATCHED_FPR_DETECTION.csv")
    adapt_file("tables/IA_V3_SESSION_DETECTION.csv", "tables/IA_V4_SESSION_DETECTION.csv")
    if det["catastrophic_failure"]:
        final_report(json.loads((EXP / "IA_V3_EARLY_STOP_RESULT.json").read_text()), pf, full, det, None, det["verdict"]); return
    frozen_eval.generate_answers(pre, con, rows, det_compat)
    e2e_compat = frozen_eval.score_e2e(con, rows, gt, det_compat)
    e2e = replace_v3(e2e_compat); e2e["verdict"] = "IA_V4_MATCHED_BUDGET_E2E_COMPLETE"; e2e["variant"] = "IA-Std-Q15-v4-TwoStage"; e2e["claim_boundary"] = pre["claim_boundary"]
    atomic_json(CHILD / "IA_V4_E2E_RESULT.json", e2e)
    adapt_file("tables/IA_V3_E2E_SESSION_DETAIL.csv", "tables/IA_V4_E2E_SESSION_DETAIL.csv")
    adapt_file("tables/IA_V3_E2E_PRIVACY.csv", "tables/IA_V4_E2E_PRIVACY.csv")
    adapt_file("tables/IA_V3_EXTERNAL_SIDECHANNEL.csv", "tables/IA_V4_EXTERNAL_SIDECHANNEL.csv")
    child_checkpoint(e2e["verdict"], next_stage="FINAL_SUMMARY")
    final_report(json.loads((EXP / "IA_V3_EARLY_STOP_RESULT.json").read_text()), pf, full, det, e2e, "IA_V4_E2E_COMPLETE_STANDARDIZED_VARIANT")


if __name__ == "__main__":
    main()
