#!/usr/bin/env python3
"""One-shot structured IA-Q15-v3 recovery, detection, and matched-budget E2E."""
from __future__ import annotations

import bisect
import csv
import json
import math
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BGE, BUDGETS, CORE3, EXP, FINAL8, K_LOCAL, LC, QWEN, ROOT, SEED, atomic_json, checkpoint, now, read_jsonl, sha_file, sha_text, verify_hashed_json, write_csv, write_jsonl
from ia_schema import parse_exact

sys.path.insert(0, str(ROOT / "code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments, MIRABEL_FORMULA_VERSION

SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any required output format exactly. "
                 "If the context is insufficient, answer exactly: I don't know.")


def db() -> sqlite3.Connection:
    con = sqlite3.connect(EXP / "runtime" / "ia_v3.sqlite3", timeout=120)
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS query_generation(
      target_id TEXT PRIMARY KEY, membership TEXT, raw_outputs_json TEXT, parsed_json TEXT,
      valid INTEGER, reason TEXT, selected_attempt INTEGER, prompt_sha256 TEXT,
      wall_seconds REAL, completed_utc TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS ground_truth(
      target_id TEXT PRIMARY KEY, raw_outputs_json TEXT, labels_json TEXT,
      valid INTEGER, reason TEXT, selected_attempt INTEGER, prompt_sha256 TEXT,
      wall_seconds REAL, completed_utc TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS answer(
      query_id TEXT, branch TEXT, answer TEXT, answer_sha256 TEXT, prompt_sha256 TEXT,
      source_ids_json TEXT, source_caps_json TEXT, hidden_source_id TEXT,
      answer_tokens INTEGER, wall_seconds REAL, completed_utc TEXT,
      PRIMARY KEY(query_id,branch))""")
    con.commit(); return con


def render(tokenizer, prompt: str) -> tuple[str, str]:
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return text, sha_text(json.dumps(ids, separators=(",", ":")))


def generate(model, tokenizer, prompts: list[str], max_new: int) -> tuple[list[str], float]:
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True, max_length=4096, return_tensors="pt").to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new, do_sample=False, num_beams=1,
                                pad_token_id=tokenizer.eos_token_id, use_cache=True)
    elapsed = time.perf_counter() - started
    values = tokenizer.batch_decode(output[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
    del encoded, output
    return [value.strip() for value in values], elapsed


def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    return model, tokenizer


def generate_queries(pre: dict, con: sqlite3.Connection, targets: list[dict]) -> tuple[list[dict], dict]:
    completed = {row[0] for row in con.execute("SELECT target_id FROM query_generation")}
    tasks = []
    for target in targets:
        if target["document_id"] in completed: continue
        prompt = pre["query_prompt"].format(target_text=target["source_text"])
        tasks.append((target, prompt))
    if tasks:
        checkpoint("IA_V3_QUERY_GENERATOR_LOADING", remaining=len(tasks))
        model, tokenizer = load_generator(); started = time.monotonic(); done_before = len(targets) - len(tasks)
        for start in range(0, len(tasks), 6):
            batch = tasks[start:start + 6]; rendered = [render(tokenizer, prompt) for _, prompt in batch]
            outputs, elapsed = generate(model, tokenizer, [item[0] for item in rendered], pre["decoding"]["query_max_new_tokens"])
            for (target, _), (_, prompt_sha), raw in zip(batch, rendered, outputs):
                attempts = [raw]; parsed = parse_exact(raw); selected = 1 if parsed.valid else None
                if not parsed.valid:
                    retry_prompt = pre["query_retry_prompt"].format(target_text=target["source_text"])
                    retry_rendered, prompt_sha = render(tokenizer, retry_prompt)
                    retry_output, retry_elapsed = generate(model, tokenizer, [retry_rendered], pre["decoding"]["query_max_new_tokens"])
                    attempts.append(retry_output[0]); elapsed += retry_elapsed; parsed = parse_exact(retry_output[0]); selected = 2 if parsed.valid else None
                con.execute("INSERT INTO query_generation VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    target["document_id"], target["membership"], json.dumps(attempts, ensure_ascii=False),
                    json.dumps(parsed.items, ensure_ascii=False), int(parsed.valid), parsed.reason, selected,
                    prompt_sha, elapsed / len(batch), now()))
            con.commit(); completed_now = done_before + min(start + len(batch), len(tasks))
            rate = (completed_now - done_before) / max(time.monotonic() - started, 1e-9)
            checkpoint("IA_V3_QUERY_GENERATION_PROGRESS", completed=completed_now, total=2000,
                       valid=con.execute("SELECT count(*) FROM query_generation WHERE valid=1").fetchone()[0],
                       eta_seconds=round((2000 - completed_now) / max(rate, 1e-9)))
        del model; torch.cuda.empty_cache()
    rows = list(con.execute("SELECT target_id,membership,raw_outputs_json,parsed_json,valid,reason,selected_attempt,prompt_sha256 FROM query_generation ORDER BY target_id"))
    if len(rows) != 2000: raise RuntimeError("IA-v3 query-generation row count drift")
    audit, queries = [], []
    target_map = {row["document_id"]: row for row in targets}
    for target_id, membership, raw, parsed_json, valid, reason, selected, prompt_sha in rows:
        items = json.loads(parsed_json)
        audit.append({"target_id": target_id, "membership": membership, "valid": bool(valid), "reason": reason,
                      "selected_attempt": selected or "", "entries": len(items), "inferred_fields": 0,
                      "fuzzy_parsing": 0, "raw_outputs_sha256": sha_text(raw), "prompt_sha256": prompt_sha})
        if valid:
            target = target_map[target_id]
            for item_id, question, judgment in items:
                queries.append({"attack": "IA-Std-Q15-v3", "session_id": f"ia_v3::{target_id}",
                                "query_id": f"ia_v3::{target_id}::q{item_id:02d}", "query_index": item_id,
                                "target_id": target_id, "membership": membership, "domain": target["domain"],
                                "query": question, "query_sha256": sha_text(question),
                                "query_generator_judgment_not_scoring_gt": judgment,
                                "protocol": "IA-Std-Q15-v3", "claim_boundary": "STANDARDIZED_STRUCTURED_Q15_NOT_PAPER_EXACT_Q30"})
    write_csv(EXP / "audits" / "IA_V3_VALIDITY_AUDIT.csv", audit)
    write_jsonl(EXP / "inputs" / "IA_STD_Q15_V3_QUERIES.jsonl", queries)
    valid_rows = [row for row in audit if row["valid"]]; member = sum(row["membership"] == "member" for row in valid_rows)
    nonmember = sum(row["membership"] == "nonmember" for row in valid_rows); coverage = len(valid_rows) / 2000; gap = abs(member - nonmember) / 1000
    checks = {"coverage_at_least_95pct": coverage >= .95, "validity_gap_at_most_2pp": gap <= .02 + 1e-12,
              "exact_q15": all(row["entries"] == 15 for row in valid_rows), "inferred_fields_zero": True, "fuzzy_parsing_zero": True}
    verdict = "IA_STD_Q15_V3_READY" if all(checks.values()) else "IA_STD_Q15_V3_FAILED"
    result = {"campaign": EXP.name, "verdict": verdict, "completed_utc": now(), "sessions": 2000,
              "valid_sessions": len(valid_rows), "coverage": coverage, "member_valid": member, "nonmember_valid": nonmember,
              "validity_gap": gap, "valid_queries": len(queries), "failure_reasons": dict(Counter(row["reason"] for row in audit if not row["valid"])),
              "checks": checks, "inferred_fields": 0, "fuzzy_parsing": 0,
              "next_stage": "IA_V3_DETECTION" if verdict.endswith("READY") else "STOP_IA_NO_MORE_TUNING"}
    atomic_json(EXP / "IA_V3_VALIDITY_RESULT.json", result)
    checkpoint(verdict, valid_sessions=len(valid_rows), coverage=round(coverage, 6), member_valid=member,
               nonmember_valid=nonmember, validity_gap=round(gap, 6), next_stage=result["next_stage"])
    return queries, result


def threshold(values: list[float], alpha: float) -> tuple[float, int]:
    ordered = sorted(map(float, values), reverse=True); allowance = math.floor(alpha * len(ordered) + 1e-12)
    tau = ordered[allowance] if allowance < len(ordered) else -math.inf; alarms = sum(value > tau for value in ordered)
    if alarms > allowance: raise RuntimeError("benign budget exceeded")
    return tau, alarms


def bootstrap_delta(rows: list[dict], mt: float, lt: float) -> tuple[float, float]:
    groups = defaultdict(list)
    for row in rows: groups[row["session_id"]].append(row)
    units = sorted(groups); mh = np.asarray([sum(float(x["M"]) > mt for x in groups[u]) for u in units]); lh = np.asarray([sum(float(x["R_LC"]) > lt for x in groups[u]) for u in units])
    sizes = np.asarray([len(groups[u]) for u in units]); rng = np.random.default_rng(SEED); values = np.empty(2000)
    for i in range(2000):
        idx = rng.integers(0, len(units), len(units)); values[i] = (lh[idx].sum() - mh[idx].sum()) / sizes[idx].sum()
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def detection(pre: dict, queries: list[dict]) -> tuple[list[dict], dict]:
    prior = read_jsonl(LC / "cache" / "LARGE_DETECTION_SCORES.jsonl")
    prior_embeddings = np.asarray(np.load(LC / "cache" / "QUERY_EMBEDDINGS.float16.npy"), dtype=np.float32)
    ref_indices = [index for index, row in enumerate(prior) if row["split"] == "REFERENCE"]
    ref_embeddings = prior_embeddings[ref_indices]; ref_margins = [float(prior[index]["M"]) for index in ref_indices]
    documents = read_jsonl(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")
    document_embeddings = np.asarray(np.load(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy"), dtype=np.float32)
    checkpoint("IA_V3_BGE_LOADING", queries=len(queries))
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True); model.max_seq_length = 512
    embeddings = np.asarray(model.encode([row["query"] for row in queries], batch_size=48, show_progress_bar=True,
                                         convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    del model; torch.cuda.empty_cache(); np.save(EXP / "cache" / "IA_V3_QUERY_EMBEDDINGS.float16.npy", embeddings.astype(np.float16))
    doc_ids = [row["document_id"] for row in documents]; global_sorted = sorted(ref_margins); output = []
    for start in range(0, len(queries), 128):
        scores_batch = embeddings[start:start + 128] @ document_embeddings.T
        similarity_batch = embeddings[start:start + 128] @ ref_embeddings.T
        for offset, scores in enumerate(scores_batch):
            source = queries[start + offset]; idx = np.argpartition(-scores, 4)[:4]; idx = idx[np.argsort(-scores[idx], kind="stable")]
            ids = [doc_ids[int(i)] for i in idx]; top_scores = [float(scores[int(i)]) for i in idx]
            stat = canonical_mirabel_from_moments(top1=top_scores[0], sum_all=float(scores.sum(dtype=np.float64)),
                sumsq_all=float(np.square(scores, dtype=np.float64).sum(dtype=np.float64)), corpus_size=len(doc_ids), confidence=.95)
            sims = similarity_batch[offset]; neighbors = np.argpartition(-sims, K_LOCAL)[:K_LOCAL]
            neighbors = neighbors[np.argsort(-sims[neighbors], kind="stable")]; local = sorted(ref_margins[int(i)] for i in neighbors)
            p_local = (1 + len(local) - bisect.bisect_left(local, float(stat.margin))) / (len(local) + 1)
            p_global = (1 + len(global_sorted) - bisect.bisect_left(global_sorted, float(stat.margin))) / (len(global_sorted) + 1)
            output.append({**source, "top_document_ids": ids, "top_scores": top_scores, "selected_source_id": ids[0],
                           "target_rank": ids.index(source["target_id"]) + 1 if source["target_id"] in ids else 0,
                           "M": float(stat.margin), "mirabel_formula_version": MIRABEL_FORMULA_VERSION,
                           "p_local": p_local, "R_LC": -math.log(p_local), "p_global": p_global, "R_GLOBAL": -math.log(p_global)})
        checkpoint("IA_V3_DETECTION_PROGRESS", completed=min(start + 128, len(queries)), total=len(queries))
    path = EXP / "cache" / "IA_V3_RETRIEVAL_AND_DETECTION.jsonl"; write_jsonl(path, output)
    old = read_jsonl(FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl"); benign = [row for row in old if row["attack"] == "BENIGN"]
    thresholds, table = {}, []
    member_rows = [row for row in output if row["membership"] == "member"]
    for budget in BUDGETS:
        record = {"budget": budget, "member_queries": len(member_rows), "benign_queries": len(benign)}
        for method, key in (("MIRABEL", "M"), ("Final LC", "R_LC")):
            tau, alarms = threshold([row[key] for row in benign], budget); thresholds[(method, budget)] = tau
            record[f"{method}_threshold"] = tau; record[f"{method}_benign_alarms"] = alarms
            record[f"{method}_tpr"] = statistics.fmean(float(row[key]) > tau for row in member_rows)
        if budget == .025:
            record["delta"] = record["Final LC_tpr"] - record["MIRABEL_tpr"]
            record["delta_ci95_low"], record["delta_ci95_high"] = bootstrap_delta(member_rows, thresholds[("MIRABEL", budget)], thresholds[("Final LC", budget)])
        table.append(record)
    write_csv(EXP / "tables" / "IA_V3_MATCHED_FPR_DETECTION.csv", table)
    sessions = []
    for membership in ("member", "nonmember"):
        groups = defaultdict(list)
        for row in output:
            if row["membership"] == membership: groups[row["session_id"]].append(row)
        for session, rows in sorted(groups.items()):
            rows.sort(key=lambda row: row["query_index"])
            if len(rows) != 15: raise RuntimeError("IA-v3 Q15 detection drift")
            for method, key in (("MIRABEL", "M"), ("Final LC", "R_LC")):
                flags = [float(row[key]) > thresholds[(method, .025)] for row in rows]
                sessions.append({"membership": membership, "session_id": session, "method": method,
                                 "alarms": sum(flags), "alarm_fraction": sum(flags) / 15, "any_alarm": any(flags),
                                 "all_15_alarm": all(flags), "protected_queries": sum(flags)})
    write_csv(EXP / "tables" / "IA_V3_SESSION_DETECTION.csv", sessions)
    primary = next(row for row in table if row["budget"] == .025)
    catastrophic = primary["Final LC_tpr"] < .05
    result = {"campaign": EXP.name, "verdict": "IA_V3_DETECTION_CATASTROPHIC_FAILURE" if catastrophic else "IA_V3_DETECTION_READY_FOR_E2E",
              "completed_utc": now(), "table": table, "primary": primary, "catastrophic_failure": catastrophic,
              "thresholds": {f"{method}@{budget}": value for (method, budget), value in thresholds.items()},
              "next_stage": "STOP_IA_E2E" if catastrophic else "IA_V3_GT_AND_E2E"}
    atomic_json(EXP / "IA_V3_DETECTION_RESULT.json", result)
    checkpoint(result["verdict"], mirabel_tpr=round(primary["MIRABEL_tpr"], 6), final_lc_tpr=round(primary["Final LC_tpr"], 6), next_stage=result["next_stage"])
    return output, result


def generate_gt(pre: dict, con: sqlite3.Connection, targets: dict[str, dict], queries: list[dict]) -> dict[str, list[str]] | None:
    grouped = defaultdict(list)
    for row in queries: grouped[row["target_id"]].append(row)
    completed = {row[0] for row in con.execute("SELECT target_id FROM ground_truth")}; tasks = []
    for target_id in sorted(grouped):
        if target_id in completed: continue
        rows = sorted(grouped[target_id], key=lambda row: row["query_index"])
        question_json = json.dumps([{"id": row["query_index"], "question": row["query"]} for row in rows], ensure_ascii=False)
        tasks.append((target_id, rows, pre["gt_prompt"].format(target_text=targets[target_id]["source_text"], questions_json=question_json), question_json))
    if tasks:
        checkpoint("IA_V3_GT_GENERATOR_LOADING", remaining=len(tasks)); model, tokenizer = load_generator(); started = time.monotonic()
        for start in range(0, len(tasks), 6):
            batch = tasks[start:start + 6]; rendered = [render(tokenizer, task[2]) for task in batch]
            outputs, elapsed = generate(model, tokenizer, [item[0] for item in rendered], pre["decoding"]["gt_max_new_tokens"])
            for (target_id, rows, _, question_json), (_, prompt_sha), raw in zip(batch, rendered, outputs):
                attempts = [raw]; parsed = parse_exact(raw)
                exact = parsed.valid and [item[1] for item in parsed.items] == [row["query"] for row in rows]
                reason = parsed.reason if parsed.valid else parsed.reason; selected = 1 if exact else None
                if not exact:
                    retry_prompt = pre["gt_retry_prompt"].format(target_text=targets[target_id]["source_text"], questions_json=question_json)
                    retry_rendered, prompt_sha = render(tokenizer, retry_prompt); retry, retry_elapsed = generate(model, tokenizer, [retry_rendered], pre["decoding"]["gt_max_new_tokens"])
                    attempts.append(retry[0]); elapsed += retry_elapsed; parsed = parse_exact(retry[0])
                    exact = parsed.valid and [item[1] for item in parsed.items] == [row["query"] for row in rows]
                    reason = "VALID_EXACT_GT" if exact else (parsed.reason if not parsed.valid else "QUESTION_TEXT_CHANGED"); selected = 2 if exact else None
                else: reason = "VALID_EXACT_GT"
                labels = [item[2] for item in parsed.items] if exact else []
                con.execute("INSERT INTO ground_truth VALUES (?,?,?,?,?,?,?,?,?)", (target_id, json.dumps(attempts, ensure_ascii=False),
                    json.dumps(labels), int(exact), reason, selected, prompt_sha, elapsed / len(batch), now()))
            con.commit(); completed_now = len(completed) + min(start + len(batch), len(tasks)); rate = (completed_now - len(completed)) / max(time.monotonic() - started, 1e-9)
            checkpoint("IA_V3_GT_PROGRESS", completed=completed_now, total=len(grouped), valid=con.execute("SELECT count(*) FROM ground_truth WHERE valid=1").fetchone()[0],
                       eta_seconds=round((len(grouped) - completed_now) / max(rate, 1e-9)))
        del model; torch.cuda.empty_cache()
    rows = list(con.execute("SELECT target_id,labels_json,valid,reason FROM ground_truth ORDER BY target_id"))
    audit = [{"target_id": target, "valid": bool(valid), "labels": len(json.loads(labels)), "reason": reason,
              "unknown_labels": sum(value == "Unknown" for value in json.loads(labels))} for target, labels, valid, reason in rows]
    write_csv(EXP / "audits" / "IA_V3_GT_AUDIT.csv", audit)
    if len(rows) != len(grouped) or any(not row[2] for row in rows):
        atomic_json(EXP / "IA_V3_GT_RESULT.json", {"verdict": "IA_V3_GT_FAILED", "sessions": len(grouped),
                    "valid": sum(row[2] for row in rows), "no_session_excluded": True})
        checkpoint("IA_V3_GT_FAILED", sessions=len(grouped), valid=sum(row[2] for row in rows), next_stage="STOP_IA_E2E")
        return None
    result = {target: json.loads(labels) for target, labels, _, _ in rows}
    atomic_json(EXP / "IA_V3_GT_RESULT.json", {"verdict": "IA_V3_GT_READY", "sessions": len(result),
                "unknown_labels": sum(value == "Unknown" for labels in result.values() for value in labels), "inferred_fields": 0})
    return result


def waterfill(lengths: list[int], total: int = 2048) -> list[int]:
    caps = np.zeros(len(lengths), dtype=int); remaining = total; active = [i for i, value in enumerate(lengths) if value > 0]
    while remaining and active:
        share = max(1, remaining // len(active)); changed = False
        for index in list(active):
            add = min(share, lengths[index] - int(caps[index]), remaining); caps[index] += add; remaining -= add; changed |= bool(add)
            if caps[index] >= lengths[index]: active.remove(index)
            if not remaining: break
        if not changed: break
    return caps.tolist()


def build_answer_prompt(tokenizer, row: dict, docs: dict[str, str], hidden: str | None) -> tuple[str, dict]:
    kept = [source for source in row["top_document_ids"] if source != hidden]; encoded = [tokenizer(docs[source], add_special_tokens=False).input_ids for source in kept]
    caps = waterfill([len(value) for value in encoded]); visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip() for value, cap in zip(encoded, caps)]
    context = "\n\n".join(f"[Document {index}]\n{text}" for index, text in enumerate(visible, 1))
    user = f"Retrieved context:\n{context}\n\nUser query:\n{row['query']}"
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(ids) > 3072: ids = ids[-3072:]; rendered = tokenizer.decode(ids, skip_special_tokens=False)
    return rendered, {"prompt_sha256": sha_text(json.dumps(ids, separators=(",", ":"))), "source_ids": kept, "caps": caps, "hidden": hidden}


def generate_answers(pre: dict, con: sqlite3.Connection, rows: list[dict], detection_result: dict) -> None:
    thresholds = detection_result["thresholds"]; mt = thresholds["MIRABEL@0.025"]; lt = thresholds["Final LC@0.025"]
    docs = {row["document_id"]: row["source_text"] for row in read_jsonl(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")}
    existing = {(row[0], row[1]) for row in con.execute("SELECT query_id,branch FROM answer")}; tasks = []
    for row in rows:
        branches = [("A0", None)]
        if float(row["M"]) > mt or float(row["R_LC"]) > lt: branches.append(("A_HIDE", row["selected_source_id"]))
        for branch, hidden in branches:
            if (row["query_id"], branch) not in existing: tasks.append((row, branch, hidden))
    if not tasks: return
    checkpoint("IA_V3_E2E_GENERATOR_LOADING", remaining=len(tasks)); model, tokenizer = load_generator(); started = time.monotonic()
    for start in range(0, len(tasks), 24):
        batch = tasks[start:start + 24]; built = [build_answer_prompt(tokenizer, row, docs, hidden) for row, _, hidden in batch]
        outputs, elapsed = generate(model, tokenizer, [item[0] for item in built], pre["decoding"]["answer_max_new_tokens"])
        for (row, branch, _), answer, (_, provenance) in zip(batch, outputs, built):
            con.execute("INSERT INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?)", (row["query_id"], branch, answer,
                sha_text(answer), provenance["prompt_sha256"], json.dumps(provenance["source_ids"]), json.dumps(provenance["caps"]),
                provenance["hidden"], len(tokenizer(answer, add_special_tokens=False).input_ids), elapsed / len(batch), now()))
        con.commit(); completed = len(existing) + min(start + len(batch), len(tasks)); rate = (completed - len(existing)) / max(time.monotonic() - started, 1e-9)
        checkpoint("IA_V3_E2E_GENERATION_PROGRESS", completed=completed, total=len(existing) + len(tasks),
                   eta_seconds=round((len(existing) + len(tasks) - completed) / max(rate, 1e-9)))
    del model; torch.cuda.empty_cache()


def normalize_yes_no(answer: str) -> str:
    lowered = " ".join(answer.casefold().split())
    if any(term in lowered for term in ("i don't know", "i do not know", "not sure", "unable to")): return "UNK"
    found = set(re.findall(r"\b(?:yes|no)\b", lowered))
    if found == {"yes"}: return "Yes"
    if found == {"no"}: return "No"
    return "UNK"


def auc_summary(labels: list[int], scores: list[float], seed: int) -> dict:
    labels_array = np.asarray(labels); scores_array = np.asarray(scores); raw = float(roc_auc_score(labels_array, scores_array))
    member, nonmember = np.where(labels_array == 1)[0], np.where(labels_array == 0)[0]; rng = np.random.default_rng(seed); boot = np.empty(2000)
    for index in range(2000):
        chosen = np.concatenate((rng.choice(member, len(member), replace=True), rng.choice(nonmember, len(nonmember), replace=True)))
        boot[index] = roc_auc_score(labels_array[chosen], scores_array[chosen])
    return {"native_auc": raw, "e_auc_secondary": max(raw, 1 - raw), "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)), "n": len(labels), "member_n": len(member), "nonmember_n": len(nonmember)}


def score_e2e(con: sqlite3.Connection, rows: list[dict], gt: dict[str, list[str]], detection_result: dict) -> dict:
    columns = [item[0] for item in con.execute("SELECT * FROM answer LIMIT 1").description]
    answer_map = {(row["query_id"], row["branch"]): dict(zip(columns, values)) for values in con.execute("SELECT * FROM answer") for row in [dict(zip(columns, values))]}
    mt = detection_result["thresholds"]["MIRABEL@0.025"]; lt = detection_result["thresholds"]["Final LC@0.025"]
    conditions = {"NO_DEFENSE": lambda row: False, "MIRABEL_MATCHED_2_5": lambda row: float(row["M"]) > mt,
                  "FINAL_LC_MATCHED_2_5": lambda row: float(row["R_LC"]) > lt}
    results, side_rows = [], []
    for condition, rule in conditions.items():
        groups = defaultdict(list)
        for row in rows:
            branch = "A_HIDE" if rule(row) else "A0"; answer = answer_map[(row["query_id"], branch)]["answer"]
            groups[row["session_id"]].append((int(row["query_index"]), row, answer))
        labels, scores = [], []
        for session, group in sorted(groups.items()):
            group.sort(); target = group[0][1]["target_id"]
            if len(group) != 15 or len(gt[target]) != 15: raise RuntimeError("IA-v3 E2E Q15 drift")
            responses = [normalize_yes_no(answer) for _, _, answer in group]; total = sum(-5 if response == "UNK" else int(response == truth) for response, truth in zip(responses, gt[target]))
            membership = group[0][1]["membership"]; labels.append(int(membership == "member")); scores.append(total / 15)
            answers = [answer for _, _, answer in group]
            side_rows.append({"condition": condition, "session_id": session, "membership": membership,
                              "score": total / 15, "total_answer_length": sum(len(answer) for answer in answers),
                              "mean_answer_length": statistics.fmean(len(answer) for answer in answers),
                              "refusal_count": sum(normalize_yes_no(answer) == "UNK" for answer in answers),
                              "empty_count": sum(not answer.strip() for answer in answers)})
        summary = auc_summary(labels, scores, SEED + sum(map(ord, condition))); summary.update({"condition": condition, "metric": "IA-v3 session ROC-AUC"}); results.append(summary)
    write_csv(EXP / "tables" / "IA_V3_E2E_SESSION_DETAIL.csv", side_rows); write_csv(EXP / "tables" / "IA_V3_E2E_PRIVACY.csv", results)
    sidechannels = []
    for condition in conditions:
        subset = [row for row in side_rows if row["condition"] == condition]; labels = [int(row["membership"] == "member") for row in subset]
        for feature in ("total_answer_length", "mean_answer_length", "refusal_count", "empty_count"):
            result = auc_summary(labels, [float(row[feature]) for row in subset], SEED + sum(map(ord, condition + feature)))
            sidechannels.append({"condition": condition, "feature": feature, **result})
    write_csv(EXP / "tables" / "IA_V3_EXTERNAL_SIDECHANNEL.csv", sidechannels)
    by = {row["condition"]: row for row in results}; verdict = "IA_V3_MATCHED_BUDGET_E2E_COMPLETE"
    result = {"campaign": EXP.name, "verdict": verdict, "completed_utc": now(), "conditions": results,
              "final_lc_vs_no_defense_delta": by["FINAL_LC_MATCHED_2_5"]["native_auc"] - by["NO_DEFENSE"]["native_auc"],
              "final_lc_vs_matched_mirabel_delta": by["FINAL_LC_MATCHED_2_5"]["native_auc"] - by["MIRABEL_MATCHED_2_5"]["native_auc"],
              "worst_external_sidechannel_e_auc": max(row["e_auc_secondary"] for row in sidechannels if row["condition"] == "FINAL_LC_MATCHED_2_5"),
              "claim_boundary": "IA-Std-Q15-v3 standardized hard confirmation; not paper-exact IA Q30"}
    atomic_json(EXP / "IA_V3_E2E_RESULT.json", result)
    checkpoint(verdict, no_defense_auc=round(by["NO_DEFENSE"]["native_auc"], 6), mirabel_auc=round(by["MIRABEL_MATCHED_2_5"]["native_auc"], 6),
               final_lc_auc=round(by["FINAL_LC_MATCHED_2_5"]["native_auc"], 6), next_stage="FINAL_REPORT")
    return result


def main() -> None:
    pre = verify_hashed_json(EXP / "configs" / "IA_STD_Q15_V3_PRECOMMIT.json")
    for relative, digest in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != digest: raise RuntimeError(f"IA-v3 code drift: {relative}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    targets_list = list(csv.DictReader((LC / "inputs" / "LARGE_SHARED_TARGETS.csv").open(encoding="utf-8"))); targets = {row["document_id"]: row for row in targets_list}
    con = db(); queries, validity = generate_queries(pre, con, targets_list)
    if validity["verdict"] != "IA_STD_Q15_V3_READY": return
    rows, detection_result = detection(pre, queries)
    if detection_result["catastrophic_failure"]: return
    gt = generate_gt(pre, con, targets, queries)
    if gt is None: return
    generate_answers(pre, con, rows, detection_result)
    score_e2e(con, rows, gt, detection_result)


if __name__ == "__main__": main()
