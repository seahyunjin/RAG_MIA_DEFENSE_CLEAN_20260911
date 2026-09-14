#!/usr/bin/env python3
"""Run the final slotted, duplicate-constrained IA-v6 artifact recovery."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

CHILD = Path(__file__).resolve().parents[1]
ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
DB = CHILD / "runtime" / "ia_v6.sqlite3"
SEED = 20260912
sys.path.insert(0, str(CHILD / "code"))
from ia_v6_constraints import (AllowedSequenceTrieLogitsProcessor,
                               DuplicateSequenceLogitsProcessor,
                               FORBIDDEN_REFERENCES, FirstTokenMask, LABELS,
                               STARTERS, normalize_question, serialize_session,
                               validate_label, validate_question)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_text(value: str):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def heartbeat(stage: str, **fields):
    payload = {"campaign": "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED", "stage": stage,
               "updated_utc": now(), **fields}
    atomic_json(CHILD / "HEARTBEAT.json", payload)
    lines = ["# IA_STD_Q15_V6_DUPLICATE_CONSTRAINED", "", f"- Stage: `{stage}`",
             f"- Updated UTC: `{payload['updated_utc']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in fields.items())
    temporary = CHILD / "STATUS.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(CHILD / "STATUS.md")


def verify_precommit():
    path = CHILD / "configs" / "IA_STD_Q15_V6_PRECOMMIT.json"
    expected = (CHILD / "configs" / "IA_STD_Q15_V6_PRECOMMIT.sha256").read_text().split()[0]
    if sha_file(path) != expected:
        raise RuntimeError("precommit hash mismatch")
    precommit = json.loads(path.read_text(encoding="utf-8"))
    for relative, digest in precommit["code_sha256"].items():
        if sha_file(ROOT / relative) != digest:
            raise RuntimeError(f"code drift: {relative}")
    if sha_file(Path(precommit["cohort"]["path"])) != precommit["cohort"]["sha256"]:
        raise RuntimeError("cohort drift")
    if sha_file(Path(precommit["v5_frozen_failure"]["path"])) != precommit["v5_frozen_failure"]["sha256"]:
        raise RuntimeError("v5 frozen result drift")
    return precommit


def database():
    connection = sqlite3.connect(DB, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""CREATE TABLE IF NOT EXISTS question_slot(
      target_id TEXT, membership TEXT, slot INTEGER, previous_sha256 TEXT,
      raw_output TEXT, question TEXT, generated_token_ids_json TEXT,
      valid INTEGER, reason TEXT, prompt_sha256 TEXT, output_sha256 TEXT,
      wall_seconds REAL, completed_utc TEXT, PRIMARY KEY(target_id,slot))""")
    connection.execute("""CREATE TABLE IF NOT EXISTS judgment_slot(
      target_id TEXT, membership TEXT, slot INTEGER, question_sha256 TEXT,
      raw_output TEXT, judgment TEXT, generated_token_ids_json TEXT,
      valid INTEGER, reason TEXT, prompt_sha256 TEXT, wall_seconds REAL,
      completed_utc TEXT, PRIMARY KEY(target_id,slot))""")
    connection.commit()
    return connection


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    return model, tokenizer


def render(tokenizer, prompt: str):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return text, sha_text(json.dumps(ids, separators=(",", ":")))


def single_token_ids(tokenizer, values):
    output = []
    for value in values:
        for surface in (value, " " + value):
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(ids) == 1:
                output.append(ids[0])
    return sorted(set(output))


def question_mark_tokens(tokenizer):
    return sorted(token_id for token_id in range(tokenizer.vocab_size)
                  if "?" in tokenizer.decode([token_id], skip_special_tokens=False))


def forbidden_sequence_ids(tokenizer):
    output, seen = [], set()
    for phrase in FORBIDDEN_REFERENCES:
        for surface in (phrase, " " + phrase, phrase.title(), " " + phrase.title()):
            ids = tuple(tokenizer.encode(surface, add_special_tokens=False))
            if ids and ids not in seen:
                output.append(list(ids)); seen.add(ids)
    return output


def trim_generated(sequence, stop_ids, eos_token_id):
    kept = []
    for token_id in sequence:
        token_id = int(token_id)
        if token_id == eos_token_id:
            break
        kept.append(token_id)
        if token_id in stop_ids:
            break
    return kept


def generate_questions(model, tokenizer, prompts, forbidden_by_row, allowed_start, stop_ids, bad_words, max_new):
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True,
                        max_length=4096, return_tensors="pt").to("cuda")
    input_length = encoded.input_ids.shape[1]
    processors = LogitsProcessorList([
        FirstTokenMask(input_length, allowed_start),
        DuplicateSequenceLogitsProcessor(input_length, forbidden_by_row),
    ])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=max_new, do_sample=False, num_beams=1,
            logits_processor=processors, bad_words_ids=bad_words,
            eos_token_id=sorted(set([tokenizer.eos_token_id, *stop_ids])),
            pad_token_id=tokenizer.pad_token_id, use_cache=True)
    elapsed = time.perf_counter() - started
    sequences, values = [], []
    for row in output[:, input_length:].detach().cpu().tolist():
        tokens = trim_generated(row, set(stop_ids), tokenizer.eos_token_id)
        sequences.append(tokens)
        values.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
    del encoded, output
    return values, sequences, elapsed


def generate_judgments(model, tokenizer, prompts, allowed_sequences):
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True,
                        max_length=4096, return_tensors="pt").to("cuda")
    input_length = encoded.input_ids.shape[1]
    processor = AllowedSequenceTrieLogitsProcessor(input_length, allowed_sequences, tokenizer.eos_token_id)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=max(len(x) for x in allowed_sequences) + 1,
            do_sample=False, num_beams=1, logits_processor=LogitsProcessorList([processor]),
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    elapsed = time.perf_counter() - started
    sequences, values = [], []
    for row in output[:, input_length:].detach().cpu().tolist():
        tokens = trim_generated(row, set(), tokenizer.eos_token_id)
        sequences.append(tokens)
        values.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
    del encoded, output
    return values, sequences, elapsed


def previous_questions(connection, target_id, slot):
    rows = list(connection.execute(
        "SELECT slot,question,generated_token_ids_json,valid FROM question_slot "
        "WHERE target_id=? AND slot<? ORDER BY slot", (target_id, slot)))
    if len(rows) != slot - 1 or not all(row[3] for row in rows):
        return None
    return [row[1] for row in rows], [json.loads(row[2]) for row in rows]


def run_sessions(precommit, connection, targets, ids, phase, model, tokenizer):
    allowed_start = single_token_ids(tokenizer, STARTERS)
    stop_ids = question_mark_tokens(tokenizer)
    bad_words = forbidden_sequence_ids(tokenizer)
    allowed_labels = [list(ids) for ids in precommit["judgment_stage"]["allowed_token_trie"].values()]
    if not allowed_start or not stop_ids or not bad_words or not allowed_labels:
        raise RuntimeError("constraint construction failed")

    question_total = len(ids) * 15
    completed = sum(connection.execute("SELECT count(*) FROM question_slot WHERE target_id=?", (target_id,)).fetchone()[0]
                    for target_id in ids)
    started = time.monotonic()
    for slot in range(1, 16):
        tasks = []
        for target_id in ids:
            if connection.execute("SELECT 1 FROM question_slot WHERE target_id=? AND slot=?", (target_id, slot)).fetchone():
                continue
            previous = previous_questions(connection, target_id, slot)
            if previous is None:
                continue
            texts, token_sequences = previous
            tasks.append((targets[target_id], texts, token_sequences))
        batch_size = int(precommit["decoding"]["question_batch"])
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            rendered = [render(tokenizer, precommit["question_stage"]["prompt"].format(
                target_text=row["source_text"], slot=slot,
                previous_questions="NONE" if not texts else "\n".join(
                    f"{index}. {question}" for index, question in enumerate(texts, 1))))
                for row, texts, _ in batch]
            values, sequences, elapsed = generate_questions(
                model, tokenizer, [item[0] for item in rendered], [item[2] for item in batch],
                allowed_start, stop_ids, bad_words, int(precommit["question_stage"]["max_new_tokens"]))
            for (row, texts, previous_tokens), (_, prompt_sha), raw, tokens in zip(batch, rendered, values, sequences):
                parsed = validate_question(raw, tokens, texts, previous_tokens)
                connection.execute("INSERT INTO question_slot VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    row["document_id"], row["membership"], slot,
                    sha_text(json.dumps({"texts": texts, "tokens": previous_tokens}, ensure_ascii=False, sort_keys=True)),
                    raw, parsed.value, json.dumps(tokens, separators=(",", ":")), int(parsed.valid), parsed.reason,
                    prompt_sha, sha_text(raw), elapsed / len(batch), now()))
            connection.commit(); completed += len(batch)
            rate = completed / max(time.monotonic() - started, 1e-9)
            heartbeat(f"IA_V6_{phase}_QUESTION_PROGRESS", slot=slot, completed_slots=completed,
                      total_slots=question_total,
                      valid_slots=connection.execute("SELECT count(*) FROM question_slot WHERE valid=1").fetchone()[0],
                      eta_seconds=round((question_total - completed) / max(rate, 1e-9)))

    tasks = []
    for target_id in ids:
        questions = list(connection.execute(
            "SELECT slot,question FROM question_slot WHERE target_id=? AND valid=1 ORDER BY slot", (target_id,)))
        if len(questions) != 15:
            continue
        for slot, question in questions:
            if not connection.execute("SELECT 1 FROM judgment_slot WHERE target_id=? AND slot=?", (target_id, slot)).fetchone():
                tasks.append((targets[target_id], slot, question))
    started = time.monotonic()
    batch_size = int(precommit["decoding"]["judgment_batch"])
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        rendered = [render(tokenizer, precommit["judgment_stage"]["prompt"].format(
            target_text=row["source_text"], question=question)) for row, _, question in batch]
        values, sequences, elapsed = generate_judgments(
            model, tokenizer, [item[0] for item in rendered], allowed_labels)
        for (row, slot, question), (_, prompt_sha), raw, tokens in zip(batch, rendered, values, sequences):
            valid, label, reason = validate_label(raw)
            connection.execute("INSERT INTO judgment_slot VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                row["document_id"], row["membership"], slot, sha_text(question), raw, label,
                json.dumps(tokens, separators=(",", ":")), int(valid), reason, prompt_sha,
                elapsed / len(batch), now()))
        connection.commit()
        completed_judgments = min(start + len(batch), len(tasks))
        rate = completed_judgments / max(time.monotonic() - started, 1e-9)
        heartbeat(f"IA_V6_{phase}_JUDGMENT_PROGRESS", completed_slots=completed_judgments,
                  total_slots=len(tasks),
                  valid_slots=connection.execute("SELECT count(*) FROM judgment_slot WHERE valid=1").fetchone()[0],
                  eta_seconds=round((len(tasks) - completed_judgments) / max(rate, 1e-9)))


def audit(connection, targets, ids):
    rows = []
    for target_id in ids:
        questions = list(connection.execute(
            "SELECT slot,question,generated_token_ids_json,valid,reason FROM question_slot "
            "WHERE target_id=? ORDER BY slot", (target_id,)))
        judgments = list(connection.execute(
            "SELECT slot,judgment,valid,reason,question_sha256 FROM judgment_slot "
            "WHERE target_id=? ORDER BY slot", (target_id,)))
        question_valid = (len(questions) == 15 and all(row[3] for row in questions)
                          and [row[0] for row in questions] == list(range(1, 16)))
        judgment_valid = (len(judgments) == 15 and all(row[2] for row in judgments)
                          and [row[0] for row in judgments] == list(range(1, 16)))
        linked = question_valid and judgment_valid and all(
            sha_text(question[1]) == judgment[4] for question, judgment in zip(questions, judgments))
        normalized = [normalize_question(row[1]) for row in questions]
        token_sequences = [tuple(json.loads(row[2])) for row in questions]
        normalized_duplicates = len(normalized) - len(set(normalized))
        token_duplicates = len(token_sequences) - len(set(token_sequences))
        rows.append({
            "target_id": target_id, "membership": targets[target_id]["membership"],
            "questions": len(questions), "valid_question_slots": sum(row[3] for row in questions),
            "judgments": len(judgments), "valid_judgment_slots": sum(row[2] for row in judgments),
            "token_exact_duplicates": token_duplicates, "normalized_exact_duplicates": normalized_duplicates,
            "hash_link": linked,
            "valid": question_valid and judgment_valid and linked and token_duplicates == 0 and normalized_duplicates == 0,
            "question_failure": next((row[4] for row in questions if not row[3]), "") if questions else "NOT_GENERATED",
            "judgment_failure": next((row[3] for row in judgments if not row[2]), "") if judgments else "NOT_GENERATED",
            "inferred_fields": 0, "fuzzy_recovery": 0})
    return rows


def write_csv(path, rows):
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def summarize(rows):
    rates = {}
    for membership in ("member", "nonmember"):
        group = [row for row in rows if row["membership"] == membership]
        rates[membership] = sum(row["valid"] for row in group) / len(group)
    invalid = [row for row in rows if not row["valid"]]
    return {"sessions": len(rows), "valid": sum(row["valid"] for row in rows),
            "coverage": sum(row["valid"] for row in rows) / len(rows),
            "member_valid_rate": rates["member"], "nonmember_valid_rate": rates["nonmember"],
            "validity_gap": abs(rates["member"] - rates["nonmember"]),
            "token_exact_duplicates": sum(row["token_exact_duplicates"] for row in rows),
            "normalized_exact_duplicates": sum(row["normalized_exact_duplicates"] for row in rows),
            "forbidden_judgments": sum(row["judgment_failure"] == "INVALID_JUDGMENT" for row in rows),
            "failure_reasons": dict(Counter(
                ("QUESTION:" + row["question_failure"]) if row["question_failure"]
                else ("JUDGMENT:" + row["judgment_failure"]) for row in invalid)),
            "inferred_fields": 0, "fuzzy_recovery": 0}


def freeze_sessions(connection, targets, rows):
    session_lines, query_lines, gt_lines = [], [], []
    for row in rows:
        if not row["valid"]:
            continue
        target_id = row["target_id"]
        questions = list(connection.execute(
            "SELECT slot,question,output_sha256 FROM question_slot WHERE target_id=? ORDER BY slot", (target_id,)))
        judgments = list(connection.execute(
            "SELECT slot,judgment FROM judgment_slot WHERE target_id=? ORDER BY slot", (target_id,)))
        session_lines.append(serialize_session(target_id, [question[1] for question in questions]))
        gt_lines.append(json.dumps({"target_id": target_id, "membership": row["membership"],
                                   "judgments": [judgment[1] for judgment in judgments]},
                                  ensure_ascii=False, separators=(",", ":")))
        for (slot, question, output_sha), (_, judgment) in zip(questions, judgments):
            query_lines.append(json.dumps({
                "attack": "IA-Std-Q15-v6-DuplicateConstrained",
                "session_id": f"ia_v6::{target_id}", "query_id": f"ia_v6::{target_id}::q{slot:02d}",
                "query_index": slot, "target_id": target_id, "membership": row["membership"],
                "domain": targets[target_id]["domain"], "query": question,
                "query_sha256": sha_text(question), "standardized_gt": judgment,
                "question_artifact_sha256": output_sha,
                "claim_boundary": "STANDARDIZED_DUPLICATE_CONSTRAINED_NOT_ORIGINAL_OR_PAPER_EXACT_IA"},
                ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    files = {"IA_STD_Q15_V6_SESSIONS.jsonl": session_lines,
             "IA_STD_Q15_V6_QUERIES.jsonl": query_lines,
             "IA_STD_Q15_V6_GT.jsonl": gt_lines}
    for name, lines in files.items():
        (CHILD / "inputs" / name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {name: len(lines) for name, lines in files.items()}


def finish(verdict, preflight, full=None):
    result = {"campaign": "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED", "verdict": verdict,
              "completed_utc": now(), "preflight": preflight, "full": full,
              "detection_run": False, "e2e_run": False, "final_lc_modified": False,
              "training_steps": 0, "additional_recovery_allowed": False}
    atomic_json(CHILD / "FINAL_RESULT.json", result)
    lines = ["# IA-Std-Q15-v6-DuplicateConstrained", "", f"- Verdict: `{verdict}`",
             f"- Preflight valid: {preflight['valid']}/{preflight['sessions']} ({preflight['coverage']:.1%})",
             f"- Member/nonmember validity gap: {preflight['validity_gap']:.1%}",
             f"- Token/normalized duplicates: {preflight['token_exact_duplicates']}/{preflight['normalized_exact_duplicates']}",
             f"- Full: {full['valid']}/{full['sessions']} ({full['coverage']:.1%})" if full else "- Full: NOT RUN", "",
             "No detection, attack-success, AUC, E-AUC, or privacy result was opened during format gating."]
    (CHILD / "reports" / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    precommit = verify_precommit()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    target_rows = list(csv.DictReader(Path(precommit["cohort"]["path"]).open(encoding="utf-8")))
    targets = {row["document_id"]: row for row in target_rows}
    connection = database()
    heartbeat("IA_V6_PREFLIGHT_MODEL_LOADING", sessions=100)
    model, tokenizer = load_model()
    preflight_ids = list(precommit["preflight"]["ids"])
    run_sessions(precommit, connection, targets, preflight_ids, "PREFLIGHT", model, tokenizer)
    rows = audit(connection, targets, preflight_ids)
    write_csv(CHILD / "audits" / "IA_V6_PREFLIGHT_VALIDITY.csv", rows)
    preflight = summarize(rows)
    preflight["checks"] = {
        "valid_at_least_98": preflight["valid"] >= 98,
        "validity_gap_at_most_2pp": preflight["validity_gap"] <= 0.02 + 1e-12,
        "exact_q15_and_judgments": all(row["questions"] == 15 and row["judgments"] == 15 for row in rows if row["valid"]),
        "duplicates_zero_among_valid": all(row["token_exact_duplicates"] == 0 and row["normalized_exact_duplicates"] == 0 for row in rows if row["valid"]),
        "forbidden_judgments_zero": preflight["forbidden_judgments"] == 0,
        "inferred_fields_zero": True, "fuzzy_recovery_zero": True}
    preflight["verdict"] = "IA_STD_Q15_V6_FORMAT_PREFLIGHT_PASS" if all(preflight["checks"].values()) else "IA_STD_Q15_V6_FAILED_FINAL"
    atomic_json(CHILD / "IA_V6_PREFLIGHT_RESULT.json", preflight)
    if preflight["verdict"] == "IA_STD_Q15_V6_FAILED_FINAL":
        heartbeat(preflight["verdict"], valid=preflight["valid"], coverage=preflight["coverage"],
                  validity_gap=preflight["validity_gap"], next_stage="STOP_STANDARDIZED_ATTACK_UNAVAILABLE")
        finish(preflight["verdict"], preflight); return

    heartbeat(preflight["verdict"], valid=preflight["valid"], coverage=preflight["coverage"],
              validity_gap=preflight["validity_gap"], next_stage="FULL_2000")
    remaining = [row["document_id"] for row in target_rows if row["document_id"] not in set(preflight_ids)]
    run_sessions(precommit, connection, targets, remaining, "FULL", model, tokenizer)
    all_ids = [row["document_id"] for row in target_rows]
    rows = audit(connection, targets, all_ids)
    write_csv(CHILD / "audits" / "IA_V6_FULL_VALIDITY.csv", rows)
    full = summarize(rows)
    full["checks"] = {"valid_at_least_1900": full["valid"] >= 1900,
                      "coverage_at_least_95pct": full["coverage"] >= 0.95,
                      "validity_gap_at_most_2pp": full["validity_gap"] <= 0.02 + 1e-12,
                      "duplicates_zero_among_valid": all(row["token_exact_duplicates"] == 0 and row["normalized_exact_duplicates"] == 0 for row in rows if row["valid"]),
                      "forbidden_judgments_zero": full["forbidden_judgments"] == 0,
                      "inferred_fields_zero": True, "fuzzy_recovery_zero": True}
    full["verdict"] = "IA_STD_Q15_V6_READY" if all(full["checks"].values()) else "IA_STD_Q15_V6_FAILED_FINAL"
    atomic_json(CHILD / "IA_V6_FULL_RESULT.json", full)
    if full["verdict"] == "IA_STD_Q15_V6_READY":
        counts = freeze_sessions(connection, targets, rows)
        heartbeat(full["verdict"], valid=full["valid"], outputs=counts,
                  next_stage="FROZEN_DETECTION_AND_E2E_ALLOWED")
    else:
        heartbeat(full["verdict"], valid=full["valid"], coverage=full["coverage"],
                  next_stage="STOP_STANDARDIZED_ATTACK_UNAVAILABLE")
    finish(full["verdict"], preflight, full)


if __name__ == "__main__":
    main()
