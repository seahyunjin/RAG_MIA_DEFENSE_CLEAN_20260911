#!/usr/bin/env python3
"""Generate frozen slotted questions and grammar-constrained judgments."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

CHILD = Path(__file__).resolve().parents[1]
PARENT = CHILD.parent
ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
DB = CHILD / "runtime" / "ia_v5.sqlite3"
SEED = 20260912
sys.path.insert(0, str(CHILD / "code"))
from ia_v5_protocol import FORBIDDEN_REFERENCES, LABELS, STARTERS, validate_label, validate_question


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_text(value: str):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def heartbeat(stage: str, **fields):
    payload = {"campaign": "IA_STD_Q15_V5_SLOTTED_CONSTRAINED", "stage": stage, "updated_utc": now(), **fields}
    atomic_json(CHILD / "HEARTBEAT.json", payload)
    lines = ["# IA_STD_Q15_V5_SLOTTED_CONSTRAINED", "", f"- Stage: `{stage}`", f"- Updated UTC: `{payload['updated_utc']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in fields.items())
    (CHILD / "STATUS.md.tmp").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (CHILD / "STATUS.md.tmp").replace(CHILD / "STATUS.md")


def verify_precommit():
    path = CHILD / "configs" / "IA_STD_Q15_V5_PRECOMMIT.json"
    expected = (CHILD / "configs" / "IA_STD_Q15_V5_PRECOMMIT.sha256").read_text().split()[0]
    if sha_file(path) != expected:
        raise RuntimeError("precommit hash mismatch")
    pre = json.loads(path.read_text(encoding="utf-8"))
    for relative, digest in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != digest:
            raise RuntimeError(f"code drift: {relative}")
    if sha_file(Path(pre["cohort"]["path"])) != pre["cohort"]["sha256"]:
        raise RuntimeError("cohort drift")
    return pre


def database():
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS question_slot(
      target_id TEXT, membership TEXT, slot INTEGER, previous_sha256 TEXT,
      raw_output TEXT, question TEXT, valid INTEGER, reason TEXT,
      prompt_sha256 TEXT, output_sha256 TEXT, wall_seconds REAL, completed_utc TEXT,
      PRIMARY KEY(target_id,slot))""")
    con.execute("""CREATE TABLE IF NOT EXISTS judgment_slot(
      target_id TEXT, membership TEXT, slot INTEGER, question_sha256 TEXT,
      raw_output TEXT, judgment TEXT, token_id INTEGER, valid INTEGER, reason TEXT,
      prompt_sha256 TEXT, wall_seconds REAL, completed_utc TEXT,
      PRIMARY KEY(target_id,slot))""")
    con.commit()
    return con


class FirstStepMask(LogitsProcessor):
    def __init__(self, input_length: int, allowed: list[int]):
        self.input_length = input_length
        self.allowed = torch.tensor(sorted(set(allowed)), dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if input_ids.shape[1] == self.input_length:
            mask = torch.full_like(scores, float("-inf"))
            allowed = self.allowed.to(scores.device)
            mask[:, allowed] = scores[:, allowed]
            return mask
        return scores


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    return model, tokenizer


def render(tokenizer, prompt: str):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return text, sha_text(json.dumps(ids, separators=(",", ":")))


def token_ids(tokenizer, values: tuple[str, ...]):
    output = []
    for value in values:
        for surface in (value, " " + value):
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(ids) == 1:
                output.append(ids[0])
    return sorted(set(output))


def question_mark_tokens(tokenizer):
    output = []
    for token_id in range(tokenizer.vocab_size):
        if "?" in tokenizer.decode([token_id], skip_special_tokens=False):
            output.append(token_id)
    return sorted(set(output))


def forbidden_sequence_ids(tokenizer):
    """Token sequences for interface constraints already frozen in the prompt."""
    output = []
    seen = set()
    for phrase in FORBIDDEN_REFERENCES:
        for surface in (phrase, " " + phrase, phrase.title(), " " + phrase.title()):
            ids = tuple(tokenizer.encode(surface, add_special_tokens=False))
            if ids and ids not in seen:
                seen.add(ids)
                output.append(list(ids))
    return output


def generate_questions(model, tokenizer, prompts: list[str], allowed_start: list[int], stop_ids: list[int],
                       bad_words: list[list[int]], max_new: int):
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True, max_length=4096, return_tensors="pt").to("cuda")
    input_length = encoded.input_ids.shape[1]
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new, do_sample=False, num_beams=1,
                                logits_processor=LogitsProcessorList([FirstStepMask(input_length, allowed_start)]),
                                bad_words_ids=bad_words,
                                eos_token_id=sorted(set([tokenizer.eos_token_id, *stop_ids])),
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
    elapsed = time.perf_counter() - started
    values = tokenizer.batch_decode(output[:, input_length:], skip_special_tokens=True)
    del encoded, output
    return [value.strip() for value in values], elapsed


def generate_labels(model, tokenizer, prompts: list[str], allowed_labels: list[int]):
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True, max_length=4096, return_tensors="pt").to("cuda")
    input_length = encoded.input_ids.shape[1]
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=1, do_sample=False, num_beams=1,
                                logits_processor=LogitsProcessorList([FirstStepMask(input_length, allowed_labels)]),
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
    elapsed = time.perf_counter() - started
    token = output[:, input_length].detach().cpu().tolist()
    values = [tokenizer.decode([item], skip_special_tokens=True).strip() for item in token]
    del encoded, output
    return values, token, elapsed


def prior_questions(con, target_id: str, slot: int):
    rows = list(con.execute("SELECT slot,question,valid FROM question_slot WHERE target_id=? AND slot<? ORDER BY slot", (target_id, slot)))
    if len(rows) != slot - 1 or not all(row[2] for row in rows):
        return None
    return [row[1] for row in rows]


def run_sessions(pre, con, targets, ids, phase, model, tokenizer):
    allowed_start = token_ids(tokenizer, STARTERS)
    allowed_labels = token_ids(tokenizer, LABELS)
    stop_ids = question_mark_tokens(tokenizer)
    bad_words = forbidden_sequence_ids(tokenizer)
    if not allowed_start or not allowed_labels or not stop_ids or not bad_words:
        raise RuntimeError("token grammar construction failed")
    question_total = len(ids) * 15
    completed = sum(
        con.execute("SELECT count(*) FROM question_slot WHERE target_id=?", (target_id,)).fetchone()[0]
        for target_id in ids
    )
    started = time.monotonic()
    for slot in range(1, 16):
        tasks = []
        for target_id in ids:
            if con.execute("SELECT 1 FROM question_slot WHERE target_id=? AND slot=?", (target_id, slot)).fetchone():
                continue
            previous = prior_questions(con, target_id, slot)
            if previous is None:
                continue
            tasks.append((targets[target_id], previous))
        batch_size = int(pre["decoding"]["question_batch"])
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            rendered = [render(tokenizer, pre["question_stage"]["prompt"].format(
                target_text=row["source_text"], slot=slot,
                previous_questions="NONE" if not previous else "\n".join(f"{i+1}. {q}" for i, q in enumerate(previous))))
                for row, previous in batch]
            values, elapsed = generate_questions(model, tokenizer, [x[0] for x in rendered], allowed_start, stop_ids,
                                                 bad_words, int(pre["question_stage"]["max_new_tokens"]))
            for (row, previous), (_, prompt_sha), raw in zip(batch, rendered, values):
                parsed = validate_question(raw, previous)
                con.execute("INSERT INTO question_slot VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                    row["document_id"], row["membership"], slot, sha_text(json.dumps(previous, ensure_ascii=False)),
                    raw, parsed.value, int(parsed.valid), parsed.reason, prompt_sha, sha_text(raw), elapsed / len(batch), now()))
            con.commit()
            completed += len(batch)
            rate = completed / max(time.monotonic() - started, 1e-9)
            heartbeat(f"IA_V5_{phase}_QUESTION_PROGRESS", slot=slot, completed_slots=completed, total_slots=question_total,
                      valid_slots=con.execute("SELECT count(*) FROM question_slot WHERE valid=1").fetchone()[0],
                      eta_seconds=round((question_total - completed) / max(rate, 1e-9)))

    tasks = []
    for target_id in ids:
        questions = list(con.execute("SELECT slot,question FROM question_slot WHERE target_id=? AND valid=1 ORDER BY slot", (target_id,)))
        if len(questions) != 15:
            continue
        for slot, question in questions:
            if not con.execute("SELECT 1 FROM judgment_slot WHERE target_id=? AND slot=?", (target_id, slot)).fetchone():
                tasks.append((targets[target_id], slot, question))
    started = time.monotonic()
    batch_size = int(pre["decoding"]["judgment_batch"])
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        rendered = [render(tokenizer, pre["judgment_stage"]["prompt"].format(target_text=row["source_text"], question=question))
                    for row, _, question in batch]
        values, ids_out, elapsed = generate_labels(model, tokenizer, [x[0] for x in rendered], allowed_labels)
        for (row, slot, question), (_, prompt_sha), raw, token_id in zip(batch, rendered, values, ids_out):
            valid, label, reason = validate_label(raw)
            con.execute("INSERT INTO judgment_slot VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                row["document_id"], row["membership"], slot, sha_text(question), raw, label, token_id,
                int(valid), reason, prompt_sha, elapsed / len(batch), now()))
        con.commit()
        completed = min(start + len(batch), len(tasks)); rate = completed / max(time.monotonic() - started, 1e-9)
        heartbeat(f"IA_V5_{phase}_JUDGMENT_PROGRESS", completed_slots=completed, total_slots=len(tasks),
                  valid_slots=con.execute("SELECT count(*) FROM judgment_slot WHERE valid=1").fetchone()[0],
                  eta_seconds=round((len(tasks) - completed) / max(rate, 1e-9)))


def audit(con, targets, ids):
    rows = []
    for target_id in ids:
        questions = list(con.execute("SELECT slot,question,valid,reason FROM question_slot WHERE target_id=? ORDER BY slot", (target_id,)))
        judgments = list(con.execute("SELECT slot,judgment,valid,reason,question_sha256 FROM judgment_slot WHERE target_id=? ORDER BY slot", (target_id,)))
        question_valid = len(questions) == 15 and all(row[2] for row in questions) and [r[0] for r in questions] == list(range(1, 16))
        judgment_valid = len(judgments) == 15 and all(row[2] for row in judgments) and [r[0] for r in judgments] == list(range(1, 16))
        linked = question_valid and judgment_valid and all(sha_text(q[1]) == j[4] for q, j in zip(questions, judgments))
        duplicates = len(questions) - len({" ".join(r[1].casefold().split()) for r in questions})
        rows.append({"target_id": target_id, "membership": targets[target_id]["membership"],
                     "questions": len(questions), "valid_question_slots": sum(r[2] for r in questions),
                     "judgments": len(judgments), "valid_judgment_slots": sum(r[2] for r in judgments),
                     "duplicate_questions": duplicates, "hash_link": linked,
                     "valid": question_valid and judgment_valid and linked and duplicates == 0,
                     "question_failure": next((r[3] for r in questions if not r[2]), "") if questions else "NOT_GENERATED",
                     "judgment_failure": next((r[3] for r in judgments if not r[2]), "") if judgments else "NOT_GENERATED",
                     "inferred_fields": 0, "fuzzy_parsing": 0})
    return rows


def write_csv(path, rows):
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def summary(rows):
    valid = [r for r in rows if r["valid"]]
    rates = {}
    for member in ("member", "nonmember"):
        group = [r for r in rows if r["membership"] == member]
        rates[member] = sum(r["valid"] for r in group) / len(group)
    return {"sessions": len(rows), "valid": len(valid), "coverage": len(valid) / len(rows),
            "member_valid_rate": rates["member"], "nonmember_valid_rate": rates["nonmember"],
            "validity_gap": abs(rates["member"] - rates["nonmember"]),
            "duplicate_questions": sum(r["duplicate_questions"] for r in rows),
            "failure_reasons": dict(Counter(("QUESTION:" + r["question_failure"]) if r["question_failure"] else
                                            ("JUDGMENT:" + r["judgment_failure"]) for r in rows if not r["valid"])),
            "inferred_fields": 0, "fuzzy_parsing": 0}


def freeze_merged(con, targets, rows):
    queries, gt = [], []
    for row in rows:
        if not row["valid"]:
            continue
        target_id = row["target_id"]
        qs = list(con.execute("SELECT slot,question,output_sha256 FROM question_slot WHERE target_id=? ORDER BY slot", (target_id,)))
        js = list(con.execute("SELECT slot,judgment FROM judgment_slot WHERE target_id=? ORDER BY slot", (target_id,)))
        gt.append({"target_id": target_id, "membership": row["membership"], "judgments": [j[1] for j in js]})
        for (slot, question, output_sha), (_, label) in zip(qs, js):
            queries.append({"attack": "IA-Std-Q15-v5-SlottedConstrained", "session_id": f"ia_v5::{target_id}",
                            "query_id": f"ia_v5::{target_id}::q{slot:02d}", "query_index": slot,
                            "target_id": target_id, "membership": row["membership"], "domain": targets[target_id]["domain"],
                            "query": question, "query_sha256": sha_text(question), "standardized_gt": label,
                            "question_artifact_sha256": output_sha,
                            "claim_boundary": "STANDARDIZED_SLOTTED_CONSTRAINED_NOT_ORIGINAL_OR_PAPER_EXACT_IA"})
    for name, values in (("IA_STD_Q15_V5_QUERIES.jsonl", queries), ("IA_STD_Q15_V5_GT.jsonl", gt)):
        path = CHILD / "inputs" / name
        path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in values), encoding="utf-8")
    return queries, gt


def finish(status, pf, full=None):
    result = {"campaign": "IA_STD_Q15_V5_SLOTTED_CONSTRAINED", "verdict": status, "completed_utc": now(),
              "preflight": pf, "full": full, "detection_run": False, "e2e_run": False,
              "final_lc_modified": False, "training_steps": 0}
    atomic_json(CHILD / "FINAL_RESULT.json", result)
    lines = ["# IA-Std-Q15-v5-SlottedConstrained", "", f"- Verdict: `{status}`",
             f"- Preflight: {pf['valid']}/{pf['sessions']} ({pf['coverage']:.1%})",
             f"- Member/nonmember validity gap: {pf['validity_gap']:.1%}",
             f"- Duplicate questions: {pf['duplicate_questions']}",
             f"- Full: {full['valid']}/{full['sessions']} ({full['coverage']:.1%})" if full else "- Full: NOT RUN", "",
             "No detection or privacy result was opened during the format gate."]
    (CHILD / "reports" / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    pre = verify_precommit()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    target_rows = list(csv.DictReader(Path(pre["cohort"]["path"]).open(encoding="utf-8")))
    targets = {r["document_id"]: r for r in target_rows}
    con = database()
    heartbeat("IA_V5_PREFLIGHT_MODEL_LOADING", sessions=100)
    model, tokenizer = load_model()
    pre_ids = list(pre["preflight"]["session_ids"])
    run_sessions(pre, con, targets, pre_ids, "PREFLIGHT", model, tokenizer)
    pf_rows = audit(con, targets, pre_ids); write_csv(CHILD / "audits" / "IA_V5_PREFLIGHT_VALIDITY.csv", pf_rows)
    pf = summary(pf_rows)
    pf["checks"] = {"valid_at_least_98": pf["valid"] >= 98, "validity_gap_at_most_2pp": pf["validity_gap"] <= .02 + 1e-12,
                    "exact_q15_and_judgments": all(r["questions"] == 15 and r["judgments"] == 15 for r in pf_rows if r["valid"]),
                    "duplicate_questions_zero": pf["duplicate_questions"] == 0, "inferred_fields_zero": True}
    pf["verdict"] = "IA_STD_Q15_V5_FORMAT_PREFLIGHT_PASS" if all(pf["checks"].values()) else "IA_STD_Q15_V5_FORMAT_PREFLIGHT_FAILED"
    atomic_json(CHILD / "IA_V5_PREFLIGHT_RESULT.json", pf)
    if pf["verdict"].endswith("FAILED"):
        heartbeat(pf["verdict"], valid=pf["valid"], coverage=pf["coverage"], validity_gap=pf["validity_gap"], next_stage="STOP_NO_RETRY")
        finish(pf["verdict"], pf); return
    heartbeat(pf["verdict"], valid=pf["valid"], coverage=pf["coverage"], validity_gap=pf["validity_gap"], next_stage="FULL_2000")
    remaining = [r["document_id"] for r in target_rows if r["document_id"] not in set(pre_ids)]
    run_sessions(pre, con, targets, remaining, "FULL", model, tokenizer)
    all_ids = [r["document_id"] for r in target_rows]
    full_rows = audit(con, targets, all_ids); write_csv(CHILD / "audits" / "IA_V5_FULL_VALIDITY.csv", full_rows)
    full = summary(full_rows)
    full["checks"] = {"valid_at_least_1900": full["valid"] >= 1900, "coverage_at_least_95pct": full["coverage"] >= .95,
                      "validity_gap_at_most_2pp": full["validity_gap"] <= .02 + 1e-12,
                      "duplicate_questions_zero": full["duplicate_questions"] == 0, "inferred_fields_zero": True}
    full["verdict"] = "IA_STD_Q15_V5_READY" if all(full["checks"].values()) else "IA_STD_Q15_V5_FAILED"
    atomic_json(CHILD / "IA_V5_FULL_RESULT.json", full)
    if full["verdict"] == "IA_STD_Q15_V5_READY":
        queries, gt = freeze_merged(con, targets, full_rows)
        heartbeat("IA_V5_READY_FOR_FROZEN_DETECTION", valid=full["valid"], queries=len(queries), next_stage="STOP_AND_REPORT")
    else:
        heartbeat(full["verdict"], valid=full["valid"], coverage=full["coverage"], next_stage="STOP_NO_RETRY")
    finish(full["verdict"], pf, full)


if __name__ == "__main__":
    main()
