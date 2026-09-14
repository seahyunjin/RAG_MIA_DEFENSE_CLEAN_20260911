#!/usr/bin/env python3
"""Resumable Qwen generation for the four frozen LC-MIRABEL conditions."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import EXP, PARENT, QWEN, atomic_json, checkpoint, now, read_jsonl, sha_file


CONDITIONS = ("NO_DEFENSE", "ORIGINAL_MIRABEL_SIMPLE_HIDE", "GLOBAL_BC_SIMPLE_HIDE", "LC_MIRABEL_SIMPLE_HIDE")
SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any required output format exactly. "
                 "If the context is insufficient, answer exactly: I don't know.")
SOURCE_TOKEN_BUDGET = 2048
MAX_PROMPT_TOKENS = 3072
MAX_NEW_TOKENS = 160
BATCH_SIZE = 16
SEED = 20260912
PRECOMMIT = EXP / "configs/LC_MIRABEL_LARGE_V1_PRECOMMIT.json"


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def verify() -> None:
    expected = PRECOMMIT.with_suffix(".sha256").read_text().split()[0]
    pre = json.loads(PRECOMMIT.read_text())
    if sha_file(PRECOMMIT) != expected or pre["lineage"]["code_sha256"].get(Path(__file__).name) != sha_file(Path(__file__)):
        raise RuntimeError("precommit/code checksum mismatch")
    phase = json.loads((EXP / "PHASE1_RESULT.json").read_text())
    if phase["verdict"] != "LC_MIRABEL_DETECTION_PHASE1_PASS" or not phase["e2e_allowed"]:
        raise RuntimeError("E2E forbidden because detection gate did not pass")
    cache = EXP / "cache/LARGE_DETECTION_SCORES.jsonl"
    manifest = json.loads((EXP / "cache/LARGE_DETECTION_CACHE_MANIFEST.json").read_text())
    if sha_file(cache) != manifest["sha256"]:
        raise RuntimeError("detection cache drift")


def waterfill(lengths: list[int], total: int) -> list[int]:
    caps = np.zeros(len(lengths), dtype=int)
    remaining = total
    active = [i for i, length in enumerate(lengths) if length > 0]
    while remaining and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, lengths[index]-int(caps[index]), remaining)
            if add:
                caps[index] += add
                remaining -= add
                changed = True
            if caps[index] >= lengths[index]:
                active.remove(index)
            if not remaining:
                break
        if not changed:
            break
    return caps.tolist()


def build_prompt(tokenizer, query: str, source_ids: list[str], documents: dict[str, str], hidden: str | None) -> tuple[str, dict]:
    kept = [source for source in source_ids if source != hidden]
    encoded = [tokenizer(documents[source], add_special_tokens=False).input_ids for source in kept]
    caps = waterfill([len(value) for value in encoded], SOURCE_TOKEN_BUDGET)
    visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip() for value, cap in zip(encoded, caps)]
    context = "\n\n".join(f"[Document {index}]\n{text}" for index, text in enumerate(visible, 1))
    user = f"Retrieved context:\n{context}\n\nUser query:\n{query}"
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                               {"role": "user", "content": user}], tokenize=False,
                                              add_generation_prompt=True)
    prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(prompt_ids) > MAX_PROMPT_TOKENS:
        prompt_ids = prompt_ids[-MAX_PROMPT_TOKENS:]
        rendered = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    return rendered, {"prompt_sha256": sha_text(json.dumps(prompt_ids, separators=(",", ":"))),
                      "prompt_tokens": len(prompt_ids), "source_ids": kept, "source_caps": caps,
                      "source_token_count": sum(caps), "context_sha256": sha_text(context), "hidden_source_id": hidden}


def database() -> sqlite3.Connection:
    path = EXP / "runtime/generation.sqlite3"
    con = sqlite3.connect(path, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS answer (
        query_id TEXT NOT NULL, condition TEXT NOT NULL, answer TEXT NOT NULL, answer_sha256 TEXT NOT NULL,
        prompt_sha256 TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, answer_tokens INTEGER NOT NULL,
        source_ids_json TEXT NOT NULL, source_caps_json TEXT NOT NULL, source_token_count INTEGER NOT NULL,
        context_sha256 TEXT NOT NULL, hidden_source_id TEXT, detector_alarm INTEGER NOT NULL,
        generated INTEGER NOT NULL, reused_from TEXT, wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL,
        PRIMARY KEY(query_id, condition))""")
    con.commit()
    return con


def main() -> None:
    verify()
    rows = [row for row in read_jsonl(EXP / "cache/LARGE_DETECTION_SCORES.jsonl") if row["split"] != "REFERENCE"]
    if len(rows) != 15000:
        raise RuntimeError(f"expected 15,000 evaluation queries, got {len(rows)}")
    documents = {row["document_id"]: row["source_text"] for row in read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")}
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint("LARGE_E2E_GENERATOR_LOADING", queries=len(rows), logical_condition_rows=len(rows)*4)
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    con = database()
    conditions = {
        "NO_DEFENSE": lambda row: False,
        "ORIGINAL_MIRABEL_SIMPLE_HIDE": lambda row: row["M"] > 0,
        "GLOBAL_BC_SIMPLE_HIDE": lambda row: row["p_global"] <= .03,
        "LC_MIRABEL_SIMPLE_HIDE": lambda row: row["p_local"] <= .03,
    }
    total = len(rows)*len(conditions)
    started = time.monotonic()
    for condition, alarm_fn in conditions.items():
        tasks = []
        for row in sorted(rows, key=lambda item: item["query_id"]):
            if con.execute("SELECT 1 FROM answer WHERE query_id=? AND condition=?", (row["query_id"], condition)).fetchone():
                continue
            alarm = bool(alarm_fn(row))
            hidden = row["selected_source_id"] if alarm else None
            rendered, provenance = build_prompt(tokenizer, row["query"], row["top_document_ids"], documents, hidden)
            reusable = con.execute("SELECT answer,answer_sha256,prompt_tokens,answer_tokens,source_ids_json,source_caps_json,"
                "source_token_count,context_sha256,hidden_source_id,condition FROM answer WHERE query_id=? AND prompt_sha256=? LIMIT 1",
                (row["query_id"], provenance["prompt_sha256"])).fetchone()
            if reusable:
                con.execute("INSERT INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["query_id"], condition, reusable[0], reusable[1], provenance["prompt_sha256"], reusable[2], reusable[3],
                     reusable[4], reusable[5], reusable[6], reusable[7], reusable[8], int(alarm), 0, reusable[9], 0.0, now()))
            else:
                tasks.append((row, rendered, provenance, alarm))
        con.commit()
        phase_started = time.monotonic()
        checkpoint("LARGE_E2E_CONDITION_STARTED", condition=condition, pending_generations=len(tasks),
                   database_rows=con.execute("SELECT count(*) FROM answer").fetchone()[0])
        for offset in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[offset:offset+BATCH_SIZE]
            encoded = tokenizer([item[1] for item in batch], add_special_tokens=False, padding=True, truncation=True,
                                max_length=MAX_PROMPT_TOKENS, return_tensors="pt").to(model.device)
            before = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                                         use_cache=True, pad_token_id=tokenizer.eos_token_id)
            elapsed = time.perf_counter()-before
            generated = outputs[:, encoded.input_ids.shape[1]:]
            answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for (row, _, provenance, alarm), answer in zip(batch, answers):
                clean = answer.strip()
                con.execute("INSERT OR REPLACE INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["query_id"], condition, clean, sha_text(clean), provenance["prompt_sha256"], provenance["prompt_tokens"],
                     len(tokenizer(clean, add_special_tokens=False).input_ids), json.dumps(provenance["source_ids"]),
                     json.dumps(provenance["source_caps"]), provenance["source_token_count"], provenance["context_sha256"],
                     provenance["hidden_source_id"], int(alarm), 1, None, elapsed/len(batch), now()))
            con.commit()
            done = min(offset+len(batch), len(tasks))
            rate = done/max(time.monotonic()-phase_started, 1e-6)
            checkpoint("LARGE_E2E_GENERATION_PROGRESS", condition=condition, condition_generated=done,
                       condition_total=len(tasks), database_rows=con.execute("SELECT count(*) FROM answer").fetchone()[0],
                       eta_seconds=round((len(tasks)-done)/max(rate, 1e-9)))
            del encoded, outputs, generated
    count = con.execute("SELECT count(*) FROM answer").fetchone()[0]
    if count != total:
        raise RuntimeError(f"incomplete logical generation rows {count}/{total}")
    columns = [item[0] for item in con.execute("SELECT * FROM answer LIMIT 1").description]
    output = EXP / "runtime/LARGE_GENERATED_ANSWERS.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for values in con.execute("SELECT * FROM answer ORDER BY query_id,condition"):
            handle.write(json.dumps(dict(zip(columns, values)), ensure_ascii=False, sort_keys=True)+"\n")
    manifest = {"condition_rows": count, "evaluation_queries": len(rows),
                "physical_generations": con.execute("SELECT count(*) FROM answer WHERE generated=1").fetchone()[0],
                "exact_reuses": con.execute("SELECT count(*) FROM answer WHERE generated=0").fetchone()[0],
                "condition_counts": {condition: con.execute("SELECT count(*) FROM answer WHERE condition=?", (condition,)).fetchone()[0]
                                     for condition in CONDITIONS},
                "output_sha256": sha_file(output), "runtime_seconds": time.monotonic()-started, "completed_utc": now()}
    atomic_json(EXP / "runtime/LARGE_GENERATION_MANIFEST.json", manifest)
    checkpoint("LARGE_E2E_GENERATION_COMPLETE", **manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
