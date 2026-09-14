#!/usr/bin/env python3
"""Resumable, cache-keyed, one-shot generation for every supported attack and condition."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONDITIONS, CORE3, EXP, QWEN, ROOT, atomic_json, checkpoint, now, read_jsonl, sha_file


SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any required output format exactly. "
                 "If the context is insufficient, answer exactly: I don't know.")
SOURCE_TOKEN_BUDGET = 2048
MAX_PROMPT_TOKENS = 3072
MAX_NEW = {"BENIGN": 160, "MEntA": 160, "MBA": 160, "RAG-MIA": 12, "S²-MIA": 160}
BATCH_SIZE = 16
SEED = 20260912


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def verify() -> tuple[dict, list[dict]]:
    pre_path = EXP / "configs" / "FINAL_8ATTACK_E2E_PRECOMMIT.json"
    expected = pre_path.with_suffix(".sha256").read_text().split()[0]
    if sha_file(pre_path) != expected:
        raise RuntimeError("precommit hash mismatch")
    pre = json.loads(pre_path.read_text())
    if pre["code_sha256"][str(__import__("pathlib").Path(__file__).relative_to(ROOT))] != sha_file(__import__("pathlib").Path(__file__)):
        raise RuntimeError("generation code changed after freeze")
    cache_path = EXP / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl"
    manifest = json.loads((EXP / "cache" / "FINAL_RETRIEVAL_MANIFEST.json").read_text())
    if sha_file(cache_path) != manifest["sha256"]:
        raise RuntimeError("retrieval cache drift")
    return pre, read_jsonl(cache_path)


def waterfill(lengths: list[int], total: int) -> list[int]:
    caps = np.zeros(len(lengths), dtype=int)
    remaining = total
    active = [index for index, length in enumerate(lengths) if length > 0]
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


def build_prompt(tokenizer, row: dict, documents: dict[str, str], hidden: str | None) -> tuple[str, dict]:
    kept = [source for source in row["top_document_ids"] if source != hidden]
    encoded = [tokenizer(documents[source], add_special_tokens=False).input_ids for source in kept]
    caps = waterfill([len(value) for value in encoded], SOURCE_TOKEN_BUDGET)
    visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip() for value, cap in zip(encoded, caps)]
    context = "\n\n".join(f"[Document {index}]\n{text}" for index, text in enumerate(visible, 1))
    user = f"Retrieved context:\n{context}\n\nUser query:\n{row['query']}"
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
    path = EXP / "runtime" / "generation.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS answer (
        query_id TEXT NOT NULL, attack TEXT NOT NULL, condition TEXT NOT NULL, output_budget INTEGER NOT NULL,
        answer TEXT NOT NULL, answer_sha256 TEXT NOT NULL, prompt_sha256 TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,
        answer_tokens INTEGER NOT NULL, mean_token_logprob REAL, perplexity REAL, source_ids_json TEXT NOT NULL,
        source_caps_json TEXT NOT NULL, source_token_count INTEGER NOT NULL, context_sha256 TEXT NOT NULL,
        hidden_source_id TEXT, detector_alarm INTEGER NOT NULL, generated INTEGER NOT NULL, reused_from TEXT,
        wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL,
        PRIMARY KEY(query_id, condition, output_budget))""")
    con.commit()
    return con


def alarm(row: dict, condition: str) -> bool:
    if condition == "NO_DEFENSE":
        return False
    if condition == "ORIGINAL_MIRABEL_SIMPLE_HIDE":
        return float(row["M"]) > 0.0
    if condition == "GLOBAL_BC_SIMPLE_HIDE":
        return float(row["R_GLOBAL"]) > 3.428580914764567
    if condition == "FINAL_LC_OUTER_SIMPLE_HIDE":
        return float(row["R_LC"]) > 3.80543877128208
    raise KeyError(condition)


def transition_statistics(model, generated, scores, eos_id: int) -> tuple[list[float | None], list[float | None]]:
    transition = model.compute_transition_scores(generated, scores, normalize_logits=True).detach().float().cpu().numpy()
    new_tokens = generated[:, -transition.shape[1]:].detach().cpu().numpy()
    means, ppls = [], []
    for tokens, values in zip(new_tokens, transition):
        length = len(values)
        eos_positions = np.where(tokens == eos_id)[0]
        if len(eos_positions):
            length = max(1, int(eos_positions[0]))
        valid = values[:length]
        mean = float(valid.mean()) if len(valid) else None
        means.append(mean)
        ppls.append(float(np.exp(min(50.0, -mean))) if mean is not None else None)
    return means, ppls


def main() -> None:
    pre, rows = verify()
    if len(rows) != 17000:
        raise RuntimeError(f"expected 17,000 query rows, got {len(rows)}")
    documents = {row["document_id"]: row["source_text"] for row in read_jsonl(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")}
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    if free < 14 * (1 << 30):
        raise RuntimeError(f"GPU_WAIT_REQUIRED free_bytes={free} total_bytes={total}")
    checkpoint("PHASE_E_GENERATOR_LOADING", queries=len(rows), logical_condition_rows=len(rows)*len(CONDITIONS), gpu_free_bytes=free)
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    con = database()
    total_rows = len(rows) * len(CONDITIONS)
    started = time.monotonic()
    for condition in CONDITIONS:
        tasks = []
        for row in sorted(rows, key=lambda item: (item["attack"], item["query_id"])):
            budget = MAX_NEW[row["attack"]]
            if con.execute("SELECT 1 FROM answer WHERE query_id=? AND condition=? AND output_budget=?",
                           (row["query_id"], condition, budget)).fetchone():
                continue
            detector_alarm = alarm(row, condition)
            hidden = row["selected_source_id"] if detector_alarm else None
            rendered, provenance = build_prompt(tokenizer, row, documents, hidden)
            reusable = con.execute("SELECT answer,answer_sha256,prompt_tokens,answer_tokens,mean_token_logprob,perplexity,"
                "source_ids_json,source_caps_json,source_token_count,context_sha256,hidden_source_id,condition "
                "FROM answer WHERE query_id=? AND prompt_sha256=? AND output_budget=? LIMIT 1",
                (row["query_id"], provenance["prompt_sha256"], budget)).fetchone()
            if reusable:
                con.execute("INSERT INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["query_id"], row["attack"], condition, budget, reusable[0], reusable[1], provenance["prompt_sha256"],
                     reusable[2], reusable[3], reusable[4], reusable[5], reusable[6], reusable[7], reusable[8], reusable[9],
                     reusable[10], int(detector_alarm), 0, reusable[11], 0.0, now()))
            else:
                tasks.append((row, rendered, provenance, detector_alarm, budget))
        con.commit()
        checkpoint("PHASE_E_CONDITION_STARTED", condition=condition, pending_generations=len(tasks),
                   database_rows=con.execute("SELECT count(*) FROM answer").fetchone()[0])
        offset = 0
        phase_started = time.monotonic()
        while offset < len(tasks):
            attack = tasks[offset][0]["attack"]
            budget = tasks[offset][4]
            end = offset
            while end < len(tasks) and tasks[end][0]["attack"] == attack and tasks[end][4] == budget and end-offset < BATCH_SIZE:
                end += 1
            batch = tasks[offset:end]
            encoded = tokenizer([item[1] for item in batch], add_special_tokens=False, padding=True, truncation=True,
                                max_length=MAX_PROMPT_TOKENS, return_tensors="pt").to(model.device)
            before = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=budget, do_sample=False, num_beams=1, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                    output_scores=(attack == "S²-MIA"))
            elapsed = time.perf_counter() - before
            sequences = generated.sequences
            answer_ids = sequences[:, encoded.input_ids.shape[1]:]
            answers = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
            if attack == "S²-MIA":
                means, ppls = transition_statistics(model, sequences, generated.scores, tokenizer.eos_token_id)
            else:
                means, ppls = [None]*len(batch), [None]*len(batch)
            for (row, _, provenance, detector_alarm, output_budget), answer, mean, ppl in zip(batch, answers, means, ppls):
                clean = answer.strip()
                con.execute("INSERT OR REPLACE INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["query_id"], row["attack"], condition, output_budget, clean, sha_text(clean), provenance["prompt_sha256"],
                     provenance["prompt_tokens"], len(tokenizer(clean, add_special_tokens=False).input_ids), mean, ppl,
                     json.dumps(provenance["source_ids"]), json.dumps(provenance["source_caps"]), provenance["source_token_count"],
                     provenance["context_sha256"], provenance["hidden_source_id"], int(detector_alarm), 1, None,
                     elapsed/len(batch), now()))
            con.commit()
            offset = end
            done = offset
            rate = done / max(time.monotonic()-phase_started, 1e-6)
            checkpoint("PHASE_E_GENERATION_PROGRESS", condition=condition, attack=attack, condition_generated=done,
                       condition_total=len(tasks), database_rows=con.execute("SELECT count(*) FROM answer").fetchone()[0],
                       eta_seconds=round((len(tasks)-done)/max(rate, 1e-9)))
            del encoded, generated, sequences, answer_ids
    count = con.execute("SELECT count(*) FROM answer").fetchone()[0]
    if count != total_rows:
        raise RuntimeError(f"logical answer count {count}/{total_rows}")
    columns = [item[0] for item in con.execute("SELECT * FROM answer LIMIT 1").description]
    output = EXP / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for values in con.execute("SELECT * FROM answer ORDER BY query_id,condition,output_budget"):
            handle.write(json.dumps(dict(zip(columns, values)), ensure_ascii=False, sort_keys=True) + "\n")
    cache_rows = []
    for values in con.execute("SELECT query_id,attack,condition,output_budget,context_sha256,answer_sha256,prompt_sha256,generated,reused_from FROM answer ORDER BY query_id,condition"):
        cache_rows.append(dict(zip(("query_id","attack","condition","budget","context_hash","answer_hash","prompt_hash","generated","reused_from"), values)))
    cache_path = EXP / "manifests" / "GLOBAL_CACHE_MANIFEST.jsonl"
    with cache_path.open("w", encoding="utf-8") as handle:
        for row in cache_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"logical_rows": count, "physical_generations": con.execute("SELECT count(*) FROM answer WHERE generated=1").fetchone()[0],
                "exact_reuses": con.execute("SELECT count(*) FROM answer WHERE generated=0").fetchone()[0],
                "condition_counts": {condition: con.execute("SELECT count(*) FROM answer WHERE condition=?", (condition,)).fetchone()[0] for condition in CONDITIONS},
                "answer_sha256": sha_file(output), "cache_manifest_sha256": sha_file(cache_path),
                "runtime_seconds": time.monotonic()-started, "completed_utc": now()}
    atomic_json(EXP / "runtime" / "FINAL_GENERATION_MANIFEST.json", manifest)
    checkpoint("PHASE_E_ONE_SHOT_GENERATION_COMPLETE", **manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

