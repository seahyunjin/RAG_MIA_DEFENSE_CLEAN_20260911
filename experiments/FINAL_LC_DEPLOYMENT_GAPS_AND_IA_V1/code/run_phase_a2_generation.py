#!/usr/bin/env python3
"""Checkpointed Qwen generation for Phase-A2 churn causal E2E."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (CHURN, CONDITIONS, CORE3, CORE6, EXP, FINAL8, QWEN, ROOT,
                    SEED, VERSIONS, atomic_json, checkpoint, now, read_jsonl,
                    sha_file, sha_text, verify_hashed_json)


SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any required output format exactly. "
                 "If the context is insufficient, answer exactly: I don't know.")
SOURCE_TOKEN_BUDGET = 2048
MAX_PROMPT_TOKENS = 3072
MAX_NEW = {"MEntA": 160, "MBA": 160, "RAG-MIA": 12, "S²-MIA": 160, "DCMI-Std-Q2": 12}
BATCH_SIZE = 16


def verify(pre: dict) -> None:
    for key in ("frozen_final_lc", "phase_a1", "churn_result", "query_manifest", "s2_provenance",
                "base_corpus", "added_corpus", "old_answers", "old_dcmi_answers"):
        value = pre[key]
        if sha_file(Path(value["path"])) != value["sha256"]:
            raise RuntimeError(f"Phase-A2 input drift: {key}")
    for relative, digest in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != digest:
            raise RuntimeError(f"Phase-A2 code drift: {relative}")
    substrate = json.loads((EXP / "PHASE_A2_SUBSTRATE_AUDIT.json").read_text(encoding="utf-8"))
    if sha_file(EXP / "cache" / "PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl") != substrate["retrieval_sha256"]:
        raise RuntimeError("Phase-A2 retrieval substrate drift")


def waterfill(lengths: list[int], total: int = SOURCE_TOKEN_BUDGET) -> list[int]:
    caps = np.zeros(len(lengths), dtype=int)
    remaining = total
    active = [index for index, value in enumerate(lengths) if value > 0]
    while remaining and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, lengths[index] - int(caps[index]), remaining)
            caps[index] += add
            remaining -= add
            changed = changed or bool(add)
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
    caps = waterfill([len(value) for value in encoded])
    visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip() for value, cap in zip(encoded, caps)]
    context = "\n\n".join(f"[Document {index}]\n{text}" for index, text in enumerate(visible, 1))
    user = f"Retrieved context:\n{context}\n\nUser query:\n{row['query']}"
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                               {"role": "user", "content": user}],
                                              tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(ids) > MAX_PROMPT_TOKENS:
        ids = ids[-MAX_PROMPT_TOKENS:]
        rendered = tokenizer.decode(ids, skip_special_tokens=False)
    provenance = {"prompt_sha256": sha_text(json.dumps(ids, separators=(",", ":"))), "prompt_tokens": len(ids),
                  "context_sha256": sha_text(context), "source_ids_json": json.dumps(kept),
                  "source_caps_json": json.dumps(caps), "source_token_count": sum(caps),
                  "hidden_source_id": hidden}
    return rendered, provenance


def database() -> sqlite3.Connection:
    path = EXP / "runtime" / "phase_a2_generation.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS answer (
      db TEXT NOT NULL, query_id TEXT NOT NULL, attack TEXT NOT NULL, membership TEXT NOT NULL,
      primary_evaluation INTEGER NOT NULL, s2_evaluation_split TEXT, condition TEXT NOT NULL,
      output_budget INTEGER NOT NULL, answer TEXT NOT NULL, answer_sha256 TEXT NOT NULL,
      prompt_sha256 TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, answer_tokens INTEGER NOT NULL,
      mean_token_logprob REAL, perplexity REAL, source_ids_json TEXT NOT NULL, source_caps_json TEXT NOT NULL,
      source_token_count INTEGER NOT NULL, context_sha256 TEXT NOT NULL, hidden_source_id TEXT,
      detector_alarm INTEGER NOT NULL, generated INTEGER NOT NULL, reused_from TEXT,
      wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL,
      PRIMARY KEY(db,query_id,condition))""")
    con.commit()
    return con


def old_cache() -> dict[tuple[str, str, int], dict]:
    output = {}
    paths = (FINAL8 / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl",
             CORE6 / "runtime" / "STANDARDIZED_BRANCH_ANSWERS.jsonl")
    for path in paths:
        for row in read_jsonl(path):
            key = (row["query_id"], row["prompt_sha256"], int(row.get("output_budget") or MAX_NEW.get(row["attack"], 160)))
            if key in output:
                prior = output[key]
                if prior["answer_sha256"] != row["answer_sha256"] or prior.get("perplexity") != row.get("perplexity"):
                    raise RuntimeError(f"old exact-prompt cache conflict: {key}")
            else:
                output[key] = row
    return output


def insert(con, row: dict, condition: str, budget: int, answer: str, provenance: dict,
           detector_alarm: bool, generated: bool, reused_from: str | None,
           answer_tokens: int, mean: float | None, ppl: float | None, wall: float) -> None:
    clean = answer.strip()
    con.execute("INSERT OR REPLACE INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        row["db"], row["query_id"], row["attack"], row["membership"], int(row["primary_evaluation"]),
        row.get("s2_evaluation_split"), condition, budget, clean, sha_text(clean), provenance["prompt_sha256"],
        provenance["prompt_tokens"], answer_tokens, mean, ppl, provenance["source_ids_json"],
        provenance["source_caps_json"], provenance["source_token_count"], provenance["context_sha256"],
        provenance["hidden_source_id"], int(detector_alarm), int(generated), reused_from, wall, now()))


def transition_statistics(model, generated, scores, eos_id: int) -> tuple[list[float | None], list[float | None]]:
    transition = model.compute_transition_scores(generated, scores, normalize_logits=True).detach().float().cpu().numpy()
    new_tokens = generated[:, -transition.shape[1]:].detach().cpu().numpy()
    means, ppls = [], []
    for tokens, values in zip(new_tokens, transition):
        length = len(values)
        eos = np.where(tokens == eos_id)[0]
        if len(eos):
            length = max(1, int(eos[0]))
        valid = values[:length]
        mean = float(valid.mean()) if len(valid) else None
        means.append(mean)
        ppls.append(float(np.exp(min(50.0, -mean))) if mean is not None else None)
    return means, ppls


def main() -> None:
    pre = verify_hashed_json(EXP / "configs" / "PHASE_A2_CHURN_E2E_PRECOMMIT.json")
    verify(pre)
    rows = read_jsonl(EXP / "cache" / "PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl")
    base = read_jsonl(Path(pre["base_corpus"]["path"]))
    added = read_jsonl(Path(pre["added_corpus"]["path"]))
    documents = {row["document_id"]: row["source_text"] for row in base + added}
    missing_docs = {source for row in rows for source in row["top_document_ids"] if source not in documents}
    if missing_docs:
        raise RuntimeError(f"document text missing: {sorted(missing_docs)[:5]}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    con = database()
    reused = old_cache()
    expected = len(rows) * len(CONDITIONS)
    started = time.monotonic()
    checkpoint("PHASE_A2_GENERATOR_LOADING", retrieval_rows=len(rows), logical_rows=expected,
               existing_rows=con.execute("SELECT count(*) FROM answer").fetchone()[0])
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    for version in VERSIONS:
        version_rows = [row for row in rows if row["db"] == version]
        for condition in CONDITIONS:
            tasks = []
            for row in sorted(version_rows, key=lambda value: (value["attack"], value["query_id"])):
                if con.execute("SELECT 1 FROM answer WHERE db=? AND query_id=? AND condition=?",
                               (version, row["query_id"], condition)).fetchone():
                    continue
                alarm = bool(row["final_lc_alarm"]) if condition == "FINAL_LC_REFRESH" else False
                hidden = row["selected_source_id"] if alarm else None
                rendered, provenance = build_prompt(tokenizer, row, documents, hidden)
                budget = MAX_NEW[row["attack"]]
                local = con.execute("SELECT answer,answer_tokens,mean_token_logprob,perplexity,condition FROM answer "
                                    "WHERE db=? AND query_id=? AND prompt_sha256=? LIMIT 1",
                                    (version, row["query_id"], provenance["prompt_sha256"])).fetchone()
                exact = reused.get((row["query_id"], provenance["prompt_sha256"], budget))
                if local:
                    insert(con, row, condition, budget, local[0], provenance, alarm, False,
                           f"LOCAL:{local[4]}", int(local[1]), local[2], local[3], 0.0)
                elif exact:
                    insert(con, row, condition, budget, exact["answer"], provenance, alarm, False,
                           f"OLD:{exact.get('condition') or exact.get('branch')}", int(exact["answer_tokens"]),
                           exact.get("mean_token_logprob"), exact.get("perplexity"), 0.0)
                else:
                    tasks.append((row, rendered, provenance, alarm, budget))
            con.commit()
            checkpoint("PHASE_A2_CONDITION_STARTED", db=version, condition=condition, pending=len(tasks),
                       database_rows=con.execute("SELECT count(*) FROM answer").fetchone()[0])
            done = 0
            phase_started = time.monotonic()
            while done < len(tasks):
                attack = tasks[done][0]["attack"]
                budget = tasks[done][4]
                end = done
                while end < len(tasks) and tasks[end][0]["attack"] == attack and tasks[end][4] == budget and end - done < BATCH_SIZE:
                    end += 1
                batch = tasks[done:end]
                encoded = tokenizer([item[1] for item in batch], add_special_tokens=False, padding=True,
                                    truncation=True, max_length=MAX_PROMPT_TOKENS, return_tensors="pt").to(model.device)
                before = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**encoded, max_new_tokens=budget, do_sample=False, num_beams=1,
                        use_cache=True, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                        output_scores=(attack == "S²-MIA"))
                wall = (time.perf_counter() - before) / len(batch)
                answer_ids = generated.sequences[:, encoded.input_ids.shape[1]:]
                answers = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
                if attack == "S²-MIA":
                    means, ppls = transition_statistics(model, generated.sequences, generated.scores, tokenizer.eos_token_id)
                else:
                    means, ppls = [None] * len(batch), [None] * len(batch)
                for item, answer, token_ids, mean, ppl in zip(batch, answers, answer_ids, means, ppls):
                    row, _, provenance, alarm, output_budget = item
                    length = int((token_ids != tokenizer.pad_token_id).sum().item())
                    insert(con, row, condition, output_budget, answer, provenance, alarm, True, None,
                           length, mean, ppl, wall)
                con.commit()
                done = end
                rate = done / max(time.monotonic() - phase_started, 1e-9)
                checkpoint("PHASE_A2_GENERATION_PROGRESS", db=version, condition=condition, attack=attack,
                           condition_generated=done, condition_pending=len(tasks),
                           database_rows=con.execute("SELECT count(*) FROM answer").fetchone()[0],
                           expected_rows=expected, eta_condition_seconds=round((len(tasks) - done) / max(rate, 1e-9)))
                del encoded, generated, answer_ids
    count = con.execute("SELECT count(*) FROM answer").fetchone()[0]
    if count != expected:
        raise RuntimeError(f"Phase-A2 answer count mismatch: {count}/{expected}")
    columns = [item[0] for item in con.execute("SELECT * FROM answer LIMIT 1").description]
    path = EXP / "runtime" / "PHASE_A2_GENERATED_ANSWERS.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for values in con.execute("SELECT * FROM answer ORDER BY db,query_id,condition"):
            handle.write(json.dumps(dict(zip(columns, values)), ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"verdict": "PHASE_A2_GENERATION_COMPLETE", "logical_rows": count,
                "physical_generations": con.execute("SELECT count(*) FROM answer WHERE generated=1").fetchone()[0],
                "exact_reuses": con.execute("SELECT count(*) FROM answer WHERE generated=0").fetchone()[0],
                "answer_sha256": sha_file(path), "runtime_seconds": time.monotonic() - started, "completed_utc": now()}
    atomic_json(EXP / "runtime" / "PHASE_A2_GENERATION_MANIFEST.json", manifest)
    checkpoint("PHASE_A2_GENERATION_COMPLETE", **manifest, next="PHASE_A2_SCORING")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
