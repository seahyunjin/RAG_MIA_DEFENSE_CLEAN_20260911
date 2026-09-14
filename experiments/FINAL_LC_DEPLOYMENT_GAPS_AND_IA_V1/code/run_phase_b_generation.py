#!/usr/bin/env python3
"""Generate only precommitted closed-book fallback branches."""
from __future__ import annotations

import csv
import json
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (EXP, QWEN, ROOT, SEED, atomic_json, checkpoint, now,
                    read_jsonl, sha_file, sha_text, verify_hashed_json,
                    write_jsonl)


MAX_NEW = {"GOLD": 160, "MEntA": 160, "MBA": 160, "RAG-MIA": 12, "S²-MIA": 160, "DCMI-Std-Q2": 12}
BATCH_SIZE = 16


def refusal(value: str) -> bool:
    text = " ".join(value.casefold().split())
    return any(term in text for term in ("i don't know", "i do not know", "cannot determine",
                                         "insufficient information", "not enough information"))


def build_prompt(tokenizer, query: str, system: str) -> tuple[str, str, int]:
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": system},
                                               {"role": "user", "content": f"User query:\n{query}"}],
                                              tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(ids) > 3072:
        ids = ids[-3072:]
        rendered = tokenizer.decode(ids, skip_special_tokens=False)
    return rendered, sha_text(json.dumps(ids, separators=(",", ":"))), len(ids)


def database() -> sqlite3.Connection:
    path = EXP / "runtime" / "phase_b_closed_book.sqlite3"
    con = sqlite3.connect(path, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS answer (
      scope TEXT NOT NULL, db TEXT, query_id TEXT NOT NULL, attack TEXT NOT NULL, membership TEXT,
      primary_evaluation INTEGER NOT NULL, s2_evaluation_split TEXT, query TEXT NOT NULL,
      simple_hide_answer TEXT NOT NULL, closed_book_answer TEXT NOT NULL,
      prompt_sha256 TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, answer_tokens INTEGER NOT NULL,
      mean_token_logprob REAL, perplexity REAL, wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL,
      PRIMARY KEY(scope,query_id))""")
    con.commit()
    return con


def transition_statistics(model, generated, scores, eos_id: int):
    transition = model.compute_transition_scores(generated, scores, normalize_logits=True).detach().float().cpu().numpy()
    new_tokens = generated[:, -transition.shape[1]:].detach().cpu().numpy()
    output = []
    for tokens, values in zip(new_tokens, transition):
        length = len(values)
        eos = np.where(tokens == eos_id)[0]
        if len(eos):
            length = max(1, int(eos[0]))
        valid = values[:length]
        mean = float(valid.mean()) if len(valid) else None
        output.append((mean, float(np.exp(min(50.0, -mean))) if mean is not None else None))
    return output


def main() -> None:
    pre = verify_hashed_json(EXP / "configs" / "FP_CLOSED_BOOK_FALLBACK_PRECOMMIT.json")
    for relative, digest in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != digest:
            raise RuntimeError(f"Phase-B code drift: {relative}")
    fp = list(csv.DictReader(Path(pre["gold_fp"]["path"]).open(encoding="utf-8")))
    gold_queries = {row["query_id"]: row for row in read_jsonl(Path(pre["gold_queries"]["path"]))}
    retrieval = [row for row in read_jsonl(Path(pre["a2_retrieval"]["path"])) if row["db"] == "V0"]
    answers = {(row["db"], row["query_id"], row["condition"]): row for row in read_jsonl(Path(pre["a2_answers"]["path"]))}
    tasks = []
    for row in fp:
        answer = row["final_answer"]
        if refusal(answer) or not answer.strip():
            query = gold_queries[row["query_id"]].get("raw_question") or gold_queries[row["query_id"]].get("query")
            tasks.append({"scope": "GOLD_FP", "db": None, "query_id": row["query_id"], "attack": "GOLD",
                          "membership": None, "primary_evaluation": True, "s2_evaluation_split": None,
                          "query": query, "simple_hide_answer": answer})
    for row in retrieval:
        simple = answers[("V0", row["query_id"], "FINAL_LC_REFRESH")]
        if simple["detector_alarm"] and (refusal(simple["answer"]) or not simple["answer"].strip()):
            tasks.append({"scope": "CORE5", "db": "V0", "query_id": row["query_id"], "attack": row["attack"],
                          "membership": row["membership"], "primary_evaluation": bool(row["primary_evaluation"]),
                          "s2_evaluation_split": row.get("s2_evaluation_split"), "query": row["query"],
                          "simple_hide_answer": simple["answer"]})
    con = database()
    tasks = [row for row in tasks if not con.execute("SELECT 1 FROM answer WHERE scope=? AND query_id=?",
                                                      (row["scope"], row["query_id"])).fetchone()]
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    checkpoint("PHASE_B_GENERATOR_LOADING", pending=len(tasks))
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    system = pre["closed_book_prompt"]["system"]
    done = 0
    started = time.monotonic()
    while done < len(tasks):
        attack = tasks[done]["attack"]
        end = done
        while end < len(tasks) and tasks[end]["attack"] == attack and end - done < BATCH_SIZE:
            end += 1
        batch = tasks[done:end]
        built = [build_prompt(tokenizer, row["query"], system) for row in batch]
        encoded = tokenizer([row[0] for row in built], add_special_tokens=False, padding=True,
                            truncation=True, max_length=3072, return_tensors="pt").to(model.device)
        before = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=MAX_NEW[attack], do_sample=False, num_beams=1,
                use_cache=True, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                output_scores=(attack == "S²-MIA"))
        wall = (time.perf_counter() - before) / len(batch)
        answer_ids = generated.sequences[:, encoded.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
        stats = transition_statistics(model, generated.sequences, generated.scores, tokenizer.eos_token_id) if attack == "S²-MIA" else [(None, None)] * len(batch)
        for row, answer, token_ids, build, (mean, ppl) in zip(batch, decoded, answer_ids, built, stats):
            con.execute("INSERT OR REPLACE INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                row["scope"], row["db"], row["query_id"], row["attack"], row["membership"],
                int(row["primary_evaluation"]), row["s2_evaluation_split"], row["query"], row["simple_hide_answer"],
                answer.strip(), build[1], build[2], int((token_ids != tokenizer.pad_token_id).sum().item()),
                mean, ppl, wall, now()))
        con.commit()
        done = end
        checkpoint("PHASE_B_GENERATION_PROGRESS", generated=done, pending=len(tasks),
                   database_rows=con.execute("SELECT count(*) FROM answer").fetchone()[0])
        del encoded, generated, answer_ids
    columns = [item[0] for item in con.execute("SELECT * FROM answer LIMIT 1").description]
    path = EXP / "runtime" / "PHASE_B_CLOSED_BOOK_ANSWERS.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for values in con.execute("SELECT * FROM answer ORDER BY scope,query_id"):
            handle.write(json.dumps(dict(zip(columns, values)), ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"verdict": "PHASE_B_CLOSED_BOOK_GENERATION_COMPLETE", "triggered_rows": len(tasks),
                "total_cached_rows": con.execute("SELECT count(*) FROM answer").fetchone()[0],
                "gold_rows": con.execute("SELECT count(*) FROM answer WHERE scope='GOLD_FP'").fetchone()[0],
                "attack_rows": con.execute("SELECT count(*) FROM answer WHERE scope='CORE5'").fetchone()[0],
                "answers_sha256": sha_file(path), "runtime_seconds": time.monotonic() - started, "completed_utc": now()}
    atomic_json(EXP / "runtime" / "PHASE_B_GENERATION_MANIFEST.json", manifest)
    checkpoint("PHASE_B_CLOSED_BOOK_GENERATION_COMPLETE", **manifest, next="PHASE_B_SCORING")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
