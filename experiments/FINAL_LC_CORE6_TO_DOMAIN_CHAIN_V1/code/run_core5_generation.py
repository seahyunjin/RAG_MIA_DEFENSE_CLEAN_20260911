#!/usr/bin/env python3
"""Generate only the missing Core5 A0/A_HIDE branches; IA is excluded."""
from __future__ import annotations

import csv
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CORE3, EXP, LC, QWEN, ROOT, SEED, atomic_json, checkpoint, now, read_jsonl, sha_file, verify_hashed_json, write_jsonl
from run_phase_b_generation import generate_answers

CORE4 = {"MEntA", "MBA", "RAG-MIA", "S²-MIA"}


def database() -> sqlite3.Connection:
    con = sqlite3.connect(EXP / "runtime" / "core5_phase_b_generation.sqlite3", timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS answer (
      query_id TEXT NOT NULL, attack TEXT NOT NULL, branch TEXT NOT NULL, answer TEXT NOT NULL,
      answer_sha256 TEXT NOT NULL, prompt_sha256 TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,
      answer_tokens INTEGER NOT NULL, mean_token_logprob REAL, perplexity REAL,
      source_ids_json TEXT NOT NULL, source_caps_json TEXT NOT NULL,
      source_token_count INTEGER NOT NULL, context_sha256 TEXT NOT NULL, hidden_source_id TEXT,
      wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL, PRIMARY KEY(query_id,branch))""")
    con.commit()
    return con


def main() -> None:
    pre = verify_hashed_json(EXP / "configs" / "CORE5_MATCHED_BUDGET_E2E_PRECOMMIT.json")
    for relative, expected in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != expected:
            raise RuntimeError(f"Core5 Phase-B code drift: {relative}")
    new_rows = read_jsonl(Path(pre["new_detection"]["path"]))
    if len(new_rows) != 4000 or {row["attack"] for row in new_rows} != {"DCMI-Std-Q2"}:
        raise RuntimeError("Core5 DCMI detection input drift")
    docs = {row["document_id"]: row["source_text"] for row in read_jsonl(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")}
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free, total = torch.cuda.mem_get_info()
    if free < 14 * (1 << 30):
        raise RuntimeError(f"GPU_WAIT_REQUIRED free={free} total={total}")
    checkpoint("CORE5_GENERATOR_LOADING", gpu_free_bytes=free, new_dcmi_queries=len(new_rows))
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    con = database()
    old_rows = [row for row in read_jsonl(Path(pre["old_detection"]["path"])) if row["attack"] in CORE4]
    old_answer_rows = read_jsonl(Path(pre["old_answers"]["path"]))
    old_branches = defaultdict(set)
    old_selected = {row["query_id"]: row["selected_source_id"] for row in old_rows}
    old_ids = set(old_selected)
    for row in old_answer_rows:
        if row["query_id"] not in old_ids:
            continue
        if row["condition"] == "NO_DEFENSE":
            old_branches[row["query_id"]].add("A0")
        elif row.get("detector_alarm") and row.get("hidden_source_id") == old_selected.get(row["query_id"]):
            old_branches[row["query_id"]].add("A_HIDE")
        else:
            old_branches[row["query_id"]].add("OTHER")
    generate_answers(model, tokenizer, con, old_rows, docs, pre, old_branches)
    generate_answers(model, tokenizer, con, new_rows, docs, pre)
    columns = [row[0] for row in con.execute("SELECT * FROM answer LIMIT 1").description]
    answer_rows = [dict(zip(columns, values)) for values in con.execute("SELECT * FROM answer ORDER BY query_id,branch")]
    output = EXP / "runtime" / "STANDARDIZED_BRANCH_ANSWERS.jsonl"
    write_jsonl(output, answer_rows)
    manifest = {"answer_rows": len(answer_rows), "a0": sum(row["branch"] == "A0" for row in answer_rows),
                "a_hide": sum(row["branch"] == "A_HIDE" for row in answer_rows), "answers_sha256": sha_file(output),
                "ia_rows": 0, "ia_regenerated": False, "completed_utc": now()}
    atomic_json(EXP / "runtime" / "PHASE_B_GENERATION_MANIFEST.json", manifest)
    checkpoint("CORE5_GENERATION_COMPLETE", **manifest, next_stage="CORE5_SCORING")


if __name__ == "__main__":
    main()
