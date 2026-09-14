#!/usr/bin/env python3
"""Generate DCMI-Std-Q2 and IA-Std-Q15 once, with resumable exact provenance."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (EXP, LC, QWEN, ROOT, SEED, atomic_json, checkpoint, freeze_json,
                    now, sha_file, sha_text, verify_hashed_json, write_jsonl)


def parse_ia(raw: str) -> list[dict]:
    items: list[dict] = []
    seen = set()
    for raw_line in raw.splitlines():
        line = raw_line.strip().strip("` ")
        if not line or "?" not in line:
            continue
        line = re.sub(r"^\s*(?:\d+[\).:-]\s*|[-*]\s*)", "", line)
        question_part, tail = line.split("?", 1)
        question = question_part.strip().strip('"') + "?"
        labels = re.findall(r"\b(?:yes|no)\b", tail, re.I)
        if not question or question.casefold() in seen or len(labels) != 1:
            continue
        seen.add(question.casefold())
        items.append({"question": question, "query_generator_answer_not_scoring_gt": labels[0].title()})
    return items if len(items) == 15 else []


def database() -> sqlite3.Connection:
    path = EXP / "runtime" / "standardized_query_generation.sqlite3"
    con = sqlite3.connect(path, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS generation (
      attack TEXT NOT NULL, target_id TEXT NOT NULL, membership TEXT NOT NULL, raw_output TEXT NOT NULL,
      parsed_json TEXT NOT NULL, valid INTEGER NOT NULL, error TEXT, prompt_sha256 TEXT NOT NULL,
      output_sha256 TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, answer_tokens INTEGER NOT NULL,
      wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL, PRIMARY KEY(attack,target_id))""")
    con.commit()
    return con


def render(tokenizer, prompt: str) -> tuple[str, int, str]:
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return text, len(ids), sha_text(json.dumps(ids, separators=(",", ":")))


def generate_batch(model, tokenizer, rendered: list[str], max_new_tokens: int) -> tuple[list[str], float]:
    encoded = tokenizer(rendered, add_special_tokens=False, padding=True, truncation=True,
                        max_length=4096, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                use_cache=True, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - started
    answers = tokenizer.batch_decode(output[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
    del encoded, output
    return [value.strip() for value in answers], elapsed


def process_attack(attack: str, targets: list[dict], protocol: dict, model, tokenizer, con: sqlite3.Connection) -> None:
    completed = {row[0] for row in con.execute("SELECT target_id FROM generation WHERE attack=?", (attack,))}
    tasks = []
    for target in targets:
        if target["document_id"] in completed:
            continue
        if attack == "DCMI-Std-Q2":
            count = max(1, math.floor(0.06 * len(target["source_text"].split())))
            prompt = protocol["prompt_template"].format(replace_count=count, target_text=target["source_text"])
        else:
            prompt = protocol["question_prompt"].format(target_text=target["source_text"])
        rendered, prompt_tokens, prompt_sha = render(tokenizer, prompt)
        source_tokens = len(tokenizer(target["source_text"], add_special_tokens=False).input_ids)
        tasks.append((target, rendered, prompt_tokens, prompt_sha, source_tokens))
    tasks.sort(key=lambda item: (item[4], item[0]["document_id"]))
    total = len(targets)
    stage_started = time.monotonic()
    done_before = total - len(tasks)
    offset = 0
    while offset < len(tasks):
        source_tokens = tasks[offset][4]
        if attack == "DCMI-Std-Q2":
            batch_size = 1 if source_tokens > 2500 else 2 if source_tokens > 1600 else 12
            batch = tasks[offset:offset+batch_size]
            max_new = min(4096, max(item[4] for item in batch) + 256)
        else:
            batch = tasks[offset:offset+12]
            max_new = 1000
        answers, elapsed = generate_batch(model, tokenizer, [item[1] for item in batch], max_new)
        for (target, _, prompt_tokens, prompt_sha, _), raw_first in zip(batch, answers):
            raw = raw_first
            if attack == "DCMI-Std-Q2":
                parsed = {"perturbed_text": raw}
                valid = bool(raw) and raw != target["source_text"]
                error = None if valid else "EMPTY_OR_IDENTICAL_OUTPUT"
            else:
                items = parse_ia(raw)
                parsed = {"items": items}
                valid = len(items) == 15
                error = None if valid else f"EXPECTED_15_PARSED_{len(items)}"
            attempts = [raw_first]
            if not valid:
                if attack == "DCMI-Std-Q2":
                    replace_count = max(1, math.floor(0.06 * len(target["source_text"].split())))
                    retry = protocol["fixed_retry_prompt_template"].format(
                        replace_count=replace_count, target_text=target["source_text"], previous_output=raw_first)
                    retry_max = min(4096, len(tokenizer(target["source_text"], add_special_tokens=False).input_ids)+256)
                else:
                    retry = protocol["fixed_retry_prompt"].format(target_text=target["source_text"], previous_output=raw_first)
                    retry_max = 1000
                retry_rendered, retry_prompt_tokens, retry_prompt_sha = render(tokenizer, retry)
                retry_answers, retry_elapsed = generate_batch(model, tokenizer, [retry_rendered], retry_max)
                raw = retry_answers[0]
                attempts.append(raw)
                elapsed += retry_elapsed
                prompt_tokens, prompt_sha = retry_prompt_tokens, retry_prompt_sha
                if attack == "DCMI-Std-Q2":
                    parsed = {"perturbed_text": raw}
                    valid = bool(raw) and raw != target["source_text"]
                    error = None if valid else "EMPTY_OR_IDENTICAL_OUTPUT_AFTER_FIXED_RETRY"
                else:
                    items = parse_ia(raw)
                    parsed = {"items": items}
                    valid = len(items) == 15
                    error = None if valid else f"EXPECTED_15_PARSED_{len(items)}_AFTER_FIXED_RETRY"
            answer_tokens = len(tokenizer(raw, add_special_tokens=False).input_ids)
            con.execute("INSERT INTO generation VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (attack, target["document_id"], target["membership"], json.dumps(attempts, ensure_ascii=False),
                         json.dumps(parsed, ensure_ascii=False, sort_keys=True), int(valid), error, prompt_sha,
                         sha_text(raw), prompt_tokens, answer_tokens, elapsed/len(batch), now()))
        con.commit()
        offset += len(batch)
        completed_now = done_before + offset
        rate = offset / max(time.monotonic() - stage_started, 1e-9)
        eta = (total-completed_now) / max(rate, 1e-9)
        checkpoint("STANDARDIZED_QUERY_GENERATION_PROGRESS", attack=attack, completed=completed_now, total=total,
                   valid=con.execute("SELECT count(*) FROM generation WHERE attack=? AND valid=1", (attack,)).fetchone()[0],
                   invalid=con.execute("SELECT count(*) FROM generation WHERE attack=? AND valid=0", (attack,)).fetchone()[0],
                   eta_seconds=round(eta))


def export(targets: list[dict], con: sqlite3.Connection) -> None:
    target_map = {row["document_id"]: row for row in targets}
    dcmi_rows, ia_rows, provenance = [], [], []
    invalid = []
    for values in con.execute("SELECT attack,target_id,membership,raw_output,parsed_json,valid,error,prompt_sha256,output_sha256,prompt_tokens,answer_tokens,wall_seconds,completed_utc FROM generation ORDER BY attack,target_id"):
        keys = ("attack","target_id","membership","raw_output","parsed_json","valid","error","prompt_sha256","output_sha256","prompt_tokens","answer_tokens","wall_seconds","completed_utc")
        record = dict(zip(keys, values)); provenance.append(record)
        if not record["valid"]:
            invalid.append(record); continue
        target = target_map[record["target_id"]]
        parsed = json.loads(record["parsed_json"])
        if record["attack"] == "DCMI-Std-Q2":
            perturbed = parsed["perturbed_text"]
            for index, (variant, sample) in enumerate((("original", target["source_text"]), ("perturbed", perturbed)), 1):
                query = f'Is this: "{sample}" right? Answer with Yes or No.'
                dcmi_rows.append({"attack": "DCMI-Std-Q2", "session_id": f"dcmi_std::{record['target_id']}",
                                  "query_id": f"dcmi_std::{record['target_id']}::{variant}", "query_index": index,
                                  "variant": variant, "target_id": record["target_id"], "membership": record["membership"],
                                  "domain": target["domain"], "query": query, "query_sha256": sha_text(query),
                                  "perturbed_text_sha256": sha_text(perturbed), "generation_output_sha256": record["output_sha256"],
                                  "protocol": "DCMI-Std-Q2", "claim_boundary": "STANDARDIZED_NOT_ORIGINAL_DCMI"})
        else:
            for index, item in enumerate(parsed["items"], 1):
                query = item["question"]
                ia_rows.append({"attack": "IA-Std-Q15", "session_id": f"ia_std::{record['target_id']}",
                                "query_id": f"ia_std::{record['target_id']}::q{index:02d}", "query_index": index,
                                "target_id": record["target_id"], "membership": record["membership"],
                                "domain": target["domain"], "query": query, "query_sha256": sha_text(query),
                                "query_generator_answer_not_scoring_gt": item["query_generator_answer_not_scoring_gt"],
                                "generation_output_sha256": record["output_sha256"], "protocol": "IA-Std-Q15",
                                "claim_boundary": "STANDARDIZED_Q15_NOT_PAPER_EXACT_Q30"})
    write_jsonl(EXP / "runtime" / "STANDARDIZED_QUERY_GENERATION_PROVENANCE.jsonl", provenance)
    if invalid or len(dcmi_rows) != 4000 or len(ia_rows) != 30000:
        write_jsonl(EXP / "audits" / "STANDARDIZED_QUERY_GENERATION_FAILURES.jsonl", invalid)
        checkpoint("STANDARDIZED_QUERY_GENERATION_INCOMPLETE", invalid=len(invalid), dcmi_queries=len(dcmi_rows), ia_queries=len(ia_rows))
        raise RuntimeError(f"fail-close standardized query generation invalid={len(invalid)} dcmi={len(dcmi_rows)} ia={len(ia_rows)}")
    write_jsonl(EXP / "inputs" / "DCMI_STD_Q2_QUERIES.jsonl", dcmi_rows)
    write_jsonl(EXP / "inputs" / "IA_STD_Q15_QUERIES.jsonl", ia_rows)
    manifest = {
        "created_utc": now(), "targets": 2000, "member": 1000, "nonmember": 1000,
        "dcmi_queries": len(dcmi_rows), "ia_queries": len(ia_rows),
        "dcmi": {"path": str(EXP / "inputs" / "DCMI_STD_Q2_QUERIES.jsonl"), "sha256": sha_file(EXP / "inputs" / "DCMI_STD_Q2_QUERIES.jsonl")},
        "ia": {"path": str(EXP / "inputs" / "IA_STD_Q15_QUERIES.jsonl"), "sha256": sha_file(EXP / "inputs" / "IA_STD_Q15_QUERIES.jsonl")},
        "provenance": {"path": str(EXP / "runtime" / "STANDARDIZED_QUERY_GENERATION_PROVENANCE.jsonl"), "sha256": sha_file(EXP / "runtime" / "STANDARDIZED_QUERY_GENERATION_PROVENANCE.jsonl")},
        "condition_independent_once_only": True, "regeneration_prohibited": True,
    }
    digest = freeze_json(EXP / "configs" / "STANDARDIZED_QUERY_BUNDLE_MANIFEST.json", manifest)
    checkpoint("STANDARDIZED_QUERY_BUNDLE_FROZEN", dcmi_queries=len(dcmi_rows), ia_queries=len(ia_rows), manifest_sha256=digest)


def main() -> None:
    pre = verify_hashed_json(EXP / "configs" / "CORE6_DETECTION_PRECOMMIT.json")
    dcmi = verify_hashed_json(EXP / "configs" / "DCMI_STD_Q2_PROTOCOL.json")
    ia = verify_hashed_json(EXP / "configs" / "IA_STD_Q15_PROTOCOL.json")
    if pre["dcmi_protocol_sha256"] != sha_file(EXP / "configs" / "DCMI_STD_Q2_PROTOCOL.json") or pre["ia_protocol_sha256"] != sha_file(EXP / "configs" / "IA_STD_Q15_PROTOCOL.json"):
        raise RuntimeError("protocol-precommit linkage drift")
    targets = list(csv.DictReader((LC / "inputs" / "LARGE_SHARED_TARGETS.csv").open(encoding="utf-8")))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for frozen local query generator")
    free, total = torch.cuda.mem_get_info()
    if free < 14 * (1 << 30):
        raise RuntimeError(f"GPU_WAIT_REQUIRED free={free} total={total}")
    checkpoint("STANDARDIZED_QUERY_GENERATOR_LOADING", gpu_free_bytes=free, targets=len(targets))
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    con = database()
    process_attack("DCMI-Std-Q2", targets, dcmi, model, tokenizer, con)
    process_attack("IA-Std-Q15", targets, ia, model, tokenizer, con)
    export(targets, con)


if __name__ == "__main__":
    main()
