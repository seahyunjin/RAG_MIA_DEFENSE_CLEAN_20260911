#!/usr/bin/env python3
"""Generate frozen IA shadow GT plus A0/A_HIDE answers for standardized attacks."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (CORE3, EXP, LC, QWEN, ROOT, SEED, atomic_json, checkpoint, now,
                    read_jsonl, sha_file, sha_text, verify_hashed_json, write_jsonl)

SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any required output format exactly. "
                 "If the context is insufficient, answer exactly: I don't know.")
SOURCE_TOKEN_BUDGET = 2048
MAX_PROMPT_TOKENS = 3072


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
                caps[index] += add; remaining -= add; changed = True
            if caps[index] >= lengths[index]: active.remove(index)
            if not remaining: break
        if not changed: break
    return caps.tolist()


def build_prompt(tokenizer, row: dict, docs: dict[str, str], hidden: str | None) -> tuple[str, dict]:
    kept = [source for source in row["top_document_ids"] if source != hidden]
    encoded = [tokenizer(docs[source], add_special_tokens=False).input_ids for source in kept]
    caps = waterfill([len(value) for value in encoded], SOURCE_TOKEN_BUDGET)
    visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip() for value, cap in zip(encoded, caps)]
    context = "\n\n".join(f"[Document {index}]\n{text}" for index, text in enumerate(visible, 1))
    user = f"Retrieved context:\n{context}\n\nUser query:\n{row['query']}"
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                               {"role": "user", "content": user}], tokenize=False,
                                              add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(ids) > MAX_PROMPT_TOKENS:
        ids = ids[-MAX_PROMPT_TOKENS:]
        rendered = tokenizer.decode(ids, skip_special_tokens=False)
    return rendered, {"prompt_sha256": sha_text(json.dumps(ids, separators=(",", ":"))),
                      "prompt_tokens": len(ids), "source_ids": kept, "source_caps": caps,
                      "source_token_count": sum(caps), "context_sha256": sha_text(context), "hidden_source_id": hidden}


def database() -> sqlite3.Connection:
    con = sqlite3.connect(EXP / "runtime" / "phase_b_generation.sqlite3", timeout=120)
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS ia_gt (
      target_id TEXT PRIMARY KEY, labels_json TEXT NOT NULL, raw_outputs_json TEXT NOT NULL,
      valid INTEGER NOT NULL, error TEXT, prompt_sha256 TEXT NOT NULL, completed_utc TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS answer (
      query_id TEXT NOT NULL, attack TEXT NOT NULL, branch TEXT NOT NULL, answer TEXT NOT NULL,
      answer_sha256 TEXT NOT NULL, prompt_sha256 TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,
      answer_tokens INTEGER NOT NULL, mean_token_logprob REAL, perplexity REAL,
      source_ids_json TEXT NOT NULL, source_caps_json TEXT NOT NULL,
      source_token_count INTEGER NOT NULL, context_sha256 TEXT NOT NULL, hidden_source_id TEXT,
      wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL, PRIMARY KEY(query_id,branch))""")
    con.commit(); return con


def render_user(tokenizer, prompt: str) -> tuple[str, str]:
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return rendered, sha_text(json.dumps(tokenizer(rendered, add_special_tokens=False).input_ids, separators=(",", ":")))


def transition_statistics(model, sequences, scores, eos_id: int) -> tuple[list[float | None], list[float | None]]:
    transition = model.compute_transition_scores(sequences, scores, normalize_logits=True).float().cpu().numpy()
    generated = sequences[:, -transition.shape[1]:].detach().cpu().numpy()
    means, ppls = [], []
    for token_ids, token_scores in zip(generated, transition):
        usable = []
        for token_id, token_score in zip(token_ids, token_scores):
            if int(token_id) == eos_id:
                break
            usable.append(float(token_score))
        mean = float(np.mean(usable)) if usable else None
        means.append(mean)
        ppls.append(float(np.exp(min(50.0, -mean))) if mean is not None else None)
    return means, ppls


def generate(model, tokenizer, prompts: list[str], max_input: int, max_new: int,
             need_transition_scores: bool = False) -> tuple[list[str], float, list[float | None], list[float | None]]:
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True, max_length=max_input,
                        return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new, do_sample=False, num_beams=1, use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                                return_dict_in_generate=need_transition_scores,
                                output_scores=need_transition_scores)
    elapsed = time.perf_counter()-started
    sequences = output.sequences if need_transition_scores else output
    answers = tokenizer.batch_decode(sequences[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
    if need_transition_scores:
        means, ppls = transition_statistics(model, sequences, output.scores, tokenizer.eos_token_id)
    else:
        means, ppls = [None] * len(prompts), [None] * len(prompts)
    del encoded, output, sequences
    return [answer.strip() for answer in answers], elapsed, means, ppls


def parse_gt(raw: str) -> list[str]:
    values = []
    for line in raw.splitlines():
        match = re.match(r"^\s*\d+[\).:-]?\s*(Yes|No)\s*[.!]?\s*$", line, re.I)
        if match: values.append(match.group(1).title())
    return values if len(values) == 15 else []


def generate_gt(model, tokenizer, con: sqlite3.Connection, targets: dict[str, dict], ia_queries: list[dict], ia_protocol: dict) -> None:
    grouped = {}
    for row in ia_queries: grouped.setdefault(row["target_id"], []).append(row)
    completed = {row[0] for row in con.execute("SELECT target_id FROM ia_gt")}
    tasks = []
    for target_id in sorted(grouped):
        if target_id in completed: continue
        rows = sorted(grouped[target_id], key=lambda row: row["query_index"])
        questions = "\n".join(f"{index}. {row['query']}" for index, row in enumerate(rows, 1))
        prompt = ia_protocol["ground_truth"]["prompt"].format(target_text=targets[target_id]["source_text"], questions=questions)
        rendered, prompt_sha = render_user(tokenizer, prompt)
        tasks.append((target_id, rendered, prompt_sha, prompt, questions))
    offset = 0; started = time.monotonic()
    while offset < len(tasks):
        batch = tasks[offset:offset+8]
        outputs, elapsed, _, _ = generate(model, tokenizer, [row[1] for row in batch], 6144, 160)
        for (target_id, _, prompt_sha, prompt, questions), raw in zip(batch, outputs):
            attempts = [raw]; labels = parse_gt(raw)
            if len(labels) != 15:
                retry = ("The previous response failed the exact parser. Return exactly 15 numbered lines and nothing else. "
                         "Each line must contain only its number and Yes or No.\n\n" + prompt + "\n\nInvalid previous output:\n" + raw)
                retry_rendered, prompt_sha = render_user(tokenizer, retry)
                retry_output, _, _, _ = generate(model, tokenizer, [retry_rendered], 6144, 160)
                raw = retry_output[0]; attempts.append(raw); labels = parse_gt(raw)
            valid = len(labels) == 15
            con.execute("INSERT INTO ia_gt VALUES (?,?,?,?,?,?,?)", (target_id, json.dumps(labels),
                        json.dumps(attempts, ensure_ascii=False), int(valid), None if valid else f"PARSED_{len(labels)}",
                        prompt_sha, now()))
        con.commit(); offset += len(batch)
        rate = offset/max(time.monotonic()-started, 1e-9)
        checkpoint("PHASE_B_IA_GT_PROGRESS", completed=len(completed)+offset, total=len(grouped),
                   valid=con.execute("SELECT count(*) FROM ia_gt WHERE valid=1").fetchone()[0],
                   invalid=con.execute("SELECT count(*) FROM ia_gt WHERE valid=0").fetchone()[0],
                   eta_seconds=round((len(tasks)-offset)/max(rate,1e-9)))
    invalid = con.execute("SELECT count(*) FROM ia_gt WHERE valid=0").fetchone()[0]
    if invalid or con.execute("SELECT count(*) FROM ia_gt").fetchone()[0] != 2000:
        checkpoint("PHASE_B_IA_GT_INCOMPLETE", invalid=invalid)
        raise RuntimeError(f"IA GT fail-close invalid={invalid}")


def needs_hidden(row: dict, pre: dict) -> bool:
    for budget in (.01, .025, .03, .05):
        if float(row["M"]) > pre["thresholds"][f"MIRABEL@{budget}"]["threshold"]: return True
        if float(row["R_LC"]) > pre["thresholds"][f"Final LC@{budget}"]["threshold"]: return True
    return float(row["M"]) > 0


def generate_answers(model, tokenizer, con: sqlite3.Connection, rows: list[dict], docs: dict[str, str], pre: dict,
                     old_branches: dict[str, set[str]] | None = None) -> None:
    existing = {(a,b) for a,b in con.execute("SELECT query_id,branch FROM answer")}
    tasks = []
    for row in sorted(rows, key=lambda item: (item["attack"], item["query_id"])):
        if old_branches is not None and row["query_id"] in old_branches:
            branches = []
            if needs_hidden(row, pre) and "A_HIDE" not in old_branches[row["query_id"]]:
                branches.append(("A_HIDE", row["selected_source_id"]))
        else:
            branches = [("A0", None)]
            if needs_hidden(row, pre): branches.append(("A_HIDE", row["selected_source_id"]))
        for branch, hidden in branches:
            if (row["query_id"], branch) in existing: continue
            prompt, provenance = build_prompt(tokenizer, row, docs, hidden)
            tasks.append((row, branch, prompt, provenance))
    max_new_by_attack = {"RAG-MIA": 12, "DCMI-Std-Q2": 12, "IA-Std-Q15": 12,
                         "MEntA": 160, "MBA": 160, "S²-MIA": 160}
    offset=0; started=time.monotonic()
    while offset < len(tasks):
        attack = tasks[offset][0]["attack"]
        end = offset
        while end < len(tasks) and tasks[end][0]["attack"] == attack and end-offset < 16:
            end += 1
        batch=tasks[offset:end]
        answers, elapsed, means, ppls=generate(
            model,tokenizer,[row[2] for row in batch],MAX_PROMPT_TOKENS,
            max_new_by_attack[attack],need_transition_scores=(attack == "S²-MIA"))
        for (row,branch,_,p), answer, mean, ppl in zip(batch,answers,means,ppls):
            con.execute("INSERT INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                row["query_id"],row["attack"],branch,answer,sha_text(answer),p["prompt_sha256"],p["prompt_tokens"],
                len(tokenizer(answer,add_special_tokens=False).input_ids),mean,ppl,
                json.dumps(p["source_ids"]),json.dumps(p["source_caps"]),
                p["source_token_count"],p["context_sha256"],p["hidden_source_id"],elapsed/len(batch),now()))
        con.commit(); offset=end
        rate=offset/max(time.monotonic()-started,1e-9)
        checkpoint("PHASE_B_STANDARDIZED_ANSWER_PROGRESS", completed=len(existing)+offset,
                   total=len(existing)+len(tasks), database_rows=con.execute("SELECT count(*) FROM answer").fetchone()[0],
                   eta_seconds=round((len(tasks)-offset)/max(rate,1e-9)))


def main() -> None:
    pre=verify_hashed_json(EXP/"configs"/"CORE6_MATCHED_BUDGET_E2E_PRECOMMIT.json")
    for relative,expected in pre["code_sha256"].items():
        if sha_file(ROOT/relative)!=expected: raise RuntimeError(f"Phase-B code drift: {relative}")
    ia_protocol=verify_hashed_json(EXP/"configs"/"IA_STD_Q15_PROTOCOL.json")
    new_rows=read_jsonl(EXP/"cache"/"STANDARDIZED_RETRIEVAL_AND_DETECTION.jsonl")
    ia_queries=read_jsonl(EXP/"inputs"/"IA_STD_Q15_QUERIES.jsonl")
    targets={row["document_id"]:row for row in csv.DictReader((LC/"inputs"/"LARGE_SHARED_TARGETS.csv").open(encoding="utf-8"))}
    docs={row["document_id"]:row["source_text"] for row in read_jsonl(CORE3/"inputs"/"CLEAN_CORE3_PROTECTED_DB.jsonl")}
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    free,total=torch.cuda.mem_get_info()
    if free<14*(1<<30): raise RuntimeError(f"GPU_WAIT_REQUIRED free={free} total={total}")
    checkpoint("PHASE_B_GENERATOR_LOADING", gpu_free_bytes=free)
    tokenizer=AutoTokenizer.from_pretrained(QWEN,local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(QWEN,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa",low_cpu_mem_usage=True).to("cuda").eval()
    con=database()
    generate_gt(model,tokenizer,con,targets,ia_queries,ia_protocol)
    old_rows=[row for row in read_jsonl(Path(pre["old_detection"]["path"])) if row["attack"]!="BENIGN"]
    old_answer_rows=read_jsonl(Path(pre["old_answers"]["path"]))
    old_branches=defaultdict(set)
    old_selected={row["query_id"]:row["selected_source_id"] for row in old_rows}
    for row in old_answer_rows:
        if row["condition"]=="NO_DEFENSE":
            old_branches[row["query_id"]].add("A0")
        elif row.get("detector_alarm") and row.get("hidden_source_id")==old_selected.get(row["query_id"]):
            old_branches[row["query_id"]].add("A_HIDE")
        else:
            old_branches[row["query_id"]].add("OTHER")
    generate_answers(model,tokenizer,con,old_rows,docs,pre,old_branches)
    generate_answers(model,tokenizer,con,new_rows,docs,pre)
    gt_rows=[]
    for target_id,labels,raw,valid,error,prompt_sha,completed in con.execute("SELECT * FROM ia_gt ORDER BY target_id"):
        gt_rows.append({"target_id":target_id,"labels":json.loads(labels),"raw_outputs":json.loads(raw),"valid":bool(valid),
                        "error":error,"prompt_sha256":prompt_sha,"completed_utc":completed})
    write_jsonl(EXP/"runtime"/"IA_STD_Q15_GROUND_TRUTH.jsonl",gt_rows)
    columns=[row[0] for row in con.execute("SELECT * FROM answer LIMIT 1").description]
    answer_rows=[dict(zip(columns,values)) for values in con.execute("SELECT * FROM answer ORDER BY query_id,branch")]
    write_jsonl(EXP/"runtime"/"STANDARDIZED_BRANCH_ANSWERS.jsonl",answer_rows)
    manifest={"gt_rows":len(gt_rows),"answer_rows":len(answer_rows),
              "a0":sum(row["branch"]=="A0" for row in answer_rows),"a_hide":sum(row["branch"]=="A_HIDE" for row in answer_rows),
              "gt_sha256":sha_file(EXP/"runtime"/"IA_STD_Q15_GROUND_TRUTH.jsonl"),
              "answers_sha256":sha_file(EXP/"runtime"/"STANDARDIZED_BRANCH_ANSWERS.jsonl"),"completed_utc":now()}
    atomic_json(EXP/"runtime"/"PHASE_B_GENERATION_MANIFEST.json",manifest)
    checkpoint("PHASE_B_GENERATION_COMPLETE",**manifest)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
