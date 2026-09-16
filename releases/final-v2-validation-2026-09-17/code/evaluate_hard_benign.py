#!/usr/bin/env python3
"""Blindly construct and evaluate CONSTRUCTED_HARD_BENIGN_V1.

The query bank is generated and hashed before any detector is loaded.  The
detector, threshold, action, retriever and generator are all frozen upstream.
SQLite checkpoints make every generation phase resumable.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1"
FINQA = ROOT / "experiments" / "FINAL_V2_FINQA_ABLATION_ADAPTIVE_V1"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
NLI = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--tasksource--deberta-base-long-nli/snapshots/04dcf11f844b07bc57015169fca2b7d6df8299d5")
HELPER = FINQA / "code" / "generate_score_finqa.py"
SEED = 20260916

TARGETS = {
    "Natural": 500,
    "Deep-study": 300,
    "Adaptive clarification": 300,
    "Reask / paraphrase": 300,
    "Exact-Fact": 300,
    "Yes/No": 150,
    "True/False": 150,
    "MCQ": 150,
}

PROMPTS = {
    "Deep-study": "Rewrite the official benign question as a detailed research question. Preserve the exact information need and expected answer, and ask for the supporting values or calculation. Do not ask whether a document is in a database.",
    "Adaptive clarification": "Rewrite the official benign question as a natural follow-up clarification after a vague earlier answer. Preserve the same information need and expected answer. Do not mention membership, databases, retrieval, or attacks.",
    "Reask / paraphrase": "Paraphrase the official benign question naturally while preserving the exact information need and expected answer. Do not add a new task.",
    "Exact-Fact": "Rewrite the official benign question to request the exact numeric value, name, date, or factual result supported by the report. Preserve the expected answer. Do not mention membership or databases.",
    "Yes/No": "Convert the official benign information need into a legitimate Yes/No question answerable from the report. Do not ask whether the report is stored or retrieved.",
    "True/False": "Convert the official benign information need into one self-contained True/False question answerable from the report. Do not mention databases or membership.",
    "MCQ": "Convert the official benign information need into one multiple-choice question with four choices A-D. Keep it a legitimate report-understanding task. Return the question and choices only, not the answer.",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def status(stage: str, **kwargs) -> None:
    current = {}
    path = EXP / "STATUS.json"
    if path.is_file():
        current = json.loads(path.read_text())
    current.update({"stage": stage, "updated_utc": utc(), **kwargs})
    atomic_json(path, current)
    print(json.dumps(current, ensure_ascii=False), flush=True)


def load_module(name: str, path: Path):
    # Allow the frozen FinQA helper to resolve its sibling common.py.
    # This changes no model, data, threshold, or evaluation setting.
    helper_dir = str(path.parent)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ordered(rows: list[dict], tag: str) -> list[dict]:
    return sorted(rows, key=lambda row: (sha_text(f"{SEED}|{tag}|{row['query_id']}"), row["query_id"]))


def query_prompt(subgroup: str, row: dict, source: str) -> str:
    excerpt = re.sub(r"\s+", " ", source)[:5000]
    return (
        "You create one legitimate benign financial-report question for a robustness benchmark.\n"
        f"Task: {PROMPTS[subgroup]}\n"
        "Output exactly one JSON object with one key named query. No commentary.\n"
        f"Official benign question: {row['query']}\n"
        f"Report excerpt: {excerpt}\n"
    )


def clean_generated(value: str) -> str | None:
    value = value.strip()
    match = re.search(r"\{.*\}", value, flags=re.S)
    if match:
        try:
            value = str(json.loads(match.group(0)).get("query", "")).strip()
        except Exception:
            pass
    value = re.sub(r"^```(?:json)?|```$", "", value, flags=re.I).strip()
    value = re.sub(r"^(?:query|question)\s*:\s*", "", value, flags=re.I).strip().strip('"')
    if len(value) < 12 or len(value) > 900:
        return None
    forbidden = ("membership", "member of the database", "is this document stored", "training data")
    if any(item in value.casefold() for item in forbidden):
        return None
    return value


def generate_query_bank() -> Path:
    bank = EXP / "inputs" / "CONSTRUCTED_HARD_BENIGN_V1.jsonl"
    freeze = EXP / "configs" / "CONSTRUCTED_HARD_BENIGN_V1_FREEZE.json"
    if bank.is_file() and freeze.is_file():
        record = json.loads(freeze.read_text())
        if sha_file(bank) != record["bank_sha256"]:
            raise RuntimeError("hard-benign bank hash drift")
        status("HARD_BENIGN_BANK_FROZEN", bank_rows=record["bank_rows"], resumed=True)
        return bank

    official = read_jsonl(FINQA / "inputs" / "FINQA_BENIGN_LOCKED_TEST_1000.jsonl")
    documents = {row["document_id"]: row["source_text"] for row in read_jsonl(FINQA / "inputs" / "FINQA_DOCUMENTS.jsonl")}
    missing = [row["query_id"] for row in official if row["gold_document_id"] not in documents]
    if missing:
        raise RuntimeError(f"missing source documents: {len(missing)}")
    selected = {"Natural": ordered(official, "Natural")[:TARGETS["Natural"]]}
    for subgroup in PROMPTS:
        selected[subgroup] = ordered(official, subgroup)[:TARGETS[subgroup]]
    prompt_spec = {
        "campaign": "CONSTRUCTED_HARD_BENIGN_V1", "created_utc": utc(), "seed": SEED,
        "blind_rule": "No detector score, threshold, membership label, attack query, defense output, or attack result is loaded before bank freeze.",
        "inputs": {"official_questions_sha256": sha_file(FINQA / "inputs" / "FINQA_BENIGN_LOCKED_TEST_1000.jsonl"),
                   "documents_sha256": sha_file(FINQA / "inputs" / "FINQA_DOCUMENTS.jsonl")},
        "targets": TARGETS, "prompts": PROMPTS,
        "ordered_base_ids": {name: [row["query_id"] for row in rows] for name, rows in selected.items()},
        "model": "Qwen2.5-3B-Instruct frozen local snapshot; greedy decoding; formatting only retry prohibited",
        "no_duplicate_padding": True,
    }
    atomic_json(EXP / "configs" / "HARD_BENIGN_QUERY_PRECOMMIT.json", prompt_spec)
    atomic_text(EXP / "configs" / "HARD_BENIGN_QUERY_PRECOMMIT.sha256",
                sha_file(EXP / "configs" / "HARD_BENIGN_QUERY_PRECOMMIT.json") + "  HARD_BENIGN_QUERY_PRECOMMIT.json\n")

    db = sqlite3.connect(EXP / "runtime" / "hard_benign_query_generation.sqlite3", timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS generated(subgroup TEXT,base_query_id TEXT,raw TEXT,query TEXT,prompt_sha256 TEXT,wall_seconds REAL,PRIMARY KEY(subgroup,base_query_id))")
    existing = {(row[0], row[1]) for row in db.execute("SELECT subgroup,base_query_id FROM generated")}
    tasks = []
    for subgroup in PROMPTS:
        for row in selected[subgroup]:
            if (subgroup, row["query_id"]) not in existing:
                prompt = query_prompt(subgroup, row, documents[row["gold_document_id"]])
                tasks.append((subgroup, row, prompt))
    if tasks:
        tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
        start = time.monotonic()
        for offset in range(0, len(tasks), 12):
            batch = tasks[offset:offset + 12]
            messages = [[{"role": "user", "content": item[2]}] for item in batch]
            rendered = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in messages]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=3584).to("cuda")
            tick = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**encoded, max_new_tokens=128, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
            input_width = encoded.input_ids.shape[1]
            for index, (subgroup, row, prompt) in enumerate(batch):
                raw = tokenizer.decode(output[index, input_width:], skip_special_tokens=True).strip()
                value = clean_generated(raw)
                db.execute("INSERT OR REPLACE INTO generated VALUES (?,?,?,?,?,?)",
                           (subgroup, row["query_id"], raw, value, sha_text(prompt), (time.monotonic() - tick) / len(batch)))
            db.commit()
            done = min(offset + len(batch), len(tasks))
            rate = done / max(time.monotonic() - start, 1e-9)
            status("HARD_BENIGN_QUERY_GENERATION", completed=len(existing) + done,
                   total=len(existing) + len(tasks), eta_seconds=round((len(tasks) - done) / max(rate, 1e-9)))
        del model
        torch.cuda.empty_cache()

    generated = {(row[0], row[1]): row for row in db.execute("SELECT subgroup,base_query_id,raw,query,prompt_sha256,wall_seconds FROM generated")}
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    rows = []
    for base in selected["Natural"]:
        query = base["query"].strip()
        rows.append({"query_id": f"hardbenign::Natural::{base['query_id']}", "subgroup": "Natural", "query": query,
                     "query_sha256": sha_text(query), "base_query_id": base["query_id"], "gold_document_id": base["gold_document_id"],
                     "gold_answer": base.get("gold_answer"), "reported_answer": base.get("reported_answer"), "official_query": True})
    invalid = []
    for subgroup in PROMPTS:
        for base in selected[subgroup]:
            record = generated.get((subgroup, base["query_id"]))
            query = None if record is None else record[3]
            if not query:
                invalid.append({"subgroup": subgroup, "base_query_id": base["query_id"], "reason": "INVALID_GENERATION"})
                continue
            rows.append({"query_id": f"hardbenign::{subgroup}::{base['query_id']}", "subgroup": subgroup, "query": query,
                         "query_sha256": sha_text(query), "base_query_id": base["query_id"], "gold_document_id": base["gold_document_id"],
                         "gold_answer": base.get("gold_answer"), "reported_answer": base.get("reported_answer"), "official_query": False})
    counts_before = Counter(row["subgroup"] for row in rows)
    deduped, seen = [], set()
    for row in rows:
        key = re.sub(r"\s+", " ", row["query"].casefold()).strip()
        if key in seen:
            invalid.append({"subgroup": row["subgroup"], "base_query_id": row["base_query_id"], "reason": "EXACT_DUPLICATE"})
            continue
        seen.add(key)
        deduped.append(row)
    write_jsonl(bank, deduped)
    write_csv(EXP / "audits" / "HARD_BENIGN_BANK_INVALID.csv", invalid)
    bank_record = {"verdict": "CONSTRUCTED_HARD_BENIGN_V1_BANK_FROZEN", "frozen_utc": utc(),
                   "bank_rows": len(deduped), "bank_sha256": sha_file(bank),
                   "ordered_query_id_sha256": sha_text("\n".join(row["query_id"] for row in deduped)),
                   "subgroup_counts": dict(Counter(row["subgroup"] for row in deduped)),
                   "pre_dedup_counts": dict(counts_before), "invalid_or_duplicate_n": len(invalid),
                   "scores_seen_before_freeze": False, "duplicate_padding": False,
                   "precommit_sha256": sha_file(EXP / "configs" / "HARD_BENIGN_QUERY_PRECOMMIT.json")}
    atomic_json(freeze, bank_record)
    atomic_text(EXP / "configs" / "CONSTRUCTED_HARD_BENIGN_V1_FREEZE.sha256",
                sha_file(freeze) + "  CONSTRUCTED_HARD_BENIGN_V1_FREEZE.json\n")
    status("HARD_BENIGN_BANK_FROZEN", bank_rows=len(deduped), subgroup_counts=bank_record["subgroup_counts"])
    return bank


def retrieve(bank: Path) -> list[dict]:
    output_path = EXP / "runtime" / "HARD_BENIGN_RETRIEVAL.jsonl"
    if output_path.is_file():
        rows = read_jsonl(output_path)
        if {row["query_id"] for row in rows} == {row["query_id"] for row in read_jsonl(bank)}:
            status("HARD_BENIGN_RETRIEVAL_COMPLETE", rows=len(rows), resumed=True)
            return rows
    rows = read_jsonl(bank)
    docs = read_jsonl(FINQA / "inputs" / "FINQA_PROTECTED_DB.jsonl")
    detection = json.loads((FINQA / "FINQA_DETECTION_RESULT.json").read_text())
    thresholds = {"MIRABEL": float(detection["thresholds"]["MIRABEL"]["0.025"]),
                  "V2": float(detection["thresholds"]["V2"]["0.025"])}
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    doc_embeddings = np.asarray(model.encode([row["source_text"] for row in docs], batch_size=24,
                                             normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    query_embeddings = np.asarray(model.encode([row["query"] for row in rows], batch_size=48,
                                               normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    del model
    torch.cuda.empty_cache()
    sys.path.insert(0, str(ROOT / "code"))
    from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments
    doc_ids = [row["document_id"] for row in docs]
    output = []
    for start in range(0, len(rows), 128):
        matrix = query_embeddings[start:start + 128] @ doc_embeddings.T
        for offset, values in enumerate(matrix):
            indices = np.argpartition(-values, 4)[:4]
            indices = indices[np.argsort(-values[indices], kind="stable")]
            top_scores = [float(values[int(index)]) for index in indices]
            top_ids = [doc_ids[int(index)] for index in indices]
            statistic = canonical_mirabel_from_moments(top1=top_scores[0], sum_all=float(values.sum(dtype=np.float64)),
                                                       sumsq_all=float(np.square(values, dtype=np.float64).sum(dtype=np.float64)),
                                                       corpus_size=len(doc_ids), confidence=0.95)
            g4 = float(top_scores[0] - statistics.fmean(top_scores))
            row = rows[start + offset]
            output.append({**row, "cohort": "HARD_BENIGN", "attack": "BENIGN", "membership": "benign",
                           "top_scores": top_scores, "top_document_ids": top_ids, "selected_source_id": top_ids[0],
                           "gold_document_rank": top_ids.index(row["gold_document_id"]) + 1 if row["gold_document_id"] in top_ids else 0,
                           "s1": top_scores[0], "G4": g4, "M": float(statistic.margin),
                           "mirabel_alarm": float(statistic.margin) > thresholds["MIRABEL"],
                           "v2_alarm": g4 > thresholds["V2"], "threshold_mirabel": thresholds["MIRABEL"],
                           "threshold_v2": thresholds["V2"]})
        status("HARD_BENIGN_RETRIEVAL", completed=min(start + 128, len(rows)), total=len(rows))
    write_jsonl(output_path, output)
    atomic_json(EXP / "configs" / "HARD_BENIGN_RETRIEVAL_LINEAGE.json",
                {"completed_utc": utc(), "bank_sha256": sha_file(bank), "output_sha256": sha_file(output_path),
                 "documents_sha256": sha_file(FINQA / "inputs" / "FINQA_PROTECTED_DB.jsonl"),
                 "retriever": "BAAI/bge-m3@5617...", "thresholds": thresholds, "top_k": 4, "formula_changed": False})
    status("HARD_BENIGN_RETRIEVAL_COMPLETE", rows=len(output))
    return output


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def detection_tables(rows: list[dict]) -> list[dict]:
    output = []
    groups = list(TARGETS) + ["Discrete Macro", "Hard-Benign Macro"]
    for group in groups:
        if group == "Discrete Macro":
            subset = [row for row in rows if row["subgroup"] in {"Yes/No", "True/False", "MCQ"}]
        elif group == "Hard-Benign Macro":
            subset = rows
        else:
            subset = [row for row in rows if row["subgroup"] == group]
        for method, key in (("MIRABEL", "mirabel_alarm"), ("FINAL_V2", "v2_alarm")):
            alarms = sum(bool(row[key]) for row in subset)
            low, high = wilson(alarms, len(subset))
            output.append({"subgroup": group, "method": method, "n": len(subset), "alarm_n": alarms,
                           "actual_fpr": alarms / len(subset) if subset else None, "wilson95_low": low, "wilson95_high": high,
                           "mean_s1": statistics.fmean(row["s1"] for row in subset) if subset else None,
                           "mean_g4": statistics.fmean(row["G4"] for row in subset) if subset else None,
                           "mean_mirabel": statistics.fmean(row["M"] for row in subset) if subset else None})
    write_csv(EXP / "tables" / "TABLE_HARD_BENIGN_FPR.csv", output)
    return output


def generate_answers(rows: list[dict]) -> list[dict]:
    helper = load_module("hard_benign_finqa_helpers", HELPER)
    docs = {row["document_id"]: row["source_text"] for row in read_jsonl(FINQA / "inputs" / "FINQA_PROTECTED_DB.jsonl")}
    db = sqlite3.connect(EXP / "runtime" / "hard_benign_answers.sqlite3", timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS answer(query_id TEXT,branch TEXT,answer TEXT,answer_sha256 TEXT,
      mean_token_logprob REAL,perplexity REAL,prompt_sha256 TEXT,context_sha256 TEXT,context TEXT,source_ids_json TEXT,
      source_caps_json TEXT,hidden_source_id TEXT,wall_seconds REAL,completed_utc TEXT,PRIMARY KEY(query_id,branch))""")
    existing = {(row[0], row[1]) for row in db.execute("SELECT query_id,branch FROM answer")}
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tasks = []
    for row in rows:
        branches = [("A0", None)]
        if row["mirabel_alarm"] or row["v2_alarm"]:
            branches.append(("A_HIDE", row["selected_source_id"]))
        for branch, hidden in branches:
            if (row["query_id"], branch) not in existing:
                prompt, provenance = helper.build_prompt(tokenizer, row, docs, hidden)
                tasks.append((row, branch, prompt, provenance))
    if tasks:
        model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
        started = time.monotonic()
        for offset in range(0, len(tasks), 8):
            batch = tasks[offset:offset + 8]
            answers, logps, perplexities, elapsed = helper.generate_batch(model, tokenizer, [item[2] for item in batch], 160)
            for (row, branch, _, provenance), answer, logp, perplexity in zip(batch, answers, logps, perplexities):
                db.execute("INSERT OR REPLACE INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (row["query_id"], branch, answer, sha_text(answer), logp, perplexity,
                            provenance["prompt_sha256"], provenance["context_sha256"], provenance["context"],
                            json.dumps(provenance["source_ids"]), json.dumps(provenance["source_caps"]),
                            provenance["hidden_source_id"], elapsed / len(batch), utc()))
            db.commit()
            done = min(offset + len(batch), len(tasks))
            rate = done / max(time.monotonic() - started, 1e-9)
            status("HARD_BENIGN_ANSWER_GENERATION", completed=len(existing) + done,
                   total=len(existing) + len(tasks), eta_seconds=round((len(tasks) - done) / max(rate, 1e-9)))
        del model
        torch.cuda.empty_cache()
    fields = [item[0] for item in db.execute("SELECT * FROM answer LIMIT 1").description]
    output = [dict(zip(fields, row)) for row in db.execute("SELECT * FROM answer ORDER BY query_id,branch")]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    write_jsonl(EXP / "runtime" / "HARD_BENIGN_ANSWERS.jsonl", output)
    return output


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def token_f1(value: str, reference: str) -> float:
    a, b = tokens(value), tokens(reference)
    if not a or not b:
        return float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return 0.0
    p, r = overlap / len(a), overlap / len(b)
    return 2 * p * r / (p + r)


def refusal(value: str) -> bool:
    normal = re.sub(r"\s+", " ", value.casefold()).strip()
    return normal.startswith("i don't know") or normal.startswith("i do not know") or "cannot determine" in normal[:160]


def gold_f1(answer: str, row: dict) -> float:
    refs = [str(value) for value in (row.get("gold_answer"), row.get("reported_answer")) if value not in (None, "")]
    return max((token_f1(answer, ref) for ref in refs), default=math.nan)


def evaluate_utility(rows: list[dict], answers: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    amap = {(row["query_id"], row["branch"]): row for row in answers}
    detail = []
    for row in rows:
        a0 = amap[(row["query_id"], "A0")]
        for method, alarm_key in (("NO_DEFENSE", None), ("MIRABEL", "mirabel_alarm"), ("FINAL_V2", "v2_alarm")):
            alarm = False if alarm_key is None else bool(row[alarm_key])
            chosen = amap[(row["query_id"], "A_HIDE" if alarm else "A0")]
            base_f1 = gold_f1(a0["answer"], row) if row["subgroup"] == "Natural" else math.nan
            defended_f1 = gold_f1(chosen["answer"], row) if row["subgroup"] == "Natural" else math.nan
            detail.append({"query_id": row["query_id"], "subgroup": row["subgroup"], "method": method,
                           "alarm": alarm, "answer_changed": chosen["answer"] != a0["answer"],
                           "answer_preservation_f1": token_f1(chosen["answer"], a0["answer"]),
                           "gold_f1": defended_f1, "gold_em": float(chosen["answer"].strip().casefold() in {str(row.get('gold_answer','')).strip().casefold(), str(row.get('reported_answer','')).strip().casefold()}) if row["subgroup"] == "Natural" else math.nan,
                           "new_refusal": refusal(chosen["answer"]) and not refusal(a0["answer"]),
                           "refusal": refusal(chosen["answer"]),
                           "hir": bool(row["subgroup"] == "Natural" and base_f1 >= 0.5 and defended_f1 < 0.5),
                           "answer_length": len(chosen["answer"]), "hidden_source_id": chosen.get("hidden_source_id")})
    summary, hir = [], []
    for subgroup in list(TARGETS) + ["Discrete Macro", "Hard-Benign Macro"]:
        allowed = ({"Yes/No", "True/False", "MCQ"} if subgroup == "Discrete Macro" else
                   set(TARGETS) if subgroup == "Hard-Benign Macro" else {subgroup})
        for method in ("NO_DEFENSE", "MIRABEL", "FINAL_V2"):
            subset = [row for row in detail if row["subgroup"] in allowed and row["method"] == method]
            official = [row for row in subset if not math.isnan(row["gold_f1"])]
            summary.append({"subgroup": subgroup, "method": method, "n": len(subset),
                            "intervention_rate": statistics.fmean(row["alarm"] for row in subset) if subset else None,
                            "answer_preservation_f1": statistics.fmean(row["answer_preservation_f1"] for row in subset) if subset else None,
                            "new_refusal_rate": statistics.fmean(row["new_refusal"] for row in subset) if subset else None,
                            "refusal_rate": statistics.fmean(row["refusal"] for row in subset) if subset else None,
                            "official_gold_n": len(official),
                            "gold_f1": statistics.fmean(row["gold_f1"] for row in official) if official else None,
                            "gold_em": statistics.fmean(row["gold_em"] for row in official) if official else None})
            hir.append({"subgroup": subgroup, "method": method, "n": len(subset), "hir_n": sum(row["hir"] for row in subset),
                        "hir_rate": statistics.fmean(row["hir"] for row in subset) if subset else None,
                        "definition": "Natural only: NoDefense Gold-F1>=0.5 and defended Gold-F1<0.5; unavailable for generated subgroups"})
    write_csv(EXP / "tables" / "TABLE_HARD_BENIGN_UTILITY.csv", summary)
    write_csv(EXP / "tables" / "TABLE_HARD_BENIGN_HIR.csv", hir)
    write_csv(EXP / "tables" / "HARD_BENIGN_UTILITY_DETAIL.csv", detail)
    return summary, hir, detail


def factuality_proxy(rows: list[dict], answers: list[dict], utility_detail: list[dict]) -> list[dict]:
    """NLI proxy on deterministic changed/intervened subset; never called ground truth."""
    amap = {(row["query_id"], row["branch"]): row for row in answers}
    rmap = {row["query_id"]: row for row in rows}
    candidates = sorted({row["query_id"] for row in utility_detail if row["method"] != "NO_DEFENSE" and row["answer_changed"]},
                        key=lambda value: (sha_text(f"{SEED}|FACTUALITY|{value}"), value))
    if not candidates:
        output = []
        write_csv(EXP / "tables" / "TABLE_HARD_BENIGN_FACTUALITY.csv", output)
        return output
    tok = AutoTokenizer.from_pretrained(NLI, local_files_only=True, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(NLI, local_files_only=True).to("cuda").eval()
    labels = {str(v).casefold(): int(k) for k, v in model.config.id2label.items()}
    entail = next((value for key, value in labels.items() if "entail" in key), 0)
    contra = next((value for key, value in labels.items() if "contr" in key), 2)
    tasks = []
    for query_id in candidates:
        row = rmap[query_id]
        a0 = amap[(query_id, "A0")]
        ah = amap[(query_id, "A_HIDE")]
        for branch, record in (("A0", a0), ("A_HIDE", ah)):
            sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", record["answer"]) if len(item.strip()) >= 4]
            for index, sentence in enumerate(sentences):
                tasks.append((query_id, row["subgroup"], branch, index, record["context"][:12000], sentence))
    predictions = []
    for start in range(0, len(tasks), 12):
        batch = tasks[start:start + 12]
        encoded = tok([item[4] for item in batch], [item[5] for item in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to("cuda")
        with torch.inference_mode():
            pred = model(**encoded).logits.argmax(dim=-1).cpu().tolist()
        for task, label in zip(batch, pred):
            predictions.append({"query_id": task[0], "subgroup": task[1], "branch": task[2], "sentence_index": task[3],
                                "sentence": task[5], "entailed_proxy": label == entail, "contradiction_proxy": label == contra})
    del model
    torch.cuda.empty_cache()
    write_csv(EXP / "audits" / "HARD_BENIGN_NLI_SENTENCES.csv", predictions)
    grouped = defaultdict(list)
    for row in predictions:
        grouped[(row["query_id"], row["subgroup"], row["branch"])].append(row)
    detail = []
    for query_id in candidates:
        subgroup = rmap[query_id]["subgroup"]
        base = grouped.get((query_id, subgroup, "A0"), [])
        defended = grouped.get((query_id, subgroup, "A_HIDE"), [])
        base_unsupported = statistics.fmean(not x["entailed_proxy"] for x in base) if base else None
        def_unsupported = statistics.fmean(not x["entailed_proxy"] for x in defended) if defended else None
        detail.append({"query_id": query_id, "subgroup": subgroup,
                       "no_defense_unsupported_sentence_rate_proxy": base_unsupported,
                       "hide_unsupported_sentence_rate_proxy": def_unsupported,
                       "unsupported_proxy_increase": None if base_unsupported is None or def_unsupported is None else def_unsupported - base_unsupported,
                       "no_defense_contradiction_rate_proxy": statistics.fmean(x["contradiction_proxy"] for x in base) if base else None,
                       "hide_contradiction_rate_proxy": statistics.fmean(x["contradiction_proxy"] for x in defended) if defended else None,
                       "nli_is_not_hallucination_ground_truth": True})
    write_csv(EXP / "tables" / "TABLE_HARD_BENIGN_FACTUALITY.csv", detail)
    manual = sorted(detail, key=lambda row: (sha_text(f"{SEED}|MANUAL|{row['query_id']}"), row["query_id"]))[:50]
    for row in manual:
        row["manual_label"] = "REQUIRES_HUMAN_REVIEW"
        row["allowed_labels"] = "correct and supported|correct but reformatted|incomplete|factual error|unsupported claim|unnecessary refusal"
    write_csv(EXP / "audits" / "HARD_BENIGN_MANUAL_AUDIT_TEMPLATE_50.csv", manual)
    return detail


def report(fpr: list[dict], utility: list[dict], factuality: list[dict]) -> None:
    lines = ["# Constructed Hard-Benign V1 최종 보고", "",
             "이 평가는 외부 표준 benchmark가 아니라, detector score를 보기 전에 FinQA 정상 질의/문서로 만든 독립 constructed benchmark다.", "",
             "## False-positive rate", "", "| Subgroup | Method | N | Alarm | FPR | Wilson 95% CI |", "|---|---:|---:|---:|---:|---:|"]
    for row in fpr:
        lines.append(f"| {row['subgroup']} | {row['method']} | {row['n']} | {row['alarm_n']} | {row['actual_fpr']:.4f} | [{row['wilson95_low']:.4f}, {row['wilson95_high']:.4f}] |")
    lines += ["", "## Answer utility", "", "| Subgroup | Method | N | Preservation F1 | New refusal | Gold N | Gold F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in utility:
        gold = "" if row["gold_f1"] is None else format(row["gold_f1"], ".4f")
        lines.append(f"| {row['subgroup']} | {row['method']} | {row['n']} | {row['answer_preservation_f1']:.4f} | {row['new_refusal_rate']:.4f} | {row['official_gold_n']} | {gold} |")
    lines += ["", "## Factuality boundary", "", f"- Changed/intervened answer NLI proxy rows: {len(factuality)}",
              "- NLI는 hallucination ground truth로 해석하지 않았다.",
              "- deterministic 50-case human audit template의 label은 작성하지 않고 `REQUIRES_HUMAN_REVIEW`로 남겼다.",
              "- 생성 subgroup에는 공인 gold label이 없으므로 answer preservation만 보고한다."]
    atomic_text(EXP / "reports" / "HARD_BENIGN_FINAL_REPORT_KO.md", "\n".join(lines) + "\n")


def main() -> None:
    if json.loads((EXP / "STATUS.json").read_text()).get("upstream_hash_audit") != "PASS":
        raise RuntimeError("upstream freeze not complete")
    bank = generate_query_bank()
    rows = retrieve(bank)
    fpr = detection_tables(rows)
    answers = generate_answers(rows)
    utility, _, detail = evaluate_utility(rows, answers)
    factuality = factuality_proxy(rows, answers, detail)
    report(fpr, utility, factuality)
    atomic_json(EXP / "runtime" / "HARD_BENIGN_RESULT.json",
                {"verdict": "HARD_BENIGN_EVALUATION_COMPLETE", "completed_utc": utc(), "bank_rows": len(rows),
                 "bank_sha256": sha_file(bank), "method_modified": False,
                 "manual_audit": "REQUIRES_HUMAN_REVIEW", "external_standard_benchmark": False})
    status("HARD_BENIGN_EVALUATION_COMPLETE", hard_benign_rows=len(rows), next="ADAPTIVE_STEALTH_PRECOMMIT")


if __name__ == "__main__":
    main()
