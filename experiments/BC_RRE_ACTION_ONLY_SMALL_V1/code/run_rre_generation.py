#!/usr/bin/env python3
"""Generate only BC-RRE answers for the frozen BC-q97 risk set."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments/CLEAN_CORE3_DEV_V1"
CAMPAIGN = ROOT / "experiments/BC_RRE_ACTION_ONLY_SMALL_V1"
GENERATOR = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any "
                 "required output format exactly. If the context is insufficient, answer "
                 "exactly: I don't know.")
SOURCE_TOKEN_BUDGET = 2048
MAX_PROMPT_TOKENS = 3072
MAX_NEW_TOKENS = 160
BATCH_SIZE = 16
SEED = 20260911


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"campaign": "BC_RRE_ACTION_ONLY_SMALL_V1", "stage": stage,
               "updated_utc": now(), "pid": os.getpid(), **details}
    write_json(CAMPAIGN / "HEARTBEAT.json", payload)
    write_json(CAMPAIGN / f"checkpoints/{stage}.json", payload)
    lines = ["# BC_RRE_ACTION_ONLY_SMALL_V1 STATUS", "", f"- Current stage: `{stage}`",
             f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{os.getpid()}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    (CAMPAIGN / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def waterfill(lengths: list[int], total: int) -> list[int]:
    values = np.asarray(lengths, dtype=int)
    caps = np.zeros(len(values), dtype=int)
    remaining = total
    active = [index for index, length in enumerate(values) if length > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, int(values[index] - caps[index]), remaining)
            if add:
                caps[index] += add; remaining -= add; changed = True
            if caps[index] >= values[index]:
                active.remove(index)
            if remaining <= 0:
                break
        if not changed:
            break
    return caps.tolist()


def build_prompt(tokenizer, query: str, source_ids: list[str], documents: dict[str, str]) -> tuple[str, dict]:
    encoded = [tokenizer(documents[source], add_special_tokens=False).input_ids for source in source_ids]
    caps = waterfill([len(value) for value in encoded], SOURCE_TOKEN_BUDGET)
    visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip() for value, cap in zip(encoded, caps)]
    context = "\n\n".join(f"[Document {rank}]\n{text}" for rank, text in enumerate(visible, 1))
    user = f"Retrieved context:\n{context}\n\nUser query:\n{query}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(prompt_ids) > MAX_PROMPT_TOKENS:
        prompt_ids = prompt_ids[-MAX_PROMPT_TOKENS:]
        rendered = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    return rendered, {"source_ids": source_ids, "source_caps": caps,
                      "source_token_count": int(sum(caps)), "context_sha256": sha_text(context),
                      "prompt_sha256": sha_text(json.dumps(prompt_ids, separators=(",", ":"))),
                      "prompt_tokens": len(prompt_ids)}


def main() -> None:
    expected = (CAMPAIGN / "configs/BC_RRE_ACTION_ONLY_SMALL_V1_PRECOMMIT.sha256").read_text().split()[0]
    if sha256(CAMPAIGN / "configs/BC_RRE_ACTION_ONLY_SMALL_V1_PRECOMMIT.json") != expected:
        raise RuntimeError("precommit checksum mismatch")
    retrieval = read_jsonl(CAMPAIGN / "cache/RRE_REFERENCE_RETRIEVAL.jsonl")
    protected = {row["document_id"]: row["source_text"] for row in read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")}
    reference = {row["document_id"]: row["source_text"] for row in read_jsonl(CAMPAIGN / "inputs/REFERENCE_POOL.jsonl")}
    documents = {**protected, **reference}
    tasks = []
    for row in sorted(retrieval, key=lambda item: item["query_id"]):
        sources = list(row["original_top_document_ids"])
        slot = sources.index(row["selected_source_id"])
        sources[slot] = row["replacement_source_id"]
        if len(sources) != 4 or len(set(sources)) != 4:
            raise RuntimeError(f"invalid RRE source list for {row['query_id']}")
        tasks.append((row, sources, slot))

    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint("RRE_GENERATOR_LOADING", risk_queries=len(tasks))
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda").eval()
    database = sqlite3.connect(CAMPAIGN / "runtime/rre_generation.sqlite3", timeout=120)
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=FULL")
    database.execute("""CREATE TABLE IF NOT EXISTS answer(
        query_id TEXT PRIMARY KEY, answer TEXT, answer_sha256 TEXT, prompt_sha256 TEXT,
        prompt_tokens INTEGER, answer_tokens INTEGER, source_ids_json TEXT, source_caps_json TEXT,
        source_token_count INTEGER, context_sha256 TEXT, removed_source_id TEXT,
        replacement_source_id TEXT, replacement_slot INTEGER, wall_seconds REAL, completed_utc TEXT)""")
    completed = {row[0] for row in database.execute("SELECT query_id FROM answer")}
    pending = [task for task in tasks if task[0]["query_id"] not in completed]
    started = time.monotonic()
    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset:offset + BATCH_SIZE]
        built = [build_prompt(tokenizer, row["query"], sources, documents) for row, sources, _ in batch]
        encoded = tokenizer([item[0] for item in built], add_special_tokens=False, padding=True,
                            truncation=True, max_length=MAX_PROMPT_TOKENS, return_tensors="pt").to(model.device)
        before = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    num_beams=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - before
        generated = output[:, encoded.input_ids.shape[1]:]
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for (row, sources, slot), (_, provenance), answer in zip(batch, built, text):
            clean = answer.strip()
            database.execute("INSERT OR REPLACE INTO answer VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                row["query_id"], clean, sha_text(clean), provenance["prompt_sha256"], provenance["prompt_tokens"],
                len(tokenizer(clean, add_special_tokens=False).input_ids), json.dumps(sources),
                json.dumps(provenance["source_caps"]), provenance["source_token_count"], provenance["context_sha256"],
                row["selected_source_id"], row["replacement_source_id"], slot + 1, elapsed / len(batch), now()))
        database.commit()
        done = len(completed) + min(offset + len(batch), len(pending))
        rate = (offset + len(batch)) / max(time.monotonic() - started, 1e-9)
        checkpoint("RRE_GENERATION_PROGRESS", completed=done, total=len(tasks),
                   eta_seconds=round((len(tasks)-done) / max(rate, 1e-9), 1))
    columns = [item[0] for item in database.execute("SELECT * FROM answer LIMIT 1").description]
    output_path = CAMPAIGN / "runtime/RRE_GENERATED_ANSWERS.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for values in database.execute("SELECT * FROM answer ORDER BY query_id"):
            handle.write(json.dumps(dict(zip(columns, values)), ensure_ascii=False, sort_keys=True) + "\n")
    count = database.execute("SELECT COUNT(*) FROM answer").fetchone()[0]
    if count != len(tasks):
        raise RuntimeError(f"incomplete RRE generation {count}/{len(tasks)}")
    manifest = {"verdict": "RRE_GENERATION_COMPLETE", "new_generation_count": count,
                "safe_query_new_generation_count": 0, "output_sha256": sha256(output_path),
                "precommit_sha256": expected, "runtime_seconds": time.monotonic()-started}
    write_json(CAMPAIGN / "runtime/RRE_GENERATION_MANIFEST.json", manifest)
    checkpoint("RRE_GENERATION_COMPLETE", **manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
