#!/usr/bin/env python3
"""Generate each frozen IA-ST1 A0/A_HIDE branch once with the Final-LC Qwen stack."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1"
POST = EXP / "post_ready"
PRECOMMIT = POST / "configs" / "IA_ST1_E2E_PRECOMMIT.json"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
DB = POST / "runtime" / "ia_st1_generation.sqlite3"
SYSTEM_PROMPT = "Answer the user's question using only the retrieved context. Follow any required output format exactly. If the context is insufficient, answer exactly: I don't know."
SOURCE_TOKEN_BUDGET = 2048
MAX_PROMPT_TOKENS = 3072


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def checkpoint(stage: str, **extra: object) -> None:
    value = {"campaign": "IA_STEALTH_CONFIRMATION_V1", "phase": "IA_MATCHED_BUDGET_E2E", "stage": stage, "updated_utc": now(), "pid": os.getpid(), **extra}
    atomic_json(POST / "HEARTBEAT.json", value)
    atomic_text(POST / "STATUS.md", "\n".join(["# IA-ST1 post-ready evaluation", "", f"- stage: `{stage}`", f"- updated_utc: `{value['updated_utc']}`"] + [f"- {key}: `{item}`" for key, item in extra.items()]) + "\n")


def verify_precommit() -> dict:
    expected = PRECOMMIT.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if sha(PRECOMMIT) != expected:
        raise RuntimeError("E2E precommit hash mismatch")
    pre = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    for key in ("detection_result", "detection_cache", "queries", "gt", "corpus", "targets", "final_lc_manifest", "generator_config"):
        if sha(Path(pre[key]["path"])) != pre[key]["sha256"]:
            raise RuntimeError(f"frozen E2E input drift: {key}")
    for path, expected_hash in pre["code"].items():
        if sha(Path(path)) != expected_hash:
            raise RuntimeError(f"E2E code drift: {path}")
    return pre


def waterfill(lengths: list[int], total: int) -> list[int]:
    caps = np.zeros(len(lengths), dtype=int)
    remaining = total
    active = [index for index, length in enumerate(lengths) if length > 0]
    while remaining and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, lengths[index] - int(caps[index]), remaining)
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
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(ids) > MAX_PROMPT_TOKENS:
        ids = ids[-MAX_PROMPT_TOKENS:]
        rendered = tokenizer.decode(ids, skip_special_tokens=False)
    return rendered, {"prompt_sha256": sha_text(json.dumps(ids, separators=(",", ":"))), "prompt_tokens": len(ids), "source_ids": kept, "source_caps": caps, "source_token_count": sum(caps), "context_sha256": sha_text(context), "hidden_source_id": hidden}


def database() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA synchronous=FULL")
    con.execute("""CREATE TABLE IF NOT EXISTS answer (
      query_id TEXT NOT NULL, branch TEXT NOT NULL, answer TEXT NOT NULL, answer_sha256 TEXT NOT NULL,
      prompt_sha256 TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, answer_tokens INTEGER NOT NULL,
      source_ids_json TEXT NOT NULL, source_caps_json TEXT NOT NULL, source_token_count INTEGER NOT NULL,
      context_sha256 TEXT NOT NULL, hidden_source_id TEXT, wall_seconds REAL NOT NULL,
      completed_utc TEXT NOT NULL, PRIMARY KEY(query_id,branch))""")
    con.commit()
    return con


def needs_hidden(row: dict, pre: dict) -> bool:
    return float(row["M"]) > float(pre["thresholds"]["MIRABEL"]["threshold"]) or float(row["R_LC"]) > float(pre["thresholds"]["Final LC"]["threshold"]) or float(row["M"]) > 0.0


def generate_batch(model, tokenizer, prompts: list[str]) -> tuple[list[str], float]:
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True, max_length=MAX_PROMPT_TOKENS, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(**encoded, max_new_tokens=12, do_sample=False, num_beams=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - started
    answers = tokenizer.batch_decode(sequences[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
    del encoded, sequences
    return [answer.strip() for answer in answers], elapsed


def main() -> None:
    pre = verify_precommit()
    rows = read_jsonl(Path(pre["detection_cache"]["path"]))
    docs = {row["document_id"]: row["source_text"] for row in read_jsonl(Path(pre["corpus"]["path"]))}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for frozen Qwen generation")
    free, total = torch.cuda.mem_get_info()
    if free < 14 * (1 << 30):
        raise RuntimeError(f"GPU_WAIT_REQUIRED free={free} total={total}")
    checkpoint("QWEN_LOADING", rows=len(rows), gpu_free_bytes=free)
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    con = database()
    existing = {(query_id, branch) for query_id, branch in con.execute("SELECT query_id,branch FROM answer")}
    tasks = []
    for row in sorted(rows, key=lambda item: item["query_id"]):
        branches = [("A0", None)]
        if needs_hidden(row, pre): branches.append(("A_HIDE", row["selected_source_id"]))
        for branch, hidden in branches:
            if (row["query_id"], branch) in existing: continue
            prompt, provenance = build_prompt(tokenizer, row, docs, hidden)
            tasks.append((row["query_id"], branch, prompt, provenance))
    completed_before = len(existing)
    started = time.monotonic()
    for offset in range(0, len(tasks), 32):
        batch = tasks[offset:offset + 32]
        answers, elapsed = generate_batch(model, tokenizer, [item[2] for item in batch])
        for (query_id, branch, _, provenance), answer in zip(batch, answers):
            con.execute("INSERT INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (query_id, branch, answer, sha_text(answer), provenance["prompt_sha256"], provenance["prompt_tokens"], len(tokenizer(answer, add_special_tokens=False).input_ids), json.dumps(provenance["source_ids"]), json.dumps(provenance["source_caps"]), provenance["source_token_count"], provenance["context_sha256"], provenance["hidden_source_id"], elapsed / len(batch), now()))
        con.commit()
        done = min(offset + len(batch), len(tasks))
        if done % 320 < len(batch) or done == len(tasks):
            rate = done / max(time.monotonic() - started, 1e-9)
            checkpoint("ANSWER_PROGRESS", new_completed=done, new_total=len(tasks), database_rows=completed_before + done, eta_seconds=round((len(tasks) - done) / max(rate, 1e-9)))
    columns = [item[0] for item in con.execute("SELECT * FROM answer LIMIT 1").description]
    output = [dict(zip(columns, values)) for values in con.execute("SELECT * FROM answer ORDER BY query_id,branch")]
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)"); con.close()
    out = POST / "runtime" / "IA_ST1_BRANCH_ANSWERS.jsonl"
    write_jsonl(out, output)
    manifest = {"completed_utc": now(), "rows": len(output), "a0": sum(row["branch"] == "A0" for row in output), "a_hide": sum(row["branch"] == "A_HIDE" for row in output), "answers": {"path": str(out), "sha256": sha(out)}, "sqlite": {"path": str(DB), "sha256": sha(DB)}, "generator_unchanged": True}
    atomic_json(POST / "runtime" / "IA_ST1_GENERATION_MANIFEST.json", manifest)
    checkpoint("ANSWER_GENERATION_COMPLETE", rows=len(output), a0=manifest["a0"], a_hide=manifest["a_hide"])
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
