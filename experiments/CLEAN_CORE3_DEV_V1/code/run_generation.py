#!/usr/bin/env python3
"""Generate CLEAN_CORE3 answers under the three precommitted conditions.

The SQLite store makes the expensive GPU stage resumable.  A defended answer is
newly generated only when its frozen detector alarms; otherwise it is an exact
reuse of the freshly generated no-defense answer from this campaign.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
CAMPAIGN = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
INPUTS = CAMPAIGN / "inputs"
CACHE = CAMPAIGN / "cache"
CONFIGS = CAMPAIGN / "configs"
RUNTIME = CAMPAIGN / "runtime"
CHECKPOINTS = CAMPAIGN / "checkpoints"
GENERATOR = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")

SYSTEM_PROMPT = (
    "Answer the user's question using only the retrieved context. Follow any "
    "required output format exactly. If the context is insufficient, answer "
    "exactly: I don't know."
)
SOURCE_TOKEN_BUDGET = 2048
MAX_PROMPT_TOKENS = 3072
MAX_NEW_TOKENS = 160
SEED = 20260911
BATCH_SIZE = 16


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"campaign": "CLEAN_CORE3_DEV_V1", "stage": stage, "updated_utc": now(),
               "pid": os.getpid(), **details}
    write_json(CAMPAIGN / "HEARTBEAT.json", payload)
    write_json(CHECKPOINTS / f"{stage}.json", payload)
    lines = ["# CLEAN_CORE3_DEV_V1 STATUS", "", f"- Current stage: `{stage}`",
             f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{os.getpid()}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    (CAMPAIGN / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def waterfill(lengths: list[int], total: int = SOURCE_TOKEN_BUDGET) -> list[int]:
    values = np.asarray(lengths, dtype=int)
    caps = np.zeros(len(values), dtype=int)
    remaining = int(total)
    active = [index for index, length in enumerate(values) if length > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, int(values[index] - caps[index]), remaining)
            if add > 0:
                caps[index] += add
                remaining -= add
                changed = True
            if caps[index] >= values[index]:
                active.remove(index)
            if remaining <= 0:
                break
        if not changed:
            break
    return caps.tolist()


def build_prompt(tokenizer, query: str, source_ids: list[str], documents: dict[str, str],
                 hidden_source: str | None) -> tuple[str, dict]:
    kept_ids = [source for source in source_ids if source != hidden_source]
    encoded = [tokenizer(documents[source], add_special_tokens=False).input_ids for source in kept_ids]
    caps = waterfill([len(value) for value in encoded])
    visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip()
               for value, cap in zip(encoded, caps)]
    blocks = [f"[Document {rank}]\n{text}" for rank, text in enumerate(visible, 1)]
    context = "\n\n".join(blocks)
    user = f"Retrieved context:\n{context}\n\nUser query:\n{query}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if len(prompt_ids) > MAX_PROMPT_TOKENS:
        prompt_ids = prompt_ids[-MAX_PROMPT_TOKENS:]
        rendered = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    provenance = {
        "source_ids": kept_ids,
        "source_caps": caps,
        "source_token_count": int(sum(caps)),
        "context_sha256": sha_text(context),
        "prompt_sha256": sha_text(json.dumps(prompt_ids, separators=(",", ":"))),
        "prompt_tokens": len(prompt_ids),
        "hidden_source_id": hidden_source,
    }
    return rendered, provenance


def open_database() -> sqlite3.Connection:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(RUNTIME / "generation.sqlite3", timeout=120)
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=FULL")
    database.execute("""CREATE TABLE IF NOT EXISTS answer (
        query_id TEXT NOT NULL,
        condition TEXT NOT NULL,
        answer TEXT NOT NULL,
        answer_sha256 TEXT NOT NULL,
        prompt_sha256 TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        answer_tokens INTEGER NOT NULL,
        source_ids_json TEXT NOT NULL,
        source_caps_json TEXT NOT NULL,
        source_token_count INTEGER NOT NULL,
        context_sha256 TEXT NOT NULL,
        hidden_source_id TEXT,
        detector_alarm INTEGER NOT NULL,
        generated INTEGER NOT NULL,
        reused_from TEXT,
        wall_seconds REAL NOT NULL,
        completed_utc TEXT NOT NULL,
        PRIMARY KEY(query_id, condition)
    )""")
    database.commit()
    return database


def main() -> None:
    precommit_path = CONFIGS / "CLEAN_CORE3_DEV_V1_PRECOMMIT.json"
    expected = (CONFIGS / "CLEAN_CORE3_DEV_V1_PRECOMMIT.sha256").read_text(encoding="utf-8").split()[0]
    if sha_file(precommit_path) != expected:
        raise RuntimeError("precommit checksum mismatch")
    retrieval_path = CACHE / "CLEAN_CORE3_RETRIEVAL_CACHE.jsonl"
    retrieval_manifest = json.loads((CACHE / "CLEAN_CORE3_RETRIEVAL_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    if sha_file(retrieval_path) != retrieval_manifest["cache_sha256"]:
        raise RuntimeError("retrieval cache checksum mismatch")

    all_rows = read_jsonl(retrieval_path)
    # Calibration and non-intervened holdout queries need no generation.
    # End-to-end privacy uses every attack query; the specified false-positive
    # damage audit uses every locked-holdout query alarmed by primary BC q97.
    rows = [row for row in all_rows if row["cohort"] == "ATTACK" or
            (row["split"] == "HOLDOUT" and row["bc_q97_alarm"])]
    documents = {row["document_id"]: row["source_text"]
                 for row in read_jsonl(INPUTS / "CLEAN_CORE3_PROTECTED_DB.jsonl")}
    if len(all_rows) != 1280 or len(rows) != 299 or len(documents) != 3000:
        raise RuntimeError(f"unexpected substrate size rows={len(rows)} documents={len(documents)}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checkpoint("GENERATOR_LOADING", rows=len(rows), generator=str(GENERATOR))
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    database = open_database()
    completed = {(row[0], row[1]) for row in database.execute("SELECT query_id,condition FROM answer")}

    conditions = [
        ("NO_DEFENSE", None),
        ("ORIGINAL_MIRABEL_TOP1_HIDE", "original_alarm"),
        ("BC_MIRABEL_Q97_TOP1_HIDE", "bc_q97_alarm"),
    ]
    # Generate no-defense first.  Defended non-alarm rows can then safely reuse it.
    total_condition_rows = len(rows) * len(conditions)
    start_time = time.monotonic()
    newly_generated = 0

    for condition, alarm_field in conditions:
        tasks: list[tuple[dict, str, dict, bool]] = []
        reused = 0
        for row in sorted(rows, key=lambda value: value["query_id"]):
            key = (row["query_id"], condition)
            if key in completed:
                continue
            alarm = True if alarm_field is None else bool(row[alarm_field])
            # BC q97 alarms are a strict subset of Original MIRABEL alarms and
            # invoke the identical top-1-hide prompt.  Reuse that already-new
            # deterministic answer rather than executing the same forward pass
            # twice.  Prompt/context hashes remain available for verification.
            if condition == "BC_MIRABEL_Q97_TOP1_HIDE" and alarm:
                source = database.execute(
                    "SELECT answer,answer_sha256,prompt_sha256,prompt_tokens,answer_tokens,"
                    "source_ids_json,source_caps_json,source_token_count,context_sha256,"
                    "hidden_source_id,wall_seconds FROM answer WHERE query_id=? AND "
                    "condition='ORIGINAL_MIRABEL_TOP1_HIDE'", (row["query_id"],),
                ).fetchone()
                if source is None or source[9] != row["selected_source_id"]:
                    raise RuntimeError(f"identical Original hide answer unavailable: {row['query_id']}")
                database.execute(
                    "INSERT INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["query_id"], condition, source[0], source[1], source[2], source[3], source[4],
                     source[5], source[6], source[7], source[8], source[9], 1, 0,
                     "ORIGINAL_MIRABEL_TOP1_HIDE_IDENTICAL_PROMPT", 0.0, now()),
                )
                reused += 1
                continue
            if condition != "NO_DEFENSE" and not alarm:
                source = database.execute(
                    "SELECT answer,answer_sha256,prompt_sha256,prompt_tokens,answer_tokens,"
                    "source_ids_json,source_caps_json,source_token_count,context_sha256,wall_seconds "
                    "FROM answer WHERE query_id=? AND condition='NO_DEFENSE'", (row["query_id"],),
                ).fetchone()
                if source is None:
                    raise RuntimeError(f"no-defense answer missing before reuse: {row['query_id']}")
                database.execute(
                    "INSERT INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["query_id"], condition, source[0], source[1], source[2], source[3], source[4],
                     source[5], source[6], source[7], source[8], None, 0, 0, "NO_DEFENSE", 0.0, now()),
                )
                reused += 1
                continue
            hidden = row["selected_source_id"] if condition != "NO_DEFENSE" else None
            rendered, provenance = build_prompt(
                tokenizer, row["query"], list(row["top_document_ids"]), documents, hidden,
            )
            tasks.append((row, rendered, provenance, alarm))
        database.commit()
        checkpoint("GENERATION_CONDITION_STARTED", condition=condition, pending_generation=len(tasks),
                   reused_without_generation=reused, already_complete=sum(1 for x in completed if x[1] == condition))

        phase_started = time.monotonic()
        for offset in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[offset:offset + BATCH_SIZE]
            rendered = [item[1] for item in batch]
            encoded = tokenizer(rendered, add_special_tokens=False, padding=True, truncation=True,
                                max_length=MAX_PROMPT_TOKENS, return_tensors="pt").to(model.device)
            before = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                    use_cache=True, pad_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - before
            generated = output[:, encoded.input_ids.shape[1]:]
            answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for (row, _, provenance, alarm), answer, answer_ids in zip(batch, answers, generated):
                clean = answer.strip()
                answer_tokens = len(tokenizer(clean, add_special_tokens=False).input_ids)
                database.execute(
                    "INSERT OR REPLACE INTO answer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["query_id"], condition, clean, sha_text(clean), provenance["prompt_sha256"],
                     provenance["prompt_tokens"], answer_tokens, json.dumps(provenance["source_ids"]),
                     json.dumps(provenance["source_caps"]), provenance["source_token_count"],
                     provenance["context_sha256"], provenance["hidden_source_id"], int(alarm), 1,
                     None, elapsed / len(batch), now()),
                )
            database.commit()
            newly_generated += len(batch)
            done = min(offset + len(batch), len(tasks))
            rate = done / max(time.monotonic() - phase_started, 1e-9)
            checkpoint("GENERATION_PROGRESS", condition=condition, condition_generated=done,
                       condition_total=len(tasks), percent=round(100 * done / max(len(tasks), 1), 2),
                       generations_per_second=round(rate, 3),
                       eta_seconds=round((len(tasks) - done) / max(rate, 1e-9)),
                       database_rows=database.execute("SELECT COUNT(*) FROM answer").fetchone()[0])
            del encoded, output, generated
        completed = {(row[0], row[1]) for row in database.execute("SELECT query_id,condition FROM answer")}

    count = database.execute("SELECT COUNT(*) FROM answer").fetchone()[0]
    if count != total_condition_rows:
        raise RuntimeError(f"generation incomplete: {count}/{total_condition_rows}")

    output_path = RUNTIME / "CLEAN_CORE3_GENERATED_ANSWERS.jsonl"
    columns = [description[0] for description in database.execute("SELECT * FROM answer LIMIT 1").description]
    with output_path.open("w", encoding="utf-8") as handle:
        for values in database.execute("SELECT * FROM answer ORDER BY query_id,condition"):
            handle.write(json.dumps(dict(zip(columns, values)), ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "created_utc": now(), "condition_rows": count, "query_rows": len(rows),
        "retrieval_rows_not_requiring_generation": len(all_rows) - len(rows),
        "new_generations": database.execute("SELECT COUNT(*) FROM answer WHERE generated=1").fetchone()[0],
        "exact_reuses": database.execute("SELECT COUNT(*) FROM answer WHERE generated=0").fetchone()[0],
        "logical_defense_regenerations": database.execute(
            "SELECT COUNT(*) FROM answer WHERE condition!='NO_DEFENSE' AND detector_alarm=1"
        ).fetchone()[0],
        "condition_counts": {condition: database.execute("SELECT COUNT(*) FROM answer WHERE condition=?", (condition,)).fetchone()[0]
                             for condition, _ in conditions},
        "output_sha256": sha_file(output_path), "precommit_sha256": expected,
        "runtime_seconds_this_invocation": time.monotonic() - start_time,
    }
    write_json(RUNTIME / "CLEAN_CORE3_GENERATION_MANIFEST.json", manifest)
    checkpoint("GENERATION_COMPLETE", **manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
