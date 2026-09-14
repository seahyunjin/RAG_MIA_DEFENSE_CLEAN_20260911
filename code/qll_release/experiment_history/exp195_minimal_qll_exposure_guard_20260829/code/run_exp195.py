#!/usr/bin/env python3
"""Exp195: Minimal QLL-anchored leave-one-source exposure guard.

The policy has no learned parameters.  QLL selects exactly one source, the
generated vanilla answer is teacher-forced once without that source, and the
positive per-token NLL increase is charged to a per-session/source ledger.
Budgets are fixed exclusively from the benign TopiOCQA calibration cohort.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp195_minimal_qll_exposure_guard_20260829"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP152 = PROJECT / "exp152_exp150_clean_3k_confirmation_20260825"
EXP153 = PROJECT / "exp153_native_dcmi2_ia15_20260826"
EXP158 = PROJECT / "exp158_global_cap64_native_sessions_20260826"
EXP166 = PROJECT / "exp166_topiocqa_gold_utility_20260827"
EXP179 = PROJECT / "exp179_cross_family_qwen_source_influence_20260828"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
EXP189 = PROJECT / "exp189_simple_global_disclosure_ledger_20260829"
EXP191 = PROJECT / "exp191_two_channel_signal_compression_audit_20260829"
EXP193 = PROJECT / "exp193_top4_context_rebase_20260829"
EXP194 = PROJECT / "exp194_minimal_final_defense_mechanism_novelty_gate_20260829"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")

ABSTENTION = "I can't provide additional source-specific detail for this request."
ALPHAS = (0.005, 0.01, 0.02, 0.05)
CONDITION_BY_ALPHA = {
    0.005: "QLL_GUARD_0P5PCT",
    0.01: "QLL_GUARD_1PCT",
    0.02: "QLL_GUARD_2PCT",
    0.05: "QLL_GUARD_5PCT",
}
PRIMARY = "QLL_GUARD_1PCT"
BASELINES = ("NO_DEFENSE", "ORIGINAL_MIRABEL")
SEED = 19520260829
SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any required output "
                 "format exactly. If the context is insufficient, answer exactly: I don't know.")

NORMAL_CASES = ROOT / "private/EXP195_NORMAL_CASES.private.pkl.gz"
NORMAL_GENERATIONS = ROOT / "private/EXP195_NORMAL_VANILLA.private.csv.gz"
NORMAL_GEN_DB = ROOT / "private/EXP195_NORMAL_GENERATION.sqlite3"
NORMAL_TASKS = ROOT / "private/EXP195_NORMAL_SCORE_TASKS.private.csv.gz"
NORMAL_SCORE_DB = ROOT / "private/EXP195_NORMAL_TOKEN_SCORES.sqlite3"
NORMAL_EXPOSURE = ROOT / "private/EXP195_NORMAL_EXPOSURE.private.csv.gz"
ATTACK_EXPOSURE = ROOT / "private/EXP195_ATTACK_EXPOSURE.private.csv.gz"
POLICY_DETAIL = ROOT / "private/EXP195_POLICY_DETAIL.private.csv.gz"
ATTACK_RESPONSES = ROOT / "private/EXP195_ATTACK_RESPONSES.private.csv.gz"
NORMAL_RESPONSES = ROOT / "private/EXP195_NORMAL_RESPONSES.private.csv.gz"

INPUTS = {
    "exp193_cases": EXP193 / "private/EXP193_CASES.private.pkl.gz",
    "exp193_responses": EXP193 / "private/EXP193_TOP4_RESPONSES.private.csv.gz",
    "exp193_qll": EXP193 / "private/EXP193_QLL_DOMINANT_SOURCE.private.csv.gz",
    "exp193_claims": EXP193 / "private/EXP193_CLAIMS.private.pkl.gz",
    "exp193_tasks": EXP193 / "private/EXP193_C1_TASKS.private.pkl.gz",
    "exp193_scores": EXP193 / "private/EXP193_C1_TOKEN_SCORES.sqlite3",
    "exp193_final": EXP193 / "FINAL_RESULT.json",
    "exp194_final": EXP194 / "FINAL_RESULT.json",
    "exp188_cases": EXP188 / "private/EXP188_CASES.private.pkl.gz",
    "exp179_qll": EXP179 / "private/EXP179_QUERY_SOURCE_QLL.private.csv.gz",
    "exp191_final": EXP191 / "FINAL_RESULT.json",
    "orientation": EXP188 / "configs/ATTACK_SCORE_ORIENTATION_FROZEN.json",
    "normal_gold": EXP166 / "private/TOPIOCQA_GOLD_1000.private.csv.gz",
    "normal_corpus": EXP166 / "private/TOPIOCQA_CORPUS.private.csv.gz",
    "normal_factuality": EXP189 / "private/BENIGN_FACTUALITY_ROWS.private.csv.gz",
    "qwen_config": QWEN / "config.json",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                 default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def atomic_csv(frame, path, compression=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent); os.close(fd)
    try:
        frame.to_csv(temporary, index=False, compression=compression); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def checkpoint(stage, **details):
    payload = {
        "experiment": "Exp195", "stage": stage, "updated_utc": now(), "pid": os.getpid(),
        "new_trainable_parameters": 0, "attack_specific_thresholds": 0,
        "rank_specific_thresholds": 0, "dataset_specific_thresholds": 0,
        "defense_induced_regeneration": 0, "fresh_blind": False, "paid_api_calls": 0,
        **details,
    }
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp195 Status", "", f"- Stage: **{stage}**", f"- Updated UTC: `{payload['updated_utc']}`",
             f"- PID: `{payload['pid']}`"] + [f"- {key}: `{value}`" for key, value in details.items()]
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensure_layout():
    for name in ("audits", "checkpoints", "code", "configs", "logs", "private", "provenance", "reports", "scripts", "tables", "tests"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def preflight():
    ensure_layout()
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing: raise RuntimeError(f"missing frozen inputs: {missing}")
    prior193 = json.loads(INPUTS["exp193_final"].read_text())
    prior194 = json.loads(INPUTS["exp194_final"].read_text())
    prior191 = json.loads(INPUTS["exp191_final"].read_text())
    if prior193.get("final_verdict") != "EXP193_TOP4_REBASE_VALID_REVERSAL_NOT_REPRODUCED" or prior193.get("exact_c1_overall") != 1.0:
        raise RuntimeError("Exp193 lineage mismatch")
    if prior194.get("mechanism_verdict") != "EXP194_MECHANISM_UNRESOLVED" or prior194.get("pair_coverage") != 1.0:
        raise RuntimeError("Exp194 lineage mismatch")
    if prior191.get("selected_instantaneous_candidate") != "I4" or prior191.get("selected_cumulative_candidate") != "C1":
        raise RuntimeError("Exp191 reference mismatch")
    manifest = pd.DataFrame([{"key": key, "path": str(path), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path), "access": "READ_ONLY"}
                             for key, path in INPUTS.items()])
    atomic_csv(manifest, ROOT / "provenance/FROZEN_INPUTS.csv")
    cases = pd.read_pickle(INPUTS["exp193_cases"], compression="gzip")
    qll = pd.read_csv(INPUTS["exp193_qll"], keep_default_na=False, low_memory=False, dtype={"case_id": str})
    responses = pd.read_csv(INPUTS["exp193_responses"], keep_default_na=False, low_memory=False, dtype={"case_id": str})
    if len(cases) != 35680 or cases.session_id.nunique() != 5600:
        raise RuntimeError("Exp193 attack cohort size mismatch")
    if len(qll) != len(cases) or not qll.qll_status.eq("AVAILABLE").all() or len(responses) != len(cases):
        raise RuntimeError("Exp193 exact answer/QLL coverage mismatch")
    if set(cases.case_id.astype(str)) != set(qll.case_id.astype(str)) or set(cases.case_id.astype(str)) != set(responses.case_id.astype(str)):
        raise RuntimeError("Exp193 case identity mismatch")
    precommit = {
        "experiment": "Exp195", "candidate": "QLL_ANCHORED_LEAVE_ONE_SOURCE_EXPOSURE_GUARD",
        "written_before_candidate_attack_results": True,
        "source_rule": "argmax frozen QLL; tie lower retrieval rank then source ID",
        "exposure": "max(0, mean_token_NLL_minus_selected_source - mean_token_NLL_full)",
        "ledger": "one nonnegative scalar per active session/source pair",
        "primary_benign_intervention_rate": 0.01,
        "reporting_rates": list(ALPHAS), "strict_policy": "intervene iff cumulative exposure > B",
        "abstention": ABSTENTION, "trainable_parameters": 0,
        "forbidden": ["attack classifier", "family threshold", "rank threshold", "dataset threshold",
                      "manual fusion", "I4+C1 fusion", "source-use U", "attack-tuned budget",
                      "alternative regeneration", "paraphrase defense", "token noise", "new hiding rule"],
        "decision_gate": {"candidate": PRIMARY, "privacy_direction": "lower standard ROC-AUC is better",
                          "mirabel_noninferiority_margin": 0.0},
        "input_hashes": {row.key: row.sha256 for row in manifest.itertuples(index=False)},
        "created_utc": now(),
    }
    atomic_json(ROOT / "configs/PRECOMMIT.json", precommit)
    atomic_text(ROOT / "configs/PRECOMMIT.sha256", sha256_file(ROOT / "configs/PRECOMMIT.json") + "  PRECOMMIT.json\n")
    checkpoint("PREFLIGHT_COMPLETE", attack_queries=len(cases), attack_sessions=cases.session_id.nunique(),
               exact_qll_coverage=1.0, prior_artifacts_read_only=True)
    return cases, qll, responses


def build_normal_cases():
    if NORMAL_CASES.exists(): return pd.read_pickle(NORMAL_CASES, compression="gzip")
    parent = pd.read_pickle(INPUTS["exp188_cases"], compression="gzip")
    normal = parent[parent.panel.eq("NORMAL_GOLD")].sort_values("case_id").reset_index(drop=True).copy()
    if len(normal) != 1000 or normal.session_id.nunique() != 1000:
        raise RuntimeError("normal calibration cohort must be 1000 independent frozen sessions")
    corpus = pd.read_csv(INPUTS["normal_corpus"], keep_default_na=False, low_memory=False,
                         dtype={"document_id": str})
    documents = {str(row.document_id): f"{row.title}\n{row.text}" for row in corpus.itertuples(index=False)}
    normal["source_texts"] = normal.source_ids.map(lambda ids: [documents[str(source)] for source in ids])
    cross = pd.read_csv(INPUTS["exp179_qll"], keep_default_na=False, low_memory=False,
                        dtype={"case_id": str, "source_id": str})
    cross = cross[cross.cohort.eq("TOPIOCQA_NORMAL")]
    selected = {}
    for case_id, cell in cross.groupby("case_id", sort=False):
        ordered = cell.sort_values(["mean_query_log_probability", "source_rank", "source_id"],
                                   ascending=[False, True, True])
        top, second = ordered.iloc[0], ordered.iloc[1]
        row_id = str(case_id).split("|", 1)[1]
        selected[row_id] = {"selected_source_id": str(top.source_id),
                            "selected_source_rank": int(top.source_rank),
                            "selected_qll": float(top.mean_query_log_probability),
                            "second_qll": float(second.mean_query_log_probability),
                            "qll_margin": float(top.mean_query_log_probability-second.mean_query_log_probability)}
    if set(normal.row_id.astype(str)) != set(selected): raise RuntimeError("normal QLL ID coverage mismatch")
    for column in ("selected_source_id", "selected_source_rank", "selected_qll", "second_qll", "qll_margin"):
        normal[column] = normal.row_id.astype(str).map({key: value[column] for key, value in selected.items()})
    if not all(str(source) in set(map(str, ids)) for source, ids in zip(normal.selected_source_id, normal.source_ids)):
        raise RuntimeError("normal QLL selected source outside Top-4")
    normal.to_pickle(NORMAL_CASES, compression="gzip")
    atomic_csv(normal[["case_id", "row_id", "session_id", "target_rank", "selected_source_id",
                       "selected_source_rank", "qll_margin"]], ROOT / "audits/NORMAL_QLL_SELECTION.csv")
    checkpoint("NORMAL_SUBSTRATE_FROZEN", normal_queries=len(normal), sessions=normal.session_id.nunique(),
               qll_selection_rate=1.0, top4_sources_every_query=bool(normal.source_ids.map(len).eq(4).all()))
    return normal


def generate_normal(normal):
    if NORMAL_GENERATIONS.exists():
        frame = pd.read_csv(NORMAL_GENERATIONS, keep_default_na=False, dtype={"case_id": str})
        if len(frame) == len(normal): return frame
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    models.ROOT = ROOT; models.heartbeat = lambda stage, **details: checkpoint(stage, **details)
    models.GENERATION_CONFIG["batch_size"] = 32
    tasks, metadata = [], []
    for row in normal.itertuples(index=False):
        prompt = models.normal_prompt(str(row.query), list(row.source_texts))
        task = models.make_task(task_type="EXP195_NORMAL_TOP4_VANILLA", row_id=str(row.case_id),
                                prompt=prompt, system_prompt=str(row.system_prompt), max_new_tokens=96)
        tasks.append(task); metadata.append({"case_id": str(row.case_id), "row_id": str(row.row_id),
                                             "prompt_hash": task["prompt_hash"]})
    checkpoint("NORMAL_GENERATION_STARTED", tasks=len(tasks), device="cuda:0", generator="Qwen2.5-3B-Instruct")
    answers = models.run_generation(tasks, NORMAL_GEN_DB, None, "NORMAL_GENERATION_PROGRESS")
    output = pd.DataFrame(metadata); output["response"] = output.case_id.map(answers)
    if output.response.isna().any() or len(output) != len(normal): raise RuntimeError("normal generation incomplete")
    output["response_sha256"] = output.response.map(sha256_text)
    atomic_csv(output, NORMAL_GENERATIONS, "gzip")
    checkpoint("NORMAL_GENERATION_COMPLETE", base_generation_rows=len(output),
               defense_induced_regeneration=0, alternate_answer_generation=0)
    return output


def render_prompt(tokenizer, system_prompt, query, source_ids, source_texts, removed_source=""):
    kept = [(rank, str(source), str(text)) for rank, (source, text) in
            enumerate(zip(source_ids, source_texts), 1) if str(source) != str(removed_source)]
    context = "\n\n".join(f"[Document {rank}]\n{text}" for rank, _, text in kept)
    user = f"Retrieved context:\n{context}\n\nUser query:\n{query}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": str(system_prompt)}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, add_special_tokens=False).input_ids[-3072:]


def build_normal_tasks(normal, generations):
    if NORMAL_TASKS.exists(): return pd.read_csv(NORMAL_TASKS, keep_default_na=False, dtype={"case_id": str})
    response_map = generations.set_index("case_id").response.to_dict(); rows = []
    for row in normal.itertuples(index=False):
        response = str(response_map[str(row.case_id)])
        for condition, removed in (("FULL", ""), ("REMOVE", str(row.selected_source_id))):
            task_id = sha256_text("\0".join([str(row.case_id), condition, removed, response]))
            rows.append({"task_id": task_id, "case_id": str(row.case_id), "condition": condition,
                         "removed_source_id": removed, "selected_source_id": str(row.selected_source_id),
                         "selected_source_rank": int(row.selected_source_rank), "response": response})
    tasks = pd.DataFrame(rows)
    if len(tasks) != 2*len(normal) or tasks.task_id.duplicated().any(): raise RuntimeError("normal task identity failure")
    atomic_csv(tasks, NORMAL_TASKS, "gzip")
    checkpoint("NORMAL_SCORE_TASKS_FROZEN", tasks=len(tasks), queries=len(normal),
               scoring_variants=["FULL", "REMOVE_SELECTED_SOURCE"])
    return tasks


def init_normal_score_db():
    connection = sqlite3.connect(NORMAL_SCORE_DB); connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""CREATE TABLE IF NOT EXISTS score (
      task_id TEXT PRIMARY KEY, token_logp BLOB NOT NULL, answer_tokens INTEGER NOT NULL,
      prompt_tokens INTEGER NOT NULL, wall_seconds REAL NOT NULL, peak_vram_bytes INTEGER NOT NULL,
      completed_utc TEXT NOT NULL)""")
    return connection


def score_normal(normal, tasks):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    connection = init_normal_score_db(); complete = {row[0] for row in connection.execute("SELECT task_id FROM score")}
    pending = tasks[~tasks.task_id.isin(complete)].sort_values("task_id")
    checkpoint("NORMAL_SCORE_STARTED", total=len(tasks), cached=len(complete), pending=len(pending), device="cuda:0")
    if pending.empty: connection.close(); return
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    normal_map = {str(row.case_id): row for row in normal.itertuples(index=False)}
    started = time.monotonic(); initial = len(complete); done = initial
    torch.cuda.reset_peak_memory_stats()
    for task in pending.itertuples(index=False):
        row = normal_map[str(task.case_id)]
        prompt_ids = render_prompt(tokenizer, row.system_prompt, row.query, row.source_ids, row.source_texts,
                                   task.removed_source_id if task.condition == "REMOVE" else "")
        answer_ids = tokenizer(str(task.response), add_special_tokens=False).input_ids
        if not answer_ids: answer_ids = [tokenizer.eos_token_id]
        sequence = prompt_ids + answer_ids; count = len(answer_ids)
        ids = torch.tensor([sequence], dtype=torch.long, device="cuda"); attention = torch.ones_like(ids)
        before = time.monotonic()
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False,
                           logits_to_keep=count+1).logits.float()[0, :-1]
            targets = ids[0, -count:]
            values = torch.log_softmax(logits[-count:], dim=-1).gather(1, targets[:, None]).squeeze(1)
        elapsed = time.monotonic()-before
        connection.execute("INSERT OR REPLACE INTO score VALUES (?,?,?,?,?,?,?)",
                           (str(task.task_id), sqlite3.Binary(zlib.compress(values.cpu().numpy().astype(np.float32).tobytes())),
                            count, len(prompt_ids), elapsed, int(torch.cuda.max_memory_allocated()), now()))
        done += 1
        if done % 50 == 0 or done == len(tasks):
            connection.commit(); rate = max(1, done-initial)/max(time.monotonic()-started, 1e-9)
            checkpoint("NORMAL_SCORE_PROGRESS", total=len(tasks), completed=done, pending=len(tasks)-done,
                       percent=round(100*done/len(tasks), 3), tasks_per_second=round(rate, 3),
                       eta_seconds=round((len(tasks)-done)/max(rate, 1e-9)))
        del ids, attention, logits, values
    connection.commit(); connection.close(); del model; gc.collect(); torch.cuda.empty_cache()


def decode(blob, count):
    values = np.frombuffer(zlib.decompress(blob), dtype=np.float32).copy()
    if len(values) != int(count): raise RuntimeError("token score corruption")
    return values


def aggregate_normal_exposure(normal, tasks):
    if NORMAL_EXPOSURE.exists(): return pd.read_csv(NORMAL_EXPOSURE, keep_default_na=False, dtype={"case_id": str})
    connection = sqlite3.connect(NORMAL_SCORE_DB)
    vectors = {task_id: (decode(blob, count), int(prompt), float(wall), int(vram))
               for task_id, blob, count, prompt, wall, vram in connection.execute(
                   "SELECT task_id,token_logp,answer_tokens,prompt_tokens,wall_seconds,peak_vram_bytes FROM score")}
    connection.close()
    if len(vectors) != len(tasks): raise RuntimeError("normal score cache incomplete")
    rows = []
    for case_id, cell in tasks.groupby("case_id", sort=True):
        full_task = cell[cell.condition.eq("FULL")].iloc[0]; minus_task = cell[cell.condition.eq("REMOVE")].iloc[0]
        full, pf, wf, vf = vectors[str(full_task.task_id)]; minus, pm, wm, vm = vectors[str(minus_task.task_id)]
        if len(full) != len(minus): raise RuntimeError("normal token alignment failure")
        signed = float(np.sum(full-minus)); normalized = float(np.mean(full-minus))
        meta = normal[normal.case_id.astype(str).eq(str(case_id))].iloc[0]
        rows.append({"case_id": str(case_id), "row_id": str(meta.row_id), "session_id": str(meta.session_id),
                     "selected_source_id": str(meta.selected_source_id), "selected_source_rank": int(meta.selected_source_rank),
                     "qll_margin": float(meta.qll_margin), "answer_tokens": len(full),
                     "raw_signed_nll_delta_total": signed, "normalized_signed_nll_delta_per_token": normalized,
                     "D_t": max(0.0, normalized), "full_nll": float(-full.mean()), "minus_nll": float(-minus.mean()),
                     "full_wall_seconds": wf, "remove_wall_seconds": wm,
                     "full_prompt_tokens": pf, "remove_prompt_tokens": pm, "peak_vram_bytes": max(vf, vm)})
    output = pd.DataFrame(rows); atomic_csv(output, NORMAL_EXPOSURE, "gzip")
    checkpoint("NORMAL_EXPOSURE_COMPLETE", rows=len(output), positive_rate=float(output.D_t.gt(0).mean()),
               normalization="per generated-answer token")
    return output


def aggregate_attack_exposure(cases, qll):
    if ATTACK_EXPOSURE.exists(): return pd.read_csv(ATTACK_EXPOSURE, keep_default_na=False, low_memory=False,
                                                    dtype={"case_id": str})
    tasks = pd.read_pickle(INPUTS["exp193_tasks"], compression="gzip")
    claims = pd.read_pickle(INPUTS["exp193_claims"], compression="gzip")
    connection = sqlite3.connect(INPUTS["exp193_scores"])
    vectors = {task_id: decode(blob, count) for task_id, blob, count in
               connection.execute("SELECT task_id,token_logp,claim_tokens FROM score")}
    timing = {task_id: float(wall) for task_id, wall in connection.execute("SELECT task_id,wall_seconds FROM score")}
    connection.close()
    if len(vectors) != len(tasks): raise RuntimeError("Exp193 token cache incomplete")
    task_index = {(str(row.claim_id), str(row.condition), str(row.removed_source_id)): str(row.task_id)
                  for row in tasks.itertuples(index=False)}
    claim_map = {str(key): cell.sort_values("claim_index").claim_id.astype(str).tolist()
                 for key, cell in claims.groupby("score_key", sort=False)}
    # EXP193_CASES preserves the parent top-3 score_key.  Exp193's exact-C1
    # cache is keyed by the newly generated top-4 response, so reconstruct
    # that frozen key exactly as run_exp193.py did instead of joining on the
    # stale parent key.
    responses = pd.read_csv(INPUTS["exp193_responses"], keep_default_na=False, low_memory=False,
                            dtype={"case_id": str})
    response_map = responses.set_index("case_id").response.to_dict()
    frame = cases.copy()
    frame["response"] = frame.case_id.astype(str).map(response_map)
    if frame.response.isna().any():
        raise RuntimeError("Exp193 response coverage incomplete while reconstructing score_key")
    frame["score_key"] = [sha256_text("\0".join([str(row.system_prompt), str(row.prompt_key), str(row.response)]))
                          for row in frame.itertuples(index=False)]
    if not set(frame.score_key.astype(str)).issubset(claim_map):
        missing = sorted(set(frame.score_key.astype(str)) - set(claim_map))
        raise RuntimeError(f"Exp193 reconstructed score_key coverage incomplete: {len(missing)} missing")
    frame = frame.merge(qll[["case_id", "dominant_source_id", "dominant_source_retrieval_rank", "qll_margin"]],
                        on="case_id", validate="one_to_one")
    unique_pairs = frame[["score_key", "dominant_source_id"]].drop_duplicates()
    pair_values = {}
    for row in unique_pairs.itertuples(index=False):
        full_values, minus_values, wall = [], [], 0.0
        for claim_id in claim_map[str(row.score_key)]:
            full_id = task_index[(claim_id, "FULL", "")]
            minus_id = task_index[(claim_id, "REMOVE", str(row.dominant_source_id))]
            full_values.append(vectors[full_id]); minus_values.append(vectors[minus_id]); wall += timing[minus_id]
        full = np.concatenate(full_values); minus = np.concatenate(minus_values)
        signed = float(np.sum(full-minus)); normalized = float(np.mean(full-minus))
        pair_values[(str(row.score_key), str(row.dominant_source_id))] = (
            len(full), signed, normalized, max(0.0, normalized), float(-full.mean()), float(-minus.mean()), wall)
    rows = []
    for row in frame.itertuples(index=False):
        count, signed, normalized, d_t, full_nll, minus_nll, wall = pair_values[(str(row.score_key), str(row.dominant_source_id))]
        rows.append({"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": str(row.panel),
                     "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
                     "session_id": str(row.session_id), "turn_order": int(row.turn_order),
                     "target_rank": int(row.target_rank), "selected_source_id": str(row.dominant_source_id),
                     "selected_source_rank": int(row.dominant_source_retrieval_rank), "qll_margin": float(row.qll_margin),
                     "answer_tokens": count, "raw_signed_nll_delta_total": signed,
                     "normalized_signed_nll_delta_per_token": normalized, "D_t": d_t,
                     "full_nll": full_nll, "minus_nll": minus_nll, "remove_wall_seconds": wall})
    output = pd.DataFrame(rows); atomic_csv(output, ATTACK_EXPOSURE, "gzip")
    selection = output.groupby(["family", "member", "selected_source_rank"], as_index=False).agg(
        queries=("case_id", "size"), sessions=("session_id", "nunique"), mean_D=("D_t", "mean"))
    atomic_csv(selection, ROOT / "tables/TABLE_195_01_QLL_SELECTION.csv")
    rank4 = output[(output.family.eq("BudgetLeak-Z")) & output.member.eq(1) & output.target_rank.eq(4)]
    checkpoint("ATTACK_EXPOSURE_COMPLETE", queries=len(output), sessions=output.session_id.nunique(),
               qll_selection_rate=1.0, budgetleak_rank4_queries=len(rank4),
               budgetleak_rank4_selected_target_rate=float(rank4.selected_source_rank.eq(4).mean()))
    return output


def quantile_budget(values, alpha):
    ordered = np.sort(np.asarray(values, dtype=float))
    if not len(ordered): raise RuntimeError("empty benign calibration")
    index = max(0, min(len(ordered)-1, int(math.ceil((1-alpha)*len(ordered)))-1))
    return float(ordered[index])


def calibrate(normal_exposure):
    path = ROOT / "configs/BENIGN_BUDGETS_FROZEN.json"
    if path.exists(): return {float(k): float(v) for k, v in json.loads(path.read_text())["budgets"].items()}
    budgets = {alpha: quantile_budget(normal_exposure.D_t, alpha) for alpha in ALPHAS}
    rows = []
    for alpha, budget in budgets.items():
        rate = float(normal_exposure.D_t.gt(budget).mean())
        rows.append({"target_benign_intervention_rate": alpha, "B": budget,
                     "empirical_benign_intervention_rate": rate, "strict_operator": ">",
                     "calibration_queries": len(normal_exposure), "attack_examples_used": 0})
    payload = {"experiment": "Exp195", "calibration": "TopiOCQA normal Top-4 only",
               "normal_sessions": int(normal_exposure.session_id.nunique()),
               "normal_queries": len(normal_exposure), "single_turn_sessions": True,
               "budgets": {str(k): v for k, v in budgets.items()}, "attack_examples_used": 0,
               "frozen_utc": now()}
    atomic_json(path, payload); atomic_csv(pd.DataFrame(rows), ROOT / "tables/TABLE_195_02_BENIGN_BUDGETS.csv")
    atomic_text(ROOT / "configs/BENIGN_BUDGETS_FROZEN.sha256", sha256_file(path) + "  BENIGN_BUDGETS_FROZEN.json\n")
    checkpoint("BENIGN_BUDGETS_FROZEN", primary_B=budgets[0.01], primary_empirical_rate=rows[1]["empirical_benign_intervention_rate"],
               attack_examples_used=0)
    return budgets


def replay_policy(attack_exposure, budgets):
    if POLICY_DETAIL.exists(): return pd.read_csv(POLICY_DETAIL, keep_default_na=False, low_memory=False,
                                                  dtype={"case_id": str})
    rows = []
    for alpha, budget in budgets.items():
        condition = CONDITION_BY_ALPHA[alpha]
        for session_id, session in attack_exposure.groupby("session_id", sort=True):
            ledger = defaultdict(float)
            for row in session.sort_values(["turn_order", "case_id"]).itertuples(index=False):
                key = str(row.selected_source_id); before = ledger[key]; after = before + float(row.D_t); ledger[key] = after
                rows.append({"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": str(row.panel),
                             "family": str(row.family), "member": int(row.member), "session_id": str(session_id),
                             "turn_order": int(row.turn_order), "target_rank": int(row.target_rank),
                             "selected_source_id": key, "selected_source_rank": int(row.selected_source_rank),
                             "D_t": float(row.D_t), "ledger_before": before, "ledger_after": after,
                             "B": budget, "alpha": alpha, "condition": condition, "intervened": bool(after > budget)})
    detail = pd.DataFrame(rows); atomic_csv(detail, POLICY_DETAIL, "gzip")
    summary = detail.groupby(["condition", "family"], as_index=False).agg(
        queries=("case_id", "size"), sessions=("session_id", "nunique"), intervention_rate=("intervened", "mean"),
        mean_D=("D_t", "mean"), mean_final_exposure=("ledger_after", "mean"))
    atomic_csv(summary, ROOT / "tables/TABLE_195_03_ATTACK_INTERVENTION.csv")
    checkpoint("POLICY_REPLAY_COMPLETE", rows=len(detail), conditions=detail.condition.nunique(),
               primary_intervention_rate=float(detail[detail.condition.eq(PRIMARY)].intervened.mean()))
    return detail


def construct_responses(cases, exp193_responses, policy):
    if ATTACK_RESPONSES.exists(): return pd.read_csv(ATTACK_RESPONSES, keep_default_na=False, low_memory=False,
                                                     dtype={"case_id": str, "row_id": str})
    vanilla = exp193_responses.set_index("case_id").response.to_dict()
    parent = pd.read_pickle(INPUTS["exp188_cases"], compression="gzip")
    parent = parent[~parent.panel.eq("NORMAL_GOLD")]
    if set(parent.case_id.astype(str)) != set(cases.case_id.astype(str)): raise RuntimeError("Mirabel cohort identity mismatch")
    mirabel = parent.set_index("case_id").mirabel_response.to_dict(); rows = []
    for row in cases.itertuples(index=False):
        common = {"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": str(row.panel),
                  "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
                  "session_id": str(row.session_id), "turn_order": int(row.turn_order),
                  "target_rank": int(row.target_rank), "query": str(row.query)}
        rows.append({**common, "condition": "NO_DEFENSE", "response": str(vanilla[str(row.case_id)]), "intervened": False})
        rows.append({**common, "condition": "ORIGINAL_MIRABEL", "response": str(mirabel[str(row.case_id)]),
                     "intervened": str(mirabel[str(row.case_id)]) != str(vanilla[str(row.case_id)])})
    detail_map = policy.set_index(["case_id", "condition"]).to_dict("index")
    for row in cases.itertuples(index=False):
        common = {"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": str(row.panel),
                  "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
                  "session_id": str(row.session_id), "turn_order": int(row.turn_order),
                  "target_rank": int(row.target_rank), "query": str(row.query)}
        for condition in CONDITION_BY_ALPHA.values():
            item = detail_map[(str(row.case_id), condition)]; intervened = bool(item["intervened"])
            rows.append({**common, "condition": condition,
                         "response": ABSTENTION if intervened else str(vanilla[str(row.case_id)]),
                         "intervened": intervened})
    output = pd.DataFrame(rows); atomic_csv(output, ATTACK_RESPONSES, "gzip")
    checkpoint("ATTACK_RESPONSES_COMPLETE", rows=len(output), conditions=output.condition.nunique(),
               defense_induced_regeneration=0, nonfactual_abstention_template_count=1)
    return output


def construct_normal_responses(normal, generations, exposure, budgets):
    if NORMAL_RESPONSES.exists(): return pd.read_csv(NORMAL_RESPONSES, keep_default_na=False,
                                                     dtype={"case_id": str, "row_id": str})
    vanilla = generations.set_index("case_id").response.to_dict(); mirabel = normal.set_index("case_id").mirabel_response.to_dict()
    exposure_map = exposure.set_index("case_id").D_t.to_dict(); rows = []
    for row in normal.itertuples(index=False):
        common = {"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": "NORMAL_GOLD",
                  "family": "NORMAL_GOLD", "member": -1, "domain": "TopiOCQA",
                  "session_id": str(row.session_id), "turn_order": 1, "target_rank": int(row.target_rank),
                  "query": str(row.query)}
        base = str(vanilla[str(row.case_id)])
        rows.append({**common, "condition": "NO_DEFENSE", "response": base, "intervened": False})
        rows.append({**common, "condition": "ORIGINAL_MIRABEL", "response": str(mirabel[str(row.case_id)]),
                     "intervened": str(mirabel[str(row.case_id)]) != base})
        for alpha, condition in CONDITION_BY_ALPHA.items():
            intervened = bool(float(exposure_map[str(row.case_id)]) > budgets[alpha])
            rows.append({**common, "condition": condition, "response": ABSTENTION if intervened else base,
                         "intervened": intervened})
    output = pd.DataFrame(rows); atomic_csv(output, NORMAL_RESPONSES, "gzip")
    checkpoint("NORMAL_RESPONSES_COMPLETE", rows=len(output), conditions=output.condition.nunique())
    return output


def utility_evaluation(normal_responses, normal_exposure):
    destination = ROOT / "tables/TABLE_195_04_NORMAL_UTILITY.csv"
    if destination.exists(): return pd.read_csv(destination)
    sys.path.insert(0, str(EXP188 / "code")); import exp188_core as core
    gold = pd.read_csv(INPUTS["normal_gold"], keep_default_na=False, dtype={"row_id": str})
    frame = normal_responses.merge(gold[["row_id", "gold_answers"]], on="row_id", validate="many_to_one")
    baseline = frame[frame.condition.eq("NO_DEFENSE")].set_index("row_id").response.to_dict()
    frame["vanilla_response"] = frame.row_id.map(baseline)
    frame["exact_vanilla_preservation"] = frame.response.astype(str).eq(frame.vanilla_response.astype(str))
    frame["token_f1"] = [max(core.token_f1(answer, truth) for truth in json.loads(golds))
                         for answer, golds in zip(frame.response, frame.gold_answers)]
    frame["exact_match"] = [max(core.normalize_answer(answer) == core.normalize_answer(truth)
                                for truth in json.loads(golds)) for answer, golds in zip(frame.response, frame.gold_answers)]
    frame["refusal"] = frame.response.map(core.refusal)
    frame["base_refusal"] = frame.vanilla_response.map(core.refusal)
    frame["new_refusal"] = frame.refusal & ~frame.base_refusal
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MPNET), device="cuda", local_files_only=True); model.max_seq_length = 512
    unique = pd.unique(pd.concat([frame.response, frame.vanilla_response]).astype(str))
    vectors = model.encode(unique.tolist(), normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=256, show_progress_bar=False)
    vector_map = dict(zip(unique, vectors))
    frame["response_preservation"] = [float(np.clip(np.dot(vector_map[str(a)], vector_map[str(b)]), 0, 1))
                                      for a, b in zip(frame.response, frame.vanilla_response)]
    del model; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    summary = frame.groupby("condition", as_index=False).agg(
        normal_queries=("row_id", "size"), intervention_rate=("intervened", "mean"),
        exact_vanilla_preservation=("exact_vanilla_preservation", "mean"), token_f1=("token_f1", "mean"),
        exact_match=("exact_match", "mean"), response_preservation=("response_preservation", "mean"),
        refusal_rate=("refusal", "mean"), new_refusal_rate=("new_refusal", "mean"))
    atomic_csv(frame, ROOT / "private/EXP195_NORMAL_UTILITY_DETAIL.private.csv.gz", "gzip")
    atomic_csv(summary, destination)
    primary = frame[frame.condition.eq(PRIMARY)]
    nontrigger = primary[~primary.intervened]
    identity = float(nontrigger.response.astype(str).eq(nontrigger.vanilla_response.astype(str)).mean())
    atomic_json(ROOT / "audits/NONTRIGGER_IDENTITY.json", {"condition": PRIMARY, "nontriggered": len(nontrigger),
                "exact_identity": identity, "defense_induced_regeneration": 0})
    if identity != 1.0: raise RuntimeError("non-triggered answer identity contract failed")
    checkpoint("NORMAL_UTILITY_COMPLETE", primary_intervention=float(primary.intervened.mean()),
               primary_exact_preservation=float(primary.exact_vanilla_preservation.mean()),
               nontrigger_identity=identity)
    return summary


def factuality_inheritance(normal_responses):
    rows_path = ROOT / "tables/TABLE_195_05_FACTUALITY_INHERITANCE.csv"
    if rows_path.exists(): return pd.read_csv(rows_path)
    old = pd.read_csv(INPUTS["normal_factuality"], keep_default_na=False, dtype={"row_id": str})
    old = old[old.condition.eq("NO_DEFENSE")][["row_id", "factuality_risk", "safe_abstention"]]
    subset = normal_responses[normal_responses.row_id.astype(str).isin(set(old.row_id))].merge(
        old, on="row_id", validate="many_to_one")
    records = []
    for condition, cell in subset.groupby("condition"):
        if condition.startswith("QLL_GUARD"):
            risk = np.where(cell.intervened, False, cell.factuality_risk.astype(bool))
            safe = np.where(cell.intervened, True, cell.safe_abstention.astype(bool))
            induced_unsupported = 0
            status = "STRUCTURAL_INHERITANCE_NO_REGENERATION"
        elif condition == "NO_DEFENSE":
            risk = cell.factuality_risk.astype(bool).to_numpy(); safe = cell.safe_abstention.astype(bool).to_numpy()
            induced_unsupported = 0; status = "FROZEN_NLI_MEASURED"
        else:
            risk = np.full(len(cell), np.nan); safe = np.full(len(cell), np.nan)
            induced_unsupported = np.nan; status = "MIRABEL_SAME_SUBSTRATE_NLI_UNAVAILABLE"
        records.append({"condition": condition, "matched_rows": len(cell),
                        "factuality_risk_rate": float(np.nanmean(risk)) if np.isfinite(np.asarray(risk, float)).any() else np.nan,
                        "safe_abstention_rate": float(np.nanmean(safe)) if np.isfinite(np.asarray(safe, float)).any() else np.nan,
                        "defense_induced_unsupported_claims": induced_unsupported, "status": status})
    output = pd.DataFrame(records); atomic_csv(output, rows_path)
    checkpoint("FACTUALITY_INHERITANCE_COMPLETE", matched_normal_rows=old.row_id.nunique(),
               candidate_regeneration=0, candidate_induced_unsupported_claims=0)
    return output


def _normalize_native_manifest(frame, single_query=False):
    frame = frame.copy()
    if "purpose" not in frame: frame["purpose"] = "PRIVACY_EVAL"
    if "kind" not in frame: frame["kind"] = "attack"
    if "turn" not in frame: frame["turn"] = 1 if single_query else frame["selected_turn"]
    frame["exp87_row_id"] = frame["row_id"].astype(str)
    return frame


def _attack_response_frame(responses, panel):
    return responses[responses.panel.eq(panel)].rename(columns={"row_id": "exp87_row_id"})[[
        "exp87_row_id", "condition", "response"]].assign(
            A_R=lambda x: x.response, action="EXP195_EXPOSURE_GUARD", lola_alarm=False,
            lola_score=float("nan"), beta=float("nan"))


def score_native(responses, evaluator):
    destination = ROOT / "private/EXP195_NATIVE_SCORES.private.csv.gz"
    if destination.exists(): return pd.read_csv(destination, keep_default_na=False, low_memory=False)
    clean = pd.read_csv(EXP152 / "private/EXP152_CLEAN_3K_MANIFEST.private.csv.gz",
                        keep_default_na=False, low_memory=False, dtype={"row_id": str})
    clean = _normalize_native_manifest(clean[clean.family.isin(["RAG-MIA", "S²-MIA", "MBA"])], True)
    native = _normalize_native_manifest(pd.read_csv(EXP153 / "private/EXP153_NATIVE_TURNS.private.csv.gz",
                                       keep_default_na=False, low_memory=False, dtype={"row_id": str}))
    menta = _normalize_native_manifest(pd.read_csv(
        EXP153 / "menta5_followup/private/MENTA5_NATIVE_TURNS.private.csv.gz",
        keep_default_na=False, low_memory=False, dtype={"row_id": str}))
    required = {(str(row.domain), str(row.target_document_id)) for frame in (clean, native, menta)
                for row in frame.itertuples(index=False)}
    documents = evaluator._native_documents(required)
    truth_frame = pd.read_csv(EXP158 / "private/EXP158_IA_GROUND_TRUTH.private.csv.gz",
                              keep_default_na=False, dtype={"row_id": str})
    truth = truth_frame.set_index("row_id").response.to_dict(); pieces = []
    scorer_path = EXP87 / "code/exp87_scoring.py"
    for suite, panel, frame, local_truth in (("Q1", "NATIVE_Q1", clean, {}),
            ("DCMI_IA", "NATIVE_SESSION", native, truth), ("MENTA", "NATIVE_MENTA", menta, {})):
        scorer = load_module(f"exp195_scorer_{suite}", scorer_path)
        scorer.ROOT = ROOT; scorer.RESPONSE_DB = ROOT / f"private/EXP195_{suite}_SCORER.sqlite3"
        scorer.atomic_csv = atomic_csv; scorer.atomic_json = atomic_json
        scorer.checkpoint = lambda stage, suite=suite, **details: checkpoint(f"SCORER_{suite}_{stage}", **details)
        shared = ROOT / "private/EXP87_MIA_SCORES.private.csv.gz"
        if shared.exists(): shared.unlink()
        measured = scorer.mia_scores(frame, _attack_response_frame(responses, panel), local_truth, documents)
        measured["suite"] = suite; pieces.append(measured)
    output = pd.concat(pieces, ignore_index=True); atomic_csv(output, destination, "gzip")
    checkpoint("NATIVE_SCORING_COMPLETE", rows=len(output), invalid=int(output.attack_score.isna().sum()))
    return output


def copy_table(source, destination):
    frame = pd.read_csv(source, keep_default_na=False, low_memory=False); atomic_csv(frame, destination); return frame


def privacy_evaluation(attack_responses, attack_cases):
    sys.path.insert(0, str(EXP188 / "code")); import run_exp188 as evaluator
    evaluator.ROOT = ROOT; evaluator.ORIENTATION = INPUTS["orientation"]
    native_scores = score_native(attack_responses, evaluator)
    evaluator.evaluate_native(native_scores)
    native_auc = copy_table(ROOT / "tables/TABLE_188_02_NATIVE_ROC_AUC.csv", ROOT / "tables/TABLE_195_06_NATIVE_AUC.csv")
    native_low = copy_table(ROOT / "tables/TABLE_188_03_NATIVE_LOW_FPR.csv", ROOT / "tables/TABLE_195_07_NATIVE_LOW_FPR.csv")
    metrics = evaluator.external_response_metrics(attack_responses, attack_cases)
    external_scores = evaluator.external_attack_scores(metrics)
    evaluator.evaluate_external(external_scores)
    rag = copy_table(ROOT / "tables/TABLE_188_05_RAGLEAK_ROC_AUC.csv", ROOT / "tables/TABLE_195_08_RAGLEAK_AUC.csv")
    budget = copy_table(ROOT / "tables/TABLE_188_07_BUDGETLEAK_ROC_AUC.csv", ROOT / "tables/TABLE_195_09_BUDGETLEAK_AUC.csv")
    ranks = copy_table(ROOT / "tables/TABLE_188_08_BUDGETLEAK_BY_RANK.csv", ROOT / "tables/TABLE_195_10_BUDGETLEAK_BY_RANK.csv")
    copy_table(ROOT / "tables/TABLE_188_06_RAGLEAK_LOW_FPR.csv", ROOT / "tables/TABLE_195_11_RAGLEAK_LOW_FPR.csv")
    copy_table(ROOT / "tables/TABLE_188_09_BUDGETLEAK_LOW_FPR_BY_RANK.csv", ROOT / "tables/TABLE_195_12_BUDGETLEAK_LOW_FPR_BY_RANK.csv")
    checkpoint("PRIVACY_EVALUATION_COMPLETE", native_families=native_auc.attack_family.nunique(),
               external_attacks=2, conditions=native_auc.condition.nunique())
    return native_auc, native_low, rag, budget, ranks


def efficiency(normal_exposure):
    remove = normal_exposure.remove_wall_seconds.to_numpy(float)
    table = pd.DataFrame([{
        "condition": PRIMARY, "new_trainable_parameters": 0, "answer_generations_per_query": 1,
        "defense_induced_regenerations_per_query": 0, "additional_teacher_forced_passes_per_query": 1,
        "stored_state": "one float scalar per active session/source pair",
        "mean_additional_latency_seconds": float(remove.mean()), "median_additional_latency_seconds": float(np.median(remove)),
        "p95_additional_latency_seconds": float(np.quantile(remove, .95)),
        "scoring_throughput_queries_per_second": float(1/remove.mean()),
        "peak_vram_bytes": int(normal_exposure.peak_vram_bytes.max()),
        "measurement_note": "microbatch-1 audit; FULL likelihood is available from generation, REMOVE is the one extra pass",
    }])
    atomic_csv(table, ROOT / "tables/TABLE_195_13_EFFICIENCY.csv")
    return table


def reference_table():
    prior = json.loads(INPUTS["exp191_final"].read_text())
    table = pd.DataFrame([{"condition": "EXP191_I4_C1_REFERENCE", "status": "SIGNAL_REFERENCE_ONLY_NOT_A_DEFENSE",
                           "instantaneous_signal": prior["selected_instantaneous_candidate"],
                           "cumulative_signal": prior["selected_cumulative_candidate"],
                           "release_rule": "NONE", "privacy_auc": np.nan,
                           "reason": "Exp191 did not define a user-visible release policy; fabricating responses is prohibited."}])
    atomic_csv(table, ROOT / "tables/TABLE_195_14_I4_C1_REFERENCE.csv")
    return table


def final_decision(native, rag, budget, ranks, utility, factuality, efficiency_table, policy):
    def raw_auc(table, condition, family=None, rank=None):
        cell = table[table.condition.eq(condition)]
        if family is not None: cell = cell[cell.attack_family.eq(family)]
        if rank is not None: cell = cell[cell.target_rank_stratum.eq(rank)]
        if len(cell) != 1: raise RuntimeError(f"missing metric {condition} {family} {rank}: {len(cell)}")
        return float(cell.roc_auc.iloc[0])
    # A defense may reverse a frozen attack score without removing membership
    # information.  Such an AUC below 0.5 is still exploitable by an attacker
    # that flips score direction, so all privacy gates use effective AUC.
    def effective_auc(value):
        return max(float(value), 1.0-float(value))
    cand_rag_raw, mir_rag_raw = raw_auc(rag, PRIMARY, "RAGLeak"), raw_auc(rag, "ORIGINAL_MIRABEL", "RAGLeak")
    cand_budget_raw, mir_budget_raw = raw_auc(budget, PRIMARY, "BudgetLeak-Z"), raw_auc(budget, "ORIGINAL_MIRABEL", "BudgetLeak-Z")
    cand_rank4_raw = raw_auc(ranks, PRIMARY, "BudgetLeak-Z", "rank4")
    mir_rank4_raw = raw_auc(ranks, "ORIGINAL_MIRABEL", "BudgetLeak-Z", "rank4")
    no_budget_raw = raw_auc(budget, "NO_DEFENSE", "BudgetLeak-Z")
    cand_rag, mir_rag = effective_auc(cand_rag_raw), effective_auc(mir_rag_raw)
    cand_budget, mir_budget = effective_auc(cand_budget_raw), effective_auc(mir_budget_raw)
    cand_rank4, mir_rank4 = effective_auc(cand_rank4_raw), effective_auc(mir_rank4_raw)
    no_budget = effective_auc(no_budget_raw)
    u = utility.set_index("condition"); cand_u, mir_u = u.loc[PRIMARY], u.loc["ORIGINAL_MIRABEL"]
    nontrigger_identity = json.loads((ROOT / "audits/NONTRIGGER_IDENTITY.json").read_text())["exact_identity"]
    primary_policy = policy[policy.condition.eq(PRIMARY)].copy()
    budget_policy = primary_policy[primary_policy.family.eq("BudgetLeak-Z")]
    first = budget_policy.groupby("session_id").first().intervened.mean()
    terminal = budget_policy.groupby("session_id").last().intervened.mean()
    checks = {
        "budgetleak_cumulative_protection": bool(cand_budget <= no_budget and terminal > first),
        "budgetleak_rank4_at_least_mirabel": bool(cand_rank4 <= mir_rank4 + 1e-12),
        "ragleak_at_least_mirabel": bool(cand_rag <= mir_rag + 1e-12),
        "benign_utility_above_mirabel": bool(cand_u.exact_vanilla_preservation > mir_u.exact_vanilla_preservation and
                                              cand_u.token_f1 >= mir_u.token_f1),
        "nontrigger_identity_100pct": bool(nontrigger_identity == 1.0),
        "defense_induced_regeneration_zero": True,
        "attack_or_rank_specific_tuning_zero": True,
    }
    if all(checks.values()): verdict = "MINIMAL_EXPOSURE_GUARD_SUPPORTED"
    elif checks["budgetleak_cumulative_protection"] and not checks["ragleak_at_least_mirabel"]:
        verdict = "CUMULATIVE_ONLY_PARTIAL"
    elif not checks["budgetleak_cumulative_protection"] and not checks["ragleak_at_least_mirabel"]:
        verdict = "MINIMAL_EXPOSURE_GUARD_FAILED"
    else:
        verdict = "MINIMAL_EXPOSURE_GUARD_FAILED"
    native_primary = native[native.condition.eq(PRIMARY)].copy()
    native_mirabel = native[native.condition.eq("ORIGINAL_MIRABEL")].copy()
    native_primary["effective_auc"] = native_primary.roc_auc.map(effective_auc)
    native_mirabel["effective_auc"] = native_mirabel.roc_auc.map(effective_auc)
    comparison = pd.DataFrame([
        {"metric": "RAGLeak effective ROC-AUC", "ours": cand_rag, "mirabel": mir_rag, "lower_is_better": True},
        {"metric": "BudgetLeak effective ROC-AUC", "ours": cand_budget, "mirabel": mir_budget, "lower_is_better": True},
        {"metric": "BudgetLeak rank4 effective ROC-AUC", "ours": cand_rank4, "mirabel": mir_rank4, "lower_is_better": True},
        {"metric": "normal exact vanilla preservation", "ours": cand_u.exact_vanilla_preservation,
         "mirabel": mir_u.exact_vanilla_preservation, "lower_is_better": False},
        {"metric": "normal gold token F1", "ours": cand_u.token_f1, "mirabel": mir_u.token_f1, "lower_is_better": False},
    ])
    atomic_csv(comparison, ROOT / "tables/TABLE_195_15_MIRABEL_COMPARISON.csv")
    gate = pd.DataFrame([{"gate": key, "passed": value} for key, value in checks.items()])
    atomic_csv(gate, ROOT / "tables/TABLE_195_16_DECISION_GATES.csv")
    result = {
        "experiment": "Exp195", "verdict": verdict, "primary_condition": PRIMARY,
        "primary_B": float(json.loads((ROOT / "configs/BENIGN_BUDGETS_FROZEN.json").read_text())["budgets"]["0.01"]),
        "benign_intervention_rate": float(cand_u.intervention_rate),
        "benign_exact_preservation": float(cand_u.exact_vanilla_preservation),
        "nontrigger_identity": float(nontrigger_identity), "defense_induced_regeneration": 0,
        "ragleak_raw_auc": cand_rag_raw, "mirabel_ragleak_raw_auc": mir_rag_raw,
        "ragleak_effective_auc": cand_rag, "mirabel_ragleak_effective_auc": mir_rag,
        "budgetleak_raw_auc": cand_budget_raw, "mirabel_budgetleak_raw_auc": mir_budget_raw,
        "budgetleak_effective_auc": cand_budget, "mirabel_budgetleak_effective_auc": mir_budget,
        "budgetleak_rank4_raw_auc": cand_rank4_raw, "mirabel_budgetleak_rank4_raw_auc": mir_rank4_raw,
        "budgetleak_rank4_effective_auc": cand_rank4, "mirabel_budgetleak_rank4_effective_auc": mir_rank4,
        "native_family_mean_effective_auc": float(native_primary.effective_auc.mean()),
        "native_worst_effective_auc": float(native_primary.effective_auc.max()),
        "native_mirabel_mean_effective_auc": float(native_mirabel.effective_auc.mean()),
        "normal_token_f1": float(cand_u.token_f1), "mirabel_normal_token_f1": float(mir_u.token_f1),
        "new_trainable_parameters": 0, "additional_teacher_forced_passes": 1,
        "stored_state": "one scalar per active session/source", "checks": checks,
        "exp196_needed": verdict == "CUMULATIVE_ONLY_PARTIAL",
        "limitations": ["benign calibration sessions are frozen single-query TopiOCQA sessions",
                        "Mirabel responses are frozen method-native answers, not regenerated on Exp193 prompts",
                        "RAGLeak/BudgetLeak-Z are disclosed external adaptations, not fresh blind"],
        "completed_utc": now(),
    }
    atomic_json(ROOT / "FINAL_RESULT.json", result)
    factual_primary = factuality[factuality.condition.eq(PRIMARY)].iloc[0]
    eff = efficiency_table.iloc[0]
    rank_primary = ranks[ranks.condition.eq(PRIMARY)][["target_rank_stratum", "roc_auc"]].copy()
    rank_primary["effective_auc"] = rank_primary.roc_auc.map(effective_auc)
    native_view = native_primary[["attack_family", "roc_auc", "effective_auc"]]
    report = f"""# Exp195 — Minimal QLL-Anchored Exposure Guard

- 최종 판정: **{verdict}**
- primary budget B: `{result['primary_B']:.8f}` (정상 1,000개, 목표 개입률 1%, 공격 사용 0)
- 실제 정상 개입률: `{result['benign_intervention_rate']:.4f}`
- 정상 vanilla 답변 exact 보존율: `{result['benign_exact_preservation']:.4f}`
- non-triggered answer identity: `{result['nontrigger_identity']:.4f}`
- defense-induced regeneration: `0`
- 새 학습 parameter: `0`

## 핵심 privacy

| 지표 | QLL Guard | Original Mirabel |
|---|---:|---:|
| RAGLeak E-AUC | {cand_rag:.4f} | {mir_rag:.4f} |
| BudgetLeak E-AUC | {cand_budget:.4f} | {mir_budget:.4f} |
| BudgetLeak rank-4 E-AUC | {cand_rank4:.4f} | {mir_rank4:.4f} |

### BudgetLeak rank별

{rank_primary.to_markdown(index=False, floatfmt='.4f')}

### 다른 공격 family

{native_view.to_markdown(index=False, floatfmt='.4f')}

## Utility·환각

- 정상 gold token F1: ours `{float(cand_u.token_f1):.4f}`, Mirabel `{float(mir_u.token_f1):.4f}`
- 정상 response preservation: ours `{float(cand_u.response_preservation):.4f}`, Mirabel `{float(mir_u.response_preservation):.4f}`
- 동결 NLI 500행 inheritance factuality risk: `{float(factual_primary.factuality_risk_rate):.4f}`
- defense-induced unsupported factual claim: `0` (대체 답변을 생성하지 않고 고정 비사실 abstention만 사용)

## Efficiency

- 추가 teacher-forced pass: `1/query`
- 평균 추가 scoring latency: `{float(eff.mean_additional_latency_seconds):.4f}s`
- scoring throughput: `{float(eff.scoring_throughput_queries_per_second):.2f} query/s`
- peak VRAM: `{int(eff.peak_vram_bytes)}` bytes
- 저장 state: source/session마다 scalar 1개

## Gate

{gate.to_markdown(index=False)}

## 해석 경계

Exp191 I4+C1은 user-visible release rule이 없는 signal reference라 AUC를 만들지 않았다. 정상 calibration은 공격을 전혀 보지 않았지만 1,000개의 single-query TopiOCQA session이다. Mirabel은 동결된 method-native 응답을 사용했다. 외부 두 공격은 fresh blind가 아니다.

Exp196 instantaneous spike gate 필요 여부: **{result['exp196_needed']}**
"""
    atomic_text(ROOT / "reports/REPORT_195_FINAL_KO.md", report)
    checkpoint("COMPLETE", verdict=verdict, exp196_needed=result["exp196_needed"])
    return result


def smoke():
    cases, qll, responses = preflight(); normal = build_normal_cases()
    sample = cases.head(8).merge(qll[["case_id", "dominant_source_id"]], on="case_id", validate="one_to_one")
    checks = {
        "attack_sample_queries": len(sample), "normal_sample_queries": min(8, len(normal)),
        "attack_selected_source_in_top4": bool(all(str(s) in set(map(str, ids)) for s, ids in zip(sample.dominant_source_id, sample.source_ids))),
        "normal_selected_source_in_top4": bool(all(str(s) in set(map(str, ids)) for s, ids in zip(normal.head(8).selected_source_id, normal.head(8).source_ids))),
        "response_coverage": len(responses) == len(cases), "abstention_is_fixed_nonfactual_template": ABSTENTION.endswith("request."),
    }
    checks["pass"] = all(v for k, v in checks.items() if k.endswith("top4") or k in {"response_coverage", "abstention_is_fixed_nonfactual_template"})
    atomic_json(ROOT / "tests/SMOKE_RESULTS.json", checks)
    checkpoint("SMOKE_COMPLETE", **checks)
    return checks


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), ensure_ascii=False, indent=2)); return
    cases, qll, exp193_responses = preflight()
    if args.preflight_only: return
    normal = build_normal_cases(); normal_generations = generate_normal(normal)
    normal_tasks = build_normal_tasks(normal, normal_generations); score_normal(normal, normal_tasks)
    normal_exposure = aggregate_normal_exposure(normal, normal_tasks)
    attack_exposure = aggregate_attack_exposure(cases, qll)
    budgets = calibrate(normal_exposure); policy = replay_policy(attack_exposure, budgets)
    attack_responses = construct_responses(cases, exp193_responses, policy)
    normal_responses = construct_normal_responses(normal, normal_generations, normal_exposure, budgets)
    utility = utility_evaluation(normal_responses, normal_exposure)
    factuality = factuality_inheritance(normal_responses); efficiency_table = efficiency(normal_exposure); reference_table()
    native, native_low, rag, budget, ranks = privacy_evaluation(attack_responses, cases)
    result = final_decision(native, rag, budget, ranks, utility, factuality, efficiency_table, policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
