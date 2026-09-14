#!/usr/bin/env python3
"""Exp193: top-4 context rebase for the frozen Exp191 signal audit.

This is an evaluation-substrate reconstruction, not a defense experiment.
It regenerates the complete frozen Exp189 attack cohort with the fourth
retrieved source present in the actual Qwen input, computes exact leave-one-
source claim dependence for all four sources, and only then re-evaluates the
frozen Exp191 signals I4 and C1.  Every expensive stage is resumable.
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
import sqlite3
import sys
import tempfile
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp193_top4_context_rebase_20260829"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
EXP189 = PROJECT / "exp189_simple_global_disclosure_ledger_20260829"
EXP190 = PROJECT / "exp190_leakage_channel_decomposition_audit_20260829"
EXP191 = PROJECT / "exp191_two_channel_signal_compression_audit_20260829"
EXP192 = PROJECT / "exp192_qll_anchored_cumulative_20260829"
EXP179 = PROJECT / "exp179_cross_family_qwen_source_influence_20260828"
EXP181 = PROJECT / "exp181_native_session_continuous_qll_20260828"
EXP160 = PROJECT / "exp160_global_cap64_ragleak_budgetleak_20260826"
EXP160E = PROJECT / "exp160e_global_cap64_external_attacks_exploratory_20260826"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")

CASES = ROOT / "private/EXP193_CASES.private.pkl.gz"
BUNDLES = ROOT / "private/EXP193_TOP4_SOURCE_BUNDLES.private.pkl.gz"
PROMPTS = ROOT / "private/EXP193_TOP4_PROMPTS.private.pkl.gz"
RESPONSES = ROOT / "private/EXP193_TOP4_RESPONSES.private.csv.gz"
CLAIMS = ROOT / "private/EXP193_CLAIMS.private.pkl.gz"
TASKS = ROOT / "private/EXP193_C1_TASKS.private.pkl.gz"
SCORE_DB = ROOT / "private/EXP193_C1_TOKEN_SCORES.sqlite3"
COSTS = ROOT / "private/EXP193_EXACT_C1.private.pkl.gz"
LEDGER = ROOT / "private/EXP193_FROZEN_LEDGER.private.csv.gz"
FEATURES = ROOT / "private/EXP193_SESSION_FEATURES.private.csv.gz"
QLL_DETAIL = ROOT / "private/EXP193_QLL_DOMINANT_SOURCE.private.csv.gz"
GEN_DB = ROOT / "private/EXP193_TOP4_RESPONSES.sqlite3"

FAMILIES = ["RAG-MIA", "S²-MIA", "MBA", "RAGLeak", "DCMI", "MEntA", "BudgetLeak-Z", "IA"]
PRIMARY_I = ["RAG-MIA", "S²-MIA", "MBA", "RAGLeak"]
PRIMARY_C = ["DCMI", "MEntA", "BudgetLeak-Z"]
REPLICATES = 20_000
SEED = 19320260829
FAMILY_MAX = {"RAG-MIA": 12, "DCMI": 12, "IA": 32, "S²-MIA": 96,
              "MEntA": 96, "MBA": 160, "RAGLeak": 128}

INPUTS = {
    "exp188_cases": EXP188 / "private/EXP188_CASES.private.pkl.gz",
    "exp188_code": EXP188 / "code/run_exp188.py",
    "exp189_cases": EXP189 / "private/ATTACK_CASES.private.pkl.gz",
    "exp189_costs": EXP189 / "private/ATTACK_CLAIM_COSTS.private.pkl.gz",
    "exp189_ledger": EXP189 / "private/ATTACK_LEDGER.private.csv.gz",
    "exp189_beta": EXP189 / "configs/BETA_FROZEN.json",
    "exp189_generation_code": EXP189 / "code/run_attack_scoring.py",
    "exp190_features": EXP190 / "private/EXP190_SESSION_FEATURES.private.csv.gz",
    "exp191_final": EXP191 / "FINAL_RESULT.json",
    "exp191_precommit": EXP191 / "configs/PRECOMMIT.json",
    "exp191_rank": EXP191 / "tables/TABLE_191_11_BUDGETLEAK_BY_RANK.csv",
    "exp192_final": EXP192 / "FINAL_RESULT.json",
    "exp192_alignment": EXP192 / "tables/TABLE_192_01_INPUT_ALIGNMENT.csv",
    "exp179_qll": EXP179 / "private/EXP179_QUERY_SOURCE_QLL.private.csv.gz",
    "exp181_qll": EXP181 / "private/EXP181_SOURCE_QLL.private.csv.gz",
    "qwen_config": QWEN / "config.json",
    "qwen_generation_code": EXP87 / "code/exp87_models.py",
    "ledger_core": EXP189 / "code/gdcel_core.py",
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


def stable_seed(*parts):
    return (SEED + int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)) % (2**32 - 1)


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
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "Exp193", "stage": stage, "updated_utc": now(), "pid": os.getpid(),
               "new_defense": False, "threshold_search": False, "alpha_search": False,
               "qll_refinement": False, "llama": False, "fresh_blind": False,
               "e_mia_open_count": 0, "selective_session_analysis": False, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp193 Status", "", f"- Stage: **{stage}**", f"- Updated UTC: `{payload['updated_utc']}`",
             f"- PID: `{payload['pid']}`"] + [f"- {key}: `{value}`" for key, value in details.items()]
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def frozen_tree_manifest():
    rows = []
    for root in (EXP189, EXP191, EXP192):
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rows.append({"experiment": root.name, "relative_path": str(path.relative_to(root)),
                         "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def preflight():
    for directory in ["audits", "checkpoints", "configs", "logs", "private", "provenance", "reports", "tables", "tests", "validation"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing: raise RuntimeError(f"missing frozen inputs: {missing}")
    prior = json.loads(INPUTS["exp191_final"].read_text())
    if prior.get("selected_instantaneous_candidate") != "I4" or prior.get("selected_cumulative_candidate") != "C1":
        raise RuntimeError("Exp191 selected-signal lineage mismatch")
    blocked = json.loads(INPUTS["exp192_final"].read_text())
    if blocked.get("final_verdict") != "EXP192_INPUT_INSUFFICIENT" or int(blocked.get("rank4_exact_per_source_c1", -1)) != 0:
        raise RuntimeError("Exp192 input-block lineage mismatch")

    before_path = ROOT / "provenance/FROZEN_TREE_BEFORE.csv"
    if not before_path.exists(): atomic_csv(frozen_tree_manifest(), before_path)
    rows = [{"key": key, "path": str(path), "sha256": sha256_file(path),
             "bytes": path.stat().st_size, "access": "READ_ONLY"} for key, path in INPUTS.items()]
    atomic_csv(pd.DataFrame(rows), ROOT / "provenance/FROZEN_INPUTS.csv")

    old = pd.read_pickle(INPUTS["exp189_cases"], compression="gzip").sort_values("case_id").reset_index(drop=True)
    parent = pd.read_pickle(INPUTS["exp188_cases"], compression="gzip")
    parent = parent[~parent.panel.eq("NORMAL_GOLD")].sort_values("case_id").reset_index(drop=True)
    identity = ["case_id", "row_id", "panel", "family", "member", "domain", "session_id",
                "turn_order", "query", "target_document_id", "target_rank", "system_prompt"]
    if len(old) != 35680 or old.session_id.nunique() != 5600 or len(parent) != len(old):
        raise RuntimeError("frozen cohort size mismatch")
    for column in identity:
        if not old[column].astype(str).equals(parent[column].astype(str)):
            raise RuntimeError(f"Exp188/189 cohort identity mismatch: {column}")
    if not old.source_ids.map(len).eq(3).all() or not parent.source_ids.map(len).eq(4).all():
        raise RuntimeError("top3/top4 source-count lineage mismatch")
    if not all(list(a) == list(b)[:3] for a, b in zip(old.source_ids, parent.source_ids)):
        raise RuntimeError("Exp189 sources are not exact Exp188 top-3 prefixes")
    if not parent.source_ids.map(lambda x: len(set(map(str, x))) == 4).all():
        raise RuntimeError("top-4 source identity is not unique")
    evidence = pd.DataFrame([
        {"fact": "Exp189 actual claim-scoring generation context source count", "value": 3,
         "evidence": "run_attack_scoring.py: source_ids = list(row.source_ids)[:3]", "verified": True},
        {"fact": "Exp189 cohort cases", "value": len(old), "evidence": str(INPUTS["exp189_cases"]), "verified": True},
        {"fact": "Exp189 sessions", "value": old.session_id.nunique(), "evidence": "exact session_id", "verified": True},
        {"fact": "Exp188 exact fourth source available for every case", "value": len(parent),
         "evidence": str(INPUTS["exp188_cases"]), "verified": True},
    ])
    atomic_csv(evidence, ROOT / "audits/TOP3_LINEAGE_EVIDENCE.csv")
    cohort = old.groupby(["family", "member"], as_index=False).agg(rows=("case_id", "size"), sessions=("session_id", "nunique"))
    atomic_csv(cohort, ROOT / "audits/COHORT_IDENTITY.csv")
    label_rows = []
    for column in ["member", "family", "target_document_id", "target_rank", "session_id", "turn_order", "row_id"]:
        label_rows.append({"field": column, "source": str(INPUTS["exp189_cases"]),
                           "parent_crosscheck": str(INPUTS["exp188_cases"]), "rows": len(old),
                           "exact_agreement": float(old[column].astype(str).eq(parent[column].astype(str)).mean()),
                           "modified_for_exp193": False})
    atomic_csv(pd.DataFrame(label_rows), ROOT / "provenance/LABEL_AND_SESSION_PROVENANCE.csv")
    cohort_hash = sha256_text("\n".join(old.case_id.astype(str)) + "\n")
    parent_hash = sha256_text("\n".join(parent.case_id.astype(str)) + "\n")
    if cohort_hash != parent_hash: raise RuntimeError("ordered case ID hash mismatch")
    precommit = {
        "experiment": "Exp193", "purpose": "TOP4_CONTEXT_REBASE_ONLY",
        "cohort_cases": len(old), "cohort_sessions": old.session_id.nunique(),
        "ordered_case_id_sha256": cohort_hash, "frozen_signals": {"I": "I4=qll_margin", "C": "C1=final_cumulative_l1"},
        "only_experimental_change": "actual generation context includes retrieved source ranks 1-4 instead of 1-3",
        "generator": "Qwen2.5-3B-Instruct", "generator_revision": QWEN.name,
        "decoding": {"temperature": 0.0, "do_sample": False, "num_beams": 1,
                     "max_input_tokens": 3072, "dtype": "bfloat16", "seed": 42},
        "required_exact_c1_alignment": 1.0, "bootstrap_replicates": REPLICATES,
        "membership_permutations": REPLICATES,
        "forbidden": ["new defense", "threshold search", "alpha search", "QLL refinement optimization",
                      "Llama", "fresh blind", "E-MIA", "selective-session analysis", "official downstream claim"],
        "inputs": {r["key"]: {"path": r["path"], "sha256": r["sha256"]} for r in rows},
        "created_utc": now(),
    }
    atomic_json(ROOT / "configs/PRECOMMIT.json", precommit)
    atomic_text(ROOT / "configs/PRECOMMIT.sha256", sha256_file(ROOT / "configs/PRECOMMIT.json") + "  PRECOMMIT.json\n")
    checkpoint("PREFLIGHT_COMPLETE", cases=len(old), sessions=old.session_id.nunique(),
               top3_lineage_verified=True, cohort_hash=cohort_hash)
    return old, parent


def load_documents(parent):
    native = parent[parent.panel.isin(["NATIVE_Q1", "NATIVE_SESSION", "NATIVE_MENTA"])]
    required_native = {(str(row.domain), str(source)) for row in native.itertuples(index=False) for source in row.source_ids}
    sys.path.insert(0, str(EXP87 / "code"))
    import exp87_scoring
    documents = exp87_scoring.document_lookup(required_native)
    if required_native - set(documents): raise RuntimeError("native top4 document lookup incomplete")
    external = parent[parent.panel.eq("EXTERNAL")]
    required_external = {(str(row.domain), str(source)) for row in external.itertuples(index=False) for source in row.source_ids}
    retrieval = pd.read_csv(EXP160E / "private/EXP160_RETRIEVAL.private.csv.gz",
                            keep_default_na=False, low_memory=False, dtype={"row_id": str})
    module = load_module("exp193_external_docs", EXP160 / "code/run_exp160.py")
    ext = module.all_documents(retrieval)
    documents.update({(str(domain), str(source)): str(text) for (domain, source), text in ext.items()
                      if (str(domain), str(source)) in required_external})
    if required_external - set(documents): raise RuntimeError("external top4 document lookup incomplete")
    return documents


def build_substrate(old, parent):
    if CASES.exists() and BUNDLES.exists() and PROMPTS.exists():
        return (pd.read_pickle(CASES, compression="gzip"), pd.read_pickle(BUNDLES, compression="gzip"),
                pd.read_pickle(PROMPTS, compression="gzip"))
    documents = load_documents(parent)
    bundle_rows, bundle_seen, case_rows = [], set(), []
    parent_map = parent.set_index("case_id")
    for row in old.itertuples(index=False):
        source_ids = list(map(str, parent_map.loc[str(row.case_id), "source_ids"]))
        bundle_key = sha256_text("\0".join([str(row.domain), *source_ids]))
        if bundle_key not in bundle_seen:
            texts = [str(documents[(str(row.domain), source)]) for source in source_ids]
            bundle_rows.append({"bundle_key": bundle_key, "domain": str(row.domain), "source_ids": source_ids,
                                "source_texts": texts, "source_sha256": [sha256_text(x) for x in texts]})
            bundle_seen.add(bundle_key)
        maximum = int(row.turn_order) if str(row.family) == "BudgetLeak-Z" else int(FAMILY_MAX[str(row.family)])
        prompt_key = sha256_text("\0".join([str(row.system_prompt), str(row.query), bundle_key]))
        case_rows.append({**row._asdict(), "source_ids_top3": list(row.source_ids), "source_ids": source_ids,
                          "bundle_key": bundle_key, "prompt_key": prompt_key,
                          "max_new_tokens": maximum, "old_top3_score_key": str(row.score_key)})
    cases = pd.DataFrame(case_rows)
    bundles = pd.DataFrame(bundle_rows)
    bundle_map = bundles.set_index("bundle_key").source_texts.to_dict()
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    prompt_rows = []
    for prompt_key, cell in cases.groupby("prompt_key", sort=True):
        row = cell.iloc[0]; texts = bundle_map[str(row.bundle_key)]
        user_prompt = models.normal_prompt(str(row.query), texts)
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": str(row.system_prompt)}, {"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        actual_ids = token_ids[-3072:]
        context = "\n\n".join(f"[Document {index}]\n{text}" for index, text in enumerate(texts, 1))
        prompt_rows.append({"prompt_key": prompt_key, "bundle_key": str(row.bundle_key),
                            "system_prompt": str(row.system_prompt), "query": str(row.query),
                            "complete_context": context, "user_prompt": user_prompt,
                            "rendered_prompt": rendered,
                            "actual_generator_prompt": tokenizer.decode(actual_ids, skip_special_tokens=False),
                            "rendered_prompt_sha256": sha256_text(rendered),
                            "actual_generator_prompt_ids_sha256": sha256_text(json.dumps(actual_ids, separators=(",", ":"))),
                            "original_prompt_tokens": len(token_ids), "actual_prompt_tokens": len(actual_ids),
                            "left_truncated_tokens": max(0, len(token_ids)-len(actual_ids))})
    prompts = pd.DataFrame(prompt_rows)
    cases.to_pickle(CASES, compression="gzip"); bundles.to_pickle(BUNDLES, compression="gzip"); prompts.to_pickle(PROMPTS, compression="gzip")
    source_audit = pd.DataFrame([{"case_id": r.case_id, "family": r.family, "member": r.member,
                                  "session_id": r.session_id, "target_rank": r.target_rank,
                                  **{f"source_rank_{i}_id": r.source_ids[i-1] for i in range(1,5)}}
                                 for r in cases.itertuples(index=False)])
    atomic_csv(source_audit, ROOT / "audits/TOP4_SOURCE_ORDER.csv.gz", "gzip")
    checkpoint("TOP4_SUBSTRATE_FROZEN", cases=len(cases), sessions=cases.session_id.nunique(),
               source_bundles=len(bundles), unique_prompts=len(prompts), top4_every_case=bool(cases.source_ids.map(len).eq(4).all()))
    return cases, bundles, prompts


def audit_substrate(cases, bundles, prompts):
    rows = []
    for i in range(1, 5):
        rows.append({"source_rank": i, "cases_with_source_id": int(cases.source_ids.map(lambda x: len(x) >= i).sum()),
                     "unique_prompts_with_label_in_complete_context": int(prompts.complete_context.str.contains(
                         f"[Document {i}]", regex=False).sum()),
                     "unique_prompts_with_label_in_actual_3072_token_input": int(prompts.actual_generator_prompt.str.contains(
                         f"[Document {i}]", regex=False).sum()), "unique_prompts": len(prompts)})
    frame = pd.DataFrame(rows); atomic_csv(frame, ROOT / "audits/ACTUAL_GENERATOR_INPUT_TOP4.csv")
    if not cases.source_ids.map(len).eq(4).all() or not frame.unique_prompts_with_label_in_complete_context.eq(len(prompts)).all():
        raise RuntimeError("complete top-4 generation context contract failed")
    atomic_json(ROOT / "audits/GENERATOR_INPUT_AUDIT.json", {
        "status": "PASS_COMPLETE_CONTEXT", "cases": len(cases), "unique_prompts": len(prompts),
        "complete_context_contains_rank1_4": True,
        "actual_3072_token_input_all_rank_labels": bool(frame.unique_prompts_with_label_in_actual_3072_token_input.eq(len(prompts)).all()),
        "left_truncation_is_frozen_generator_behavior": True,
        "prompts_left_truncated": int((prompts.left_truncated_tokens > 0).sum()),
        "note": "The full user context contains ranks 1-4 for every case. Frozen left truncation may remove an early label for exceptionally long queries; both strings are stored exactly."
    })
    return frame


def generate(cases, bundles, prompts):
    if RESPONSES.exists():
        frame = pd.read_csv(RESPONSES, keep_default_na=False, low_memory=False, dtype={"case_id": str})
        if len(frame) == len(cases) and frame.case_id.nunique() == len(cases): return frame
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    models.ROOT = ROOT
    models.heartbeat = lambda stage, **details: checkpoint(stage, **details)
    models.GENERATION_CONFIG["batch_size"] = 16
    pmap = prompts.set_index("prompt_key").user_prompt.to_dict()
    tasks, metadata = [], []
    for row in cases.sort_values("case_id").itertuples(index=False):
        tasks.append(models.make_task(task_type="EXP193_TOP4_REBASE", row_id=str(row.case_id),
                                      prompt=str(pmap[row.prompt_key]), system_prompt=str(row.system_prompt),
                                      max_new_tokens=int(row.max_new_tokens)))
        metadata.append({"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": str(row.panel),
                         "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
                         "session_id": str(row.session_id), "turn_order": int(row.turn_order),
                         "target_document_id": str(row.target_document_id), "target_rank": int(row.target_rank),
                         "bundle_key": str(row.bundle_key), "prompt_key": str(row.prompt_key),
                         "max_new_tokens": int(row.max_new_tokens), "generation_config": json.dumps(
                             {**models.GENERATION_CONFIG, "max_new_tokens": int(row.max_new_tokens)}, sort_keys=True)})
    checkpoint("TOP4_GENERATION_STARTED", tasks=len(tasks), unique_prompts=len(prompts), device="cuda:0",
               generator="Qwen2.5-3B-Instruct", paid_api_calls=0)
    answers = models.run_generation(tasks, GEN_DB, None, "TOP4_GENERATION_PROGRESS")
    output = pd.DataFrame(metadata); output["response"] = output.case_id.map(answers)
    if output.response.isna().any() or output.case_id.nunique() != len(cases): raise RuntimeError("top4 generation incomplete")
    output["response_sha256"] = output.response.map(sha256_text)
    atomic_csv(output, RESPONSES, "gzip")
    checkpoint("TOP4_GENERATION_COMPLETE", new_answer_rows=len(output), unique_responses=output.response.nunique(),
               generation_store_sha256=sha256_file(GEN_DB))
    return output


def build_claim_tasks(cases, bundles, prompts, responses):
    if CLAIMS.exists() and TASKS.exists():
        return pd.read_pickle(CLAIMS, compression="gzip"), pd.read_pickle(TASKS, compression="gzip")
    sys.path.insert(0, str(EXP189 / "code")); import gdcel_core as core
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    response_map = responses.set_index("case_id").response.to_dict()
    pmap = prompts.set_index("prompt_key")
    unique = cases.copy(); unique["response"] = unique.case_id.map(response_map)
    unique["score_key"] = [sha256_text("\0".join([str(r.system_prompt), str(r.prompt_key), str(r.response)]))
                           for r in unique.itertuples(index=False)]
    unique = unique.sort_values("case_id").drop_duplicates("score_key")
    claims, tasks = [], []
    for row in unique.itertuples(index=False):
        response = str(row.response); source_ids = list(map(str, row.source_ids))
        for claim_index, (start, end, claim) in enumerate(core.claim_spans(response), 1):
            claim_id = sha256_text(f"EXP193\0{row.score_key}\0{start}\0{end}")
            count = len(tokenizer(claim, add_special_tokens=False).input_ids)
            claims.append({"score_key": row.score_key, "claim_id": claim_id, "claim_index": claim_index,
                           "claim_start": start, "claim_end": end, "claim_text": claim,
                           "claim_key": core.exact_claim_key(claim, tokenizer), "claim_tokens": count})
            variants = [("FULL", "", 0), *[("REMOVE", source, index) for index, source in enumerate(source_ids, 1)]]
            for condition, removed, source_index in variants:
                tasks.append({"task_id": sha256_text(f"{claim_id}\0{condition}\0{removed}"),
                              "score_key": row.score_key, "claim_id": claim_id, "claim_index": claim_index,
                              "claim_start": start, "claim_end": end, "condition": condition,
                              "removed_source_id": str(removed), "source_index": int(source_index), "claim_tokens": count,
                              "prompt_key": str(row.prompt_key), "bundle_key": str(row.bundle_key)})
    claims, tasks = pd.DataFrame(claims), pd.DataFrame(tasks)
    if claims.claim_id.duplicated().any() or tasks.task_id.duplicated().any() or len(tasks) != 5*len(claims):
        raise RuntimeError("Exp193 claim task identity mismatch")
    claims.to_pickle(CLAIMS, compression="gzip"); tasks.to_pickle(TASKS, compression="gzip")
    checkpoint("C1_TASKS_FROZEN", unique_answers=len(unique), claims=len(claims), tasks=len(tasks), sources_per_claim=4)
    return claims, tasks


def init_score_db():
    connection = sqlite3.connect(SCORE_DB); connection.execute("PRAGMA journal_mode=WAL"); connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""CREATE TABLE IF NOT EXISTS score (
      task_id TEXT PRIMARY KEY, token_logp BLOB NOT NULL, claim_tokens INTEGER NOT NULL,
      wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL)""")
    return connection


def score_c1(cases, bundles, prompts, responses, claims, tasks):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    connection = init_score_db(); complete = {row[0] for row in connection.execute("SELECT task_id FROM score")}
    pending = tasks[~tasks.task_id.isin(complete)].sort_values("task_id")
    checkpoint("C1_SCORING_STARTED", total=len(tasks), cached=len(complete), pending=len(pending),
               deterministic_microbatch=1, device="cuda:0")
    if pending.empty: connection.close(); return
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    response_map = responses.set_index("case_id").response.to_dict()
    case_frame = cases.copy(); case_frame["response"] = case_frame.case_id.map(response_map)
    case_frame["score_key"] = [sha256_text("\0".join([str(r.system_prompt), str(r.prompt_key), str(r.response)]))
                               for r in case_frame.itertuples(index=False)]
    rows = {str(r.score_key): r for r in case_frame.sort_values("case_id").drop_duplicates("score_key").itertuples(index=False)}
    bundle_map = bundles.set_index("bundle_key").source_texts.to_dict()
    started = time.monotonic(); initial = len(complete); done = initial
    for task in pending.itertuples(index=False):
        row = rows[str(task.score_key)]; sources = list(map(str, row.source_ids)); texts = list(bundle_map[str(row.bundle_key)])
        if task.condition == "FULL": kept_ids, kept_texts = sources, texts
        else:
            kept = [(source, text) for source, text in zip(sources, texts) if source != str(task.removed_source_id)]
            kept_ids, kept_texts = [x[0] for x in kept], [x[1] for x in kept]
        ranks = [sources.index(source)+1 for source in kept_ids]
        context = "\n\n".join(f"[Document {rank}]\n{text}" for rank, text in zip(ranks, kept_texts))
        user = f"Retrieved context:\n{context}\n\nUser query:\n{row.query}"
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": str(row.system_prompt)}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids[-3072:]
        response = str(row.response); prior = response[:int(task.claim_start)]; claim = response[int(task.claim_start):int(task.claim_end)]
        prior_ids = tokenizer(prior, add_special_tokens=False).input_ids
        claim_ids = tokenizer(claim, add_special_tokens=False).input_ids
        if len(claim_ids) != int(task.claim_tokens): raise RuntimeError("claim tokenization drift")
        sequence = prompt_ids + prior_ids + claim_ids; count = len(claim_ids)
        ids = torch.tensor([sequence], dtype=torch.long, device="cuda"); attention = torch.ones_like(ids)
        before = time.monotonic()
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False,
                           logits_to_keep=count+1).logits.float()[0, :-1]
            targets = ids[0, -count:]
            values = torch.log_softmax(logits[-count:], dim=-1).gather(1, targets[:, None]).squeeze(1)
        connection.execute("INSERT OR REPLACE INTO score VALUES (?,?,?,?,?)",
                           (task.task_id, sqlite3.Binary(zlib.compress(values.cpu().numpy().astype(np.float32).tobytes())),
                            count, time.monotonic()-before, now()))
        done += 1
        if done % 100 == 0 or done == len(tasks):
            connection.commit(); rate = max(1, done-initial)/max(time.monotonic()-started, 1e-9)
            checkpoint("C1_SCORING_PROGRESS", total=len(tasks), completed=done, pending=len(tasks)-done,
                       percent=round(100*done/len(tasks), 3), tasks_per_second=round(rate, 3),
                       eta_seconds=round((len(tasks)-done)/max(rate, 1e-9)))
        del ids, attention, logits, values
    connection.commit(); connection.close(); del model; gc.collect(); torch.cuda.empty_cache()


def decode(blob, count):
    values = np.frombuffer(zlib.decompress(blob), dtype=np.float32).copy()
    if len(values) != int(count): raise RuntimeError("token cache corruption")
    return values


def aggregate_c1(claims, tasks):
    if COSTS.exists(): return pd.read_pickle(COSTS, compression="gzip")
    sys.path.insert(0, str(EXP189 / "code")); import gdcel_core as core
    connection = sqlite3.connect(SCORE_DB)
    vectors = {task_id: decode(blob, count) for task_id, blob, count in
               connection.execute("SELECT task_id,token_logp,claim_tokens FROM score")}; connection.close()
    if len(vectors) != len(tasks): raise RuntimeError(f"incomplete C1 cache: {len(vectors)}/{len(tasks)}")
    meta = claims.set_index("claim_id"); rows = []
    for claim_id, cell in tasks.groupby("claim_id", sort=False):
        full = vectors[str(cell[cell.condition.eq("FULL")].iloc[0].task_id)]
        # Source order is explicit retrieval order, never lexical source ID.
        removed_cell = cell[cell.condition.eq("REMOVE")].sort_values("source_index")
        removed = [vectors[str(task_id)] for task_id in removed_cell.task_id]
        c1, cinf, per_source = core.token_dependence(full, removed)
        source_ids = removed_cell.removed_source_id.astype(str).tolist()
        for item, source_id in zip(per_source, source_ids): item["source_id"] = source_id
        m = meta.loc[claim_id]
        rows.append({"score_key": m.score_key, "claim_id": claim_id, "claim_index": int(m.claim_index),
                     "claim_text": m.claim_text, "claim_key": m.claim_key, "claim_tokens": int(m.claim_tokens),
                     "c1": c1, "cinf": cinf, "per_source": per_source})
    costs = pd.DataFrame(rows); costs.to_pickle(COSTS, compression="gzip")
    checkpoint("EXACT_C1_COMPLETE", claims=len(costs), source_claim_pairs=4*len(costs), scoring_tasks=len(tasks))
    return costs


def qll_dominant(cases):
    if QLL_DETAIL.exists(): return pd.read_csv(QLL_DETAIL, keep_default_na=False, low_memory=False, dtype={"case_id": str})
    cross = pd.read_csv(INPUTS["exp179_qll"], keep_default_na=False, low_memory=False,
                        dtype={"source_id": str, "target_document_id": str})
    native = pd.read_csv(INPUTS["exp181_qll"], keep_default_na=False, low_memory=False,
                         dtype={"row_id": str, "source_id": str, "target_document_id": str})
    cross["query_sha256"] = cross.query_sha256.astype(str); native["query_sha256"] = native.query_sha256.astype(str)
    cross_groups = {key: cell for key, cell in cross.groupby(
        ["attack_family", "query_sha256", "member", "target_document_id"], sort=False)}
    native_groups = {str(key): cell for key, cell in native.groupby("row_id", sort=False)}
    rows = []
    for row in cases.itertuples(index=False):
        qhash = sha256_text(row.query)
        if row.family in {"DCMI", "MEntA", "IA"} and int(row.turn_order) > 1:
            cell = native_groups.get(str(row.row_id)); artifact = "EXP181_EXACT_ROW_ID"
        else:
            cell = cross_groups.get((str(row.family), qhash, int(row.member), str(row.target_document_id)))
            artifact = "EXP179_EXACT_QUERY_MEMBER_TARGET"
        if cell is None or len(cell) != 4 or cell.source_id.astype(str).nunique() != 4:
            rows.append({"case_id": row.case_id, "qll_status": "UNAVAILABLE_OR_NONUNIQUE", "qll_artifact": artifact}); continue
        ordered = cell.sort_values(["mean_query_log_probability", "source_rank", "source_id"], ascending=[False, True, True])
        top, second = ordered.iloc[0], ordered.iloc[1]
        rows.append({"case_id": row.case_id, "qll_status": "AVAILABLE", "qll_artifact": artifact,
                     "dominant_source_id": str(top.source_id), "dominant_source_retrieval_rank": int(top.source_rank),
                     "dominant_source_qll": float(top.mean_query_log_probability),
                     "second_source_qll": float(second.mean_query_log_probability),
                     "qll_margin": float(top.mean_query_log_probability-second.mean_query_log_probability),
                     "qll_source_count": 4})
    frame = pd.DataFrame(rows); atomic_csv(frame, QLL_DETAIL, "gzip")
    checkpoint("QLL_DOMINANT_SOURCE_MAPPED", cases=len(frame), available=int(frame.qll_status.eq("AVAILABLE").sum()))
    return frame


def apply_frozen_ledger(cases, responses, costs):
    if LEDGER.exists(): return pd.read_csv(LEDGER, keep_default_na=False, low_memory=False)
    sys.path.insert(0, str(EXP189 / "code")); import gdcel_core as core
    frozen = json.loads(INPUTS["exp189_beta"].read_text()); beta, m = float(frozen["beta"]), float(frozen["m"])
    response_map = responses.set_index("case_id").response.to_dict()
    frame = cases.copy(); frame["response"] = frame.case_id.map(response_map)
    frame["score_key"] = [sha256_text("\0".join([str(r.system_prompt), str(r.prompt_key), str(r.response)]))
                          for r in frame.itertuples(index=False)]
    cost_map = {str(key): [{"claim_index": int(r.claim_index), "claim_text": str(r.claim_text),
                            "claim_key": str(r.claim_key), "c1": float(r.c1), "cinf": float(r.cinf)}
                           for r in cell.sort_values("claim_index").itertuples(index=False)]
                for key, cell in costs.groupby("score_key", sort=False)}
    audit = []
    for session_id, session in frame.groupby("session_id", sort=True):
        ordered = session.sort_values(["turn_order", "case_id"])
        turns = [{"case_id": r.case_id, "claims": cost_map[str(r.score_key)]} for r in ordered.itertuples(index=False)]
        _, rows = core.apply_global_ledger(turns, beta=beta, median_claim_tokens=m)
        audit += [{"session_id": session_id, **item} for item in rows]
    ledger = pd.DataFrame(audit); atomic_csv(ledger, LEDGER, "gzip")
    checkpoint("FROZEN_LEDGER_REPLAY_COMPLETE", rows=len(ledger), sessions=frame.session_id.nunique(), beta=beta, m=m)
    return ledger


def build_features(cases, responses, costs, ledger, qll):
    if FEATURES.exists(): return pd.read_csv(FEATURES, keep_default_na=False, low_memory=False)
    response_map = responses.set_index("case_id").response.to_dict()
    frame = cases.copy(); frame["response"] = frame.case_id.map(response_map)
    frame["score_key"] = [sha256_text("\0".join([str(r.system_prompt), str(r.prompt_key), str(r.response)]))
                          for r in frame.itertuples(index=False)]
    key_map = frame[["case_id", "score_key", "turn_order", "target_rank"]]
    details = ledger.merge(key_map, on="case_id", validate="many_to_one").merge(
        costs[["score_key", "claim_index", "per_source"]], on=["score_key", "claim_index"], validate="many_to_one")
    rows = []
    qmap = qll.set_index("case_id")
    for (session_id, family), cell in frame.sort_values(["turn_order", "case_id"]).groupby(["session_id", "family"], sort=True):
        first_turn = cell.turn_order.min(); first_case = sorted(cell[cell.turn_order.eq(first_turn)].case_id.astype(str))[0]
        ld = details[details.session_id.astype(str).eq(str(session_id))].sort_values(["turn_order", "case_id", "claim_index"])
        first = ld[ld.case_id.astype(str).eq(first_case)]
        final_l1 = float(ld.cumulative_l1.iloc[-1]) if len(ld) else 0.0
        q = qmap.loc[first_case]
        row = cell.iloc[0]
        rows.append({"session_id": str(session_id), "family": str(family), "member": int(row.member),
                     "domain": str(row.domain), "row_id": str(row.row_id), "first_case_id": first_case,
                     "target_rank": int(row.target_rank), "qll_margin": float(q.qll_margin),
                     "qll_dominant_source_id": str(q.dominant_source_id),
                     "qll_dominant_source_rank": int(q.dominant_source_retrieval_rank),
                     "first_max_c1": float(first.c1.max()) if len(first) else 0.0,
                     "final_cumulative_l1": final_l1, "query_or_budget_count": int(cell.case_id.nunique())})
    features = pd.DataFrame(rows); atomic_csv(features, FEATURES, "gzip")
    checkpoint("SESSION_FEATURES_COMPLETE", sessions=len(features), families=features.family.nunique())
    return features


def coverage_gate(cases, costs, qll):
    cost_map = {str(key): cell for key, cell in costs.groupby("score_key", sort=False)}
    responses = pd.read_csv(RESPONSES, keep_default_na=False, low_memory=False, dtype={"case_id": str})
    response_map = responses.set_index("case_id").response.to_dict()
    frame = cases.copy(); frame["response"] = frame.case_id.map(response_map)
    frame["score_key"] = [sha256_text("\0".join([str(r.system_prompt), str(r.prompt_key), str(r.response)]))
                          for r in frame.itertuples(index=False)]
    detail = frame.merge(qll, on="case_id", how="left", validate="one_to_one")
    exact = []
    for r in detail.itertuples(index=False):
        ok = r.qll_status == "AVAILABLE" and str(r.dominant_source_id) in set(map(str, r.source_ids))
        cell = cost_map.get(str(r.score_key))
        if ok and cell is not None:
            for p in cell.per_source:
                if not isinstance(p, list) or str(r.dominant_source_id) not in {str(x.get("source_id")) for x in p}:
                    ok = False; break
        else: ok = False
        exact.append(bool(ok))
    detail["exact_c1_for_qll_dominant"] = exact
    rows = []
    def add(scope, cell, family="ALL", member="ALL", rank="ALL"):
        session_complete = cell.groupby("session_id").exact_c1_for_qll_dominant.all()
        rows.append({"scope": scope, "family": family, "member": member, "target_rank": rank,
                     "queries": len(cell), "sessions": cell.session_id.nunique(),
                     "exact": int(cell.exact_c1_for_qll_dominant.sum()),
                     "coverage": float(cell.exact_c1_for_qll_dominant.mean()) if len(cell) else np.nan,
                     "complete_sessions": int(session_complete.sum()),
                     "session_coverage": float(session_complete.mean()) if len(session_complete) else np.nan})
    add("OVERALL", detail)
    for member, cell in detail.groupby("member"): add("MEMBER", cell, member=int(member))
    for family, cell in detail.groupby("family"): add("FAMILY", cell, family=family)
    for rank, cell in detail.groupby("target_rank"): add("TARGET_RANK", cell, rank=int(rank))
    coverage = pd.DataFrame(rows); atomic_csv(coverage, ROOT / "tables/TABLE_193_01_EXACT_C1_COVERAGE.csv")
    atomic_csv(detail[["case_id", "family", "member", "session_id", "target_rank", "dominant_source_id",
                       "dominant_source_retrieval_rank", "exact_c1_for_qll_dominant"]],
               ROOT / "audits/EXACT_C1_ALIGNMENT_DETAIL.csv.gz", "gzip")
    passed = bool(detail.exact_c1_for_qll_dominant.all())
    checkpoint("EXACT_C1_COVERAGE_GATE", passed=passed, exact=int(sum(exact)), total=len(exact), coverage=float(np.mean(exact)))
    return detail, coverage, passed


def hedges_g(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2: return np.nan
    pooled = math.sqrt(((len(pos)-1)*np.var(pos,ddof=1)+(len(neg)-1)*np.var(neg,ddof=1))/max(len(pos)+len(neg)-2,1))
    if pooled == 0: return 0.0 if np.mean(pos)==np.mean(neg) else math.copysign(math.inf,np.mean(pos)-np.mean(neg))
    return float(((np.mean(pos)-np.mean(neg))/pooled)*(1-3/max(4*(len(pos)+len(neg))-9,1)))


def bootstrap_difference(pos, neg, seed):
    pos, neg = np.asarray(pos,float), np.asarray(neg,float); rng=np.random.default_rng(seed); out=np.empty(REPLICATES)
    batch=250
    for start in range(0,REPLICATES,batch):
        size=min(batch,REPLICATES-start)
        out[start:start+size]=pos[rng.integers(0,len(pos),(size,len(pos)))].mean(1)-neg[rng.integers(0,len(neg),(size,len(neg)))].mean(1)
    return float(np.quantile(out,.025)),float(np.quantile(out,.975)),float(np.mean(out>0)),float(np.mean(out<0))


def permutation_difference(pos, neg, seed):
    pos,neg=np.asarray(pos,float),np.asarray(neg,float); values=np.r_[pos,neg]; labels=np.r_[np.ones(len(pos),int),np.zeros(len(neg),int)]
    observed=abs(pos.mean()-neg.mean());rng=np.random.default_rng(seed);exceed=0
    for _ in range(REPLICATES):
        p=rng.permutation(labels); d=values[p==1].mean()-values[p==0].mean(); exceed+=abs(d)>=observed-1e-15
    return float((exceed+1)/(REPLICATES+1))


def analyze(features, coverage):
    rows=[]
    for channel, signal, families in (("INSTANTANEOUS","qll_margin",FAMILIES),("CUMULATIVE","final_cumulative_l1",PRIMARY_C+["IA"])):
        for family in families:
            cell=features[features.family.eq(family)]; pos=cell.loc[cell.member.eq(1),signal].to_numpy(float);neg=cell.loc[cell.member.eq(0),signal].to_numpy(float)
            lo,hi,pgt,plt=bootstrap_difference(pos,neg,stable_seed(channel,family));p=permutation_difference(pos,neg,stable_seed("perm",channel,family))
            rows.append({"channel":channel,"signal":signal,"family":family,"member_n":len(pos),"nonmember_n":len(neg),
                         "member_mean":pos.mean(),"member_median":np.median(pos),"nonmember_mean":neg.mean(),"nonmember_median":np.median(neg),
                         "mean_difference":pos.mean()-neg.mean(),"difference_ci95_low":lo,"difference_ci95_high":hi,
                         "direction_stability_positive":pgt,"direction_stability_negative":plt,"permutation_p":p,
                         "standardized_effect_hedges_g":hedges_g(pos,neg)})
    family=pd.DataFrame(rows);atomic_csv(family,ROOT/"tables/TABLE_193_02_TWO_SIGNAL_BY_FAMILY.csv")
    budget=features[features.family.eq("BudgetLeak-Z")];nonmember=budget.loc[budget.member.eq(0),"final_cumulative_l1"].to_numpy(float)
    rank_rows=[]
    for rank in (1,2,3,4):
        member=budget.loc[budget.member.eq(1)&budget.target_rank.eq(rank),"final_cumulative_l1"].to_numpy(float)
        lo,hi,pgt,plt=bootstrap_difference(member,nonmember,stable_seed("rank",rank));p=permutation_difference(member,nonmember,stable_seed("rankperm",rank))
        rank_rows.append({"target_rank":rank,"member_n":len(member),"common_nonmember_n":len(nonmember),
                          "member_mean":member.mean(),"member_median":np.median(member),"nonmember_mean":nonmember.mean(),
                          "nonmember_median":np.median(nonmember),"mean_difference":member.mean()-nonmember.mean(),
                          "difference_ci95_low":lo,"difference_ci95_high":hi,"direction_stability_positive":pgt,
                          "direction_stability_negative":plt,"permutation_p":p,"standardized_effect_hedges_g":hedges_g(member,nonmember)})
    rank=pd.DataFrame(rank_rows);atomic_csv(rank,ROOT/"tables/TABLE_193_03_BUDGETLEAK_C1_BY_RANK.csv")
    old=pd.read_csv(INPUTS["exp191_rank"]);old=old[old.channel.eq("CUMULATIVE")]
    comparison=old[["target_rank","member_n","common_nonmember_n","member_mean","nonmember_mean","mean_difference","standardized_effect_hedges_g"]].merge(
        rank,on="target_rank",suffixes=("_exp191_top3","_exp193_top4"),validate="one_to_one")
    comparison["paired_condition_claim_prohibited"]="SEPARATE_EXPERIMENTAL_CONDITIONS"
    atomic_csv(comparison,ROOT/"tables/TABLE_193_04_EXP191_VS_EXP193_CONTEXT_CONDITION.csv")
    i_primary=family[(family.channel.eq("INSTANTANEOUS"))&family.family.isin(PRIMARY_I)]
    c_primary=family[(family.channel.eq("CUMULATIVE"))&family.family.isin(PRIMARY_C)]
    two_signal_reestablished=bool((i_primary.mean_difference>0).all() and (c_primary.mean_difference>0).all())
    rank4=rank[rank.target_rank.eq(4)].iloc[0];reproduced=bool(rank4.mean_difference<0 and rank4.standardized_effect_hedges_g<0)
    positive_all=bool((rank.mean_difference>0).all() and (rank.standardized_effect_hedges_g>0).all())
    verdict=("EXP193_TOP4_REBASE_VALID_REVERSAL_REPRODUCED" if reproduced else
             "EXP193_TOP4_REBASE_VALID_REVERSAL_NOT_REPRODUCED")
    decision=pd.DataFrame([{"exact_c1_alignment":float(coverage.loc[coverage.scope.eq("OVERALL"),"coverage"].iloc[0]),
                            "budgetleak_rank4_n":int(rank4.member_n),"rank4_mean_difference":float(rank4.mean_difference),
                            "rank4_hedges_g":float(rank4.standardized_effect_hedges_g),"rank4_reversal_reproduced":reproduced,
                            "rank1_4_all_positive":positive_all,"frozen_two_signal_direction_reestablished":two_signal_reestablished,
                            "final_verdict":verdict,
                            "next_step":"EXP194_QLL_ANCHORED_CUMULATIVE_REFINEMENT" if reproduced else "TOP3_TO_TOP4_MECHANISM_DIAGNOSIS"}])
    atomic_csv(decision,ROOT/"tables/TABLE_193_05_DECISION.csv")
    return family,rank,comparison,decision


def verify_frozen_tree():
    before=pd.read_csv(ROOT/"provenance/FROZEN_TREE_BEFORE.csv");after=frozen_tree_manifest();atomic_csv(after,ROOT/"provenance/FROZEN_TREE_AFTER.csv")
    merged=before.merge(after,on=["experiment","relative_path"],how="outer",suffixes=("_before","_after"),indicator=True)
    ok=bool(merged._merge.eq("both").all() and merged.sha256_before.eq(merged.sha256_after).all())
    atomic_csv(merged,ROOT/"provenance/FROZEN_TREE_COMPARISON.csv")
    if not ok: raise RuntimeError("historical frozen artifact changed")
    return ok,len(before)


def write_final(cases,responses,coverage,family,rank,comparison,decision):
    frozen_ok,frozen_files=verify_frozen_tree();row=decision.iloc[0]
    result={"experiment":"Exp193","final_verdict":row.final_verdict,"cohort_cases":len(cases),
            "cohort_sessions":cases.session_id.nunique(),"new_answer_rows":len(responses),
            "exact_c1_overall":float(coverage.loc[coverage.scope.eq("OVERALL"),"coverage"].iloc[0]),
            "budgetleak_rank4_member_sessions":int(row.budgetleak_rank4_n),
            "rank4_reversal_reproduced":bool(row.rank4_reversal_reproduced),
            "frozen_two_signal_direction_reestablished":bool(row.frozen_two_signal_direction_reestablished),
            "historical_frozen_artifacts_unchanged":frozen_ok,"historical_files_verified":frozen_files,
            "claude_independent_verification":"PENDING","codex_reverification":"PENDING",
            "next_step":row.next_step,"new_defense":False,"llama":False,"fresh_blind":False,"e_mia_open_count":0,
            "completed_utc":now()}
    atomic_json(ROOT/"FINAL_RESULT.json",result)
    report=("# Exp193 Top-4 Context Rebase\n\n"
            f"- Verdict: **{row.final_verdict}**\n- Cases/sessions: `{len(cases)}` / `{cases.session_id.nunique()}`\n"
            f"- New top-4 answers: `{len(responses)}`\n- Exact C1 coverage: `{result['exact_c1_overall']:.6f}`\n"
            f"- BudgetLeak rank-4 member sessions: `{int(row.budgetleak_rank4_n)}`\n"
            f"- Rank-4 reversal reproduced: `{bool(row.rank4_reversal_reproduced)}`\n"
            f"- Frozen historical files unchanged: `{frozen_ok}` ({frozen_files} files)\n"
            f"- Next: `{row.next_step}`\n\n## Exact C1 coverage\n\n{coverage.to_markdown(index=False)}\n\n"
            f"## Frozen two-signal result\n\n{family.to_markdown(index=False)}\n\n## BudgetLeak C1 by rank\n\n{rank.to_markdown(index=False)}\n")
    atomic_text(ROOT/"reports/REPORT_193_FINAL_KO.md",report)
    checkpoint("CODEX_PRIMARY_ANALYSIS_COMPLETE", verdict=row.final_verdict, cases=len(cases), sessions=cases.session_id.nunique(),
               new_answers=len(responses), exact_c1_alignment=result["exact_c1_overall"], rank4_reversal=bool(row.rank4_reversal_reproduced),
               next_step=row.next_step)
    return result


def run():
    for directory in ["audits","checkpoints","code","configs","figures","logs","private","provenance","reports","scripts","tables","tests","validation"]:
        (ROOT/directory).mkdir(parents=True,exist_ok=True)
    old,parent=preflight();cases,bundles,prompts=build_substrate(old,parent);audit_substrate(cases,bundles,prompts)
    responses=generate(cases,bundles,prompts)
    claims,tasks=build_claim_tasks(cases,bundles,prompts,responses)
    score_c1(cases,bundles,prompts,responses,claims,tasks)
    costs=aggregate_c1(claims,tasks);qll=qll_dominant(cases)
    detail,coverage,passed=coverage_gate(cases,costs,qll)
    if not passed:
        frozen_ok,frozen_files=verify_frozen_tree()
        result={"experiment":"Exp193","final_verdict":"EXP193_INPUT_INSUFFICIENT","cohort_cases":len(cases),
                "cohort_sessions":cases.session_id.nunique(),"new_answer_rows":len(responses),
                "exact_c1_overall":float(detail.exact_c1_for_qll_dominant.mean()),
                "historical_frozen_artifacts_unchanged":frozen_ok,"historical_files_verified":frozen_files,
                "statistics_run":False,"next_step":"STOP","completed_utc":now()}
        atomic_json(ROOT/"FINAL_RESULT.json",result);checkpoint("EXP193_INPUT_INSUFFICIENT",**result);return result
    ledger=apply_frozen_ledger(cases,responses,costs);features=build_features(cases,responses,costs,ledger,qll)
    family,rank,comparison,decision=analyze(features,coverage)
    return write_final(cases,responses,coverage,family,rank,comparison,decision)


if __name__ == "__main__":
    print(json.dumps(run(),ensure_ascii=False,indent=2,default=str))
