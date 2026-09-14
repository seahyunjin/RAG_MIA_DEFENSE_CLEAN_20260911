#!/usr/bin/env python3
"""Exp196: benign-calibrated selective rank-agnostic GlobalCap64.

G1 uses only the frozen pre-generation QLL score.  The selected source gets
the already frozen 64-token GlobalCap allocation while the remaining 1,984
tokens are distributed over the other three sources in retrieval order.
Attack labels, membership labels and retrieval ranks never enter the policy.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp196_selective_rank_agnostic_globalcap64_20260829"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP152 = PROJECT / "exp152_exp150_clean_3k_confirmation_20260825"
EXP153 = PROJECT / "exp153_native_dcmi2_ia15_20260826"
EXP158 = PROJECT / "exp158_global_cap64_native_sessions_20260826"
EXP166 = PROJECT / "exp166_topiocqa_gold_utility_20260827"
EXP174 = PROJECT / "exp174_stable_prefix64_native_sessions_20260828"
EXP176 = PROJECT / "exp176_prefix64_mirabel_external_attacks_20260828"
EXP179 = PROJECT / "exp179_cross_family_qwen_source_influence_20260828"
EXP187 = PROJECT / "exp187_dc_mcel_external_attacks_20260828"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
EXP193 = PROJECT / "exp193_top4_context_rebase_20260829"
EXP195 = PROJECT / "exp195_minimal_qll_exposure_guard_20260829"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")

G1 = "G1_SELECTIVE_CAP64"
BASELINES = ("NO_DEFENSE", "GLOBAL_CAP64", "ORIGINAL_MIRABEL")
CAP = 64
TOTAL_CONTEXT = 2048
SEED = 19620260829

ATTACK_CASES = EXP193 / "private/EXP193_CASES.private.pkl.gz"
ATTACK_VANILLA = EXP193 / "private/EXP193_TOP4_RESPONSES.private.csv.gz"
ATTACK_QLL = EXP193 / "private/EXP193_QLL_DOMINANT_SOURCE.private.csv.gz"
NORMAL_CASES = EXP195 / "private/EXP195_NORMAL_CASES.private.pkl.gz"
NORMAL_VANILLA = EXP195 / "private/EXP195_NORMAL_VANILLA.private.csv.gz"
FROZEN_BASELINES = EXP188 / "private/EXP188_CASES.private.pkl.gz"

GEN_DB = ROOT / "private/EXP196_G1_GENERATION.sqlite3"
G1_RESPONSES = ROOT / "private/EXP196_G1_RESPONSES.private.csv.gz"
ALL_RESPONSES = ROOT / "private/EXP196_ALL_RESPONSES.private.csv.gz"
NORMAL_RESPONSES = ROOT / "private/EXP196_NORMAL_RESPONSES.private.csv.gz"
NATIVE_SCORES = ROOT / "private/EXP196_G1_NATIVE_SCORES.private.csv.gz"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


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
    payload = {"experiment": "Exp196", "stage": stage, "updated_utc": now(), "pid": os.getpid(),
               "new_trainable_parameters": 0, "attack_specific_tuning": 0,
               "rank_specific_tuning": 0, "family_specific_rules": 0,
               "paid_api_calls": 0, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp196 Status", "", f"- Stage: **{stage}**", f"- Updated UTC: `{payload['updated_utc']}`",
             f"- PID: `{payload['pid']}`"] + [f"- {key}: `{value}`" for key, value in details.items()]
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def preflight():
    required = {
        "attack_cases": ATTACK_CASES, "attack_vanilla": ATTACK_VANILLA, "attack_qll": ATTACK_QLL,
        "normal_cases": NORMAL_CASES, "normal_vanilla": NORMAL_VANILLA,
        "frozen_baselines": FROZEN_BASELINES,
        "qll_implementation": EXP179 / "code/run_exp179.py",
        "globalcap_implementation": PROJECT / "exp160_global_cap64_ragleak_budgetleak_20260826/code/run_exp160.py",
        "exp195_final": EXP195 / "FINAL_RESULT.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing: raise RuntimeError(f"missing frozen input: {missing}")
    implementation = required["qll_implementation"].read_text(encoding="utf-8")
    evidence = all(text in implementation for text in (
        "def prepare_sequence(tokenizer, source: str, query: str", "Context:\\n", "User query:\\n",
        "mean_query_log_probability"))
    forbidden_answer_dependency = "generated_answer" in implementation or "protected_answer" in implementation
    if not evidence or forbidden_answer_dependency:
        result = {"experiment": "Exp196", "verdict": "EXP196_PREGEN_SIGNAL_INVALID",
                  "evidence_found": evidence, "forbidden_answer_dependency": forbidden_answer_dependency,
                  "completed_utc": now()}
        atomic_json(ROOT / "FINAL_RESULT.json", result)
        checkpoint("EXP196_PREGEN_SIGNAL_INVALID", **result)
        raise SystemExit(2)
    manifest = pd.DataFrame([{"key": key, "path": str(path), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path), "access": "READ_ONLY"}
                             for key, path in required.items()])
    atomic_csv(manifest, ROOT / "provenance/FROZEN_INPUTS.csv")
    atomic_json(ROOT / "audits/PREGEN_QLL_AUDIT.json", {
        "passed": True, "operands": ["query", "one retrieved source"],
        "target_tokens": "query tokens", "generated_answer_required": False,
        "implementation": str(required["qll_implementation"]),
        "implementation_sha256": sha256_file(required["qll_implementation"]),
    })
    checkpoint("PREFLIGHT_COMPLETE", pregen_qll=True, frozen_inputs=len(manifest), old_artifacts_read_only=True)


def quantile_threshold(values, tail_rate=.01):
    ordered = np.sort(np.asarray(values, dtype=float))
    index = max(0, min(len(ordered)-1, int(math.ceil((1-tail_rate)*len(ordered)))-1))
    return float(ordered[index])


def load_substrate():
    attacks = pd.read_pickle(ATTACK_CASES, compression="gzip")
    qll = pd.read_csv(ATTACK_QLL, keep_default_na=False, low_memory=False, dtype={"case_id": str})
    av = pd.read_csv(ATTACK_VANILLA, keep_default_na=False, low_memory=False, dtype={"case_id": str})
    attacks = attacks.drop(columns=["response"], errors="ignore").merge(
        av[["case_id", "response"]].rename(columns={"response": "vanilla_response"}),
        on="case_id", validate="one_to_one").merge(
        qll[["case_id", "dominant_source_id", "dominant_source_retrieval_rank", "dominant_source_qll"]],
        on="case_id", validate="one_to_one")
    normal = pd.read_pickle(NORMAL_CASES, compression="gzip")
    nv = pd.read_csv(NORMAL_VANILLA, keep_default_na=False, dtype={"case_id": str})
    normal = normal.drop(columns=["vanilla_response"], errors="ignore").merge(
        nv[["case_id", "response"]].rename(columns={"response": "vanilla_response"}),
        on="case_id", validate="one_to_one")
    normal["dominant_source_id"] = normal.selected_source_id.astype(str)
    normal["dominant_source_retrieval_rank"] = normal.selected_source_rank.astype(int)
    normal["dominant_source_qll"] = normal.selected_qll.astype(float)
    if len(attacks) != 35680 or len(normal) != 1000:
        raise RuntimeError("substrate count mismatch")
    if not all(str(source) in set(map(str, ids)) for source, ids in zip(
            pd.concat([attacks.dominant_source_id, normal.dominant_source_id]),
            pd.concat([attacks.source_ids, normal.source_ids]))):
        raise RuntimeError("QLL selected source outside Top-4")
    return attacks, normal


def freeze_tau(normal):
    path = ROOT / "configs/G1_TAU_I_FROZEN.json"
    if path.exists(): return float(json.loads(path.read_text())["tau_I"])
    tau = quantile_threshold(normal.dominant_source_qll, .01)
    rate = float(normal.dominant_source_qll.gt(tau).mean())
    payload = {"experiment": "Exp196", "signal": "max frozen pre-generation source QLL",
               "calibration": "TopiOCQA benign-only", "calibration_queries": len(normal),
               "calibration_sessions": int(normal.session_id.nunique()), "attack_examples_used": 0,
               "target_benign_intervention_rate": .01, "strict_operator": ">",
               "tau_I": tau, "empirical_benign_intervention_rate": rate,
               "frozen_before_attack_generation": True, "frozen_utc": now()}
    atomic_json(path, payload); atomic_text(ROOT / "configs/G1_TAU_I_FROZEN.sha256", sha256_file(path)+"  G1_TAU_I_FROZEN.json\n")
    atomic_csv(pd.DataFrame([payload]), ROOT / "tables/TABLE_196_01_G1_CALIBRATION.csv")
    checkpoint("G1_TAU_FROZEN", tau_I=tau, benign_queries=len(normal), empirical_rate=rate, attack_examples_used=0)
    return tau


def caps_for(source_ids, selected):
    ids = list(map(str, source_ids))
    if len(ids) != 4 or len(set(ids)) != 4 or str(selected) not in ids:
        raise ValueError("G1 requires four unique sources and selected source in Top-4")
    other = [source for source in ids if source != str(selected)]
    remainder = TOTAL_CONTEXT-CAP; base = remainder//len(other); extra = remainder-base*len(other)
    result = {str(selected): CAP}
    for index, source in enumerate(other): result[source] = base + int(index < extra)
    if sum(result.values()) != TOTAL_CONTEXT or result[str(selected)] != CAP:
        raise AssertionError("allocation invariant failed")
    return result


def max_new_tokens(row):
    if hasattr(row, "max_new_tokens") and not pd.isna(row.max_new_tokens): return int(row.max_new_tokens)
    return 96


def generate_g1(attacks, normal, tau):
    if G1_RESPONSES.exists():
        frame = pd.read_csv(G1_RESPONSES, keep_default_na=False, low_memory=False, dtype={"case_id": str})
        if len(frame) == len(attacks)+len(normal): return frame
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    models.ROOT = ROOT; models.heartbeat = lambda stage, **details: checkpoint(stage, **details)
    models.GENERATION_CONFIG["batch_size"] = 32
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    combined = pd.concat([attacks.assign(split="ATTACK"), normal.assign(split="NORMAL")], ignore_index=True, sort=False)
    combined["triggered"] = combined.dominant_source_qll.astype(float).gt(tau)
    frozen_globalcap = pd.read_pickle(FROZEN_BASELINES, compression="gzip").set_index("case_id").response.to_dict()
    # When QLL selects rank 1, G1's [64,662,661,661] context is byte-for-byte
    # the frozen GlobalCap64 policy. Reuse that deterministic answer and only
    # generate genuinely new rank-2/3/4 allocations.
    combined["exact_globalcap_reuse"] = combined.triggered & combined.dominant_source_retrieval_rank.astype(int).eq(1)
    tasks, metadata, packing = [], [], []
    for row in combined[combined.triggered & ~combined.exact_globalcap_reuse].itertuples(index=False):
        ids = list(map(str, row.source_ids)); texts = list(map(str, row.source_texts))
        caps = caps_for(ids, str(row.dominant_source_id)); contexts, used = [], []
        for source, text in zip(ids, texts):
            token_ids = tokenizer(text, add_special_tokens=False).input_ids[:caps[source]]
            contexts.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip()); used.append(len(token_ids))
        prompt = models.normal_prompt(str(row.query), contexts)
        task = models.make_task(task_type="EXP196_G1_SELECTIVE_CAP64", row_id=str(row.case_id),
                                prompt=prompt, system_prompt=str(row.system_prompt), max_new_tokens=max_new_tokens(row))
        tasks.append(task); metadata.append({"case_id": str(row.case_id), "prompt_hash": task["prompt_hash"]})
        packing.append({"case_id": str(row.case_id), "family": str(row.family), "member": int(row.member),
                        "selected_source_id": str(row.dominant_source_id),
                        "selected_source_rank_for_audit_only": int(row.dominant_source_retrieval_rank),
                        "source_ids": json.dumps(ids), "source_caps": json.dumps([caps[x] for x in ids]),
                        "source_tokens_used": json.dumps(used), "total_cap": sum(caps.values())})
    atomic_csv(pd.DataFrame(packing), ROOT / "audits/G1_PACKING_AUDIT.csv")
    checkpoint("G1_GENERATION_STARTED", all_queries=len(combined), policy_triggered_queries=int(combined.triggered.sum()),
               exact_frozen_globalcap_reuse=int(combined.exact_globalcap_reuse.sum()), new_generation_queries=len(tasks),
               normal_triggered=int(combined[combined.split.eq("NORMAL")].triggered.sum()), device="cuda:0")
    # The frozen task key and runtime both use the already validated batch 32.
    # Only genuinely new rank-2/3/4 contexts reach this path (90 in this run).
    models.GENERATION_CONFIG["batch_size"] = 32
    answers = models.run_generation(tasks, GEN_DB, None, "G1_GENERATION_PROGRESS") if tasks else {}
    generated = dict(zip([item["case_id"] for item in metadata], [answers[item["case_id"]] for item in metadata]))
    rows = []
    for row in combined.itertuples(index=False):
        trigger = bool(row.triggered)
        if bool(row.exact_globalcap_reuse):
            response, provenance = str(frozen_globalcap[str(row.case_id)]), "EXACT_FROZEN_GLOBALCAP64_RANK1"
        elif trigger:
            response, provenance = generated[str(row.case_id)], "NEW_G1_RANK_AGNOSTIC_GENERATION"
        else:
            response, provenance = str(row.vanilla_response), "EXACT_FROZEN_VANILLA"
        rows.append({"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": str(row.panel),
                     "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
                     "session_id": str(row.session_id), "turn_order": int(row.turn_order),
                     "target_rank": int(row.target_rank), "query": str(row.query), "condition": G1,
                     "response": response, "vanilla_response": str(row.vanilla_response),
                     "intervened": trigger, "max_qll": float(row.dominant_source_qll),
                     "response_provenance": provenance,
                     "selected_source_id": str(row.dominant_source_id),
                     "selected_source_rank_audit_only": int(row.dominant_source_retrieval_rank)})
    output = pd.DataFrame(rows); atomic_csv(output, G1_RESPONSES, "gzip")
    summary = output.groupby(["family", "member"], as_index=False).agg(
        queries=("case_id", "size"), intervention_rate=("intervened", "mean"),
        response_change_rate=("response", lambda x: np.nan))
    atomic_csv(summary, ROOT / "tables/TABLE_196_02_G1_INTERVENTION.csv")
    checkpoint("G1_GENERATION_COMPLETE", responses=len(output), generated=len(tasks),
               exact_globalcap_reused=int(combined.exact_globalcap_reuse.sum()),
               total_intervention_rate=float(output.intervened.mean()))
    return output


def build_condition_responses(g1, attacks, normal):
    if ALL_RESPONSES.exists() and NORMAL_RESPONSES.exists():
        return (pd.read_csv(ALL_RESPONSES, keep_default_na=False, low_memory=False, dtype={"case_id": str}),
                pd.read_csv(NORMAL_RESPONSES, keep_default_na=False, dtype={"case_id": str}))
    baseline = pd.read_pickle(FROZEN_BASELINES, compression="gzip")
    base = baseline.set_index("case_id")
    gmap = g1.set_index("case_id")
    attack_rows, normal_rows = [], []
    for frame, destination in ((attacks, attack_rows), (normal, normal_rows)):
        for row in frame.itertuples(index=False):
            cid = str(row.case_id)
            common = {"case_id": cid, "row_id": str(row.row_id), "panel": str(row.panel),
                      "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
                      "session_id": str(row.session_id), "turn_order": int(row.turn_order),
                      "target_rank": int(row.target_rank), "query": str(row.query)}
            destination.append({**common, "condition": "NO_DEFENSE", "response": str(row.vanilla_response), "intervened": False})
            destination.append({**common, "condition": "GLOBAL_CAP64", "response": str(base.loc[cid, "response"]), "intervened": True})
            destination.append({**common, "condition": "ORIGINAL_MIRABEL", "response": str(base.loc[cid, "mirabel_response"]),
                                "intervened": str(base.loc[cid, "mirabel_response"]) != str(row.vanilla_response)})
            destination.append({**common, "condition": G1, "response": str(gmap.loc[cid, "response"]),
                                "intervened": bool(gmap.loc[cid, "intervened"])})
    a = pd.DataFrame(attack_rows); n = pd.DataFrame(normal_rows)
    atomic_csv(a, ALL_RESPONSES, "gzip"); atomic_csv(n, NORMAL_RESPONSES, "gzip")
    checkpoint("CONDITION_RESPONSES_COMPLETE", attack_rows=len(a), normal_rows=len(n), conditions=4)
    return a, n


def normalize_manifest(frame, single_query=False):
    output = frame.copy()
    output["purpose"] = "PRIVACY_EVAL"; output["kind"] = "attack"
    output["source_document_id"] = output.target_document_id.astype(str)
    if "turn" not in output: output["turn"] = 1 if single_query else output["selected_turn"]
    output["exp87_row_id"] = output.row_id.astype(str)
    return output


def score_native_g1(responses):
    if NATIVE_SCORES.exists(): return pd.read_csv(NATIVE_SCORES, keep_default_na=False, low_memory=False)
    clean = pd.read_csv(EXP152 / "private/EXP152_CLEAN_3K_MANIFEST.private.csv.gz",
                        keep_default_na=False, low_memory=False, dtype={"row_id": str})
    clean = normalize_manifest(clean[clean.family.isin(["RAG-MIA", "S²-MIA", "MBA"])], True)
    native = normalize_manifest(pd.read_csv(EXP153 / "private/EXP153_NATIVE_TURNS.private.csv.gz",
                               keep_default_na=False, low_memory=False, dtype={"row_id": str}))
    menta = normalize_manifest(pd.read_csv(EXP153 / "menta5_followup/private/MENTA5_NATIVE_TURNS.private.csv.gz",
                              keep_default_na=False, low_memory=False, dtype={"row_id": str}))
    sys.path.insert(0, str(EXP195 / "code")); import run_exp195 as parent
    sys.path.insert(0, str(EXP188 / "code")); import run_exp188 as evaluator
    required = {(str(row.domain), str(row.target_document_id)) for frame in (clean, native, menta)
                for row in frame.itertuples(index=False)}
    documents = evaluator._native_documents(required)
    truth_frame = pd.read_csv(EXP158 / "private/EXP158_IA_GROUND_TRUTH.private.csv.gz",
                              keep_default_na=False, dtype={"row_id": str})
    truth = truth_frame.set_index("row_id").response.to_dict(); pieces = []
    candidate = responses[responses.condition.eq(G1)]
    for suite, panel, frame, local_truth in (("Q1", "NATIVE_Q1", clean, {}),
            ("DCMI_IA", "NATIVE_SESSION", native, truth), ("MENTA", "NATIVE_MENTA", menta, {})):
        scorer = load_module(f"exp196_scorer_{suite}", EXP87 / "code/exp87_scoring.py")
        scorer.ROOT = ROOT; scorer.RESPONSE_DB = ROOT / f"private/EXP196_{suite}_SCORER.sqlite3"
        scorer.atomic_csv = atomic_csv; scorer.atomic_json = atomic_json
        scorer.checkpoint = lambda stage, suite=suite, **details: checkpoint(f"SCORER_{suite}_{stage}", **details)
        shared = ROOT / "private/EXP87_MIA_SCORES.private.csv.gz"
        if shared.exists(): shared.unlink()
        supplied = candidate[candidate.panel.eq(panel)].rename(columns={"row_id": "exp87_row_id"})[[
            "exp87_row_id", "condition", "response"]].assign(A_R=lambda x: x.response,
            action="EXP196_G1", lola_alarm=False, lola_score=np.nan, beta=np.nan)
        measured = scorer.mia_scores(frame, supplied, local_truth, documents)
        measured["suite"] = suite; pieces.append(measured)
    output = pd.concat(pieces, ignore_index=True); atomic_csv(output, NATIVE_SCORES, "gzip")
    checkpoint("G1_NATIVE_SCORING_COMPLETE", rows=len(output), invalid=int(output.attack_score.isna().sum()))
    return output


def metric_rows(scores, rank_column=None):
    columns = ["condition", "attack_family"] + ([rank_column] if rank_column else [])
    rows = []
    for keys, cell in scores.groupby(columns, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        # Some original attack scorers (notably IA) represent an undefined
        # score as an empty string in their immutable CSV artifact.  Preserve
        # that missingness: coerce only for metric computation and never
        # impute a score or change the attack semantics.
        numeric_score = pd.to_numeric(cell["attack_score"], errors="coerce")
        numeric_member = pd.to_numeric(cell["member"], errors="coerce")
        valid_mask = numeric_score.notna() & numeric_member.notna()
        valid = cell.loc[valid_mask].copy()
        valid["attack_score"] = numeric_score.loc[valid_mask].astype(float)
        valid["member"] = numeric_member.loc[valid_mask].astype(int)
        if valid.member.nunique() != 2: continue
        raw = float(roc_auc_score(valid.member, valid.attack_score))
        rows.append({**dict(zip(columns, keys)), "n": len(valid), "n_total": len(cell),
                     "n_undefined": int((~valid_mask).sum()), "raw_auc": raw,
                     "effective_auc": max(raw, 1-raw), "members": int(valid.member.eq(1).sum()),
                     "nonmembers": int(valid.member.eq(0).sum())})
    return pd.DataFrame(rows)


def privacy_evaluation(responses, attacks):
    native_g1 = score_native_g1(responses)
    frozen_native = pd.read_csv(EXP195 / "private/EXP195_NATIVE_SCORES.private.csv.gz",
                                keep_default_na=False, low_memory=False)
    native = pd.concat([frozen_native[frozen_native.condition.isin(BASELINES)], native_g1], ignore_index=True)
    native_table = metric_rows(native); atomic_csv(native_table, ROOT / "tables/TABLE_196_03_NATIVE_PRIVACY.csv")
    sys.path.insert(0, str(EXP188 / "code")); import run_exp188 as evaluator
    evaluator.ROOT = ROOT
    candidate = responses[(responses.condition.eq(G1)) & responses.panel.eq("EXTERNAL")]
    external_cases = attacks[attacks.panel.eq("EXTERNAL")]
    metrics = evaluator.external_response_metrics(candidate, external_cases)
    external_g1 = evaluator.external_attack_scores(metrics)
    frozen_external = pd.read_csv(EXP195 / "private/EXP188_EXTERNAL_ATTACK_SCORES.private.csv.gz",
                                  keep_default_na=False, low_memory=False)
    external = pd.concat([frozen_external[frozen_external.condition.isin(BASELINES)], external_g1], ignore_index=True)
    ext_table = metric_rows(external); atomic_csv(ext_table, ROOT / "tables/TABLE_196_04_EXTERNAL_PRIVACY.csv")
    budget = external[external.attack_family.eq("BudgetLeak-Z")].copy()
    # BudgetLeak nonmembers have no member target and therefore target_rank=0.
    # Match the frozen Exp160E/Exp188 rank audit: for each rank, compare member
    # sessions whose target occupies that rank against the complete, identical
    # 900-session nonmember reference population.
    rank_frames = []
    for rank in (1, 2, 3, 4):
        rank_frames.append(budget[(budget.member.eq(0)) |
                                  (budget.member.eq(1) & budget.target_rank.eq(rank))].assign(
                                      target_rank_stratum=f"rank{rank}"))
    rank_table = metric_rows(pd.concat(rank_frames, ignore_index=True), "target_rank_stratum")
    atomic_csv(rank_table, ROOT / "tables/TABLE_196_05_BUDGETLEAK_BY_RANK.csv")
    checkpoint("G1_PRIVACY_COMPLETE", native_families=native_table.attack_family.nunique(), external_attacks=2)
    return native_table, ext_table, rank_table


def refusal(value):
    return bool(re.search(r"\b(?:i do not know|i don't know|cannot determine|insufficient context|not enough information)\b",
                          str(value), re.I))


def utility_evaluation(normal_responses):
    destination = ROOT / "tables/TABLE_196_06_NORMAL_UTILITY.csv"
    detail_path = ROOT / "private/EXP196_NORMAL_UTILITY_DETAIL.private.csv.gz"
    if destination.exists(): return pd.read_csv(destination)
    sys.path.insert(0, str(EXP188 / "code")); import exp188_core as core
    gold = pd.read_csv(EXP166 / "private/TOPIOCQA_GOLD_1000.private.csv.gz", keep_default_na=False,
                       dtype={"row_id": str})
    frame = normal_responses.merge(gold[["row_id", "gold_answers"]], on="row_id", validate="many_to_one")
    vanilla = frame[frame.condition.eq("NO_DEFENSE")].set_index("row_id").response.to_dict()
    frame["vanilla"] = frame.row_id.map(vanilla)
    frame["exact_vanilla_preservation"] = frame.response.astype(str).eq(frame.vanilla.astype(str))
    frame["token_f1"] = [max(core.token_f1(answer, truth) for truth in json.loads(golds))
                         for answer, golds in zip(frame.response, frame.gold_answers)]
    frame["exact_match"] = [max(core.normalize_answer(answer) == core.normalize_answer(truth)
                                 for truth in json.loads(golds)) for answer, golds in zip(frame.response, frame.gold_answers)]
    frame["refusal"] = frame.response.map(refusal); frame["base_refusal"] = frame.vanilla.map(refusal)
    frame["new_refusal"] = frame.refusal & ~frame.base_refusal
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MPNET), device="cuda", local_files_only=True); model.max_seq_length = 512
    unique = pd.unique(pd.concat([frame.response, frame.vanilla]).astype(str))
    vectors = model.encode(unique.tolist(), normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=256, show_progress_bar=False); mapping = dict(zip(unique, vectors))
    frame["response_preservation"] = [float(np.clip(np.dot(mapping[str(a)], mapping[str(b)]), 0, 1))
                                      for a, b in zip(frame.response, frame.vanilla)]
    del model; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    summary = frame.groupby("condition", as_index=False).agg(
        normal_queries=("row_id", "size"), intervention_rate=("intervened", "mean"),
        exact_vanilla_preservation=("exact_vanilla_preservation", "mean"), token_f1=("token_f1", "mean"),
        exact_match=("exact_match", "mean"), response_preservation=("response_preservation", "mean"),
        refusal_rate=("refusal", "mean"), new_refusal_rate=("new_refusal", "mean"))
    atomic_csv(frame, detail_path, "gzip"); atomic_csv(summary, destination)
    g1 = frame[frame.condition.eq(G1)]; nontrigger = g1[~g1.intervened]
    identity = float(nontrigger.response.astype(str).eq(nontrigger.vanilla.astype(str)).mean())
    atomic_json(ROOT / "audits/G1_NONTRIGGER_IDENTITY.json", {"rows": len(nontrigger), "exact_identity": identity})
    if identity != 1.0: raise RuntimeError("G1 non-trigger identity failure")
    checkpoint("G1_UTILITY_COMPLETE", token_f1=float(summary.set_index("condition").loc[G1, "token_f1"]),
               vanilla_preservation=float(summary.set_index("condition").loc[G1, "exact_vanilla_preservation"]),
               nontrigger_identity=identity)
    return summary


def hallucination_audit(normal_responses):
    destination = ROOT / "tables/TABLE_196_07_HALLUCINATION_SCREEN.csv"
    if destination.exists(): return pd.read_csv(destination)
    g1 = normal_responses[normal_responses.condition.eq(G1)].copy()
    vanilla = normal_responses[normal_responses.condition.eq("NO_DEFENSE")].set_index("row_id").response.to_dict()
    g1["vanilla"] = g1.row_id.map(vanilla); changed = g1[g1.response.astype(str).ne(g1.vanilla.astype(str))].copy()
    number = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?(?!\w)")
    rows = []
    for row in changed.itertuples(index=False):
        before, after = str(row.vanilla), str(row.response)
        rows.append({"row_id": str(row.row_id), "query": str(row.query), "vanilla_response": before,
                     "g1_response": after, "new_refusal": (not refusal(before)) and refusal(after),
                     "numeric_values_changed": set(number.findall(before)) != set(number.findall(after)),
                     "status": "AUTOMATIC_CHANGE_SCREEN_NOT_HUMAN_VERIFIED"})
    detail = pd.DataFrame(rows)
    atomic_csv(detail, ROOT / "audits/G1_CHANGED_NORMAL_RESPONSES.csv")
    summary = pd.DataFrame([{"condition": G1, "normal_rows": len(g1), "changed_rows": len(changed),
                             "change_rate": len(changed)/len(g1),
                             "new_refusal_rate": float(detail.new_refusal.mean()) if len(detail) else 0.0,
                             "numeric_change_rate_among_changed": float(detail.numeric_values_changed.mean()) if len(detail) else 0.0,
                             "unsupported_claim_rate": np.nan,
                             "unsupported_claim_status": "FROZEN_NLI_PENDING_CHANGED_SUBSET",
                             "defense_induced_regeneration": int(len(changed))}])
    atomic_csv(summary, destination)
    checkpoint("G1_HALLUCINATION_SCREEN_COMPLETE", changed_rows=len(changed),
               new_refusals=int(detail.new_refusal.sum()) if len(detail) else 0,
               unsupported_claims="PENDING_FROZEN_NLI")
    return summary


def scalar(table, condition, family, column="effective_auc"):
    cell = table[(table.condition.eq(condition)) & table.attack_family.eq(family)]
    if len(cell) != 1: raise RuntimeError(f"metric missing: {condition}/{family}")
    return float(cell.iloc[0][column])


def decide_g1(native, external, ranks, utility, hallucination, tau):
    rag = scalar(external, G1, "RAGLeak"); s2 = scalar(native, G1, "S²-MIA")
    mir_rag = scalar(external, "ORIGINAL_MIRABEL", "RAGLeak")
    mir_s2 = scalar(native, "ORIGINAL_MIRABEL", "S²-MIA")
    u = utility.set_index("condition"); g = u.loc[G1]; global_u = u.loc["GLOBAL_CAP64"]
    gates = {
        "ragleak_competitive_with_mirabel_plus_0p02": rag <= mir_rag + .02,
        "s2_competitive_with_mirabel_plus_0p02": s2 <= mir_s2 + .02,
        "benign_token_f1_above_globalcap": float(g.token_f1) > float(global_u.token_f1),
        "vanilla_preservation_above_globalcap": float(g.exact_vanilla_preservation) > float(global_u.exact_vanilla_preservation),
        "nontrigger_identity_100pct": json.loads((ROOT / "audits/G1_NONTRIGGER_IDENTITY.json").read_text())["exact_identity"] == 1.0,
        "no_attack_or_rank_specific_tuning": True,
    }
    passed = all(gates.values())
    table = pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()])
    atomic_csv(table, ROOT / "tables/TABLE_196_08_G1_GATES.csv")
    if not passed:
        verdict = "EXP196_G1_ONE_SHOT_FAILED"
        result = {"experiment": "Exp196", "verdict": verdict, "final_scientific_verdict": "SELECTIVE_CAP64_FAILED",
                  "g1_executed": True, "g1_passed": False, "g2_executed": False,
                  "tau_I": tau, "tau_provenance": "TopiOCQA benign 1000; attack examples 0",
                  "ragleak_effective_auc": rag, "mirabel_ragleak_effective_auc": mir_rag,
                  "s2_effective_auc": s2, "mirabel_s2_effective_auc": mir_s2,
                  "budgetleak_effective_auc": scalar(external, G1, "BudgetLeak-Z"),
                  "budgetleak_rank_effective_auc": {str(r.target_rank_stratum): float(r.effective_auc)
                    for r in ranks[(ranks.condition.eq(G1)) & ranks.target_rank_stratum.isin(["rank1","rank2","rank3","rank4"])].itertuples(index=False)},
                  "native_mean_effective_auc": float(native[native.condition.eq(G1)].effective_auc.mean()),
                  "native_worst_effective_auc": float(native[native.condition.eq(G1)].effective_auc.max()),
                  "benign_token_f1": float(g.token_f1), "globalcap_benign_token_f1": float(global_u.token_f1),
                  "vanilla_preservation": float(g.exact_vanilla_preservation),
                  "globalcap_vanilla_preservation": float(global_u.exact_vanilla_preservation),
                  "benign_intervention_rate": float(g.intervention_rate),
                  "hallucination_screen": hallucination.iloc[0].to_dict(), "gates": gates,
                  "new_trainable_parameters": 0, "generations_per_query": 1,
                  "g1_session_state": 0, "attack_specific_tuning": 0, "rank_specific_tuning": 0,
                  "completed_utc": now()}
        atomic_json(ROOT / "FINAL_RESULT.json", result)
        report = f"""# Exp196 — Selective Rank-Agnostic GlobalCap64

- 최종 판정: **{verdict} / SELECTIVE_CAP64_FAILED**
- G1 실행: 완료
- G2 실행: 안 함 (G1 one-shot gate 실패 시 중단한다는 사전 규칙)
- tau_I: `{tau:.8f}` (정상 1,000개, 공격 0개, strict `>`)
- RAGLeak E-AUC: `{rag:.4f}` / Mirabel `{mir_rag:.4f}`
- S²-MIA E-AUC: `{s2:.4f}` / Mirabel `{mir_s2:.4f}`
- BudgetLeak E-AUC: `{result['budgetleak_effective_auc']:.4f}`
- 정상 token F1: `{float(g.token_f1):.4f}` / GlobalCap64 `{float(global_u.token_f1):.4f}`
- Vanilla exact 보존: `{float(g.exact_vanilla_preservation):.4f}` / GlobalCap64 `{float(global_u.exact_vanilla_preservation):.4f}`
- 학습 parameter: `0`, session state: `0`, generation/query: `1`

## G1 gate

{table.to_markdown(index=False)}

Raw AUC가 0.5 아래로 뒤집힌 경우에도 E-AUC=max(AUC,1-AUC)로 판정했다.
"""
        atomic_text(ROOT / "reports/REPORT_196_FINAL_KO.md", report)
        checkpoint("COMPLETE", verdict=verdict, g2_executed=False)
        return result
    checkpoint("G1_PASS_ONE_SHOT", ragleak_eauc=rag, s2_eauc=s2)
    return None


def smoke():
    preflight(); attacks, normal = load_substrate(); tau = freeze_tau(normal)
    checks = {"pregen_qll": True, "normal_rows": len(normal), "attack_rows": len(attacks),
              "normal_rate": float(normal.dominant_source_qll.gt(tau).mean()),
              "allocation_rank1": [caps_for(["a","b","c","d"], "a")[x] for x in ["a","b","c","d"]],
              "allocation_rank4": [caps_for(["a","b","c","d"], "d")[x] for x in ["a","b","c","d"]]}
    checks["pass"] = checks["normal_rate"] == .01 and checks["allocation_rank1"] == [64,662,661,661] and checks["allocation_rank4"] == [662,661,661,64]
    atomic_json(ROOT / "tests/SMOKE_RESULTS.json", checks); checkpoint("SMOKE_COMPLETE", **checks)
    print(json.dumps(checks, indent=2)); return checks


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    if args.smoke: smoke(); return
    preflight(); attacks, normal = load_substrate(); tau = freeze_tau(normal)
    g1 = generate_g1(attacks, normal, tau); responses, normal_responses = build_condition_responses(g1, attacks, normal)
    utility = utility_evaluation(normal_responses); hallucination = hallucination_audit(normal_responses)
    native, external, ranks = privacy_evaluation(responses, attacks)
    result = decide_g1(native, external, ranks, utility, hallucination, tau)
    if result is None:
        # The present frozen benign cohort contains only one-query sessions, so
        # a cumulative benign-session B cannot be estimated without fabricating
        # long sessions. Preserve G1 PASS and stop honestly if this rare branch occurs.
        blocked = {"experiment": "Exp196", "verdict": "G2_BENIGN_SESSION_CALIBRATION_REQUIRED",
                   "g1_passed": True, "g2_executed": False,
                   "reason": "frozen benign calibration sessions are all single-query; no synthetic sessions allowed",
                   "completed_utc": now()}
        atomic_json(ROOT / "FINAL_RESULT.json", blocked); checkpoint("G2_BENIGN_SESSION_CALIBRATION_REQUIRED", **blocked)
        print(json.dumps(blocked, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
