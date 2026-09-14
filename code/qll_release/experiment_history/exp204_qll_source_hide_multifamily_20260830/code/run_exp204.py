#!/usr/bin/env python3
"""Exp204: frozen multi-family extension of Exp203 selective QLL source hide."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp204_qll_source_hide_multifamily_20260830"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP166 = PROJECT / "exp166_topiocqa_gold_utility_20260827"
EXP179 = PROJECT / "exp179_cross_family_qwen_source_influence_20260828"
EXP181 = PROJECT / "exp181_native_session_continuous_qll_20260828"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
EXP193 = PROJECT / "exp193_top4_context_rebase_20260829"
EXP199 = PROJECT / "exp199_stable_evidence_budget_invariant_20260830"
EXP203 = PROJECT / "exp203_selective_qll_source_hide_20260830"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
PRECOMMIT = ROOT / "configs/PRECOMMIT.json"
CASES_PATH = EXP193 / "private/EXP193_CASES.private.pkl.gz"
QLL_MAP_PATH = EXP193 / "private/EXP193_QLL_DOMINANT_SOURCE.private.csv.gz"
QLL179_PATH = EXP179 / "private/EXP179_QUERY_SOURCE_QLL.private.csv.gz"
QLL181_PATH = EXP181 / "private/EXP181_SOURCE_QLL.private.csv.gz"
POLICY_PATH = ROOT / "private/EXP204_POLICY_ROWS.private.pkl.gz"
PACKING_PATH = ROOT / "private/EXP204_PACKING.private.pkl.gz"
RESPONSES_PATH = ROOT / "private/EXP204_RESPONSES.private.csv.gz"
GEN_DB = ROOT / "private/EXP204_QWEN_RESPONSES.sqlite3"
THRESHOLD = 0.5300846414247485
TOP_K = 4
TOTAL_BUDGET = 2048
LOCAL = "QLL_SOURCE_HIDE_PER_QUERY"
STICKY = "QLL_SOURCE_HIDE_SESSION_STICKY"
NATIVE = ("RAG-MIA", "S²-MIA", "MBA", "DCMI", "MEntA", "IA")
FAMILIES = NATIVE + ("RAGLeak",)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


common = load(EXP199 / "code/run_exp199.py", "exp204_common")


def now():
    return datetime.now(timezone.utc).isoformat()


def checkpoint(stage, **details):
    payload = {"experiment": "Exp204", "stage": stage, "updated_utc": now(), "pid": os.getpid(),
               "threshold": THRESHOLD, "trainable_parameters": 0, "paid_api_calls": 0,
               "request_rejection": False, **details}
    common.atomic_json(ROOT / "HEARTBEAT.json", payload)
    common.atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp204 Status", "", f"- Stage: **{stage}**", f"- Updated UTC: `{payload['updated_utc']}`",
             f"- PID: `{payload['pid']}`"] + [f"- {key}: `{value}`" for key, value in details.items()]
    common.atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def query_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def preflight():
    config = json.loads(PRECOMMIT.read_text())
    parent = json.loads((EXP203 / "FINAL_RESULT.json").read_text())
    if not parent.get("passed") or not np.isclose(float(parent["threshold"]), THRESHOLD, atol=1e-15, rtol=0):
        raise RuntimeError("Exp203 passed parent or threshold contract changed")
    required = {
        "precommit": PRECOMMIT, "exp203_result": EXP203 / "FINAL_RESULT.json",
        "exp203_privacy": EXP203 / "tables/TABLE_203_05_BUDGETLEAK.csv",
        "exp203_utility": EXP203 / "tables/TABLE_203_06_NORMAL_UTILITY.csv",
        "cases": CASES_PATH, "qll_map": QLL_MAP_PATH, "qll179": QLL179_PATH,
        "qll181": QLL181_PATH, "qwen": QWEN / "config.json",
        "attack_orientation": EXP188 / "configs/ATTACK_SCORE_ORIENTATION_FROZEN.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing frozen inputs: {missing}")
    manifest = pd.DataFrame([{"key": key, "path": str(path), "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path), "access": "READ_ONLY"} for key, path in required.items()])
    common.atomic_csv(manifest, ROOT / "provenance/FROZEN_INPUTS.csv")
    common.atomic_text(ROOT / "configs/PRECOMMIT.sha256", common.sha256_file(PRECOMMIT) + "  PRECOMMIT.json\n")
    common.atomic_json(ROOT / "audits/NO_TUNING_AUDIT.json", {
        "threshold_source": "Exp203", "strict_greater_than": True, "family_feature": False,
        "membership_feature": False, "target_rank_feature": False, "classifier": False,
        "lora": False, "mirabel_runtime": False, "request_rejection": False,
        "post_result_repair_forbidden": True, "fresh_blind": False})
    checkpoint("PREFLIGHT_COMPLETE", frozen_inputs=len(manifest))
    return config


def _unique_qll(frame, keys, name):
    rows = []
    for identity, cell in frame.groupby(keys, dropna=False, sort=False):
        if cell.dominance.nunique() != 1 or cell.qll_top1_source.astype(str).nunique() != 1:
            raise RuntimeError(f"{name} QLL ambiguity: {identity}")
        identity = identity if isinstance(identity, tuple) else (identity,)
        rows.append(dict(zip(keys, identity)) | {"dominance_lookup": float(cell.dominance.iloc[0]),
                    "top1_lookup": str(cell.qll_top1_source.iloc[0])})
    return pd.DataFrame(rows)


def load_policy_rows():
    if POLICY_PATH.exists():
        return pd.read_pickle(POLICY_PATH, compression="gzip")
    cases = pd.read_pickle(CASES_PATH, compression="gzip")
    cases = cases[cases.family.isin(FAMILIES)].copy()
    cases["query_sha256"] = cases["query"].map(query_hash)
    qmap = pd.read_csv(QLL_MAP_PATH, keep_default_na=False)
    cases = cases.merge(qmap, on="case_id", validate="one_to_one")
    cases["_row"] = np.arange(len(cases))
    cases["dominance"] = np.nan
    cases["qll_top1_source"] = ""

    first_mask = cases.qll_artifact.eq("EXP179_EXACT_QUERY_MEMBER_TARGET")
    q179 = pd.read_csv(QLL179_PATH, keep_default_na=False, low_memory=False,
        usecols=["attack_family", "member", "query_sha256", "target_document_id", "dominance", "qll_top1_source"])
    q179 = q179[q179.attack_family.isin(FAMILIES)]
    keys179 = ["attack_family", "member", "query_sha256", "target_document_id"]
    q179 = _unique_qll(q179, keys179, "Exp179")
    left = cases[first_mask].merge(q179, left_on=["family", "member", "query_sha256", "target_document_id"],
        right_on=keys179, how="left", validate="many_to_one")
    cases.loc[left._row, "dominance"] = left.dominance_lookup.to_numpy(float)
    cases.loc[left._row, "qll_top1_source"] = left.top1_lookup.astype(str).to_numpy()

    later_mask = cases.qll_artifact.eq("EXP181_EXACT_ROW_ID")
    q181 = pd.read_csv(QLL181_PATH, keep_default_na=False, low_memory=False,
        usecols=["row_id", "dominance", "qll_top1_source"])
    q181 = _unique_qll(q181, ["row_id"], "Exp181")
    later = cases[later_mask].merge(q181, on="row_id", how="left", validate="many_to_one")
    cases.loc[later._row, "dominance"] = later.dominance_lookup.to_numpy(float)
    cases.loc[later._row, "qll_top1_source"] = later.top1_lookup.astype(str).to_numpy()

    if cases.dominance.isna().any() or cases.qll_top1_source.eq("").any():
        raise RuntimeError("QLL coverage incomplete")
    source_match = cases.qll_top1_source.astype(str).eq(cases.dominant_source_id.astype(str))
    # Exp181 contains exact equal-QLL ties in a small MEntA subset.  Exp193
    # resolves those ties deterministically by retrieval rank; their dominance
    # is 0.25 and they can never cross the frozen threshold.  The selected
    # source below therefore always comes from the canonical Exp193 mapping.
    unsafe_mismatch = (~source_match) & cases.dominance.gt(THRESHOLD)
    if unsafe_mismatch.any():
        raise RuntimeError("triggered Exp179/181 QLL top1 disagrees with Exp193 provenance")
    common.atomic_json(ROOT / "audits/QLL_TIE_AUDIT.json", {
        "source_matches": int(source_match.sum()), "equal_qll_tie_mismatches": int((~source_match).sum()),
        "triggered_mismatches": int(unsafe_mismatch.sum()),
        "tie_break_source": "Exp193 deterministic mean-QLL, retrieval-rank, source-ID ordering"})
    cases["qll_top1_source"] = cases.dominant_source_id.astype(str)
    if not cases.qll_source_count.astype(int).eq(TOP_K).all():
        raise RuntimeError("top-4 QLL contract changed")
    cases["trigger"] = cases.dominance.gt(THRESHOLD)
    cases = cases.drop(columns=["_row"])

    local_rows = []
    for row in cases.itertuples(index=False):
        selected = str(row.qll_top1_source) if bool(row.trigger) else ""
        local_rows.append({"case_id": str(row.case_id), "condition": LOCAL,
                           "selected_source_id": selected, "ledger_source_id": "",
                           "ledger_active": False})
    sticky_rows = []
    ordered = cases.sort_values(["family", "session_id", "turn_order", "case_id"])
    for (_, _), cell in ordered.groupby(["family", "session_id"], sort=False):
        sticky = ""
        for row in cell.itertuples(index=False):
            if not sticky and bool(row.trigger):
                sticky = str(row.qll_top1_source)
            present = bool(sticky and sticky in list(map(str, row.source_ids[:TOP_K])))
            sticky_rows.append({"case_id": str(row.case_id), "condition": STICKY,
                                "selected_source_id": sticky if present else "",
                                "ledger_source_id": sticky, "ledger_active": bool(sticky)})
    policies = pd.DataFrame(local_rows + sticky_rows)
    output = cases.merge(policies, on="case_id", validate="one_to_many")
    output["intervened"] = output.selected_source_id.ne("")
    output["selected_source_rank"] = [list(map(str, ids[:TOP_K])).index(str(source)) + 1 if source else 0
        for ids, source in zip(output.source_ids, output.selected_source_id)]
    output.to_pickle(POLICY_PATH, compression="gzip")
    audit = output.groupby(["condition", "family"], as_index=False).agg(
        rows=("case_id", "size"), sessions=("session_id", "nunique"),
        trigger_rate=("trigger", "mean"), intervention_rate=("intervened", "mean"),
        ledger_active_rate=("ledger_active", "mean"))
    common.atomic_csv(audit, ROOT / "tables/TABLE_204_01_POLICY_AUDIT.csv")
    if len(cases) != 10480 or cases.case_id.nunique() != 10480:
        raise RuntimeError(f"frozen expansion cohort mismatch: {len(cases)}")
    checkpoint("POLICY_AUDIT_COMPLETE", cases=len(cases), sessions=cases.session_id.nunique(),
               expanded_rows=len(output), local_interventions=int(output[output.condition.eq(LOCAL)].intervened.sum()),
               sticky_interventions=int(output[output.condition.eq(STICKY)].intervened.sum()))
    return output


def waterfill(lengths, selected=None):
    lengths = np.asarray(lengths, dtype=int)
    caps = np.zeros(len(lengths), dtype=int)
    fixed = set()
    if selected is not None:
        caps[int(selected)] = 0
        fixed.add(int(selected))
    remaining = TOTAL_BUDGET
    active = [i for i in range(len(lengths)) if i not in fixed and lengths[i] > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, int(lengths[index] - caps[index]), remaining)
            if add > 0:
                caps[index] += add; remaining -= add; changed = True
            if caps[index] >= lengths[index]:
                active.remove(index)
            if remaining <= 0:
                break
        if not changed:
            break
    return caps.tolist()


def build_packing(policy):
    if PACKING_PATH.exists():
        return pd.read_pickle(PACKING_PATH, compression="gzip")
    from transformers import AutoTokenizer
    sys.path.insert(0, str(EXP87 / "code"))
    import exp87_models as models
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    documents = common.load_exp198().all_documents()
    needed = {(str(row.domain), str(source)) for row in policy.itertuples(index=False)
              for source in list(row.source_ids)[:TOP_K]}
    missing = sorted(needed - set(documents))
    common.atomic_json(ROOT / "audits/SOURCE_TEXT_COVERAGE.json", {
        "needed": len(needed), "resolved": len(needed)-len(missing),
        "coverage": (len(needed)-len(missing))/len(needed), "missing": missing[:100]})
    if missing:
        raise RuntimeError(f"source text coverage incomplete: {len(missing)}")
    rows = []
    for number, row in enumerate(policy.itertuples(index=False), 1):
        source_ids = list(map(str, row.source_ids[:TOP_K]))
        texts = [str(documents[(str(row.domain), source)]) for source in source_ids]
        lengths = [len(tokenizer(text, add_special_tokens=False).input_ids) for text in texts]
        selected = source_ids.index(str(row.selected_source_id)) if row.selected_source_id else None
        caps = waterfill(lengths, selected)
        contexts, used, used_ids = [], [], []
        for source, text, cap in zip(source_ids, texts, caps):
            token_ids = tokenizer(text, add_special_tokens=False).input_ids[:int(cap)]
            used.append(len(token_ids))
            if token_ids:
                contexts.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
                used_ids.append(source)
        prompt = models.normal_prompt(str(row.query), contexts)
        maximum = int(row.max_new_tokens)
        generation_key = common.sha256_text(json.dumps({"prompt": prompt, "system": str(row.system_prompt),
                                                         "max_new_tokens": maximum}, sort_keys=True))
        rows.append({"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": str(row.panel),
            "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
            "session_id": str(row.session_id), "turn_order": int(row.turn_order),
            "target_document_id": str(row.target_document_id), "target_rank": int(row.target_rank),
            "query": str(row.query), "condition": str(row.condition), "dominance": float(row.dominance),
            "trigger": bool(row.trigger), "intervened": bool(row.intervened),
            "selected_source_id": str(row.selected_source_id), "selected_source_rank": int(row.selected_source_rank),
            "ledger_source_id": str(row.ledger_source_id), "ledger_active": bool(row.ledger_active),
            "source_ids_retrieved": source_ids, "source_ids_used": used_ids, "caps": caps,
            "tokens_used": used, "total_context_tokens": int(sum(used)), "system_prompt": str(row.system_prompt),
            "reference": str(getattr(row, "reference", "")), "budget": int(getattr(row, "budget", maximum)),
            "max_new_tokens": maximum, "prompt": prompt, "generation_key": generation_key,
            "request_blocked": False})
        if number % 2500 == 0:
            checkpoint("PACKING_PROGRESS", completed=number, total=len(policy))
    output = pd.DataFrame(rows)
    output.to_pickle(PACKING_PATH, compression="gzip")
    summary = output.groupby(["condition", "family"], as_index=False).agg(
        rows=("case_id", "size"), unique_tasks=("generation_key", "nunique"),
        intervention_rate=("intervened", "mean"), mean_context_tokens=("total_context_tokens", "mean"),
        request_block_rate=("request_blocked", "mean"))
    common.atomic_csv(summary, ROOT / "tables/TABLE_204_02_PACKING.csv")
    checkpoint("PACKING_COMPLETE", rows=len(output), unique_tasks=output.generation_key.nunique())
    return output


def task_signature(task):
    return json.dumps({key: task.get(key) for key in ("generator", "generator_revision", "system_prompt_hash",
        "prompt_hash", "generation_config", "max_new_tokens")}, sort_keys=True, separators=(",", ":"))


def seed_exact_cache(tasks, models):
    sources = [
        EXP203 / "private/EXP203_QWEN_RESPONSES.sqlite3",
        EXP193 / "private/EXP193_TOP4_RESPONSES.sqlite3",
        PROJECT / "exp198_se_mirabel_soft_exposure_20260830/private/EXP198_QWEN_RESPONSES.sqlite3",
        PROJECT / "exp197_rank_agnostic_cap64_backfill_20260829/private/EXP197_QWEN_RESPONSES.sqlite3",
        PROJECT / "exp176_prefix64_mirabel_external_attacks_20260828/private/EXP175_RESPONSES.sqlite3",
        PROJECT / "exp174_stable_prefix64_native_sessions_20260828/private/EXP174_DCMI_IA_GENERATIONS.sqlite3",
        PROJECT / "exp158_global_cap64_native_sessions_20260826/private/EXP158_GENERATIONS.sqlite3",
        PROJECT / "exp157_global_cap64_menta5_20260826/private/EXP157_GENERATIONS.sqlite3",
    ]
    frozen = {}; source_rows = []
    for source in sources:
        if not source.exists():
            continue
        connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        count = 0
        for task_json, response in connection.execute("SELECT task_json,response_text FROM responses"):
            task = json.loads(task_json)
            frozen.setdefault(task_signature(task), (str(response), str(source)))
            count += 1
        connection.close()
        source_rows.append({"path": str(source), "rows": count, "sha256": common.sha256_file(source), "access": "READ_ONLY"})
    store = models.ResponseStore(GEN_DB); reused = 0; already = 0
    for task in tasks:
        if store.get(str(task["task_key"])) is not None:
            already += 1; continue
        match = frozen.get(task_signature(task))
        if match is not None:
            response, source = match
            store.put(task, response, f"EXACT_TASK_SIGNATURE:{source}"); reused += 1
    store.close()
    common.atomic_csv(pd.DataFrame(source_rows), ROOT / "provenance/READ_ONLY_GENERATION_CACHES.csv")
    common.atomic_json(ROOT / "audits/EXACT_TASK_CACHE_AUDIT.json", {
        "tasks": len(tasks), "already_present": already, "reused": reused,
        "matched_fields": ["generator", "generator_revision", "system_prompt_hash", "prompt_hash",
                           "generation_config", "max_new_tokens"]})
    return reused


def generate(packing):
    if RESPONSES_PATH.exists():
        return pd.read_csv(RESPONSES_PATH, keep_default_na=False, low_memory=False)
    sys.path.insert(0, str(EXP87 / "code"))
    import exp87_models as models
    models.ROOT = ROOT
    models.heartbeat = lambda stage, **details: checkpoint(stage, **details)
    models.GENERATION_CONFIG["batch_size"] = 16
    unique = packing.drop_duplicates("generation_key")
    tasks = [models.make_task(task_type="EXP204_FROZEN_QLL_HIDE", row_id=str(row.generation_key),
        prompt=str(row.prompt), system_prompt=str(row.system_prompt), max_new_tokens=int(row.max_new_tokens))
        for row in unique.itertuples(index=False)]
    reused = seed_exact_cache(tasks, models)
    checkpoint("GENERATION_STARTED", unique_tasks=len(tasks), expanded_rows=len(packing),
               exact_cache_reused=reused, device="cuda:0")
    started = time.perf_counter()
    answers = models.run_generation(tasks, GEN_DB, None, "GENERATION_PROGRESS")
    output = packing.drop(columns=["prompt"]).copy()
    output["response"] = output.generation_key.map(answers)
    if output.response.isna().any():
        raise RuntimeError("generation incomplete")
    common.atomic_csv(output, RESPONSES_PATH, "gzip")
    checkpoint("GENERATION_COMPLETE", expanded_rows=len(output), unique_tasks=len(tasks),
               wall_seconds=time.perf_counter()-started)
    return output


def effective(value):
    value = float(value)
    return max(value, 1.0-value)


def _normalize_native_manifest(frame, single_query=False):
    """Restore the immutable Exp87 scorer contract omitted by later manifests."""
    frame = frame.copy()
    if "purpose" not in frame:
        frame["purpose"] = "PRIVACY_EVAL"
    if "kind" not in frame:
        frame["kind"] = "attack"
    if "source_document_id" not in frame:
        frame["source_document_id"] = frame.target_document_id.astype(str)
    if "turn" not in frame:
        frame["turn"] = 1 if single_query else frame["selected_turn"]
    frame["exp87_row_id"] = frame["row_id"].astype(str)
    return frame


def score_native_fixed(responses, evaluator):
    """Run original scorers after schema-only normalization; attack semantics stay frozen."""
    destination = ROOT / "private/EXP204_NATIVE_SCORES.private.csv.gz"
    if destination.exists():
        output = pd.read_csv(destination, low_memory=False)
        output["attack_score"] = pd.to_numeric(output["attack_score"], errors="coerce")
        return output
    clean = pd.read_csv(evaluator.EXP152 / "private/EXP152_CLEAN_3K_MANIFEST.private.csv.gz",
                        keep_default_na=False, low_memory=False, dtype={"row_id": str})
    clean = _normalize_native_manifest(clean[clean.family.isin(["RAG-MIA", "S²-MIA", "MBA"])], True)
    native = _normalize_native_manifest(pd.read_csv(
        evaluator.EXP153 / "private/EXP153_NATIVE_TURNS.private.csv.gz",
        keep_default_na=False, low_memory=False, dtype={"row_id": str}))
    menta = _normalize_native_manifest(pd.read_csv(
        evaluator.EXP153 / "menta5_followup/private/MENTA5_NATIVE_TURNS.private.csv.gz",
        keep_default_na=False, low_memory=False, dtype={"row_id": str}))
    required = {(str(row.domain), str(row.target_document_id)) for frame in (clean, native, menta)
                for row in frame.itertuples(index=False)}
    documents = evaluator._native_documents(required)
    truth = pd.read_csv(evaluator.EXP158 / "private/EXP158_IA_GROUND_TRUTH.private.csv.gz",
                        keep_default_na=False, dtype={"row_id": str}).set_index("row_id").response.to_dict()
    pieces = []
    scorer_path = EXP87 / "code/exp87_scoring.py"
    for suite, panel, frame, local_truth in (("Q1", "NATIVE_Q1", clean, {}),
            ("DCMI_IA", "NATIVE_SESSION", native, truth), ("MENTA", "NATIVE_MENTA", menta, {})):
        scorer = load(scorer_path, f"exp204_scorer_{suite}")
        scorer.ROOT = ROOT
        scorer.RESPONSE_DB = ROOT / f"private/EXP204_{suite}_SCORER.sqlite3"
        scorer.atomic_csv = common.atomic_csv
        scorer.atomic_json = common.atomic_json
        scorer.checkpoint = lambda stage, suite=suite, **details: checkpoint(f"SCORER_{suite}_{stage}", **details)
        shared = ROOT / "private/EXP87_MIA_SCORES.private.csv.gz"
        if shared.exists():
            shared.unlink()
        supplied = responses[responses.panel.eq(panel)].rename(columns={"row_id": "exp87_row_id"})[[
            "exp87_row_id", "condition", "response"]].assign(
                A_R=lambda x: x.response, action="EXP204_FROZEN_QLL_HIDE", lola_alarm=False,
                lola_score=np.nan, beta=np.nan)
        checkpoint(f"NATIVE_{suite}_SCORING_STARTED", turns=len(frame), sessions=frame.session_id.nunique(),
                   conditions=supplied.condition.nunique())
        measured = scorer.mia_scores(frame, supplied, local_truth, documents)
        measured["suite"] = suite
        pieces.append(measured)
    output = pd.concat(pieces, ignore_index=True)
    common.atomic_csv(output, destination, "gzip")
    checkpoint("NATIVE_SCORING_COMPLETE", rows=len(output), invalid=int(output.attack_score.isna().sum()))
    return output


def evaluate(responses, config):
    module = load(EXP188 / "code/run_exp188.py", "exp204_scorer")
    module.ROOT = ROOT
    native_input = responses[responses.family.isin(NATIVE)].copy()
    native_scores = score_native_fixed(native_input, module)
    native_auc, native_low, _, _ = module.evaluate_native(native_scores)
    native_auc["effective_auc"] = native_auc.roc_auc.map(effective)
    common.atomic_csv(native_auc, ROOT / "tables/TABLE_204_03_NATIVE_EAUC.csv")

    rag_responses = responses[responses.family.eq("RAGLeak")].copy()
    rag_cases = rag_responses[["case_id", "reference", "budget", "panel"]].drop_duplicates("case_id")
    rag_response_input = rag_responses.drop(columns=["reference", "budget"])
    rag_metrics = module.external_response_metrics(rag_response_input, rag_cases)
    rag_scores = module.external_attack_scores(rag_metrics)
    rag_auc, rag_low, _, _, _ = module.evaluate_external(rag_scores)
    rag_auc["effective_auc"] = rag_auc.roc_auc.map(effective)
    common.atomic_csv(rag_auc, ROOT / "tables/TABLE_204_04_RAGLEAK_EAUC.csv")

    exp203_privacy = pd.read_csv(EXP203 / "tables/TABLE_203_05_BUDGETLEAK.csv")
    exp203_utility = pd.read_csv(EXP203 / "tables/TABLE_203_06_NORMAL_UTILITY.csv")
    common.atomic_csv(exp203_privacy, ROOT / "tables/TABLE_204_05_BUDGETLEAK_FROZEN_ANCHOR.csv")
    common.atomic_csv(exp203_utility, ROOT / "tables/TABLE_204_06_NORMAL_UTILITY_FROZEN_ANCHOR.csv")
    parent = json.loads((EXP203 / "FINAL_RESULT.json").read_text())

    primary_native = native_auc[native_auc.condition.eq(STICKY)]
    primary_rag = rag_auc[rag_auc.condition.eq(STICKY)]
    gates_cfg = config["gates"]
    gate_rows = []
    for row in primary_native.itertuples(index=False):
        gate_rows.append({"gate": f"native_{row.attack_family}", "value": float(row.effective_auc),
                          "criterion": f"<={gates_cfg['each_native_family_effective_auc_max']}",
                          "passed": float(row.effective_auc) <= gates_cfg["each_native_family_effective_auc_max"]})
    rag_value = float(primary_rag.effective_auc.iloc[0])
    gate_rows += [
        {"gate": "ragleak", "value": rag_value, "criterion": f"<={gates_cfg['ragleak_effective_auc_max']}",
         "passed": rag_value <= gates_cfg["ragleak_effective_auc_max"]},
        {"gate": "budgetleak_aggregate_anchor", "value": float(parent["budgetleak_aggregate_effective_auc"]),
         "criterion": f"<={gates_cfg['budgetleak_aggregate_effective_auc_max']}",
         "passed": float(parent["budgetleak_aggregate_effective_auc"]) <= gates_cfg["budgetleak_aggregate_effective_auc_max"]},
        {"gate": "budgetleak_worst_rank_anchor", "value": float(parent["budgetleak_worst_rank_effective_auc"]),
         "criterion": f"<={gates_cfg['budgetleak_worst_rank_effective_auc_max']}",
         "passed": float(parent["budgetleak_worst_rank_effective_auc"]) <= gates_cfg["budgetleak_worst_rank_effective_auc_max"]},
        {"gate": "normal_f1_retention_anchor", "value": float(parent["normal_f1_retention"]),
         "criterion": f">={gates_cfg['normal_token_f1_retention_min']}",
         "passed": float(parent["normal_f1_retention"]) >= gates_cfg["normal_token_f1_retention_min"]},
        {"gate": "normal_new_refusal_anchor", "value": float(parent["normal_new_refusal_rate"]),
         "criterion": f"<={gates_cfg['normal_new_refusal_rate_max']}",
         "passed": float(parent["normal_new_refusal_rate"]) <= gates_cfg["normal_new_refusal_rate_max"]},
        {"gate": "normal_intervention_anchor", "value": float(parent["normal_intervention_rate"]),
         "criterion": f"<={gates_cfg['normal_intervention_rate_max']}",
         "passed": float(parent["normal_intervention_rate"]) <= gates_cfg["normal_intervention_rate_max"]},
        {"gate": "request_block", "value": 0.0, "criterion": "==0", "passed": True},
    ]
    gates = pd.DataFrame(gate_rows)
    common.atomic_csv(gates, ROOT / "tables/TABLE_204_07_PRIMARY_GATES.csv")
    passed = bool(gates.passed.all())
    verdict = "QLL_SOURCE_HIDE_MULTIFAMILY_PASS" if passed else "QLL_SOURCE_HIDE_MULTIFAMILY_FAIL"

    review = responses[responses.intervened].copy()
    review = review[["family", "condition", "member", "session_id", "turn_order", "query",
                     "dominance", "selected_source_id", "selected_source_rank", "response"]]
    common.atomic_csv(review, ROOT / "query_response_intervention_review.csv")
    result = {"experiment": "Exp204", "verdict": verdict, "passed": passed,
        "primary_condition": STICKY, "threshold": THRESHOLD, "native_families": len(primary_native),
        "native_worst_effective_auc": float(primary_native.effective_auc.max()),
        "ragleak_effective_auc": rag_value,
        "budgetleak_aggregate_effective_auc_frozen_anchor": float(parent["budgetleak_aggregate_effective_auc"]),
        "budgetleak_worst_rank_effective_auc_frozen_anchor": float(parent["budgetleak_worst_rank_effective_auc"]),
        "normal_f1_retention_frozen_anchor": float(parent["normal_f1_retention"]),
        "normal_intervention_rate_frozen_anchor": float(parent["normal_intervention_rate"]),
        "normal_new_refusal_rate_frozen_anchor": float(parent["normal_new_refusal_rate"]),
        "request_block_rate": 0.0, "fresh_blind": False, "completed_utc": now()}
    common.atomic_json(ROOT / "FINAL_RESULT.json", result)
    report = f"""# Exp204 — QLL Source Hide 다중 공격 확장

- 최종 판정: **{verdict}**
- 주 조건: `{STICKY}`
- Exp203의 QLL, threshold({THRESHOLD}), strict `>`, source 전체 제외, 2048-token water-fill을 수정하지 않았다.
- 공격 family, membership label, target rank는 정책 입력이 아니며 평가에만 사용했다.
- DCMI Q2, MEntA Q5, IA Q15는 원본 turn 순서와 원본 공격 scorer를 유지했다.
- 정상 utility와 BudgetLeak 수치는 동일 정책의 동결 Exp203 anchor이며 이번 결과와 섞어 재보정하지 않았다.
- 이 실험은 development cohort 확장이며 fresh blind가 아니다.

## Native six attacks

{native_auc.to_markdown(index=False)}

## RAGLeak

{rag_auc.to_markdown(index=False)}

## Primary gates

{gates.to_markdown(index=False)}
"""
    common.atomic_text(ROOT / "reports/REPORT_204_FINAL_KO.md", report)
    checkpoint("COMPLETE", verdict=verdict, passed=passed,
               native_worst_effective_auc=result["native_worst_effective_auc"], ragleak_effective_auc=rag_value)
    return result


def main():
    config = preflight()
    policy = load_policy_rows()
    packing = build_packing(policy)
    responses = generate(packing)
    result = evaluate(responses, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
