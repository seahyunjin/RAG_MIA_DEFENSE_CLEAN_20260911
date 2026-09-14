#!/usr/bin/env python3
"""Exp199: query/rank-invariant source views with a fixed response budget."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time

import numpy as np
import pandas as pd


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp199_stable_evidence_budget_invariant_20260830"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP166 = PROJECT / "exp166_topiocqa_gold_utility_20260827"
EXP193 = PROJECT / "exp193_top4_context_rebase_20260829"
EXP195 = PROJECT / "exp195_minimal_qll_exposure_guard_20260829"
EXP196 = PROJECT / "exp196_selective_rank_agnostic_globalcap64_20260829"
EXP197 = PROJECT / "exp197_rank_agnostic_cap64_backfill_20260829"
EXP198 = PROJECT / "exp198_se_mirabel_soft_exposure_20260830"
PRECOMMIT = ROOT / "configs/PRECOMMIT.json"
ATTACK_CASES = EXP193 / "private/EXP193_CASES.private.pkl.gz"
NORMAL_CASES = EXP195 / "private/EXP195_NORMAL_CASES.private.pkl.gz"
RISK = EXP198 / "private/EXP198_MIRABEL_RISK.private.pkl.gz"
VIEWS = ROOT / "private/EXP199_STABLE_SOURCE_VIEWS.private.pkl.gz"
PACKING = ROOT / "private/EXP199_PACKING.private.pkl.gz"
RESPONSES = ROOT / "private/EXP199_RESPONSES.private.csv.gz"
GEN_DB = ROOT / "private/EXP199_QWEN_RESPONSES.sqlite3"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
CANDIDATE = "BIC_SEV64_MIRABEL"
PREFIX_BASELINE = "PREFIX64_MIRABEL_FIXED64"
SOURCE_BUDGET = 64
OUTPUT_BUDGET = 64
TOP_K = 10
MAX_RENDERED = 3072


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


def atomic_pickle(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".pkl.gz", dir=path.parent); os.close(fd)
    try:
        frame.to_pickle(temporary, compression="gzip"); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def checkpoint(stage, **details):
    payload = {"experiment": "Exp199", "candidate": CANDIDATE, "stage": stage,
        "updated_utc": now(), "pid": os.getpid(), "trainable_parameters": 0,
        "normal_calibration": False, "attack_specific_tuning": False,
        "rank_specific_tuning": False, "paid_api_calls": 0, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp199 Status", "", f"- Stage: **{stage}**", f"- Updated UTC: `{payload['updated_utc']}`",
             f"- PID: `{payload['pid']}`"] + [f"- {key}: `{value}`" for key, value in details.items()]
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def load_exp198():
    path = EXP198 / "code/run_exp198.py"
    spec = importlib.util.spec_from_file_location("exp199_exp198_inputs", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    return module


def preflight():
    required = {
        "precommit": PRECOMMIT, "attack_cases": ATTACK_CASES, "normal_cases": NORMAL_CASES,
        "mirabel_risk": RISK, "exp198_result": EXP198 / "FINAL_RESULT.json",
        "exp176_result": PROJECT / "exp176_prefix64_mirabel_external_attacks_20260828/FINAL_RESULT.json",
        "gold": EXP166 / "private/TOPIOCQA_GOLD_1000.private.csv.gz",
        "qwen": QWEN / "config.json", "mpnet": MPNET / "config.json"
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing: raise RuntimeError(f"missing frozen input: {missing}")
    config = json.loads(PRECOMMIT.read_text())
    if config["canonical_source_view"]["token_budget_per_source"] != SOURCE_BUDGET: raise RuntimeError("source budget mismatch")
    if config["budget_invariant_release"]["server_max_new_tokens"] != OUTPUT_BUDGET: raise RuntimeError("output budget mismatch")
    manifest = pd.DataFrame([{"key": key, "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path), "access": "READ_ONLY"} for key, path in required.items()])
    atomic_csv(manifest, ROOT / "provenance/FROZEN_INPUTS.csv")
    atomic_json(ROOT / "audits/NO_TUNING_AUDIT.json", {
        "source_budget": SOURCE_BUDGET, "output_budget": OUTPUT_BUDGET,
        "caller_budget_used": False, "query_used_to_construct_source_view": False,
        "rank_used_to_construct_source_view": False, "membership_used_by_policy": False,
        "attack_family_used_by_policy": False, "learned_parameters": 0,
        "request_rejection": False, "raw_document_backfill": False
    })
    checkpoint("PREFLIGHT_COMPLETE", frozen_inputs=len(manifest), precommit_sha256=sha256_file(PRECOMMIT))
    return config


def load_cases():
    attack = pd.read_pickle(ATTACK_CASES, compression="gzip").copy()
    normal = pd.read_pickle(NORMAL_CASES, compression="gzip").copy()
    attack["requested_max_new_tokens"] = attack.max_new_tokens.astype(int)
    normal["requested_max_new_tokens"] = 96
    attack["split"] = "ATTACK"; normal["split"] = "NORMAL"
    common = sorted(set(attack.columns) & set(normal.columns))
    cases = pd.concat([attack[common], normal[common]], ignore_index=True, sort=False)
    risk = pd.read_pickle(RISK, compression="gzip")
    cases = cases.drop(columns=[column for column in ("source_ids", "source_scores", "mirabel_threshold")
                                if column in cases.columns])
    cases = cases.merge(risk[["case_id", "source_ids", "source_scores", "mirabel_threshold"]],
                        on="case_id", validate="one_to_one")
    return attack, normal, cases, risk


def split_units(text, tokenizer):
    sentences = [re.sub(r"\s+", " ", item).strip() for item in re.split(r"(?<=[.!?])\s+|\n+", str(text))]
    units = []
    for sentence in sentences:
        if not sentence: continue
        ids = tokenizer(sentence, add_special_tokens=False).input_ids
        for offset in range(0, len(ids), 32):
            value = tokenizer.decode(ids[offset:offset + 32], skip_special_tokens=True).strip()
            if value: units.append(value)
    return units or [str(text).strip()]


def facility_view(units, embeddings, tokenizer, budget=SOURCE_BUDGET):
    token_ids = [tokenizer(value, add_special_tokens=False).input_ids for value in units]
    costs = np.asarray([max(1, len(value)) for value in token_ids], dtype=int)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    similarity = np.maximum(0.0, embeddings @ embeddings.T)
    covered = np.zeros(len(units), dtype=np.float32); selected = []; remaining = int(budget)
    while True:
        eligible = [index for index, cost in enumerate(costs) if index not in selected and cost <= remaining]
        if not eligible: break
        scored = []
        for index in eligible:
            gain = float(np.maximum(covered, similarity[:, index]).sum() - covered.sum())
            scored.append((gain / float(costs[index]), gain, -index, index))
        _, gain, _, chosen = max(scored)
        if selected and gain <= 1e-12: break
        selected.append(chosen); covered = np.maximum(covered, similarity[:, chosen]); remaining -= int(costs[chosen])
    if not selected:
        selected = [0]
    selected = sorted(selected)
    ids = []
    for index in selected:
        ids.extend(token_ids[index])
    ids = ids[:budget]
    return tokenizer.decode(ids, skip_special_tokens=True).strip(), selected, len(ids), float(covered.mean())


def build_views(cases):
    if VIEWS.exists():
        frame = pd.read_pickle(VIEWS, compression="gzip")
        if len(frame) and frame.view_budget.eq(SOURCE_BUDGET).all(): return frame
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    import torch
    common = load_exp198(); documents = common.all_documents()
    needed = sorted({(str(row.domain), str(source)) for row in cases.itertuples(index=False)
                     for source in list(row.source_ids)[:TOP_K]})
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    records = []; all_units = []; slices = []
    for number, key in enumerate(needed, 1):
        if key not in documents: raise RuntimeError(f"missing source text: {key}")
        units = split_units(documents[key], tokenizer); start = len(all_units); all_units.extend(units)
        slices.append((key, units, start, len(all_units), documents[key]))
        if number % 1000 == 0: checkpoint("VIEW_TOKENIZATION_PROGRESS", documents=number, total=len(needed), units=len(all_units))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(str(MPNET), device=device, local_files_only=True); encoder.max_seq_length = 128
    matrix = encoder.encode(all_units, normalize_embeddings=True, convert_to_numpy=True,
                            batch_size=256, show_progress_bar=False).astype(np.float32)
    for number, (key, units, start, stop, original) in enumerate(slices, 1):
        view, selected, used, coverage = facility_view(units, matrix[start:stop], tokenizer)
        prefix_ids = tokenizer(str(original), add_special_tokens=False).input_ids[:SOURCE_BUDGET]
        prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True).strip()
        records.append({"domain": key[0], "source_id": key[1], "stable_view": view,
            "stable_view_sha256": sha256_text(view), "prefix64": prefix,
            "view_budget": SOURCE_BUDGET, "view_tokens": used, "unit_count": len(units),
            "selected_unit_indices": selected, "facility_coverage": coverage})
        if number % 1000 == 0: checkpoint("VIEW_SELECTION_PROGRESS", documents=number, total=len(needed))
    output = pd.DataFrame(records); atomic_pickle(output, VIEWS)
    summary = output.agg(documents=("source_id", "size"), mean_view_tokens=("view_tokens", "mean"),
                         median_view_tokens=("view_tokens", "median"), mean_units=("unit_count", "mean")).reset_index(drop=True)
    atomic_csv(summary, ROOT / "tables/TABLE_199_01_VIEW_SUMMARY.csv")
    del encoder, matrix; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    checkpoint("VIEWS_COMPLETE", documents=len(output), units=len(all_units), device=device,
               mean_view_tokens=float(output.view_tokens.mean()))
    return output


def answer_tokens(value):
    return re.findall(r"[a-z0-9]+", str(value).casefold())


def token_recall(answer, text):
    target = Counter(answer_tokens(answer)); visible = Counter(answer_tokens(text))
    if not target: return 0.0
    return sum((target & visible).values()) / sum(target.values())


def exact_contains(answer, text):
    a = " ".join(answer_tokens(answer)); t = " ".join(answer_tokens(text))
    return bool(a) and a in t


def visibility(row, view_lookup, kind):
    sources = list(map(str, row.source_ids[:TOP_K])); scores = np.asarray(row.source_scores, dtype=float)
    hidden = int(np.argmax(scores)) if float(scores.max()) > float(row.mirabel_threshold) else -1
    texts = []
    for index, source in enumerate(sources):
        if index == hidden: continue
        texts.append(view_lookup[(str(row.domain), source)][kind])
    return sources, hidden, texts


def phase_a(normal, attack, risk, views, config):
    table_path = ROOT / "tables/TABLE_199_02_PHASE_A_EVIDENCE.csv"
    gate_path = ROOT / "tables/TABLE_199_03_PHASE_A_GATES.csv"
    if gate_path.exists():
        gates = pd.read_csv(gate_path); return bool(gates.passed.all())
    lookup = {(str(row.domain), str(row.source_id)): {"stable": str(row.stable_view), "prefix": str(row.prefix64)}
              for row in views.itertuples(index=False)}
    risk_index = risk.set_index("case_id")
    gold = pd.read_csv(EXP166 / "private/TOPIOCQA_GOLD_1000.private.csv.gz", keep_default_na=False, dtype={"row_id": str})
    gold_map = gold.set_index("row_id")
    details = []
    for row in normal.itertuples(index=False):
        rr = risk_index.loc[str(row.case_id)]; sources, hidden, _ = visibility(rr, lookup, "stable")
        meta = gold_map.loc[str(row.row_id)]; target = str(meta.gold_document_id); answers = json.loads(meta.gold_answers)
        for condition, kind in [("PREFIX64_MIRABEL", "prefix"), (CANDIDATE, "stable")]:
            text = "" if target not in sources or sources.index(target) == hidden else lookup[("TopiOCQA", target)][kind]
            details.append({"panel": "NORMAL_GOLD", "row_id": str(row.row_id), "condition": condition,
                "target_rank": sources.index(target) + 1 if target in sources else 0,
                "hidden_target": target in sources and sources.index(target) == hidden,
                "token_recall": max(token_recall(answer, text) for answer in answers),
                "exact_containment": max(exact_contains(answer, text) for answer in answers)})
    budget = attack[(attack.family == "BudgetLeak-Z") & (attack.member == 1) & attack.target_rank.between(1, 4)].drop_duplicates("row_id")
    for row in budget.itertuples(index=False):
        rr = risk_index.loc[str(row.case_id)]; sources, hidden, _ = visibility(rr, lookup, "stable"); target = str(row.target_document_id)
        for condition, kind in [("PREFIX64_MIRABEL", "prefix"), (CANDIDATE, "stable")]:
            text = "" if target not in sources or sources.index(target) == hidden else lookup[(str(row.domain), target)][kind]
            details.append({"panel": "BUDGET_MEMBER_TARGET", "row_id": str(row.row_id), "condition": condition,
                "target_rank": int(row.target_rank), "hidden_target": target in sources and sources.index(target) == hidden,
                "token_recall": token_recall(str(row.reference), text), "exact_containment": exact_contains(str(row.reference), text)})
    detail = pd.DataFrame(details); atomic_csv(detail, ROOT / "private/EXP199_PHASE_A_DETAIL.private.csv.gz", "gzip")
    summary = detail.groupby(["panel", "condition"], as_index=False).agg(rows=("row_id", "size"),
        mean_token_recall=("token_recall", "mean"), exact_containment_rate=("exact_containment", "mean"),
        target_hidden_rate=("hidden_target", "mean"))
    atomic_csv(summary, table_path)
    values = summary.set_index(["panel", "condition"])
    normal_recall_delta = float(values.loc[("NORMAL_GOLD", CANDIDATE), "mean_token_recall"] - values.loc[("NORMAL_GOLD", "PREFIX64_MIRABEL"), "mean_token_recall"])
    normal_exact_delta = float(values.loc[("NORMAL_GOLD", CANDIDATE), "exact_containment_rate"] - values.loc[("NORMAL_GOLD", "PREFIX64_MIRABEL"), "exact_containment_rate"])
    attack_recall_delta = float(values.loc[("BUDGET_MEMBER_TARGET", CANDIDATE), "mean_token_recall"] - values.loc[("BUDGET_MEMBER_TARGET", "PREFIX64_MIRABEL"), "mean_token_recall"])
    unique_hashes = views.groupby(["domain", "source_id"]).stable_view_sha256.nunique().max()
    settings = config["phase_a_no_generation_gates"]
    gates = pd.DataFrame([
        {"gate": "normal_gold_recall_delta", "value": normal_recall_delta,
         "criterion": f">={settings['normal_gold_target_mean_token_recall_delta_vs_prefix64_mirabel_min']}",
         "passed": normal_recall_delta >= settings["normal_gold_target_mean_token_recall_delta_vs_prefix64_mirabel_min"]},
        {"gate": "normal_exact_containment_delta", "value": normal_exact_delta,
         "criterion": f">={settings['normal_gold_target_exact_containment_delta_vs_prefix64_mirabel_min']}",
         "passed": normal_exact_delta >= settings["normal_gold_target_exact_containment_delta_vs_prefix64_mirabel_min"]},
        {"gate": "budget_target_reference_recall_delta", "value": attack_recall_delta,
         "criterion": f"<={settings['budget_member_target_mean_reference_recall_delta_vs_prefix64_mirabel_max']}",
         "passed": attack_recall_delta <= settings["budget_member_target_mean_reference_recall_delta_vs_prefix64_mirabel_max"]},
        {"gate": "source_view_hash_invariance", "value": float(unique_hashes == 1), "criterion": "==1.0",
         "passed": bool(unique_hashes == 1)}])
    atomic_csv(gates, gate_path); passed = bool(gates.passed.all())
    atomic_json(ROOT / "audits/PHASE_A_DECISION.json", {"passed": passed,
        "normal_gold_recall_delta": normal_recall_delta, "normal_exact_delta": normal_exact_delta,
        "budget_target_reference_recall_delta": attack_recall_delta, "source_hash_max_multiplicity": int(unique_hashes)})
    checkpoint("PHASE_A_PASS" if passed else "PHASE_A_FAILED", passed=passed,
        normal_gold_recall_delta=normal_recall_delta, normal_exact_delta=normal_exact_delta,
        budget_target_reference_recall_delta=attack_recall_delta)
    return passed


def render_length(tokenizer, models, query, contexts):
    prompt = models.normal_prompt(str(query), list(contexts))
    messages = [{"role": "system", "content": models.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, len(tokenizer(rendered, add_special_tokens=False).input_ids)


def build_packing(cases, views):
    if PACKING.exists():
        old = pd.read_pickle(PACKING, compression="gzip")
        if len(old) == len(cases) + int((cases.panel == "NORMAL_GOLD").sum()): return old
    from transformers import AutoTokenizer
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    lookup = {(str(row.domain), str(row.source_id)): {"stable": str(row.stable_view), "prefix": str(row.prefix64)}
              for row in views.itertuples(index=False)}
    rows = []
    for number, row in enumerate(cases.itertuples(index=False), 1):
        sources = list(map(str, row.source_ids[:TOP_K])); scores = np.asarray(row.source_scores, dtype=float)
        hidden = int(np.argmax(scores)) if float(scores.max()) > float(row.mirabel_threshold) else -1
        visible_sources = [source for index, source in enumerate(sources) if index != hidden]
        variants = [(CANDIDATE, "stable")]
        if str(row.panel) == "NORMAL_GOLD": variants.append((PREFIX_BASELINE, "prefix"))
        for condition, view_kind in variants:
            active_sources = list(visible_sources)
            contexts = [lookup[(str(row.domain), source)][view_kind] for source in active_sources]
            prompt, length = render_length(tokenizer, models, str(row.query), contexts)
            query_only_overflow = False
            if length > MAX_RENDERED:
                empty, empty_length = render_length(tokenizer, models, str(row.query), [])
                if empty_length > MAX_RENDERED:
                    prompt, contexts, active_sources, length, query_only_overflow = empty, [], [], empty_length, True
                else:
                    while length > MAX_RENDERED and contexts:
                        contexts.pop(); active_sources.pop(); prompt, length = render_length(tokenizer, models, str(row.query), contexts)
            generation_key = sha256_text(json.dumps({"prompt": prompt, "system": str(row.system_prompt),
                "max_new_tokens": OUTPUT_BUDGET}, sort_keys=True))
            case_key = str(row.case_id) if condition == CANDIDATE else f"{row.case_id}|{condition}"
            rows.append({"case_id": case_key, "source_case_id": str(row.case_id), "row_id": str(row.row_id),
                "panel": str(row.panel), "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
                "session_id": str(row.session_id), "turn_order": int(row.turn_order), "target_rank": int(row.target_rank),
                "query": str(row.query), "source_ids_top10": sources, "source_ids_used": active_sources,
                "hidden_source_id": sources[hidden] if hidden >= 0 else "", "mirabel_alarm": hidden >= 0,
                "visible_contexts": contexts, "tokens_used": [len(tokenizer(x, add_special_tokens=False).input_ids) for x in contexts],
                "total_context_tokens": sum(len(tokenizer(x, add_special_tokens=False).input_ids) for x in contexts),
                "requested_max_new_tokens": int(row.requested_max_new_tokens), "max_new_tokens": OUTPUT_BUDGET,
                "query_only_overflow": query_only_overflow, "rendered_tokens": length,
                "prompt": prompt, "prompt_hash": sha256_text(prompt), "generation_key": generation_key,
                "system_prompt": str(row.system_prompt), "condition": condition})
        if number % 4000 == 0: checkpoint("PACKING_PROGRESS", completed=number, total=len(cases))
    output = pd.DataFrame(rows); atomic_pickle(output, PACKING)
    candidate = output[output.condition == CANDIDATE]
    summary = pd.DataFrame([{"condition": CANDIDATE, "queries": len(candidate),
        "unique_generation_tasks": output.generation_key.nunique(), "top_k": TOP_K,
        "source_budget": SOURCE_BUDGET, "server_output_budget": OUTPUT_BUDGET,
        "mean_context_tokens": candidate.total_context_tokens.mean(),
        "mean_visible_sources": candidate.source_ids_used.map(len).mean(),
        "mirabel_alarm_rate": candidate.mirabel_alarm.mean(),
        "query_only_overflow_rows": int(candidate.query_only_overflow.sum())}])
    atomic_csv(summary, ROOT / "tables/TABLE_199_04_PACKING.csv")
    checkpoint("PACKING_COMPLETE", rows=len(output), unique_generation_tasks=output.generation_key.nunique(),
               mean_context_tokens=float(output.total_context_tokens.mean()))
    return output


def generate(packing):
    if RESPONSES.exists():
        old = pd.read_csv(RESPONSES, keep_default_na=False, dtype={"case_id": str})
        if len(old) == len(packing): return old
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    models.ROOT = ROOT; models.heartbeat = lambda stage, **details: checkpoint(stage, **details)
    models.GENERATION_CONFIG["batch_size"] = 16
    representatives = packing.drop_duplicates("generation_key")
    tasks = [models.make_task(task_type=CANDIDATE, row_id=str(row.generation_key), prompt=str(row.prompt),
        system_prompt=str(row.system_prompt), max_new_tokens=OUTPUT_BUDGET) for row in representatives.itertuples(index=False)]
    checkpoint("GENERATION_STARTED", unique_tasks=len(tasks), expanded_rows=len(packing),
               fixed_server_output_budget=OUTPUT_BUDGET, device="cuda:0")
    started = time.perf_counter(); answers = models.run_generation(tasks, GEN_DB, None, "GENERATION_PROGRESS")
    output = packing.drop(columns=["prompt"]).copy()
    output["response"] = output.generation_key.map(answers); output["response_sha256"] = output.response.map(sha256_text)
    if output.response.isna().any(): raise RuntimeError("generation incomplete")
    atomic_csv(output, RESPONSES, "gzip"); elapsed = time.perf_counter() - started
    invariance = output[(output.family.eq("BudgetLeak-Z")) & (output.condition.eq(CANDIDATE))].groupby("row_id").response.nunique().eq(1).mean()
    atomic_json(ROOT / "audits/BUDGET_RESPONSE_INVARIANCE.json", {"rows": int((output.family == "BudgetLeak-Z").sum()),
        "base_sessions": int(output[output.family.eq("BudgetLeak-Z")].row_id.nunique()),
        "response_invariance_rate": float(invariance), "passed": bool(invariance == 1.0)})
    checkpoint("GENERATION_COMPLETE", responses=len(output), unique_tasks=len(tasks), wall_seconds=elapsed,
               budget_response_invariance=float(invariance), database_sha256=sha256_file(GEN_DB))
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--phase-a-only", action="store_true"); parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(); config = preflight(); attack, normal, cases, risk = load_cases(); views = build_views(cases)
    if not phase_a(normal, attack, risk, views, config): return
    if args.phase_a_only: return
    packing = build_packing(cases, views)
    if args.prepare_only: checkpoint("PREPARE_COMPLETE", rows=len(packing)); return
    generate(packing); checkpoint("GENERATION_STAGE_COMPLETE", next_stage="evaluate_exp199.py")


if __name__ == "__main__": main()
