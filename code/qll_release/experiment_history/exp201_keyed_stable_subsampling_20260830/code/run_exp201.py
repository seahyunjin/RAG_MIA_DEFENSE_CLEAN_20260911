#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp201_keyed_stable_subsampling_20260830"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP166 = PROJECT / "exp166_topiocqa_gold_utility_20260827"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
EXP193 = PROJECT / "exp193_top4_context_rebase_20260829"
EXP196 = PROJECT / "exp196_selective_rank_agnostic_globalcap64_20260829"
EXP199 = PROJECT / "exp199_stable_evidence_budget_invariant_20260830"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
PRECOMMIT = ROOT / "configs/PRECOMMIT.json"
PACKING = ROOT / "private/EXP201_PACKING.private.pkl.gz"
RESPONSES = ROOT / "private/EXP201_RESPONSES.private.csv.gz"
GEN_DB = ROOT / "private/EXP201_QWEN_RESPONSES.sqlite3"
CANDIDATE = "KSS_PREFIX64_MIRABEL_FIXED64"
SEEDS = (20101, 20102, 20103)
ADMISSION = .375
OUTPUT_BUDGET = 64


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    return module


common = load(EXP199 / "code/run_exp199.py", "exp201_common")


def checkpoint(stage, **values):
    payload = {"experiment": "Exp201", "stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "candidate": CANDIDATE, "trainable_parameters": 0,
        "normal_calibration": False, "attack_specific_tuning": False, "rank_specific_tuning": False, **values}
    common.atomic_json(ROOT / "HEARTBEAT.json", payload); common.atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.jsonl").open("a") as handle: handle.write(json.dumps(payload, default=str) + "\n")
    common.atomic_text(ROOT / "STATUS.md", "\n".join(["# Exp201 Status", "", f"- Stage: **{stage}**"] +
        [f"- {key}: `{value}`" for key, value in values.items()]) + "\n")


def stable_value(seed, domain, source):
    key = f"EXP201-DEPLOYMENT-KEY-{seed}".encode()
    message = f"{domain}\0{source}".encode()
    digest = hmac.new(key, message, hashlib.sha256).digest()
    return int.from_bytes(digest, "big") / float(1 << 256)


def admitted(seed, domain, sources, hidden):
    values = [stable_value(seed, domain, source) for source in sources]
    selected = [index for index, value in enumerate(values) if value < ADMISSION and index != hidden]
    if not selected:
        eligible = [index for index in range(len(sources)) if index != hidden]
        if eligible: selected = [min(eligible, key=lambda index: (values[index], str(sources[index])))]
    return selected, values


def preflight():
    config = json.loads(PRECOMMIT.read_text())
    required = {"exp199_views": common.VIEWS, "risk": common.RISK, "attacks": common.ATTACK_CASES,
        "normal": common.NORMAL_CASES, "qwen": QWEN / "config.json", "gold": EXP166 / "private/TOPIOCQA_GOLD_1000.private.csv.gz"}
    for path in required.values():
        if not path.exists(): raise RuntimeError(f"missing input {path}")
    rows = [{"key": key, "path": str(path), "sha256": common.sha256_file(path), "bytes": path.stat().st_size,
             "access": "READ_ONLY"} for key, path in required.items()]
    common.atomic_csv(pd.DataFrame(rows), ROOT / "provenance/FROZEN_INPUTS.csv")
    common.atomic_json(ROOT / "audits/MECHANISM_AUDIT.json", {"admission_probability": ADMISSION,
        "seeds": SEEDS, "query_used": False, "rank_used": False, "membership_used": False,
        "attack_family_used": False, "caller_budget_used": False, "server_output_budget": OUTPUT_BUDGET})
    checkpoint("PREFLIGHT_COMPLETE", precommit_sha256=common.sha256_file(PRECOMMIT))
    return config


def render(tokenizer, models, query, contexts):
    prompt = models.normal_prompt(str(query), contexts)
    messages = [{"role": "system", "content": models.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, len(tokenizer(text, add_special_tokens=False).input_ids)


def build_packing():
    if PACKING.exists(): return pd.read_pickle(PACKING, compression="gzip")
    attack, normal, cases, risk = common.load_cases()
    attack = attack[attack.family.eq("BudgetLeak-Z")].copy()
    attack["requested_max_new_tokens"] = attack.max_new_tokens.astype(int); attack["split"] = "ATTACK"
    normal["requested_max_new_tokens"] = 96; normal["split"] = "NORMAL"
    columns = sorted(set(attack.columns) & set(normal.columns))
    cohort = pd.concat([attack[columns], normal[columns]], ignore_index=True, sort=False)
    cohort = cohort.drop(columns=[column for column in ("source_ids", "source_scores", "mirabel_threshold") if column in cohort.columns])
    cohort = cohort.merge(risk[["case_id", "source_ids", "source_scores", "mirabel_threshold"]], on="case_id", validate="one_to_one")
    views = pd.read_pickle(common.VIEWS, compression="gzip")
    prefix = {(str(row.domain), str(row.source_id)): str(row.prefix64) for row in views.itertuples(index=False)}
    from transformers import AutoTokenizer
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True); rows = []
    for number, row in enumerate(cohort.itertuples(index=False), 1):
        sources = list(map(str, row.source_ids[:10])); scores = np.asarray(row.source_scores, float)
        hidden = int(np.argmax(scores)) if float(scores.max()) > float(row.mirabel_threshold) else -1
        for seed in SEEDS:
            selected, values = admitted(seed, str(row.domain), sources, hidden)
            visible = [sources[index] for index in selected]; contexts = [prefix[(str(row.domain), source)] for source in visible]
            prompt, rendered = render(tokenizer, models, str(row.query), contexts)
            generation_key = common.sha256_text(json.dumps({"prompt": prompt, "system": str(row.system_prompt),
                "max": OUTPUT_BUDGET}, sort_keys=True))
            target = str(row.target_document_id); target_admitted = target in visible
            rows.append({"case_id": f"{row.case_id}|S{seed}", "source_case_id": str(row.case_id),
                "row_id": str(row.row_id), "panel": str(row.panel), "family": str(row.family),
                "member": int(row.member), "domain": str(row.domain), "session_id": str(row.session_id),
                "turn_order": int(row.turn_order), "target_rank": int(row.target_rank), "query": str(row.query),
                "condition": f"{CANDIDATE}_S{seed}", "seed": seed, "source_ids_top10": sources,
                "source_ids_used": visible, "source_admission_values": values,
                "target_admitted": target_admitted, "hidden_source_id": sources[hidden] if hidden >= 0 else "",
                "mirabel_alarm": hidden >= 0, "visible_contexts": contexts,
                "total_context_tokens": sum(len(tokenizer(x, add_special_tokens=False).input_ids) for x in contexts),
                "requested_max_new_tokens": int(row.requested_max_new_tokens), "max_new_tokens": OUTPUT_BUDGET,
                "reference": str(getattr(row, "reference", "")), "prompt": prompt,
                "generation_key": generation_key, "system_prompt": str(row.system_prompt), "rendered_tokens": rendered})
        if number % 3000 == 0: checkpoint("PACKING_PROGRESS", completed=number, total=len(cohort))
    output = pd.DataFrame(rows); common.atomic_pickle(output, PACKING)
    summary = output.groupby(["condition", "panel"], as_index=False).agg(rows=("case_id", "size"),
        unique_tasks=("generation_key", "nunique"), mean_visible_sources=("source_ids_used", lambda x: np.mean([len(v) for v in x])),
        mean_context_tokens=("total_context_tokens", "mean"), target_admission_rate=("target_admitted", "mean"),
        mirabel_alarm_rate=("mirabel_alarm", "mean"))
    common.atomic_csv(summary, ROOT / "tables/TABLE_201_01_PACKING.csv")
    checkpoint("PACKING_COMPLETE", rows=len(output), unique_tasks=output.generation_key.nunique())
    return output


def generate(packing):
    if RESPONSES.exists(): return pd.read_csv(RESPONSES, keep_default_na=False, low_memory=False)
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    models.ROOT = ROOT; models.heartbeat = lambda stage, **details: checkpoint(stage, **details); models.GENERATION_CONFIG["batch_size"] = 16
    unique = packing.drop_duplicates("generation_key")
    tasks = [models.make_task(task_type=CANDIDATE, row_id=str(row.generation_key), prompt=str(row.prompt),
        system_prompt=str(row.system_prompt), max_new_tokens=OUTPUT_BUDGET) for row in unique.itertuples(index=False)]
    checkpoint("GENERATION_STARTED", unique_tasks=len(tasks), expanded_rows=len(packing), device="cuda:0")
    started = time.perf_counter(); answers = models.run_generation(tasks, GEN_DB, None, "GENERATION_PROGRESS")
    output = packing.drop(columns=["prompt"]).copy(); output["response"] = output.generation_key.map(answers)
    common.atomic_csv(output, RESPONSES, "gzip")
    checkpoint("GENERATION_COMPLETE", responses=len(output), unique_tasks=len(tasks), wall_seconds=time.perf_counter() - started)
    return output


def external_module():
    module = load(EXP188 / "code/run_exp188.py", "exp201_external"); module.ROOT = ROOT
    return module


def auc(labels, values):
    raw = float(roc_auc_score(labels, values)); return raw, max(raw, 1 - raw)


def evaluate(responses, config):
    external = responses[responses.panel.eq("EXTERNAL")].copy()
    cases = external[["case_id", "reference", "requested_max_new_tokens", "panel"]].rename(columns={"requested_max_new_tokens": "budget"})
    response_input = external.drop(columns=["reference", "requested_max_new_tokens"]).copy()
    module = external_module(); metrics = module.external_response_metrics(response_input, cases)
    scores = module.external_attack_scores(metrics); records = []
    for condition, cell in scores.groupby("condition"):
        raw, effective = auc(cell.member, cell.attack_score)
        records.append({"condition": condition, "stratum": "aggregate", "n": len(cell), "raw_auc": raw, "effective_auc": effective})
        nonmember = cell[cell.member.eq(0)]
        for rank in (1, 2, 3, 4):
            subset = pd.concat([cell[(cell.member.eq(1)) & cell.target_rank.eq(rank)], nonmember])
            raw, effective = auc(subset.member, subset.attack_score)
            records.append({"condition": condition, "stratum": f"rank{rank}", "n": len(subset), "raw_auc": raw, "effective_auc": effective})
    privacy = pd.DataFrame(records); common.atomic_csv(privacy, ROOT / "tables/TABLE_201_02_BUDGETLEAK.csv")

    normal = responses[responses.panel.eq("NORMAL_GOLD")].copy()
    gold = pd.read_csv(EXP166 / "private/TOPIOCQA_GOLD_1000.private.csv.gz", keep_default_na=False, dtype={"row_id": str})
    normal = normal.merge(gold[["row_id", "gold_answers"]], on="row_id", validate="many_to_one")
    metric_core = load(EXP188 / "code/exp188_core.py", "exp201_metric_core")
    normal["token_f1"] = [max(metric_core.token_f1(answer, target) for target in json.loads(values))
                          for answer, values in zip(normal.response, normal.gold_answers)]
    normal["exact_match"] = [max(metric_core.normalize_answer(answer) == metric_core.normalize_answer(target)
                              for target in json.loads(values)) for answer, values in zip(normal.response, normal.gold_answers)]
    refusal = re.compile(r"\b(?:i do not know|i don't know|cannot determine|insufficient context|not enough information)\b", re.I)
    normal["refusal"] = normal.response.map(lambda x: bool(refusal.search(str(x))))
    baseline = pd.read_csv(EXP196 / "private/EXP196_NORMAL_RESPONSES.private.csv.gz", keep_default_na=False, dtype={"row_id": str})
    no_defense = baseline[baseline.condition.eq("NO_DEFENSE")].set_index("row_id").response.to_dict()
    normal["new_refusal"] = normal.refusal & ~normal.row_id.map(lambda x: bool(refusal.search(str(no_defense[str(x)]))))
    utility = normal.groupby("condition", as_index=False).agg(rows=("row_id", "size"), token_f1=("token_f1", "mean"),
        exact_match=("exact_match", "mean"), refusal_rate=("refusal", "mean"), new_refusal_rate=("new_refusal", "mean"))
    common.atomic_csv(utility, ROOT / "tables/TABLE_201_03_NORMAL_UTILITY.csv")
    invariance = external.groupby(["condition", "row_id"]).response.nunique().eq(1).groupby(level=0).mean()
    target_admission = responses[(responses.panel.eq("EXTERNAL")) & (responses.member.eq(1)) & responses.target_rank.between(1, 4)].groupby(["condition", "target_rank"]).target_admitted.mean().reset_index()
    common.atomic_csv(target_admission, ROOT / "tables/TABLE_201_04_TARGET_ADMISSION.csv")
    gate_cfg = config["primary_gates_each_seed"]; gates = []
    for condition in sorted(privacy.condition.unique()):
        aggregate = float(privacy[(privacy.condition.eq(condition)) & privacy.stratum.eq("aggregate")].effective_auc.iloc[0])
        worst = float(privacy[(privacy.condition.eq(condition)) & privacy.stratum.ne("aggregate")].effective_auc.max())
        util = utility[utility.condition.eq(condition)].iloc[0]
        passed = aggregate <= gate_cfg["budgetleak_aggregate_effective_auc_max"] and worst <= gate_cfg["budgetleak_each_rank_effective_auc_max"] and float(invariance.loc[condition]) >= gate_cfg["budget_response_invariance_min"] and float(util.token_f1) >= gate_cfg["normal_gold_token_f1_min"] and float(util.new_refusal_rate) <= gate_cfg["normal_new_refusal_rate_max"]
        gates.append({"condition": condition, "aggregate_eauc": aggregate, "worst_rank_eauc": worst,
            "response_invariance": float(invariance.loc[condition]), "normal_token_f1": float(util.token_f1),
            "normal_new_refusal": float(util.new_refusal_rate), "passed": bool(passed)})
    gate = pd.DataFrame(gates); common.atomic_csv(gate, ROOT / "tables/TABLE_201_05_GATES.csv")
    all_pass = bool(gate.passed.all()); verdict = "KSS_BUDGETLEAK_3SEED_PASS" if all_pass else "KSS_BUDGETLEAK_3SEED_FAIL"
    result = {"experiment": "Exp201", "verdict": verdict, "all_three_seeds_pass": all_pass,
        "new_generation_tasks": int(responses.generation_key.nunique()), "gates": gate.to_dict("records"),
        "completed_utc": datetime.now(timezone.utc).isoformat()}
    common.atomic_json(ROOT / "FINAL_RESULT.json", result)
    report = f"""# Exp201 — Keyed Stable Source Subsampling

- 최종 판정: **{verdict}**
- source admission probability: `{ADMISSION}`; source/output cap: `64/64`
- 요청 budget은 사용하지 않으며, 동일 source admission은 query와 rank에 무관하게 고정된다.
- 개발 실험이며 DP 보장이나 fresh blind가 아니다.

## BudgetLeak

{privacy.to_markdown(index=False)}

## Normal gold utility

{utility.to_markdown(index=False)}

## Target admission

{target_admission.to_markdown(index=False)}

## Gates

{gate.to_markdown(index=False)}
"""
    common.atomic_text(ROOT / "reports/REPORT_201_FINAL_KO.md", report); checkpoint("COMPLETE", verdict=verdict)
    print(json.dumps(result, indent=2))


def main():
    config = preflight(); packing = build_packing(); responses = generate(packing); evaluate(responses, config)


if __name__ == "__main__": main()
