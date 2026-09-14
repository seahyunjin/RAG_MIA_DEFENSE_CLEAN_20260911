#!/usr/bin/env python3
"""Exp202 selective QLL-located stable cap with deterministic backfill."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp202_selective_qll_stable_cap_20260830"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP166 = PROJECT / "exp166_topiocqa_gold_utility_20260827"
EXP179 = PROJECT / "exp179_cross_family_qwen_source_influence_20260828"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
EXP193 = PROJECT / "exp193_top4_context_rebase_20260829"
EXP195 = PROJECT / "exp195_minimal_qll_exposure_guard_20260829"
EXP196 = PROJECT / "exp196_selective_rank_agnostic_globalcap64_20260829"
EXP199 = PROJECT / "exp199_stable_evidence_budget_invariant_20260830"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
PRECOMMIT = ROOT / "configs/PRECOMMIT.json"
QLL = EXP179 / "tables/TABLE_179_02_QUERY_SUMMARY.csv"
ATTACKS = EXP193 / "private/EXP193_CASES.private.pkl.gz"
NORMAL = EXP195 / "private/EXP195_NORMAL_CASES.private.pkl.gz"
PACKING = ROOT / "private/EXP202_PACKING.private.pkl.gz"
RESPONSES = ROOT / "private/EXP202_RESPONSES.private.csv.gz"
GEN_DB = ROOT / "private/EXP202_QWEN_RESPONSES.sqlite3"
CANDIDATE = "SELECTIVE_QLL_STABLE64_BACKFILL_FIXED64"
BASELINE = "MATCHED_BALANCED_TOP4_FIXED64"
TOTAL_BUDGET = 2048
SOURCE_CAP = 64
OUTPUT_BUDGET = 64
TOP_K = 4


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


common = load(EXP199 / "code/run_exp199.py", "exp202_common")


def now():
    return datetime.now(timezone.utc).isoformat()


def checkpoint(stage, **values):
    payload = {"experiment": "Exp202", "candidate": CANDIDATE, "stage": stage,
        "updated_utc": now(), "pid": os.getpid(), "trainable_parameters": 0,
        "attack_specific_tuning": False, "rank_specific_tuning": False,
        "request_rejection": False, "paid_api_calls": 0, **values}
    common.atomic_json(ROOT / "HEARTBEAT.json", payload)
    common.atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp202 Status", "", f"- Stage: **{stage}**"] + [f"- {k}: `{v}`" for k, v in values.items()]
    common.atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def preflight():
    config = json.loads(PRECOMMIT.read_text())
    required = {"precommit": PRECOMMIT, "qll": QLL, "attacks": ATTACKS, "normal": NORMAL,
        "gold": EXP166 / "private/TOPIOCQA_GOLD_1000.private.csv.gz",
        "qwen": QWEN / "config.json", "mirabel_locator_audit": PROJECT / "exp181_mirabel_locator_rank_audit_20260828/FINAL_RESULT.json"}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing frozen inputs: {missing}")
    locator = json.loads(required["mirabel_locator_audit"].read_text())
    if not locator.get("candidate_equivalent_to_rank1_hide"):
        raise RuntimeError("canonical Mirabel rank limitation provenance changed")
    manifest = pd.DataFrame([{"key": key, "path": str(path), "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path), "access": "READ_ONLY"} for key, path in required.items()])
    common.atomic_csv(manifest, ROOT / "provenance/FROZEN_INPUTS.csv")
    common.atomic_json(ROOT / "audits/MECHANISM_AUDIT.json", {
        "qll_top_k": TOP_K, "source_cap": SOURCE_CAP, "total_context_budget": TOTAL_BUDGET,
        "output_budget": OUTPUT_BUDGET, "caller_budget_used": False, "request_rejection": False,
        "mirabel_used_at_runtime": False, "normal_calibration": True,
        "normal_calibration_scope": "TopiOCQA development only", "training": False})
    checkpoint("PREFLIGHT_COMPLETE", precommit_sha256=common.sha256_file(PRECOMMIT), frozen_inputs=len(manifest))
    return config


def query_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def load_policy_rows():
    qll = pd.read_csv(QLL, keep_default_na=False)
    normal_qll = qll[qll.attack_family.eq("NORMAL_GOLD")].copy()
    threshold = float(np.quantile(normal_qll.dominance.to_numpy(float), .95, method="higher"))
    normal_qll["trigger"] = normal_qll.dominance.gt(threshold)
    attack_qll = qll[qll.attack_family.eq("BudgetLeak-Z")].copy()
    attacks = pd.read_pickle(ATTACKS, compression="gzip")
    attacks = attacks[attacks.family.eq("BudgetLeak-Z")].copy()
    attacks["qll_case_id"] = attacks.session_id.astype(str).str.replace("EXTERNAL|", "", regex=False)
    attacks = attacks.merge(attack_qll[["case_id", "dominance", "qll_top1_source", "qll_top2_source", "margin"]],
        left_on="qll_case_id", right_on="case_id", suffixes=("", "_qll"), validate="many_to_one")
    attacks["trigger"] = attacks.dominance.gt(threshold)
    normal = pd.read_pickle(NORMAL, compression="gzip")
    normal["query_sha256"] = normal["query"].map(query_hash)
    normal = normal.merge(normal_qll[["query_sha256", "dominance", "qll_top1_source", "qll_top2_source", "margin"]],
        on="query_sha256", validate="one_to_one")
    normal["trigger"] = normal.dominance.gt(threshold)
    return attacks, normal, qll, threshold


def phase_a(attacks, normal, qll, threshold, config):
    normal_rate = float(normal.trigger.mean())
    q = qll[(qll.attack_family.eq("BudgetLeak-Z")) & qll.member.eq(1) & qll.target_in_top4.astype(bool)].copy()
    q["locator_hit"] = q.qll_top1_source.astype(str).eq(q.target_document_id.astype(str))
    q["joint_hit"] = q.locator_hit & q.dominance.gt(threshold)
    rank = q.groupby("target_rank", as_index=False).agg(samples=("case_id", "size"),
        locator_hit_at_1=("locator_hit", "mean"), trigger_rate=("dominance", lambda x: float((x > threshold).mean())),
        joint_hit_at_1=("joint_hit", "mean"))
    rank["target_rank"] = rank.target_rank.astype(int)
    common.atomic_csv(rank, ROOT / "tables/TABLE_202_01_PHASE_A_RANK_GATE.csv")
    family = qll[~qll.attack_family.eq("NORMAL_GOLD")].copy()
    family["eligible"] = family.target_in_top4.astype(bool)
    family["trigger"] = family.dominance.gt(threshold)
    family["locator_hit"] = family.qll_top1_source.astype(str).eq(family.target_document_id.astype(str))
    family["joint_hit"] = family.trigger & family.locator_hit
    rows = []
    for name, cell in family.groupby("attack_family", sort=True):
        eligible = cell[cell.eligible]
        rows.append({"attack_family": name, "queries": len(cell), "eligible_targets": len(eligible),
            "trigger_rate_all_queries": float(cell.trigger.mean()),
            "locator_hit_at_1_eligible": float(eligible.locator_hit.mean()) if len(eligible) else np.nan,
            "joint_hit_at_1_eligible": float(eligible.joint_hit.mean()) if len(eligible) else np.nan})
    common.atomic_csv(pd.DataFrame(rows), ROOT / "tables/TABLE_202_02_PHASE_A_FAMILY.csv")
    gates_cfg = config["phase_a_no_generation_gates"]
    gates = pd.DataFrame([
        {"gate": "normal_intervention", "value": normal_rate,
         "criterion": f"<={gates_cfg['actual_normal_intervention_rate_max']}",
         "passed": normal_rate <= gates_cfg["actual_normal_intervention_rate_max"]},
        {"gate": "budget_joint_hit_worst_rank", "value": float(rank.joint_hit_at_1.min()),
         "criterion": f">={gates_cfg['budgetleak_target_joint_hit_each_rank_min']}",
         "passed": float(rank.joint_hit_at_1.min()) >= gates_cfg["budgetleak_target_joint_hit_each_rank_min"]},
        {"gate": "qll_locator_hit_worst_rank", "value": float(rank.locator_hit_at_1.min()),
         "criterion": f">={gates_cfg['qll_target_hit_each_rank_min']}",
         "passed": float(rank.locator_hit_at_1.min()) >= gates_cfg["qll_target_hit_each_rank_min"]}
    ])
    common.atomic_csv(gates, ROOT / "tables/TABLE_202_03_PHASE_A_GATES.csv")
    passed = bool(gates.passed.all())
    common.atomic_json(ROOT / "audits/PHASE_A_DECISION.json", {"passed": passed, "threshold": threshold,
        "normal_intervention_rate": normal_rate, "worst_rank_joint_hit": float(rank.joint_hit_at_1.min()),
        "worst_rank_locator_hit": float(rank.locator_hit_at_1.min())})
    checkpoint("PHASE_A_PASS" if passed else "PHASE_A_FAIL", threshold=threshold,
        normal_intervention_rate=normal_rate, worst_rank_joint_hit=float(rank.joint_hit_at_1.min()))
    return passed


def waterfill(lengths, selected=None):
    lengths = np.asarray(lengths, dtype=int)
    source_count = len(lengths)
    if source_count == 0:
        return []
    caps = np.zeros(source_count, dtype=int)
    fixed = set()
    if selected is not None:
        caps[selected] = min(SOURCE_CAP, int(lengths[selected])); fixed.add(int(selected))
    remaining = TOTAL_BUDGET - int(caps.sum())
    active = [i for i in range(source_count) if i not in fixed and lengths[i] > caps[i]]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        changed = False
        for i in list(active):
            add = min(share, int(lengths[i] - caps[i]), remaining)
            if add > 0:
                caps[i] += add; remaining -= add; changed = True
            if caps[i] >= lengths[i]: active.remove(i)
            if remaining <= 0: break
        if not changed: break
    return caps.tolist()


def render(tokenizer, models, query, source_texts, caps):
    contexts = []
    used = []
    for text, cap in zip(source_texts, caps):
        ids = tokenizer(str(text), add_special_tokens=False).input_ids[:int(cap)]
        contexts.append(tokenizer.decode(ids, skip_special_tokens=True).strip()); used.append(len(ids))
    prompt = models.normal_prompt(str(query), contexts)
    messages = [{"role": "system", "content": models.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, contexts, used, len(tokenizer(rendered, add_special_tokens=False).input_ids)


def build_packing(attacks, normal, threshold):
    if PACKING.exists(): return pd.read_pickle(PACKING, compression="gzip")
    from transformers import AutoTokenizer
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    documents = common.load_exp198().all_documents()
    needed = {(str(row.domain), str(source)) for frame in (attacks, normal)
        for row in frame.itertuples(index=False) for source in list(row.source_ids)[:TOP_K]}
    missing = sorted(key for key in needed if key not in documents)
    common.atomic_json(ROOT / "audits/SOURCE_TEXT_COVERAGE.json", {
        "needed_domain_source_pairs": len(needed), "resolved_pairs": len(needed)-len(missing),
        "coverage": (len(needed)-len(missing))/len(needed) if needed else 1.0,
        "missing_pairs": missing[:100], "text_source": "immutable corpus keyed by frozen source ID",
        "legacy_exp193_source_texts_column_used": False})
    if missing:
        raise RuntimeError(f"frozen source text coverage incomplete: {len(missing)}")
    rows = []
    def add(row, condition, trigger):
        sources = list(map(str, row.source_ids[:TOP_K]))
        texts = [str(documents[(str(row.domain), source)]) for source in sources]
        lengths = [len(tokenizer(text, add_special_tokens=False).input_ids) for text in texts]
        selected = sources.index(str(row.qll_top1_source)) if trigger and str(row.qll_top1_source) in sources else None
        caps = waterfill(lengths, selected)
        prompt, contexts, used, rendered = render(tokenizer, models, str(row.query), texts, caps)
        generation_key = common.sha256_text(json.dumps({"prompt": prompt, "system": str(row.system_prompt), "max": OUTPUT_BUDGET}, sort_keys=True))
        rows.append({"case_id": str(row.case_id), "row_id": str(row.row_id), "panel": str(row.panel),
            "family": str(row.family), "member": int(row.member), "domain": str(row.domain),
            "session_id": str(row.session_id), "turn_order": int(row.turn_order), "target_rank": int(row.target_rank),
            "query": str(row.query), "condition": condition, "dominance": float(row.dominance),
            "threshold": threshold, "intervened": bool(trigger), "selected_source_id": str(row.qll_top1_source) if trigger else "",
            "selected_source_rank": selected + 1 if selected is not None else 0, "source_ids_used": sources,
            "caps": caps, "tokens_used": used, "total_context_tokens": sum(used), "visible_contexts": contexts,
            "requested_max_new_tokens": int(getattr(row, "max_new_tokens", 96)), "max_new_tokens": OUTPUT_BUDGET,
            "reference": str(getattr(row, "reference", "")), "system_prompt": str(row.system_prompt),
            "prompt": prompt, "generation_key": generation_key, "rendered_tokens": rendered,
            "request_blocked": False})
    # BudgetLeak has 14 rows per session.  The fixed prompt/output contract
    # intentionally deduplicates these to one local generation task.
    for number, row in enumerate(attacks.itertuples(index=False), 1):
        add(row, CANDIDATE, bool(row.trigger))
        if number % 5000 == 0: checkpoint("PACKING_PROGRESS", attack_rows=number, attack_total=len(attacks))
    for row in normal.itertuples(index=False):
        add(row, CANDIDATE, bool(row.trigger)); add(row, BASELINE, False)
    output = pd.DataFrame(rows)
    common.atomic_pickle(output, PACKING)
    summary = output.groupby(["condition", "panel"], as_index=False).agg(rows=("case_id", "size"),
        unique_tasks=("generation_key", "nunique"), intervention_rate=("intervened", "mean"),
        mean_context_tokens=("total_context_tokens", "mean"), mean_selected_rank=("selected_source_rank", "mean"),
        request_block_rate=("request_blocked", "mean"))
    common.atomic_csv(summary, ROOT / "tables/TABLE_202_04_PACKING.csv")
    checkpoint("PACKING_COMPLETE", rows=len(output), unique_tasks=int(output.generation_key.nunique()))
    return output


def generate(packing):
    if RESPONSES.exists(): return pd.read_csv(RESPONSES, keep_default_na=False, low_memory=False)
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    models.ROOT = ROOT; models.heartbeat = lambda stage, **details: checkpoint(stage, **details)
    models.GENERATION_CONFIG["batch_size"] = 16
    unique = packing.drop_duplicates("generation_key")
    tasks = [models.make_task(task_type=CANDIDATE, row_id=str(row.generation_key), prompt=str(row.prompt),
        system_prompt=str(row.system_prompt), max_new_tokens=OUTPUT_BUDGET) for row in unique.itertuples(index=False)]
    checkpoint("GENERATION_STARTED", unique_tasks=len(tasks), expanded_rows=len(packing), device="cuda:0")
    started = time.perf_counter(); answers = models.run_generation(tasks, GEN_DB, None, "GENERATION_PROGRESS")
    output = packing.drop(columns=["prompt"]).copy(); output["response"] = output.generation_key.map(answers)
    if output.response.isna().any(): raise RuntimeError("generation incomplete")
    common.atomic_csv(output, RESPONSES, "gzip")
    checkpoint("GENERATION_COMPLETE", responses=len(output), unique_tasks=len(tasks), wall_seconds=time.perf_counter()-started)
    return output


def auc(labels, scores):
    raw = float(roc_auc_score(labels, scores)); return raw, max(raw, 1-raw)


def evaluate(responses, config):
    external = responses[(responses.panel.eq("EXTERNAL")) & responses.condition.eq(CANDIDATE)].copy()
    cases = external[["case_id", "reference", "requested_max_new_tokens", "panel"]].rename(columns={"requested_max_new_tokens":"budget"})
    response_input = external.drop(columns=["reference", "requested_max_new_tokens"])
    module = load(EXP188 / "code/run_exp188.py", "exp202_external"); module.ROOT = ROOT
    metrics = module.external_response_metrics(response_input, cases)
    scores = module.external_attack_scores(metrics)
    privacy_rows = []
    raw, effective = auc(scores.member, scores.attack_score)
    privacy_rows.append({"stratum":"aggregate", "n":len(scores), "raw_auc":raw, "effective_auc":effective})
    nonmember = scores[scores.member.eq(0)]
    for rank in (1,2,3,4):
        subset = pd.concat([scores[(scores.member.eq(1)) & scores.target_rank.eq(rank)], nonmember])
        raw, effective = auc(subset.member, subset.attack_score)
        privacy_rows.append({"stratum":f"rank{rank}", "n":len(subset), "raw_auc":raw, "effective_auc":effective})
    privacy = pd.DataFrame(privacy_rows); common.atomic_csv(privacy, ROOT / "tables/TABLE_202_05_BUDGETLEAK.csv")
    normal = responses[responses.panel.eq("NORMAL_GOLD")].copy()
    gold = pd.read_csv(EXP166 / "private/TOPIOCQA_GOLD_1000.private.csv.gz", keep_default_na=False, dtype={"row_id":str})
    normal = normal.merge(gold[["row_id","gold_answers"]], on="row_id", validate="many_to_one")
    core = load(EXP188 / "code/exp188_core.py", "exp202_core")
    normal["token_f1"] = [max(core.token_f1(answer, target) for target in json.loads(targets)) for answer,targets in zip(normal.response,normal.gold_answers)]
    normal["exact_match"] = [max(core.normalize_answer(answer)==core.normalize_answer(target) for target in json.loads(targets)) for answer,targets in zip(normal.response,normal.gold_answers)]
    refusal_re = re.compile(r"\b(?:i do not know|i don't know|cannot determine|insufficient context|not enough information)\b",re.I)
    normal["refusal"] = normal.response.map(lambda x:bool(refusal_re.search(str(x))))
    base = normal[normal.condition.eq(BASELINE)].set_index("row_id")
    normal["new_refusal"] = normal.refusal & ~normal.row_id.map(base.refusal.to_dict())
    normal["changed_vs_matched"] = normal.response.ne(normal.row_id.map(base.response.to_dict()))
    utility = normal.groupby("condition",as_index=False).agg(rows=("row_id","size"),token_f1=("token_f1","mean"),
        exact_match=("exact_match","mean"),refusal_rate=("refusal","mean"),new_refusal_rate=("new_refusal","mean"),
        response_change_rate=("changed_vs_matched","mean"),intervention_rate=("intervened","mean"),
        request_block_rate=("request_blocked","mean"))
    common.atomic_csv(utility, ROOT / "tables/TABLE_202_06_NORMAL_UTILITY.csv")
    invariance = float(external.groupby("row_id").response.nunique().eq(1).mean())
    candidate_u = utility[utility.condition.eq(CANDIDATE)].iloc[0]; baseline_u=utility[utility.condition.eq(BASELINE)].iloc[0]
    cfg=config["phase_b_primary_gates"]; aggregate=float(privacy[privacy.stratum.eq("aggregate")].effective_auc.iloc[0]); worst=float(privacy[privacy.stratum.ne("aggregate")].effective_auc.max())
    retention=float(candidate_u.token_f1/baseline_u.token_f1) if baseline_u.token_f1 else 0.0
    gates=pd.DataFrame([
      {"gate":"budget_aggregate", "value":aggregate,"criterion":f"<={cfg['budgetleak_aggregate_effective_auc_max']}","passed":aggregate<=cfg["budgetleak_aggregate_effective_auc_max"]},
      {"gate":"budget_worst_rank", "value":worst,"criterion":f"<={cfg['budgetleak_each_rank_effective_auc_max']}","passed":worst<=cfg["budgetleak_each_rank_effective_auc_max"]},
      {"gate":"response_invariance", "value":invariance,"criterion":f">={cfg['response_invariance_min']}","passed":invariance>=cfg["response_invariance_min"]},
      {"gate":"normal_f1_retention", "value":retention,"criterion":f">={cfg['normal_token_f1_retention_vs_matched_fixed64_min']}","passed":retention>=cfg["normal_token_f1_retention_vs_matched_fixed64_min"]},
      {"gate":"normal_new_refusal", "value":float(candidate_u.new_refusal_rate),"criterion":f"<={cfg['normal_new_refusal_rate_max']}","passed":float(candidate_u.new_refusal_rate)<=cfg["normal_new_refusal_rate_max"]},
      {"gate":"normal_intervention", "value":float(candidate_u.intervention_rate),"criterion":f"<={cfg['normal_intervention_rate_max']}","passed":float(candidate_u.intervention_rate)<=cfg["normal_intervention_rate_max"]},
      {"gate":"request_block", "value":float(candidate_u.request_block_rate),"criterion":"==0","passed":float(candidate_u.request_block_rate)==0.0}
    ])
    common.atomic_csv(gates, ROOT / "tables/TABLE_202_07_GATES.csv")
    passed=bool(gates.passed.all()); verdict="SELECTIVE_QLL_BUDGETLEAK_PASS" if passed else "SELECTIVE_QLL_BUDGETLEAK_FAIL"
    result={"experiment":"Exp202","candidate":CANDIDATE,"verdict":verdict,"passed":passed,
      "threshold":float(candidate_u.intervention_rate and normal[normal.condition.eq(CANDIDATE)].threshold.iloc[0]),
      "budgetleak_aggregate_effective_auc":aggregate,"budgetleak_worst_rank_effective_auc":worst,
      "normal_token_f1":float(candidate_u.token_f1),"matched_token_f1":float(baseline_u.token_f1),"normal_f1_retention":retention,
      "normal_intervention_rate":float(candidate_u.intervention_rate),"normal_new_refusal_rate":float(candidate_u.new_refusal_rate),
      "response_invariance":invariance,"request_block_rate":float(candidate_u.request_block_rate),
      "new_generation_tasks":int(responses.generation_key.nunique()),"completed_utc":now()}
    common.atomic_json(ROOT/"FINAL_RESULT.json",result)
    report=f"""# Exp202 — Selective QLL Stable Cap

- 최종 판정: **{verdict}**
- 정상 개발 intervention target: 5%; 요청 차단: 0%; 출력 budget: 64
- QLL은 top-4를 한 batch로 teacher-forced scoring하고 답변은 한 번만 생성한다.
- 이 결과는 reused development cohort이며 fresh blind가 아니다.

## BudgetLeak

{privacy.to_markdown(index=False)}

## 정상 gold utility

{utility.to_markdown(index=False)}

## Gates

{gates.to_markdown(index=False)}
"""
    common.atomic_text(ROOT/"reports/REPORT_202_FINAL_KO.md",report); checkpoint("COMPLETE",verdict=verdict)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    config=preflight(); attacks,normal,qll,threshold=load_policy_rows()
    if not phase_a(attacks,normal,qll,threshold,config):
        common.atomic_json(ROOT/"FINAL_RESULT.json",{"experiment":"Exp202","verdict":"PHASE_A_FAIL","completed_utc":now()})
        checkpoint("COMPLETE",verdict="PHASE_A_FAIL"); return
    packing=build_packing(attacks,normal,threshold); responses=generate(packing); evaluate(responses,config)


if __name__ == "__main__": main()
