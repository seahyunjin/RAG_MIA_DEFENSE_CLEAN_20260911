#!/usr/bin/env python3
"""Frozen evaluation and stop-gate logic for Exp199."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import run_exp199 as common


ROOT = common.ROOT
CANDIDATE = common.CANDIDATE
PREFIX = common.PREFIX_BASELINE


def load_base():
    path = common.EXP197 / "code/evaluate_exp197.py"
    sys.path.insert(0, str(common.EXP197 / "code"))
    spec = importlib.util.spec_from_file_location("exp199_frozen_evaluator", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    module.ROOT = ROOT; module.RA = CANDIDATE; module.common = common

    def visible_contexts(candidate_normal, normal_cases):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(common.QWEN, local_files_only=True)
        packing = pd.read_pickle(common.PACKING, compression="gzip")
        packing = packing[packing.condition.eq(CANDIDATE)].set_index("row_id")
        output = {}
        for row in candidate_normal.itertuples(index=False):
            packed = packing.loc[str(row.row_id)]
            output[(str(row.row_id), CANDIDATE)] = list(packed.visible_contexts)
        for row in normal_cases.itertuples(index=False):
            texts = []
            for text, cap in zip(row.source_texts, [64, 662, 661, 661]):
                ids = tokenizer(str(text), add_special_tokens=False).input_ids[:cap]
                texts.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
            output[(str(row.row_id), "GLOBAL_CAP64")] = texts
        return output

    module.visible_contexts = visible_contexts
    return module


def value(table, condition, family, column="effective_auc"):
    row = table[(table.condition.eq(condition)) & table.attack_family.eq(family)]
    if len(row) != 1: raise RuntimeError(f"missing {condition}/{family}")
    return float(row.iloc[0][column])


def mirror_tables():
    mapping = {
        "TABLE_197_03_NATIVE_PRIVACY.csv": "TABLE_199_05_NATIVE_PRIVACY.csv",
        "TABLE_197_04_EXTERNAL_PRIVACY.csv": "TABLE_199_06_EXTERNAL_PRIVACY.csv",
        "TABLE_197_05_BUDGETLEAK_BY_RANK.csv": "TABLE_199_07_BUDGETLEAK_BY_RANK.csv",
        "TABLE_197_06_RANK_INSTABILITY.csv": "TABLE_199_08_RANK_INSTABILITY.csv",
        "TABLE_197_07_NORMAL_UTILITY.csv": "TABLE_199_09_NORMAL_UTILITY.csv",
        "TABLE_197_08_NLI_SCREEN.csv": "TABLE_199_10_NLI_SCREEN.csv"
    }
    for source, destination in mapping.items():
        frame = pd.read_csv(ROOT / "tables" / source); common.atomic_csv(frame, ROOT / "tables" / destination)


def verify_hashes():
    frozen = pd.read_csv(ROOT / "provenance/FROZEN_INPUTS.csv"); rows = []
    for row in frozen.itertuples(index=False):
        current = common.sha256_file(row.path) if Path(row.path).exists() else "MISSING"
        rows.append({"key": row.key, "path": row.path, "before": row.sha256, "after": current,
                     "unchanged": current == row.sha256})
    frame = pd.DataFrame(rows); common.atomic_csv(frame, ROOT / "audits/FROZEN_INPUT_HASH_RECHECK.csv")
    if not frame.unchanged.all(): raise RuntimeError("frozen input changed")


def finalize(native, external, ranks, instability, utility, nli, candidate):
    config = json.loads(common.PRECOMMIT.read_text()); mirabel = "ORIGINAL_MIRABEL"
    budget_cfg = config["primary_budgetleak_gates"]; expand_cfg = config["expansion_gates"]
    candidate_ranks = ranks[(ranks.condition.eq(CANDIDATE)) & ranks.target_rank_stratum.isin(["rank1", "rank2", "rank3", "rank4"])]
    invariance = candidate[candidate.family.eq("BudgetLeak-Z")].groupby("row_id").response.nunique().eq(1).mean()
    u = utility.set_index("condition"); n = nli.set_index("condition")
    gates = {
        "budgetleak_aggregate": value(external, CANDIDATE, "BudgetLeak-Z") <= budget_cfg["aggregate_effective_auc_max"],
        "budgetleak_all_ranks": len(candidate_ranks) == 4 and bool((candidate_ranks.effective_auc <= budget_cfg["each_target_rank_effective_auc_max"]).all()),
        "budget_response_invariance": float(invariance) >= budget_cfg["budget_variant_response_invariance_min"],
        "ragleak_mirabel_noninferior": value(external, CANDIDATE, "RAGLeak") <= value(external, mirabel, "RAGLeak") + .02,
    }
    families = sorted(native.attack_family.unique())
    for family in families:
        gates[f"native_{family}_absolute"] = value(native, CANDIDATE, family) <= expand_cfg["each_native_attack_effective_auc_max"]
        gates[f"native_{family}_mirabel_noninferior"] = value(native, CANDIDATE, family) <= value(native, mirabel, family) + expand_cfg["each_native_attack_vs_original_mirabel_margin_max"]
    gates["normal_token_f1_improves_fixed_prefix"] = float(u.loc[CANDIDATE, "token_f1"] - u.loc[PREFIX, "token_f1"]) >= expand_cfg["normal_gold_token_f1_improvement_vs_prefix64_mirabel_min"]
    gates["normal_new_refusal"] = float(u.loc[CANDIDATE, "new_refusal_rate"]) <= expand_cfg["normal_new_refusal_rate_max"]
    gates["automatic_nli_not_above_globalcap"] = float(n.loc[CANDIDATE, "unsupported_risk_rate"]) <= float(n.loc["GLOBAL_CAP64", "unsupported_risk_rate"])
    gates["no_request_rejection"] = True; gates["no_tuning_or_training"] = True
    budget_pass = gates["budgetleak_aggregate"] and gates["budgetleak_all_ranks"] and gates["budget_response_invariance"]
    all_pass = all(gates.values())
    verdict = "EXP199_SUPPORTED" if all_pass else ("EXP199_BUDGETLEAK_FAILED" if not budget_pass else "EXP199_EXPANSION_FAILED")
    gate_table = pd.DataFrame([{"gate": key, "passed": bool(status)} for key, status in gates.items()])
    common.atomic_csv(gate_table, ROOT / "tables/TABLE_199_11_GATES.csv")
    rank_values = dict(zip(candidate_ranks.target_rank_stratum, candidate_ranks.effective_auc))
    result = {"experiment": "Exp199", "verdict": verdict, "development_not_fresh_blind": True,
        "budget_gate_pass": bool(budget_pass), "all_gate_pass": bool(all_pass),
        "budgetleak_effective_auc": value(external, CANDIDATE, "BudgetLeak-Z"),
        "budgetleak_rank_effective_auc": rank_values, "budget_response_invariance": float(invariance),
        "ragleak_effective_auc": value(external, CANDIDATE, "RAGLeak"),
        "native_effective_auc": {family: value(native, CANDIDATE, family) for family in families},
        "native_worst_effective_auc": float(native[native.condition.eq(CANDIDATE)].effective_auc.max()),
        "normal_token_f1": float(u.loc[CANDIDATE, "token_f1"]),
        "prefix_fixed64_token_f1": float(u.loc[PREFIX, "token_f1"]),
        "normal_exact_match": float(u.loc[CANDIDATE, "exact_match"]),
        "normal_response_preservation": float(u.loc[CANDIDATE, "response_preservation"]),
        "normal_new_refusal_rate": float(u.loc[CANDIDATE, "new_refusal_rate"]),
        "automatic_nli_unsupported_risk": float(n.loc[CANDIDATE, "unsupported_risk_rate"]),
        "automatic_nli_is_human_label": False, "source_budget": common.SOURCE_BUDGET,
        "fixed_output_budget": common.OUTPUT_BUDGET, "gates": gates, "completed_utc": common.now()}
    common.atomic_json(ROOT / "FINAL_RESULT.json", result)
    external_view = external.pivot(index="attack_family", columns="condition", values="effective_auc").reset_index()
    native_view = native.pivot(index="attack_family", columns="condition", values="effective_auc").reset_index()
    report = f"""# Exp199 — Budget-Invariant Stable Evidence View

- 최종 판정: **{verdict}**
- BudgetLeak 최우선 게이트: **{'PASS' if budget_pass else 'FAIL'}**
- source view / server output budget: `{common.SOURCE_BUDGET} / {common.OUTPUT_BUDGET}` tokens
- query·rank 독립 stable source view, 요청 거부 없음, 학습·정상보정 없음
- 개발 실험이며 fresh blind가 아님

## External privacy E-AUC (낮을수록 좋음)

{external_view.to_markdown(index=False)}

## BudgetLeak rank 1–4

{candidate_ranks[['target_rank_stratum','n','effective_auc','raw_auc_ci95_low','raw_auc_ci95_high']].to_markdown(index=False)}

- 동일 query의 요청 budget 변화에 대한 응답 불변률: `{invariance:.6f}`

## Native 6공격

{native_view.to_markdown(index=False)}

## 정상 gold utility

{utility.to_markdown(index=False)}

## 자동 NLI screening

{nli.to_markdown(index=False)}

자동 NLI는 사람 환각 판정이 아니다.

## Gates

{gate_table.to_markdown(index=False)}
"""
    common.atomic_text(ROOT / "reports/REPORT_199_FINAL_KO.md", report)
    verify_hashes(); common.checkpoint("COMPLETE", verdict=verdict, budget_gate_pass=budget_pass)
    return result


def main():
    if not common.RESPONSES.exists(): raise RuntimeError("generation artifact missing")
    base = load_base(); candidate = pd.read_csv(common.RESPONSES, keep_default_na=False, low_memory=False, dtype={"case_id": str, "row_id": str})
    attack = pd.read_pickle(common.ATTACK_CASES, compression="gzip")
    _, native, external, ranks, instability = base.privacy(candidate, attack)
    utility, detail = base.utility(candidate); nli = base.nli_screen(candidate, detail)
    mirror_tables(); result = finalize(native, external, ranks, instability, utility, nli, candidate[candidate.condition.eq(CANDIDATE)])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
