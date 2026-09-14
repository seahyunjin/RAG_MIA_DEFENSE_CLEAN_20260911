#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp200_budget_invariant_causal_diagnostic_20260830"
EXP176 = PROJECT / "exp176_prefix64_mirabel_external_attacks_20260828"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")


def load_module():
    path = EXP188 / "code/run_exp188.py"
    spec = importlib.util.spec_from_file_location("exp200_frozen_external", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def metric(labels, values):
    raw = float(roc_auc_score(np.asarray(labels, int), np.asarray(values, float)))
    return raw, max(raw, 1.0 - raw)


def main():
    config = json.loads((ROOT / "configs/PRECOMMIT.json").read_text())
    frozen = pd.read_csv(EXP176 / "private/EXP176_GENERATIONS.private.csv.gz",
                         keep_default_na=False, low_memory=False, dtype={"row_id": str})
    frozen = frozen[frozen.attack.eq("BudgetLeak-Z")].copy()
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    source = frozen[frozen.budget.eq(70)].set_index(["condition", "row_id"]).response.to_dict()
    rows = []
    for row in frozen.itertuples(index=False):
        original = str(source[(str(row.condition), str(row.row_id))])
        ids = tokenizer(original, add_special_tokens=False).input_ids[:64]
        response = tokenizer.decode(ids, skip_special_tokens=True).strip()
        condition = f"{row.condition}_FIXED_OUTPUT64"
        case_id = f"{condition}|{row.row_id}|B{int(row.budget)}"
        rows.append({"case_id": case_id, "row_id": str(row.row_id), "panel": "EXTERNAL",
            "family": "BudgetLeak-Z", "condition": condition, "member": int(row.member),
            "target_rank": 0, "response": response, "reference": str(row.reference), "budget": int(row.budget)})
    responses = pd.DataFrame(rows)
    rank_source = pd.read_csv(EXP176 / "tables/TABLE_176_04_BUDGETLEAK_BY_RANK.csv")
    # Recover exact per-row target ranks from the frozen Exp193 case substrate.
    cases193 = pd.read_pickle(PROJECT / "exp193_top4_context_rebase_20260829/private/EXP193_CASES.private.pkl.gz", compression="gzip")
    ranks = cases193[cases193.family.eq("BudgetLeak-Z")].groupby("row_id").target_rank.first().to_dict()
    responses["target_rank"] = responses.row_id.map(ranks).fillna(0).astype(int)
    cases = responses[["case_id", "reference", "budget", "panel"]].copy()
    response_input = responses.drop(columns=["reference", "budget"])
    module = load_module(); metrics = module.external_response_metrics(response_input, cases)
    scores = module.external_attack_scores(metrics)
    output = []
    for condition, cell in scores.groupby("condition"):
        raw, effective = metric(cell.member, cell.attack_score)
        output.append({"condition": condition, "stratum": "aggregate", "n": len(cell),
                       "member_n": int(cell.member.sum()), "nonmember_n": int((cell.member == 0).sum()),
                       "raw_auc": raw, "effective_auc": effective})
        nonmember = cell[cell.member.eq(0)]
        for rank in (1, 2, 3, 4):
            subset = pd.concat([cell[(cell.member.eq(1)) & cell.target_rank.eq(rank)], nonmember], ignore_index=True)
            raw, effective = metric(subset.member, subset.attack_score)
            output.append({"condition": condition, "stratum": f"rank{rank}", "n": len(subset),
                           "member_n": int(subset.member.sum()), "nonmember_n": int((subset.member == 0).sum()),
                           "raw_auc": raw, "effective_auc": effective})
    result = pd.DataFrame(output); ROOT.joinpath("tables").mkdir(parents=True, exist_ok=True)
    result.to_csv(ROOT / "tables/TABLE_200_01_BUDGETLEAK_FIXED_OUTPUT.csv", index=False)
    invariance = responses.groupby(["condition", "row_id"]).response.nunique().eq(1).groupby(level=0).mean()
    gates = []
    for condition in sorted(result.condition.unique()):
        aggregate = float(result[(result.condition.eq(condition)) & result.stratum.eq("aggregate")].effective_auc.iloc[0])
        worst = float(result[(result.condition.eq(condition)) & result.stratum.ne("aggregate")].effective_auc.max())
        inv = float(invariance.loc[condition]); passed = (aggregate <= config["gates"]["aggregate_effective_auc_max"] and
            worst <= config["gates"]["each_rank_effective_auc_max"] and inv == 1.0)
        gates.append({"condition": condition, "aggregate_effective_auc": aggregate,
            "worst_rank_effective_auc": worst, "response_invariance": inv, "passed": passed})
    gate = pd.DataFrame(gates); gate.to_csv(ROOT / "tables/TABLE_200_02_GATES.csv", index=False)
    verdict = "FIXED_OUTPUT_CAUSAL_GO" if gate.passed.any() else "FIXED_OUTPUT_CAUSAL_NOT_SUFFICIENT"
    final = {"experiment": "Exp200", "verdict": verdict, "new_generation": 0,
        "conditions": gate.to_dict("records"), "completed_utc": datetime.now(timezone.utc).isoformat()}
    (ROOT / "FINAL_RESULT.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    (ROOT / "STATUS.md").write_text(f"# Exp200 Status\n\n- Stage: **COMPLETE**\n- verdict: `{verdict}`\n")
    report = f"""# Exp200 — Fixed-output BudgetLeak causal diagnostic

- 판정: **{verdict}**
- 새 답변 생성: `0`
- Exp176의 동결 B70 답변을 Qwen 64토큰으로 자르고 모든 요청 budget에 동일하게 반복했다.
- 이는 최종 방어가 아니라 output-budget variation 제거만의 인과 진단이다.

## 결과

{result.to_markdown(index=False)}

## Gates

{gate.to_markdown(index=False)}
"""
    (ROOT / "reports/REPORT_200_FINAL_KO.md").write_text(report)
    print(json.dumps(final, indent=2))


if __name__ == "__main__": main()
