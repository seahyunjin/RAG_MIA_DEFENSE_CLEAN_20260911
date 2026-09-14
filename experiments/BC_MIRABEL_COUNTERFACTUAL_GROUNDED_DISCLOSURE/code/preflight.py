#!/usr/bin/env python3
"""Fail-close preflight for the BC-CGD small E2E campaign."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd


WORK = Path("/home/traffic_3/workspace/workspace/SH")
ROOT = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/BC_MIRABEL_COUNTERFACTUAL_GROUNDED_DISCLOSURE"
PARENT = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/OUTPUT_CONDITIONED_SELECTIVE_LOO_GUARD_CLEAN_V1"
DV = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/DV_LOO_V2_PHASE1"
CODE = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/code/menta_official"
NLI = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7")
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")


EXPECTED = {
    PARENT / "configs/PRECOMMIT.json": "479fa2db024ca869ed883e0ee791647340c550fbc67147db671ef9c891d7a016",
    PARENT / "private/PHASE1_QUERY_SCORES.csv.gz": "5b843f73ab392b84f16f92f5886b81b23309533c3de909ea7b222903a60bd540",
    PARENT / "private/PHASE1_RETRIEVAL.csv.gz": "787302b51f45b7162f7938d5e2307f6c2f7914c9bb67ef01d0531737661d3b4d",
    PARENT / "private/PHASE1_BENIGN_SELECTION.csv.gz": "c7524bcfe6756c619ae1f4cd245e60ac411b7a97b99e4961c68c8c4d86462310",
    PARENT / "private/PHASE1_ATTACK_SELECTION.csv.gz": "6533d76513b5913f864d5e720bb8b7bb035dd9424fb1fdc20f7b8adb7424985c",
    PARENT / "private/A0_GENERATIONS.sqlite3": "9e4835bc7fa6c347cc246e2e528bbd68d0c5ce61c490c5118d280cb19327c9dd",
    PARENT / "private/QLL_SCORES.sqlite3": "04a64bba7921fdd43bd45023dfb242da051753202d433213e4ad611fb99f53bd",
    PARENT / "private/LOO_SCORES.sqlite3": "a7a2077012b9f23f33492ce00d1c1502ea1cf5b3637f0f1c20410951459b5105",
    DV / "private/MIRABEL_GUMBEL_MARGIN.csv.gz": "101d7257eb6aca4e4b11674c4a15c2728aa763664a338fe47e2b282cc5c394f5",
    CODE / "MEntA/evaluate.py": "830f57407c09d9a932a5e4eb5b7ee0ac653d2065ecff4c10241620dea90438b9",
    CODE / "S2-MIA/evaluate.py": "fe373049d977cdec30daed2013ff495bd5ad61174fd889bbf2721bcaee75fb43",
    CODE / "MBA/evaluate.py": "0523e798013a35247b6ae5583b03c15347f5f45dfa3e8b835ac31623a253104d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def table_count(database: Path, table: str) -> int:
    connection = sqlite3.connect(database)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def main() -> None:
    ROOT.joinpath("audits").mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    for path, expected in EXPECTED.items():
        actual = sha256_file(path) if path.exists() else "MISSING"
        checks.append({"requirement": f"frozen:{path.name}", "pass": actual == expected,
                       "detail": actual})

    scores = pd.read_csv(PARENT / "private/PHASE1_QUERY_SCORES.csv.gz", keep_default_na=False)
    retrieval = pd.read_csv(PARENT / "private/PHASE1_RETRIEVAL.csv.gz", keep_default_na=False)
    benign = pd.read_csv(PARENT / "private/PHASE1_BENIGN_SELECTION.csv.gz", keep_default_na=False)
    mirabel = pd.read_csv(DV / "private/MIRABEL_GUMBEL_MARGIN.csv.gz", keep_default_na=False)
    checks.extend([
        {"requirement": "clean_query_lineage", "pass": len(scores) == 780 and scores.case_id.nunique() == 780,
         "detail": f"rows={len(scores)}, unique={scores.case_id.nunique()}"},
        {"requirement": "retrieval_lineage", "pass": len(retrieval) == 780 and set(scores.case_id) == set(retrieval.case_id),
         "detail": f"rows={len(retrieval)}"},
        {"requirement": "mirabel_reconstruction_identity", "pass": len(mirabel) == 780 and bool(mirabel.top1_match.all()),
         "detail": f"top1={int(mirabel.top1_match.sum())}/{len(mirabel)}"},
        {"requirement": "a0_coverage", "pass": table_count(PARENT / "private/A0_GENERATIONS.sqlite3", "answer") == 780,
         "detail": f"rows={table_count(PARENT / 'private/A0_GENERATIONS.sqlite3', 'answer')}"},
        {"requirement": "qll_coverage", "pass": table_count(PARENT / "private/QLL_SCORES.sqlite3", "score") == 3120,
         "detail": f"rows={table_count(PARENT / 'private/QLL_SCORES.sqlite3', 'score')}"},
        {"requirement": "loo_coverage", "pass": table_count(PARENT / "private/LOO_SCORES.sqlite3", "score") == 3900,
         "detail": f"rows={table_count(PARENT / 'private/LOO_SCORES.sqlite3', 'score')}"},
    ])
    split = benign.split.value_counts().to_dict()
    checks.append({"requirement": "benign_calibration_holdout", "pass": split == {"CALIBRATION": 250, "HOLDOUT": 250},
                   "detail": json.dumps(split, sort_keys=True)})

    nli_config = json.loads((NLI / "config.json").read_text(encoding="utf-8"))
    expected_labels = {"0": "contradiction", "1": "entailment", "2": "neutral"}
    checks.append({"requirement": "nli_orientation", "pass": nli_config.get("id2label") == expected_labels,
                   "detail": json.dumps(nli_config.get("id2label"), sort_keys=True)})
    checks.append({"requirement": "qwen_checkpoint", "pass": (QWEN / "config.json").exists(),
                   "detail": str(QWEN)})

    reference_columns = {"answer", "answers", "gold", "gold_answer", "reference", "reference_answer"}
    present = sorted(reference_columns.intersection(benign.columns))
    checks.append({
        "requirement": "benign_gold_answer_provenance",
        "pass": not present,
        "detail": (
            "No gold answer is present, as expected. SMALL is precommitted to "
            "Answer Preservation Token-F1(A_final,A0), exact string match (report only), "
            "groundedness retention, unsupported-claim increase, and new refusal. "
            "It will not use the Gold-F1/EM names or claim QA correctness."
        ),
    })

    frozen_external_features = [
        "total_answer_length", "mean_answer_length", "refusal_indicator",
        "exact_idk_indicator", "empty_answer_indicator", "sentence_count",
    ]
    checks.append({
        "requirement": "side_channel_hard_gate_numeric_definition",
        "pass": True,
        "detail": json.dumps({
            "features": frozen_external_features,
            "aggregation": "single-feature membership E-AUC; maximum over features and attack families",
            "hard_gate": "worst_external_sidechannel <= 0.65",
            "internal_alarm": "diagnostic only",
        }, sort_keys=True),
    })

    frame = pd.DataFrame(checks)
    frame.to_csv(ROOT / "audits/PREFLIGHT_CHECKS.csv", index=False)
    failed = frame.loc[~frame["pass"], "requirement"].tolist()
    verdict = "BC_CGD_PREFLIGHT_PASS" if not failed else "BC_CGD_PREFLIGHT_FAILED"
    payload = {
        "campaign": "BC_MIRABEL_COUNTERFACTUAL_GROUNDED_DISCLOSURE",
        "verdict": verdict,
        "checks_total": len(frame),
        "checks_passed": int(frame["pass"].sum()),
        "failed_requirements": failed,
        "precommit_written": False,
        "small_e2e_started": False,
        "protected_generation_count": 0,
        "paid_api_calls": 0,
    }
    (ROOT / "PREFLIGHT_RESULT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "STATUS.md").write_text(
        "# BC-CGD\n\n"
        f"- Stage: **{verdict}**\n"
        f"- Passed checks: `{payload['checks_passed']}/{payload['checks_total']}`\n"
        f"- Missing: `{', '.join(failed)}`\n"
        "- Precommit: `NOT_WRITTEN`\n"
        "- Small E2E: `NOT_STARTED`\n", encoding="utf-8")
    if verdict == "BC_CGD_PREFLIGHT_PASS":
        report = (
            "# BC-CGD 독립 사전 감사\n\n"
            "## 최종 판정\n\n"
            "`BC_CGD_PREFLIGHT_PASS`\n\n"
            f"23개 검사 중 **{payload['checks_passed']}/23**이 통과했다. "
            "기존 21개 lineage·checkpoint·scorer 검사는 유지됐고, "
            "사용자가 결과 확인 전에 지정한 두 명세가 추가로 고정됐다.\n\n"
            "1. Small 정상 utility는 Gold-F1/EM이 아니라 "
            "`Answer Preservation Token-F1(A_final, A0)`을 hard metric으로 사용한다. "
            "Exact Match는 보고만 하며 QA correctness를 주장하지 않는다.\n"
            "2. 외부 side-channel은 답변 길이 합/평균, refusal, exact IDK, empty, "
            "sentence count의 단독 membership E-AUC 최댓값으로 계산하며 hard gate는 0.65다. "
            "internal alarm은 diagnostic-only다.\n\n"
            "## 동결 Small gate\n\n"
            "- intervention <= 5.1%\n"
            "- Answer Preservation Token-F1 >= 95%\n"
            "- groundedness retention >= 95%\n"
            "- unsupported claim increase <= +2%p\n"
            "- new refusal <= 3%\n"
            "- worst external side-channel E-AUC <= 0.65\n\n"
            "Small PASS 후에만 gold-answer QA calibration/test >=1,000을 별도 confirmatory evaluation으로 연다.\n"
        )
    else:
        report = (
            "# BC-CGD 독립 사전 감사\n\n"
            f"- Verdict: `{verdict}`\n"
            f"- Failed: `{', '.join(failed)}`\n"
        )
    (ROOT / "audits/PREFLIGHT_REPORT_KO.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
