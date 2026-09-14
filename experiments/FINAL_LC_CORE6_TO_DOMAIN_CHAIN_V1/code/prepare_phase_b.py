#!/usr/bin/env python3
"""Create the Phase-B matched-budget E2E precommit after and only after Phase A passes."""
from __future__ import annotations

import json
from pathlib import Path

from common import EXP, FINAL8, LC, QWEN, ROOT, checkpoint, freeze_json, now, sha_file, verify_hashed_json


def item(path: Path, **extra: object) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing Phase-B input: {path}")
    return {"path": str(path), "sha256": sha_file(path), **extra}


def main() -> None:
    phase_a = json.loads((EXP / "PHASE_A_RESULT.json").read_text(encoding="utf-8"))
    if phase_a["verdict"] != "CORE6_DETECTION_PASS":
        raise RuntimeError("Phase B prohibited because Phase A did not pass")
    frozen = verify_hashed_json(EXP / "configs" / "FINAL_LC_FROZEN_MANIFEST.json")
    thresholds = {}
    for row in phase_a["benign_alarm_rows"]:
        thresholds[f"{row['method']}@{row['nominal_budget']}"] = {
            "threshold": row["threshold"], "strict_operator": ">", "benign_alarms": row["benign_alarms"],
            "actual_benign_fpr": row["actual_benign_fpr"],
        }
    code_paths = [EXP / "code" / name for name in (
        "common.py", "prepare_phase_b.py", "run_phase_b_generation.py", "run_phase_b_scoring.py")]
    if not all(path.is_file() for path in code_paths):
        raise RuntimeError("Phase-B code set incomplete")
    pre = {
        "campaign": EXP.name, "phase": "B_CORE6_MATCHED_BUDGET_E2E", "created_utc": now(),
        "status": "PHASE_B_PRECOMMITTED_AFTER_PHASE_A_PASS",
        "phase_a_result": item(EXP / "PHASE_A_RESULT.json"),
        "final_lc_manifest": item(EXP / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
        "final_lc_source_manifest": frozen["source_of_truth"],
        "query_bundle": item(EXP / "configs" / "STANDARDIZED_QUERY_BUNDLE_MANIFEST.json"),
        "new_detection": item(EXP / "cache" / "STANDARDIZED_RETRIEVAL_AND_DETECTION.jsonl", n=34000),
        "old_detection": item(FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl", n=17000),
        "old_answers": item(FINAL8 / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl", n=68000),
        "old_query_manifest": item(FINAL8 / "inputs" / "FINAL_SUPPORTED_QUERY_MANIFEST.jsonl", n=17000),
        "old_menta_evidence": item(FINAL8 / "tables" / "MENTA_QUERY_EVIDENCE.csv"),
        "targets": item(LC / "inputs" / "LARGE_SHARED_TARGETS.csv", n=2000),
        "conditions": ["NO_DEFENSE", "MIRABEL_MATCHED_2_5", "FINAL_LC_MATCHED_2_5", "ORIGINAL_MIRABEL_FIXED"],
        "thresholds": thresholds, "primary_budget": .025,
        "branching": {"A0": "frozen normal Top-4 context", "A_HIDE": "remove MIRABEL/retriever top-1 source and deterministic waterfill",
                      "condition_composition": "alarm false selects A0; alarm true selects A_HIDE",
                      "old_core4": "exact prior A0/A_HIDE answers and scorer intermediates reused by answer hash/text",
                      "new_dcmi_ia": "generate A0 and the union-required A_HIDE branch only"},
        "generator": frozen["generator"], "generator_config": item(QWEN / "config.json"),
        "ia_ground_truth": {"method": "separate frozen local-Qwen shadow pass on source context + frozen 15 questions",
                            "allowed_labels": ["Yes", "No"], "attempts": "maximum 2 fixed parser-repair prompt", "failure": "fail-close"},
        "native_scoring": {
            "MEntA": "frozen paper-faithful NLI evidence and Q5 aggregation",
            "MBA": "mask reconstruction accuracy; every malformed/incomplete/IDK/partial output maps to 0; N=2000 retained",
            "RAG-MIA": "paper-faithful Yes/No black-box score",
            "S²-MIA": "frozen reference-split BLEU/perplexity balanced accuracy",
            "DCMI-Std-Q2": "original Yes indicator minus perturbed Yes indicator; ambiguous/IDK maps to 0; session ROC-AUC",
            "IA-Std-Q15": "sum(correct)-5*sum(UNK), divided by 15; session ROC-AUC",
        },
        "primary_metric": "native fixed-polarity ROC-AUC or S2 balanced accuracy; lower toward 0.5 means less leakage",
        "secondary_metric": "E-AUC=max(AUC,1-AUC), adaptive attacker diagnostic only",
        "bootstrap": {"iterations": 2000, "unit": "target/session", "seed": 20260912},
        "catastrophic_failure_predefinition": "Final LC native value >0.80 for DCMI-Std-Q2 or IA-Std-Q15",
        "gate": {"all6_improve_vs_no_defense": "Final LC native <= No Defense native",
                 "noninferior_to_matched_mirabel": "all Final LC native <= matched MIRABEL native + 0.03",
                 "same_or_better_at_least5": "Final LC native <= matched MIRABEL native for >=5/6",
                 "standardized_no_catastrophe": "DCMI and IA Final LC native <=0.80",
                 "preferred_each": "<=0.65; not a hard gate"},
        "original_mirabel_fixed_interpretation": "deployment reference only; prohibited for same-budget superiority claim",
        "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code_paths},
    }
    digest = freeze_json(EXP / "configs" / "CORE6_MATCHED_BUDGET_E2E_PRECOMMIT.json", pre)
    checkpoint("PHASE_B_PRECOMMIT_COMPLETE", precommit_sha256=digest, phase_a_verdict=phase_a["verdict"],
               primary_budget=.025, model_unchanged=True)
    print(json.dumps({"phase_b_precommit_sha256": digest, "thresholds": thresholds}, indent=2))


if __name__ == "__main__":
    main()

