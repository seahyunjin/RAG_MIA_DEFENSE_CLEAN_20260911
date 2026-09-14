#!/usr/bin/env python3
"""Score the frozen Core5 E2E branches with each attack's native metric."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from common import BUDGETS, EXP, LC, ROOT, SEED, atomic_json, checkpoint, now, read_jsonl, sha_file, verify_hashed_json, write_csv
from run_phase_b_scoring import (alarm, auc_record, bleu, branch_maps, complete_menta_evidence,
                                 conditions, fit_s2, mask_accuracy_or_zero, normalize_yes_no)

CORE5 = ("MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2")


def score_all(pre: dict, answer_map: dict) -> tuple[list[dict], list[dict]]:
    query_map = {row["query_id"]: row for row in read_jsonl(Path(pre["old_query_manifest"]["path"]))
                 if row["attack"] in set(CORE5[:-1])}
    query_map.update({row["query_id"]: row for row in read_jsonl(Path(pre["dcmi_queries"]["path"]))})
    target_map = {row["document_id"]: row for row in csv.DictReader((LC / "inputs" / "LARGE_SHARED_TARGETS.csv").open(encoding="utf-8"))}
    scores, missing = [], []

    evidence_map = complete_menta_evidence(pre, answer_map, query_map, target_map)
    for condition in conditions():
        grouped = defaultdict(list)
        for query_id, item in query_map.items():
            if item["attack"] != "MEntA":
                continue
            answer = answer_map[(query_id, condition)]["answer"]
            entailed, idk = evidence_map[(query_id, answer)]
            grouped[item["session_id"]].append((int(item["query_index"]), entailed, idk, item))
        rows = []
        for session, group in grouped.items():
            group.sort()
            if len(group) != 5:
                raise RuntimeError("MEntA Q5 drift")
            score = sum(-1 if idk else int(ent) for _, ent, idk, _ in group) / 5
            rows.append({"session_id": session, "membership": group[0][3]["membership"], "score": score})
        scores.append(auc_record("MEntA", condition, rows, "paper-faithful MEntA native ROC-AUC"))

    for condition in conditions():
        rows = []
        for query_id, item in query_map.items():
            if item["attack"] != "MBA":
                continue
            answer = answer_map[(query_id, condition)]["answer"]
            score, bad, error = mask_accuracy_or_zero(answer, item["mask_answers"])
            rows.append({"session_id": item["session_id"], "membership": item["membership"], "score": score})
            missing.append({"condition": condition, "query_id": query_id, "membership": item["membership"],
                            "malformed_or_incomplete": bad, "assigned_score": score, "error": error})
        if len(rows) != 2000:
            raise RuntimeError("MBA N must remain 2000")
        scores.append(auc_record("MBA", condition, rows, "paper-faithful reconstruction ROC-AUC; missing=incorrect"))

    for condition in conditions():
        rows = []
        for query_id, item in query_map.items():
            if item["attack"] == "RAG-MIA":
                response = normalize_yes_no(answer_map[(query_id, condition)]["answer"])
                rows.append({"session_id": item["session_id"], "membership": item["membership"],
                             "score": float(response == "Yes")})
        scores.append(auc_record("RAG-MIA", condition, rows, "paper-faithful Yes/No ROC-AUC"))

    for condition in conditions():
        detail = []
        for query_id, item in query_map.items():
            if item["attack"] != "S²-MIA":
                continue
            answer = answer_map[(query_id, condition)]
            if answer.get("perplexity") is None:
                raise RuntimeError(f"S2 perplexity missing {query_id}")
            detail.append({"split": item["evaluation_split"], "label": int(item["membership"] == "member"),
                           "bleu": bleu(item["s2_full_target"], answer["answer"]), "perplexity": float(answer["perplexity"])})
        reference = [row for row in detail if row["split"] == "S2_REFERENCE"]
        evaluation = [row for row in detail if row["split"] == "S2_EVALUATION"]
        ref_bacc, bt, pt = fit_s2(reference)
        labels = np.asarray([row["label"] for row in evaluation], dtype=int)
        pred = np.asarray([int(row["bleu"] >= bt and row["perplexity"] <= pt) for row in evaluation], dtype=int)
        value = float(balanced_accuracy_score(labels, pred))
        member, nonmember = np.where(labels == 1)[0], np.where(labels == 0)[0]
        rng = np.random.default_rng(SEED + sum(map(ord, condition)))
        boot = np.empty(2000)
        for index in range(2000):
            chosen = np.concatenate((rng.choice(member, len(member), replace=True), rng.choice(nonmember, len(nonmember), replace=True)))
            boot[index] = balanced_accuracy_score(labels[chosen], pred[chosen])
        scores.append({"attack": "S²-MIA", "condition": condition,
                       "native_metric": "S2-MIA-T balanced accuracy", "native_value": value,
                       "raw_auc": float(roc_auc_score(labels, pred)), "e_auc_secondary": max(value, 1 - value),
                       "ci95_low": float(np.quantile(boot, .025)), "ci95_high": float(np.quantile(boot, .975)),
                       "valid_n": len(evaluation), "member_n": len(member), "nonmember_n": len(nonmember),
                       "reference_n": len(reference), "reference_bacc": ref_bacc,
                       "bleu_threshold": bt, "perplexity_threshold": pt})

    for condition in conditions():
        grouped = defaultdict(list)
        for query_id, item in query_map.items():
            if item["attack"] != "DCMI-Std-Q2":
                continue
            response = normalize_yes_no(answer_map[(query_id, condition)]["answer"])
            grouped[item["session_id"]].append((int(item["query_index"]), item["variant"], response, item))
        rows = []
        for session, group in grouped.items():
            group.sort()
            if len(group) != 2 or {row[1] for row in group} != {"original", "perturbed"}:
                raise RuntimeError("DCMI Q2 drift")
            response = {variant: value for _, variant, value, _ in group}
            score = float(response["original"] == "Yes") - float(response["perturbed"] == "Yes")
            rows.append({"session_id": session, "membership": group[0][3]["membership"], "score": score})
        scores.append(auc_record("DCMI-Std-Q2", condition, rows, "standardized DCMI differential ROC-AUC"))
    return scores, missing


def main() -> None:
    pre = verify_hashed_json(EXP / "configs" / "CORE5_MATCHED_BUDGET_E2E_PRECOMMIT.json")
    for relative, expected in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != expected:
            raise RuntimeError(f"Core5 Phase-B code drift: {relative}")
    generation = json.loads((EXP / "runtime" / "PHASE_B_GENERATION_MANIFEST.json").read_text(encoding="utf-8"))
    if sha_file(EXP / "runtime" / "STANDARDIZED_BRANCH_ANSWERS.jsonl") != generation["answers_sha256"]:
        raise RuntimeError("Core5 answer artifact drift")
    checkpoint("CORE5_SCORING_STARTED", ia_excluded=True)
    answer_map, _, _ = branch_maps(pre)
    scores, missing = score_all(pre, answer_map)
    write_csv(EXP / "tables" / "CORE5_E2E_ALL_BUDGETS.csv", scores)
    write_csv(EXP / "tables" / "CORE5_MBA_MISSINGNESS_ALL_CONDITIONS.csv", missing)
    by = {(row["attack"], row["condition"]): row for row in scores}
    primary = []
    for attack in CORE5:
        nd, mir = by[(attack, "NO_DEFENSE")], by[(attack, "MIRABEL_MATCHED_2_5")]
        lc, orig = by[(attack, "FINAL_LC_MATCHED_2_5")], by[(attack, "ORIGINAL_MIRABEL_FIXED")]
        primary.append({"attack": attack, "no_defense": nd["native_value"],
                        "mirabel_matched_2_5": mir["native_value"], "final_lc_matched_2_5": lc["native_value"],
                        "delta_lc_vs_mirabel": lc["native_value"] - mir["native_value"],
                        "original_mirabel_fixed_reference": orig["native_value"], "native_metric": lc["native_metric"],
                        "lc_ci95_low": lc["ci95_low"], "lc_ci95_high": lc["ci95_high"],
                        "valid_n": lc["valid_n"], "lc_e_auc_secondary": lc["e_auc_secondary"]})
    write_csv(EXP / "tables" / "CORE5_MATCHED_BUDGET_E2E_PRIVACY.csv", primary)
    checks = {"all5_improve_vs_no_defense": all(row["final_lc_matched_2_5"] <= row["no_defense"] + 1e-12 for row in primary),
              "all5_noninferior_to_matched_mirabel": all(row["final_lc_matched_2_5"] <= row["mirabel_matched_2_5"] + .03 + 1e-12 for row in primary),
              "same_or_better_at_least4": sum(row["final_lc_matched_2_5"] <= row["mirabel_matched_2_5"] + 1e-12 for row in primary) >= 4,
              "dcmi_no_catastrophe": by[("DCMI-Std-Q2", "FINAL_LC_MATCHED_2_5")]["native_value"] <= .80}
    passed = all(checks.values())
    verdict = "CORE5_MATCHED_BUDGET_E2E_PASS" if passed else "CORE5_E2E_PRIVACY_FAILED"
    result = {"campaign": EXP.name, "phase": "B_CORE5_MATCHED_BUDGET_E2E", "verdict": verdict,
              "completed_utc": now(), "primary_budget": .025, "primary": primary, "checks": checks,
              "preferred_at_or_below_0_65": sum(row["final_lc_matched_2_5"] <= .65 for row in primary),
              "ia_status": "EXCLUDED_PARSER_FAILURE_NO_REGENERATION", "mba_n_excluded": 0,
              "next_stage": "PHASE_C_GOLD_QA_CHARACTERIZATION_REGARDLESS_OF_CORE5_GATE"}
    atomic_json(EXP / "PHASE_B_RESULT.json", result)
    checkpoint(verdict, passed=passed, preferred_at_or_below_0_65=result["preferred_at_or_below_0_65"],
               ia_excluded=True, next_stage=result["next_stage"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
