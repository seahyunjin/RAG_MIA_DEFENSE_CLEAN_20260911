#!/usr/bin/env python3
"""Build the 2026-09-30 ACL audit from frozen, read-only artifacts.

No model inference, retrieval, threshold calibration, or defense tuning occurs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "releases/2026-09-30/acl_final_evidence"
OUT.mkdir(parents=True, exist_ok=True)

TAU = 0.0901547372341156
GAMMA = 9.565910959243753
N_BOOT = 5000
IA_BOOT_SEED = 20261209
MENTA_BOOT_SEED = 20260935


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(name: str, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(pos, neg) -> float:
    p = np.asarray(pos, dtype=float)
    n = np.asarray(neg, dtype=float)
    return float(np.mean(p[:, None] > n[None, :]) + 0.5 * np.mean(p[:, None] == n[None, :]))


def normalize_yn(text: str) -> str:
    value = " ".join(str(text).casefold().replace("’", "'").split())
    if any(
        phrase in value
        for phrase in (
            "i don't know",
            "i do not know",
            "not enough information",
            "cannot determine",
            "can't determine",
            "unknown",
            "not specified",
            "insufficient information",
        )
    ):
        return "Unknown"
    if re.match(r"^\s*yes\b", value):
        return "Yes"
    if re.match(r"^\s*no\b", value):
        return "No"
    return "Unknown"


def percentile(values) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def ia_ablation() -> tuple[list[dict], dict]:
    source = ROOT / "paper2_csi_guard/35_query_and_ia_validity_audit/01_IA_EXACT_GENERATION.jsonl"
    methods = ("NO_DEFENSE", "CLD_ONLY", "EXACT_OLD_DUAL")
    rows = [row for row in read_jsonl(source) if row["method"] in methods]
    scored = []
    for row in rows:
        answer = normalize_yn(row["response"])
        scored.append(
            {
                **row,
                "attack_score": int(answer == row["ground_truth"]) - 5 * int(answer == "Unknown"),
            }
        )
    targets = sorted({row["target_document_id"] for row in scored})
    lookup = {
        (row["method"], row["target_document_id"], int(row["query_position"]), row["world"]): row
        for row in scored
    }
    expected = len(methods) * len(targets) * 5 * 2
    if len(lookup) != expected:
        raise RuntimeError(f"IA exact coverage mismatch: {len(lookup)} != {expected}")
    arrays = {}
    for method in methods:
        for q in range(1, 6):
            pos = np.asarray(
                [
                    np.mean([lookup[(method, target, k, "T")]["attack_score"] for k in range(1, q + 1)])
                    for target in targets
                ]
            )
            neg = np.asarray(
                [
                    np.mean([lookup[(method, target, k, "D")]["attack_score"] for k in range(1, q + 1)])
                    for target in targets
                ]
            )
            arrays[(method, q)] = (pos, neg)

    rng = np.random.default_rng(IA_BOOT_SEED)
    distributions = {(method, q): [] for method in methods for q in range(1, 6)}
    no_minus_rif = {q: [] for q in range(1, 6)}
    cld_minus_rif = {q: [] for q in range(1, 6)}
    for _ in range(N_BOOT):
        indices = rng.integers(0, len(targets), len(targets))
        values = {}
        for method in methods:
            for q in range(1, 6):
                pos, neg = arrays[(method, q)]
                values[(method, q)] = auc(pos[indices], neg[indices])
                distributions[(method, q)].append(values[(method, q)])
        for q in range(1, 6):
            no_minus_rif[q].append(values[("NO_DEFENSE", q)] - values[("EXACT_OLD_DUAL", q)])
            cld_minus_rif[q].append(values[("CLD_ONLY", q)] - values[("EXACT_OLD_DUAL", q)])

    output = []
    label = {"NO_DEFENSE": "No Defense", "CLD_ONLY": "CLD", "EXACT_OLD_DUAL": "RIF"}
    for method in methods:
        for q in range(1, 6):
            point = auc(*arrays[(method, q)])
            lo, hi = percentile(distributions[(method, q)])
            nr_lo, nr_hi = percentile(no_minus_rif[q])
            cr_lo, cr_hi = percentile(cld_minus_rif[q])
            output.append(
                {
                    "attack": "IA",
                    "cohort": "92 paired target sessions; ConvFinQA local protocol",
                    "method": label[method],
                    "Q": q,
                    "ROC_AUC": point,
                    "AUC_CI95_low": lo,
                    "AUC_CI95_high": hi,
                    "NoDefense_minus_RIF": auc(*arrays[("NO_DEFENSE", q)]) - auc(*arrays[("EXACT_OLD_DUAL", q)]),
                    "NoDefense_minus_RIF_CI95_low": nr_lo,
                    "NoDefense_minus_RIF_CI95_high": nr_hi,
                    "Delta_CF_given_CLD_CLD_minus_RIF": auc(*arrays[("CLD_ONLY", q)]) - auc(*arrays[("EXACT_OLD_DUAL", q)]),
                    "Delta_CF_given_CLD_CI95_low": cr_lo,
                    "Delta_CF_given_CLD_CI95_high": cr_hi,
                    "Q1_to_Q5_amplification": point - auc(*arrays[(method, 1)]) if q == 5 else "",
                    "lineage": "SERIAL_KVCACHE_EXACT",
                    "paired_unit": "target session (T/D resampled together)",
                    "bootstrap_replicates": N_BOOT,
                }
            )

    supplementary = read_csv(ROOT / "paper2_csi_guard/16_dual_surface_gate/DUAL_SURFACE_PRIVACY.csv")
    for row in supplementary:
        if row["Attack"] == "IA" and row["Arm"] == "CFCLIP_ONLY":
            output.append(
                {
                    "attack": "IA",
                    "cohort": "same 92 target identities; supplementary batched four-arm",
                    "method": "CF-Clip only",
                    "Q": int(row["Q"]),
                    "ROC_AUC": float(row["AUROC"]),
                    "AUC_CI95_low": float(row["CI_low"]),
                    "AUC_CI95_high": float(row["CI_high"]),
                    "NoDefense_minus_RIF": "",
                    "NoDefense_minus_RIF_CI95_low": "",
                    "NoDefense_minus_RIF_CI95_high": "",
                    "Delta_CF_given_CLD_CLD_minus_RIF": "",
                    "Delta_CF_given_CLD_CI95_low": "",
                    "Delta_CF_given_CLD_CI95_high": "",
                    "Q1_to_Q5_amplification": "",
                    "lineage": "SUPPLEMENTARY_BATCHED_FOUR_ARM_NOT_EXACT_RIF_LINEAGE",
                    "paired_unit": "target session",
                    "bootstrap_replicates": N_BOOT,
                }
            )
    summary = {
        "q5_cld_minus_rif": auc(*arrays[("CLD_ONLY", 5)]) - auc(*arrays[("EXACT_OLD_DUAL", 5)]),
        "q5_cld_minus_rif_ci95": percentile(cld_minus_rif[5]),
    }
    return sorted(output, key=lambda row: (int(row["Q"]), row["method"])), summary


def menta_ablation() -> tuple[list[dict], dict]:
    table = read_csv(ROOT / "paper2_csi_guard/37_final_task_utility_freeze/12_MENTA_Q1_Q5.csv")
    manifest = read_jsonl(ROOT / "paper2_csi_guard/30_dual_conditioned_calibration/23_FRESH_MENTA_MANIFEST.jsonl")
    evidence = {
        (row["job_id"], row["arm"]): row
        for row in read_jsonl(ROOT / "paper2_csi_guard/30_dual_conditioned_calibration/26_FRESH_MENTA_EVIDENCE.jsonl")
    }
    arms = ("NO_DEFENSE", "CLD_ONLY", "OLD_DUAL")
    grouped = defaultdict(list)
    for row in manifest:
        grouped[row["session_id"]].append(row)
    members = sorted(session for session, rows in grouped.items() if rows[0]["membership"] == "member")
    nonmembers = sorted(session for session, rows in grouped.items() if rows[0]["membership"] == "nonmember")
    arrays = {}
    for arm in arms:
        for q in range(1, 6):
            values = {}
            for session, rows in grouped.items():
                prefix = sorted(rows, key=lambda item: int(item["query_index"]))[:q]
                values[session] = float(
                    np.mean(
                        [
                            -1
                            if int(evidence[(job["job_id"], arm)]["idk"])
                            else int(evidence[(job["job_id"], arm)]["entailed"])
                            for job in prefix
                        ]
                    )
                )
            arrays[(arm, q)] = (
                np.asarray([values[session] for session in members]),
                np.asarray([values[session] for session in nonmembers]),
            )

    rng = np.random.default_rng(MENTA_BOOT_SEED)
    distributions = {(arm, q): [] for arm in arms for q in range(1, 6)}
    no_minus_rif = {q: [] for q in range(1, 6)}
    cld_minus_rif = {q: [] for q in range(1, 6)}
    for _ in range(N_BOOT):
        member_indices = rng.integers(0, len(members), len(members))
        nonmember_indices = rng.integers(0, len(nonmembers), len(nonmembers))
        values = {}
        for arm in arms:
            for q in range(1, 6):
                pos, neg = arrays[(arm, q)]
                values[(arm, q)] = auc(pos[member_indices], neg[nonmember_indices])
                distributions[(arm, q)].append(values[(arm, q)])
        for q in range(1, 6):
            no_minus_rif[q].append(values[("NO_DEFENSE", q)] - values[("OLD_DUAL", q)])
            cld_minus_rif[q].append(values[("CLD_ONLY", q)] - values[("OLD_DUAL", q)])

    label = {"NO_DEFENSE": "No Defense", "CLD_ONLY": "CLD", "OLD_DUAL": "RIF"}
    output = []
    points = {(row["arm"], int(row["Q"])): float(row["AUROC"]) for row in table}
    for arm in arms:
        for q in range(1, 6):
            lo, hi = percentile(distributions[(arm, q)])
            nr_lo, nr_hi = percentile(no_minus_rif[q])
            cr_lo, cr_hi = percentile(cld_minus_rif[q])
            output.append(
                {
                    "attack": "MEntA",
                    "cohort": "fresh zero-overlap primary; 100 member + 100 nonmember sessions",
                    "method": label[arm],
                    "Q": q,
                    "ROC_AUC": points[(arm, q)],
                    "AUC_CI95_low": lo,
                    "AUC_CI95_high": hi,
                    "NoDefense_minus_RIF": points[("NO_DEFENSE", q)] - points[("OLD_DUAL", q)],
                    "NoDefense_minus_RIF_CI95_low": nr_lo,
                    "NoDefense_minus_RIF_CI95_high": nr_hi,
                    "CLD_minus_RIF": points[("CLD_ONLY", q)] - points[("OLD_DUAL", q)],
                    "CLD_minus_RIF_CI95_low": cr_lo,
                    "CLD_minus_RIF_CI95_high": cr_hi,
                    "Q1_to_Q5_amplification": points[(arm, 5)] - points[(arm, 1)] if q == 5 else "",
                    "lineage": "FRESH_ZERO_OVERLAP_PRIMARY",
                    "paired_unit": "member and nonmember session",
                    "bootstrap_replicates": N_BOOT,
                }
            )

    exploratory = read_csv(ROOT / "paper2_csi_guard/16_dual_surface_gate/DUAL_SURFACE_PRIVACY.csv")
    for row in exploratory:
        if row["Attack"] == "MEntA" and row["Arm"] == "CFCLIP_ONLY":
            output.append(
                {
                    "attack": "MEntA",
                    "cohort": "exploratory reused four-arm cohort; not fresh primary",
                    "method": "CF-Clip only",
                    "Q": int(row["Q"]),
                    "ROC_AUC": float(row["AUROC"]),
                    "AUC_CI95_low": float(row["CI_low"]),
                    "AUC_CI95_high": float(row["CI_high"]),
                    "NoDefense_minus_RIF": "",
                    "NoDefense_minus_RIF_CI95_low": "",
                    "NoDefense_minus_RIF_CI95_high": "",
                    "CLD_minus_RIF": "",
                    "CLD_minus_RIF_CI95_low": "",
                    "CLD_minus_RIF_CI95_high": "",
                    "Q1_to_Q5_amplification": "",
                    "lineage": "EXPLORATORY_REUSE_NOT_PRIMARY",
                    "paired_unit": "session",
                    "bootstrap_replicates": N_BOOT,
                }
            )
    summary = {
        "q5_cld_minus_rif": points[("CLD_ONLY", 5)] - points[("OLD_DUAL", 5)],
        "q5_cld_minus_rif_ci95": percentile(cld_minus_rif[5]),
    }
    return sorted(output, key=lambda row: (int(row["Q"]), row["method"])), summary


def provenance_rows() -> list[dict]:
    model_hash = "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b"
    common = {
        "retriever": "BAAI/bge-m3",
        "generator": "Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "model_implementation_hash": model_hash,
        "tau": TAU,
        "gamma": GAMMA,
        "decoding": "deterministic greedy; frozen prompt/limits in source artifact",
    }
    specs = [
        ("CORE4_PROSPECTIVE", "BEIR/NFCorpus", "MBA; RAG-MIA; S2-MIA; DCMI-Std-Q2", "50/50 DEV per attack (DCMI and S2 100/100 rows)", "native attack scorers", "prospective four-arm v1", "2026-09-27", "DEV frozen-before-generation", "unified_dual_surface_dev_20260927/02_PROSPECTIVE_COHORT.jsonl", "top4 score; generator context frozen by attack"),
        ("IA_Q1_Q5_EXACT", "ConvFinQA", "IA Q1-Q5 local protocol", "92 paired target/decoy sessions", "correct - 5*unknown; prefix mean ROC-AUC", "question-only DIVERSE Q1-Q5", "2026-09-28", "fresh exact", "paper2_csi_guard/35_query_and_ia_validity_audit/01_IA_EXACT_GENERATION.jsonl", "top4 score; 3-doc generator context"),
        ("IA_CF_ONLY_SUPPLEMENT", "ConvFinQA", "IA Q1-Q5 local protocol", "92 paired target/decoy sessions", "correct - 5*unknown; prefix mean ROC-AUC", "supplementary batched four-arm", "2026-09-27", "exploratory reuse", "paper2_csi_guard/16_dual_surface_gate/DUAL_SURFACE_PRIVACY.csv", "top4 score; batched generation"),
        ("MENTA_PRIMARY", "FinQA", "MEntA Q1-Q5", "100 member + 100 nonmember sessions", "entailed/IDK native prefix ROC-AUC", "fresh zero-overlap primary", "2026-09-27", "fresh", "paper2_csi_guard/30_dual_conditioned_calibration/27_FRESH_MENTA_Q1_Q5.csv", "top4 score; 4-doc context, CLD no-refill"),
        ("BENIGN_UTILITY", "MultiDoc2Dial + frozen reference-history cohorts", "ordinary; conversational; true deep-dive", "50 ordinary; 200 conversational; 154 deep-dive", "Token-F1/NLI/refusal/answerability", "corrected frozen utility", "2026-09-28", "frozen", "releases/dualguard-final-20260928/results/FINAL_CORRECTED_BENIGN_UTILITY.csv", "method-specific stored context"),
        ("B2_CONFIRMATORY", "BEIR/NFCorpus", "IA local Q30 surrogate; MEntA Q1-Q5", "fresh disjoint 50+50 per attack", "IA native; MEntA entailment native", "fixed horizon B=2 confirmatory", "2026-09-30", "fresh confirmatory", "fixed_horizon_b2_fresh_nfcorpus_20260929/remainder_eval_20260930/08_FINAL_RESULT.json", "top8 retrieval; first two evidence-bearing RIF calls"),
        ("ADAPTIVE_B2", "BEIR/NFCorpus", "adaptive IA Q30; MEntA Q5", "same fresh B2 50+50 targets", "paired session ROC-AUC", "offline frozen-candidate reorder/reset audit", "2026-09-30", "post-freeze", "adaptive_attack_acl_fixed_horizon_b2_20260930/01_ADAPTIVE_RESULTS.csv", "stored top8 and outputs only"),
        ("MIRABEL_LOW_FPR", "mixed-domain IA audit", "IA Q15 standardized", "976 member + 965 nonmember sessions", "alarm rate/session-any", "matched benign FPR 2.5%", "2026-09-24", "audit", "mirabel_ia_low_fpr_reaudit_cpu/01_operating_points.csv", "stored scores only"),
        ("RETRIEVER_TRANSFER_COMPONENT", "same frozen domains", "Core5 + IA standardized", "stored score transfer", "CLD alarm TPR/FPR", "BGE-M3/GTE-base/MPNet", "2026-09-28", "component-only", "residual_exposure_rif_campaign_20260928/FINAL_RETRIEVER_TRANSFER_KO.md", "no full-RIF generation"),
        ("GENERATOR_TRANSFER_COMPONENT", "same frozen domains", "Core5", "stored generation transfer", "E-AUC/native metrics", "Qwen/Llama CLD component", "2026-09-28", "component-only", "residual_exposure_rif_campaign_20260928/FINAL_GENERATOR_TRANSFER_KO.md", "no Llama CF-Clip token path"),
    ]
    output = []
    for result_id, corpus, attack, cohort, scorer, query_version, date, status, source, topk in specs:
        path = ROOT / source
        output.append(
            {
                "result_id": result_id,
                "corpus": corpus,
                **common,
                "attack": attack,
                "cohort": cohort,
                "scorer": scorer,
                "query_version": query_version,
                "context_top_k": topk,
                "random_seed": "see frozen source manifest",
                "date": date,
                "status": status,
                "source_artifact": source,
                "source_sha256": sha256(path),
                "original_artifact_modified": False,
            }
        )
    return output


def main_results() -> list[dict]:
    output = []
    single = read_csv(ROOT / "releases/dualguard-final-20260928/results/TABLE_SINGLE_QUERY_ATTACKS.csv")
    method_map = {"NO_DEFENSE": "No Defense", "CLD_ONLY": "CLD", "CFCLIP_ONLY": "CF-Clip only", "CLD_CFCLIP": "RIF"}
    for row in single:
        output.append(
            {
                "paper_section": "main single-query",
                "corpus": "BEIR/NFCorpus",
                "attack": row["attack"],
                "cohort": row["cohort"],
                "method": method_map[row["method"]],
                "metric": row["metric"],
                "value": row["AUROC_or_leakage"],
                "NoDefense_minus_method": row["paired_reduction_NoDefense_minus_method"],
                "paired_CI95_low": row["paired_reduction_CI_low"],
                "paired_CI95_high": row["paired_reduction_CI_high"],
                "lineage": row["lineage"],
                "qualification": "S2 reduction CI includes zero" if row["attack"] == "S²-MIA" and row["method"] == "CLD_CFCLIP" else "",
            }
        )
    sessions = read_csv(ROOT / "releases/dualguard-final-20260928/results/TABLE_SESSION_ATTACKS_Q1_Q5.csv")
    for row in sessions:
        if row["Attack"] == "IA" and row["Method"] in ("NO_DEFENSE", "CLD_ONLY", "CFCLIP_ONLY", "EXACT_OLD_DUAL"):
            output.append(
                {
                    "paper_section": "main IA Q5 ablation" if row["Method"] != "CFCLIP_ONLY" else "supplementary IA Q5 arm",
                    "corpus": "ConvFinQA",
                    "attack": "IA Q1-Q5 local protocol",
                    "cohort": "92 paired target sessions",
                    "method": method_map.get(row["Method"], "RIF"),
                    "metric": "ROC-AUC at Q5",
                    "value": row["Q5"],
                    "NoDefense_minus_method": "",
                    "paired_CI95_low": "",
                    "paired_CI95_high": "",
                    "lineage": row["lineage"],
                    "qualification": "CF-only is supplementary batched lineage" if row["Method"] == "CFCLIP_ONLY" else "paper-exact IA not claimed",
                }
            )
        if row["Attack"] == "MEntA" and row["Method"] in ("NO_DEFENSE", "CLD_ONLY", "OLD_DUAL"):
            output.append(
                {
                    "paper_section": "main MEntA Q5",
                    "corpus": "FinQA",
                    "attack": "MEntA Q1-Q5",
                    "cohort": "fresh 100 member + 100 nonmember",
                    "method": method_map.get(row["Method"], "RIF"),
                    "metric": "ROC-AUC at Q5",
                    "value": row["Q5"],
                    "NoDefense_minus_method": "",
                    "paired_CI95_low": "",
                    "paired_CI95_high": "",
                    "lineage": row["lineage"],
                    "qualification": "RIF is 0.01065 worse than CLD at Q5" if row["Method"] == "OLD_DUAL" else "",
                }
            )
    b2 = read_csv(ROOT / "gpt_handoff_fixed_horizon_b2_20260930/02_PERFORMANCE_TABLE.csv")
    for row in b2[:2]:
        for key, method in (("No_Defense", "No Defense"), ("OLD_DUAL", "RIF"), ("FINAL_B2", "RIF + Fixed Horizon B=2")):
            output.append(
                {
                    "paper_section": "optional long-horizon stress test",
                    "corpus": "BEIR/NFCorpus",
                    "attack": row["attack"],
                    "cohort": row["lineage"],
                    "method": method,
                    "metric": row["metric"],
                    "value": row[key],
                    "NoDefense_minus_method": row["NoDefense_minus_Final"] if key == "FINAL_B2" else "",
                    "paired_CI95_low": row["CI95_low"] if key == "FINAL_B2" else "",
                    "paired_CI95_high": row["CI95_high"] if key == "FINAL_B2" else "",
                    "lineage": "FRESH_B2_CONFIRMATORY",
                    "qualification": "local Q30 surrogate; not paper-exact IA" if row["attack"].startswith("IA") else "separate cohort; do not merge with primary FinQA MEntA",
                }
            )
    return output


def utility_rows() -> list[dict]:
    output = []
    rows = read_csv(ROOT / "releases/dualguard-final-20260928/results/FINAL_CORRECTED_BENIGN_UTILITY.csv")
    for row in rows:
        output.append(
            {
                "method_scope": "RIF main utility",
                "protocol": row["protocol"],
                "subset": row["subset"],
                "method": "RIF" if row["method"] == "Frozen Dual" else row["method"],
                "n": row["n"],
                "Token_F1": row["Token_F1"],
                "NLI_support": row["NLI_support"],
                "unsupported": row["unsupported"],
                "contradiction": row["contradiction"],
                "refusal": row["refusal"],
                "answerability": row["answerability"],
                "relative_F1_change_vs_NoDefense": row["relative_F1_change_vs_NoDefense"],
                "support_delta_vs_NoDefense": row["support_delta_vs_NoDefense"],
                "answerability_delta_vs_NoDefense": row["answerability_delta_vs_NoDefense"],
                "answer_change_rate": "not in aggregate source table",
                "claim_boundary": "task-level stored utility; output identity not a gate",
            }
        )
    b2 = read_csv(ROOT / "gpt_handoff_fixed_horizon_b2_20260930/03_UTILITY_TABLE.csv")
    for row in b2:
        output.append(
            {
                "method_scope": "optional B=2 composition safeguard",
                "protocol": "HUMAN_SESSION_AUXILIARY",
                "subset": row["subset"],
                "method": "RIF + Fixed Horizon B=2",
                "n": row["queries"],
                "Token_F1": row["FIXED_HORIZON_B2_Token_F1"],
                "NLI_support": "N/A",
                "unsupported": "N/A",
                "contradiction": "N/A",
                "refusal": row["FIXED_HORIZON_B2_refusal"],
                "answerability": 1.0 - float(row["FIXED_HORIZON_B2_refusal"]),
                "relative_F1_change_vs_NoDefense": "N/A; comparator is RIF",
                "support_delta_vs_NoDefense": "N/A",
                "answerability_delta_vs_NoDefense": -float(row["answerability_loss"]),
                "answer_change_rate": 1.0 - float(row["answer_exact_identity"]),
                "claim_boundary": "explicit composition trade-off; no open-ended utility claim",
            }
        )
    return output


def runtime_rows() -> list[dict]:
    output = []
    rows = read_csv(ROOT / "paper2_csi_guard/37_final_task_utility_freeze/17_RUNTIME.csv")
    name = {"Vanilla": "No Defense", "CLD": "CLD", "Exact Dual": "RIF", "Fast Dual B=8": "Fast RIF (rejected for main)"}
    for row in rows:
        output.append(
            {
                "method": name[row["method"]],
                "engine": "SERIAL_KVCACHE" if row["method"] == "Exact Dual" else ("FAST_B8_NON_MAIN" if row["method"] == "Fast Dual B=8" else "standard frozen generation"),
                "queries": row["queries"],
                "sec_per_query": row["latency_per_query_seconds"],
                "sec_per_token": row["latency_per_token_seconds"],
                "tokens_per_sec": row["tokens_per_second"],
                "peak_VRAM_GiB": row["peak_vram_gib"],
                "scope": "920-query exact IA runtime cohort",
                "qualification": "gold/main" if row["method"] == "Exact Dual" else ("not main: MEntA exactness gate failed" if row["method"] == "Fast Dual B=8" else ""),
            }
        )
    output.append(
        {
            "method": "CF-Clip only",
            "engine": "SERIAL_KVCACHE",
            "queries": "N/A",
            "sec_per_query": "N/A",
            "sec_per_token": "N/A",
            "tokens_per_sec": "N/A",
            "peak_VRAM_GiB": "N/A",
            "scope": "same-lineage standalone timing absent",
            "qualification": "not estimated from RIF",
        }
    )
    latency = json.loads((ROOT / "fixed_horizon_b2_fresh_nfcorpus_20260929/remainder_eval_20260930/07_LATENCY_RESULT.json").read_text())
    for attack, data in (("IA Q30", latency["IA_Q30"]), ("MEntA Q5", latency["MEntA_Q5"])):
        output.append(
            {
                "method": "RIF + Fixed Horizon B=2",
                "engine": "EXACT_SERIAL_KVCACHE + deterministic abstention",
                "queries": data["evidence_queries"] + data["fallback_queries"],
                "sec_per_query": data["candidate_average_model_seconds_per_request"],
                "sec_per_token": "N/A",
                "tokens_per_sec": "N/A",
                "peak_VRAM_GiB": latency["peak_vram_gib"],
                "scope": attack,
                "qualification": f"only {data['evidence_queries']} evidence-bearing model calls; excludes retrieval/network",
            }
        )
    return output


def baseline_rows() -> list[dict]:
    return [
        {
            "method": "RIF",
            "official_or_local_implementation": "frozen local exact implementation",
            "empirical_same_cohort_result": "yes for reported main rows",
            "training_required": False,
            "attack_or_member_labels_required": False,
            "extra_LLM_or_judge": False,
            "retriever_modification": "conditional rank-1 removal",
            "generator_modification": "counterfactual logit clipping",
            "attack_specific_tuning": False,
            "runtime": "0.440092 sec/query; 8.335705 token/s; 6.156523 GiB",
            "comparison_status": "MAIN",
            "notes": "Qwen exact SERIAL_KVCACHE",
        },
        {
            "method": "MIRABEL",
            "official_or_local_implementation": "official repository + frozen local formula audit",
            "empirical_same_cohort_result": "no exact Paper-2 cohort defended-generation run",
            "training_required": False,
            "attack_or_member_labels_required": False,
            "extra_LLM_or_judge": False,
            "retriever_modification": "global EVT detector + conditional regeneration",
            "generator_modification": "conditional regeneration",
            "attack_specific_tuning": False,
            "runtime": "not measured on exact RIF cohort",
            "comparison_status": "ASSUMPTIONS_ONLY",
            "notes": "IA audit: default M>0 FPR 30.5%, query TPR 50.55%; matched FPR 2.5% query TPR 15.04%, any-Q15 64.75%; protocol mismatch",
        },
        {
            "method": "PAD",
            "official_or_local_implementation": "official code available",
            "empirical_same_cohort_result": "no",
            "training_required": False,
            "attack_or_member_labels_required": "no labels; privacy/noise hyperparameters required",
            "extra_LLM_or_judge": False,
            "retriever_modification": False,
            "generator_modification": "per-token calibrated noise",
            "attack_specific_tuning": "not allowed; exact author setting not ported",
            "runtime": "not measured",
            "comparison_status": "NOT_CURRENTLY_FAIR",
            "notes": "different decoding/threat emphasis; needs preregistered port",
        },
        {
            "method": "EPD",
            "official_or_local_implementation": "paper specification; no exact official implementation located",
            "empirical_same_cohort_result": "no",
            "training_required": False,
            "attack_or_member_labels_required": False,
            "extra_LLM_or_judge": "two candidates plus judge model",
            "retriever_modification": False,
            "generator_modification": True,
            "attack_specific_tuning": "unknown",
            "runtime": "not measured; expected high",
            "comparison_status": "INCOMPATIBLE_NOW",
            "notes": "different serving stack and MIA setup",
        },
        {
            "method": "DP-Fusion",
            "official_or_local_implementation": "official code available",
            "empirical_same_cohort_result": "no",
            "training_required": False,
            "attack_or_member_labels_required": "sensitive entity annotations and per-type privacy parameters",
            "extra_LLM_or_judge": "public model/candidate resources",
            "retriever_modification": "not drop-in",
            "generator_modification": "DP token generation",
            "attack_specific_tuning": "different threat model",
            "runtime": "not measured; released example uses four GPUs",
            "comparison_status": "INCOMPATIBLE_NOW",
            "notes": "entity/document privatization, not current unannotated paired cohort",
        },
    ]


def md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(md_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def qualitative_examples() -> str:
    selection = (
        "Selection rule fixed before rendering: use only existing fixed-seed qualitative pools; "
        "choose a representative whose No-Defense-to-RIF score change is closest to the pool median, "
        "and a boundary/failure case with the largest residual RIF score or largest utility loss. "
        "Ties use lexical IDs. No query or answer text is edited."
    )
    parts = ["# ACL Final Qualitative Examples", "", selection, ""]

    cohort = {row["query_id"]: row for row in read_jsonl(ROOT / "unified_dual_surface_dev_20260927/02_PROSPECTIVE_COHORT.jsonl")}
    answers = defaultdict(dict)
    for row in read_jsonl(ROOT / "unified_dual_surface_dev_20260927/04_FOUR_ARM_ANSWERS.jsonl"):
        answers[row["query_id"]][row["arm"]] = row["answer"]
    scores = defaultdict(dict)
    for row in read_csv(ROOT / "unified_dual_surface_dev_20260927/05_NATIVE_SCORES.csv"):
        scores[row["query_id"]][row["arm"]] = float(row["native_score"])
    eligible = []
    for query_id, arm_scores in scores.items():
        if query_id in cohort and cohort[query_id]["attack"] in ("MBA", "RAG-MIA") and all(key in answers[query_id] for key in ("NO_DEFENSE", "CLD_ONLY", "CLD_CFCLIP")):
            eligible.append((query_id, arm_scores.get("NO_DEFENSE", 0.0) - arm_scores.get("CLD_CFCLIP", 0.0), arm_scores.get("CLD_CFCLIP", 0.0)))
    median = float(np.median([row[1] for row in eligible]))
    representative = min(eligible, key=lambda row: (abs(row[1] - median), row[0]))
    boundary = max(eligible, key=lambda row: (row[2], row[0]))
    parts += ["## Single-query", "", markdown_table(
        ["type", "attack", "query", "No Defense", "CLD", "RIF"],
        [[kind, cohort[item[0]]["attack"], cohort[item[0]]["query"], answers[item[0]]["NO_DEFENSE"], answers[item[0]]["CLD_ONLY"], answers[item[0]]["CLD_CFCLIP"]]
         for kind, item in (("representative median-like", representative), ("boundary/failure", boundary))],
    ), ""]

    ia = read_csv(ROOT / "paper2_csi_guard/37_final_task_utility_freeze/23_IA_ATTACK_EXAMPLES.csv")
    ia_groups = defaultdict(list)
    for row in ia:
        if row["world"] == "T":
            ia_groups[row["target_document_id"]].append(row)
    ia_stats = []
    for key, rows in ia_groups.items():
        rows.sort(key=lambda row: int(row["Q"]))
        improvement = np.mean([float(row["No_Defense_attack_score"]) - float(row["Dual_attack_score"]) for row in rows])
        residual = np.mean([float(row["Dual_attack_score"]) for row in rows])
        ia_stats.append((key, improvement, residual))
    ia_median = float(np.median([row[1] for row in ia_stats]))
    ia_rep = min(ia_stats, key=lambda row: (abs(row[1] - ia_median), row[0]))
    ia_fail = max(ia_stats, key=lambda row: (row[2], row[0]))
    for title, item in (("representative median-like", ia_rep), ("boundary/failure", ia_fail)):
        rows = sorted(ia_groups[item[0]], key=lambda row: int(row["Q"]))
        parts += [f"## IA Q1-Q5 — {title}", "", markdown_table(
            ["Q", "actual query", "No Defense", "CLD", "RIF", "ND score", "RIF score"],
            [[row["Q"], row["actual_attack_query"], row["No_Defense_answer"], row["CLD_answer"], row["Dual_answer"], row["No_Defense_attack_score"], row["Dual_attack_score"]] for row in rows],
        ), ""]

    menta = read_csv(ROOT / "paper2_csi_guard/37_final_task_utility_freeze/24_MENTA_ATTACK_EXAMPLES.csv")
    menta_groups = defaultdict(list)
    for row in menta:
        if row["membership"] == "member":
            menta_groups[row["session_id"]].append(row)
    menta_stats = []
    for key, rows in menta_groups.items():
        rows.sort(key=lambda row: int(row["Q"]))
        improvement = np.mean([float(row["No_Defense_attack_score"]) - float(row["Dual_attack_score"]) for row in rows])
        residual = np.mean([float(row["Dual_attack_score"]) for row in rows])
        menta_stats.append((key, improvement, residual))
    menta_median = float(np.median([row[1] for row in menta_stats]))
    menta_rep = min(menta_stats, key=lambda row: (abs(row[1] - menta_median), row[0]))
    menta_fail = max(menta_stats, key=lambda row: (row[2], row[0]))
    for title, item in (("representative median-like", menta_rep), ("boundary/failure", menta_fail)):
        rows = sorted(menta_groups[item[0]], key=lambda row: int(row["Q"]))
        parts += [f"## MEntA Q1-Q5 — {title}", "", markdown_table(
            ["Q", "actual query", "No Defense", "CLD", "RIF", "ND score", "RIF score"],
            [[row["Q"], row["actual_attack_query"], row["No_Defense_answer"], row["CLD_answer"], row["Dual_answer"], row["No_Defense_attack_score"], row["Dual_attack_score"]] for row in rows],
        ), ""]

    benign = read_csv(ROOT / "paper2_csi_guard/37_final_task_utility_freeze/22_NATURAL_BENIGN_EXAMPLES.csv")
    benign_ok = [row for row in benign if row["example_type"] == "ORDINARY" and row["answer_changed"] == "True" and row["CLD_support"] == row["Dual_support"]]
    benign_losses = [float(row["CLD_F1"]) - float(row["Dual_F1"]) for row in benign_ok]
    benign_median = float(np.median(benign_losses))
    benign_rep = min(benign_ok, key=lambda row: (abs((float(row["CLD_F1"]) - float(row["Dual_F1"])) - benign_median), row["job_id"]))
    deep = [row for row in benign if row["example_type"] == "TRUE_DEEP"]
    benign_fail = max(deep, key=lambda row: (float(row["CLD_F1"]) - float(row["Dual_F1"]), row["job_id"]))
    parts += ["## Benign and deep-reading", "", markdown_table(
        ["type", "query", "No Defense", "CLD", "RIF", "CLD support", "RIF support"],
        [["representative support-preserved wording change", benign_rep["current_human_query"], benign_rep["No_Defense_answer"], benign_rep["CLD_answer"], benign_rep["Final_Dual_answer"], benign_rep["CLD_support"], benign_rep["Dual_support"]],
         ["boundary/deep-reading utility loss", benign_fail["current_human_query"], benign_fail["No_Defense_answer"], benign_fail["CLD_answer"], benign_fail["Final_Dual_answer"], benign_fail["CLD_support"], benign_fail["Dual_support"]]],
    ), ""]

    b2 = read_csv(ROOT / "gpt_handoff_fixed_horizon_b2_20260930/07_HUMAN_CONVERSATIONAL_DEEPDIVE_QUERIES_AND_OUTPUTS.csv")
    b2_groups = defaultdict(list)
    for row in b2:
        b2_groups[row["session_id"]].append(row)
    complete = sorted((session, sorted(rows, key=lambda row: int(row["query_position"]))) for session, rows in b2_groups.items() if max(int(row["query_position"]) for row in rows) >= 3)
    for index, (session, rows) in enumerate(complete[:2], 1):
        display = [row for row in rows if int(row["query_position"]) <= 3]
        parts += [f"## B=2 long-horizon example {index}", "", markdown_table(
            ["Q", "query", "RIF", "RIF + B=2", "horizon exhausted"],
            [[row["query_position"], row["query"], row["OLD_DUAL_answer"], row["FIXED_HORIZON_B2_answer"], row["horizon_exhausted"]] for row in display],
        ), ""]
    return "\n".join(parts).rstrip() + "\n"


def write_reports(ia_summary: dict, menta_summary: dict) -> None:
    claim_audit = f"""# ACL Final Claim Audit

| Claim | Verdict | Evidence boundary |
|---|---|---|
| Training-free | SUPPORT | Frozen inference-time CLD and CF-Clip; no learned classifier or attack labels. |
| Dual-surface architecture | SUPPORT | Retrieval dominance and counterfactual generation influence are distinct operations. |
| Single-query mitigation | SUPPORT | Prospective NFCorpus four-arm results reduce MBA 0.8676→0.5204, RAG-MIA 0.9600→0.5000, S² balanced accuracy 0.6800→0.5600, and DCMI-Std-Q2 differential AUC 0.9700→0.5000. S² CI includes zero. |
| CF-Clip incremental IA benefit | SUPPORT | Exact paired Q5 CLD−RIF = {ia_summary['q5_cld_minus_rif']:.6f}, 95% CI [{ia_summary['q5_cld_minus_rif_ci95'][0]:.6f}, {ia_summary['q5_cld_minus_rif_ci95'][1]:.6f}]. IA is the evaluated local protocol, not paper-exact IA. |
| MEntA repeated-query mitigation | PARTIAL | Fresh primary No Defense 0.78445→RIF 0.71555 with paired reduction CI above zero, but RIF is 0.01065 worse than CLD at Q5; CLD−RIF CI [{menta_summary['q5_cld_minus_rif_ci95'][0]:.6f}, {menta_summary['q5_cld_minus_rif_ci95'][1]:.6f}]. Separate fresh NFCorpus B=2 cohort is stronger but cannot be merged. |
| Long-horizon IA mitigation | PARTIAL | Local Q30 surrogate: 0.9746→0.8900 under stateless RIF; leakage remains high. Optional B=2 reaches 0.6830. |
| Open-ended session defense | UNSUPPORTED | Stateless residual leakage and account-reset/Sybil bypass remain. |
| Benign utility preservation | PARTIAL | RIF Token-F1 loss is 3.83% ordinary, 2.17% conversational, and 1.42% true deep-dive, but deep-dive NLI support drops about 3.20pp. B=2 has severe Q3+ cost. |
| Model/retriever generalization | PARTIAL | CLD component transfers to one Llama family; zero-shot retriever operating point is unstable. Full RIF transfer is not evaluated. |
| Universal defense | UNSUPPORTED | Evaluated attacks/settings only; DCMI and IA are standardized/local variants. |

## Allowed claims

- "RIF is a training-free inference-time composition of query-local retrieval-dominance suppression and counterfactual generation-influence clipping."
- "On the evaluated NFCorpus single-query attacks, RIF reduces the reported membership leakage metrics while preserving task-level benign utility within the disclosed cohorts."
- "In the evaluated IA Q1–Q5 local protocol, CF-Clip provides additional privacy gain beyond CLD on paired sessions."
- "Stateless RIF attenuates but does not eliminate long-horizon composition leakage."
- "An optional authenticated B=2 evidence horizon bounds evidence-bearing calls at an explicit Q3+ utility cost."

## Forbidden claims

- "RIF is a universal RAG privacy defense."
- "RIF eliminates membership leakage or solves session MIA."
- "The reported IA result is paper-exact IA."
- "Full RIF is retriever- or generator-independent."
- "B=2 preserves open-ended conversation or deep-reading utility."
- "S² improvement is statistically significant."
"""
    (OUT / "ACL_FINAL_CLAIM_AUDIT.md").write_text(claim_audit, encoding="utf-8")

    limitations = """# ACL Final Limitations

1. The exact IA Q1–Q5 evidence uses a local question-only ConvFinQA protocol; the Q30 result is explicitly a local NFCorpus surrogate, not paper-exact IA.
2. The strongest single-query results are a prospective balanced DEV, not a held-out multi-seed confirmatory test. S²'s paired reduction CI crosses zero.
3. MEntA component evidence is mixed: fresh primary RIF improves over No Defense but is slightly worse than CLD at Q5, so CF-Clip complementarity is IA-specific in current evidence.
4. Stateless RIF leaves substantial long-horizon IA leakage (Q30 AUROC 0.8900). B=2 reduces it by refusing protected-evidence use after Q2 and therefore causes severe later-turn utility loss.
5. B=2 requires durable authenticated identity. Full credential reset is an acknowledged Sybil bypass; conversation-ID reset alone is not.
6. Exact RIF runtime is about 3.95× Vanilla per query (0.440092 vs 0.111453 seconds) and 8.34 vs 29.46 tokens/s on the audited hardware.
7. Full RIF zero-shot transfer is missing. Frozen-tau CLD is unstable across GTE/MPNet, and CF-Clip has not been exactly evaluated on another generator family.
8. Strong baselines are not available on an identical end-to-end cohort. MIRABEL's matched-FPR IA audit has a protocol mismatch; PAD, EPD, and DP-Fusion require nontrivial incompatible ports or extra resources.
9. DCMI-Std-Q2 is standardized rather than paper-exact DCMI. Results must retain the qualifier.
10. Some four-arm IA/MEntA evidence is supplementary/exploratory batched lineage. It is not merged with exact/fresh primary rows.
11. Utility relies on stored automatic metrics and limited numeric-reference coverage; no broad human preference or numeric accuracy claim is warranted.
12. Adaptive robustness supports within-account candidate reordering under B=2, but full account reset restores IA leakage to AUROC 0.8900.
"""
    (OUT / "ACL_FINAL_LIMITATIONS.md").write_text(limitations, encoding="utf-8")

    recommendation = """# ACL/EMNLP 최종 투고 권고

## 1. 현재 evidence로 ACL Main이 충분한가?

현재 상태는 **ACL Main에 제출은 가능하지만, 강한 accept 정당화에는 아직 부족한 borderline evidence**다. 가장 방어 가능한 선택은 ACL Findings/EMNLP Findings까지 함께 염두에 둔 정직한 RIF 논문이다. 이유는 prospective Core4와 exact IA complementarity가 강한 반면, paper-exact IA, full-RIF transfer, 동일-cohort strong baseline, 독립 confirmatory single-query test가 비어 있기 때문이다.

## 2. 가장 강한 contribution 3개

1. **두 표면의 단순한 training-free 결합**: CLD는 retrieval rank-1 dominance를, old CF-Clip은 생성 token의 counterfactual document influence를 제어한다.
2. **공격별 routing 없는 폭넓은 실증**: prospective NFCorpus single-query 공격 네 종류에서 누출 지표가 감소하며, exact IA Q5에서는 CLD 0.8458 대비 RIF 0.7097의 추가 이득이 paired comparison으로 확인된다.
3. **정직한 composition boundary**: stateless RIF의 Q30 잔여 누출을 숨기지 않고, 별도 optional B=2 policy가 노출을 제한하지만 Q3+ utility를 희생함을 정량화한다.

## 3. reviewer가 공격할 약점 5개

1. IA가 paper-exact가 아니며 Q30은 local surrogate다.
2. MEntA fresh primary에서 RIF가 CLD보다 Q5 AUROC가 0.01065 높아 CF의 보편적 complementarity가 없다.
3. B=2는 Q3+ refusal policy이므로 defense novelty보다 access-control baseline으로 보일 수 있고 Sybil reset에 취약하다.
4. full RIF의 retriever/generator transfer와 exact strong-baseline comparison이 없다.
5. exact RIF가 Vanilla 대비 약 3.95배 느리고, single-query 주 결과가 DEV 규모다.

## 4. submission 전에 반드시 필요한 추가 실험

- **Paper-exact IA reproduction**: 공식 summary+Q30, scorer, Top-k, generator/decoding을 고정한 No Defense/CLD/CF/RIF paired test.
- **동일-lineage independent Core4 confirmatory**: 새 target, 동일 four-arm, 2,000+ paired bootstrap; 특히 S² 불확실성을 해소.
- **동일-cohort MIRABEL comparison**: 동일 benign calibration/FPR, 동일 attacks, 동일 utility/runtime assumptions. 다른 baseline은 정확한 port가 없으면 assumptions table 유지.
- **Full-RIF transfer 최소 1개**: 다른 generator에서 gamma zero-shot exact token path 또는 다른 retriever에서 frozen tau의 실패를 정식 negative result로 확정. 성공을 위해 retune하지 않는다.
- **End-to-end artifact integrity check**: 공개 코드가 frozen prompts/context policy와 일치하는지 재현 스크립트와 hash로 검증.

## 5. 있으면 좋은 실험

- 사용자 연구 또는 blinded human factuality 평가.
- 더 긴 answer/runtime profile과 다양한 GPU hardware 측정.
- 인증·credential rotation을 포함한 B=2 운영 보안 분석.
- adaptive paraphrase를 새로 생성한 prospective test. 현재 것은 저장 후보/출력 기반 offline stress test다.

## 6. B=2 배치 권고

**Main method가 아니라 optional safeguard**로 둔다. 본문 limitation/long-horizon section에 핵심 수치 한 표를 넣고, 상세 adaptive/Sybil/utility 표는 appendix로 보낸다. RIF novelty와 분리하고 `authenticated high-security deployment option`으로만 표현한다.

## 7. 최종 포지셔닝

권장 제목 개념: **RIF: Retrieval and Generation Influence Control for Membership Privacy in RAG**.

핵심 문장: "Retrieval filtering alone does not fully address membership leakage expressed through generated responses. RIF combines query-local retrieval-dominance suppression with counterfactual generation-influence clipping without retraining or attack-specific classifiers."

장기 세션 문장: "Repeated interrogation can accumulate residual leakage beyond stateless per-query defenses. An optional authenticated fixed evidence horizon bounds evidence-bearing calls at an explicit loss of open-ended deep-reading utility."

최종 판정: `ACL_MAIN_BORDERLINE__FINDINGS_READY_AFTER_REQUIRED_VALIDATION`
"""
    (OUT / "ACL_FINAL_PAPER_RECOMMENDATION_KO.md").write_text(recommendation, encoding="utf-8")


def main() -> None:
    ia_rows, ia_summary = ia_ablation()
    menta_rows, menta_summary = menta_ablation()
    write_csv("ACL_FINAL_PROVENANCE_MATRIX.csv", provenance_rows())
    write_csv("ACL_FINAL_MAIN_RESULTS.csv", main_results())
    write_csv("ACL_FINAL_IA_ABLATION.csv", ia_rows)
    write_csv("ACL_FINAL_MENTA_ABLATION.csv", menta_rows)
    write_csv("ACL_FINAL_UTILITY.csv", utility_rows())
    write_csv("ACL_FINAL_RUNTIME.csv", runtime_rows())
    write_csv("ACL_FINAL_BASELINE_COMPARISON.csv", baseline_rows())
    (OUT / "ACL_FINAL_QUALITATIVE_EXAMPLES.md").write_text(qualitative_examples(), encoding="utf-8")
    write_reports(ia_summary, menta_summary)
    manifest = {
        "created_date": "2026-09-30",
        "new_generation": False,
        "new_retrieval": False,
        "threshold_tuning": False,
        "domain_transfer_run": False,
        "tau": TAU,
        "gamma": GAMMA,
        "builder_sha256": sha256(Path(__file__)),
        "deliverables": {},
    }
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "AUDIT_MANIFEST.json":
            manifest["deliverables"][path.name] = sha256(path)
    (OUT / "AUDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

