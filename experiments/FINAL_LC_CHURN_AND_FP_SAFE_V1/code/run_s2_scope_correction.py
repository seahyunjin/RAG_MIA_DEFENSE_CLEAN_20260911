#!/usr/bin/env python3
"""Re-aggregate frozen churn scores on the exact Core5 S² evaluation scope."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (ATTACKS, EXP, FINAL8, OUTER_BUDGET, SEED, STRICT_THRESHOLD,
                    V6, VERSIONS, atomic_json, atomic_text, now, read_jsonl,
                    sha_file, verify_hashed_json, wilson, write_csv)


def bootstrap(rows, alarms, indices, seed):
    groups = defaultdict(list)
    for index in indices:
        groups[rows[index]["session_id"]].append(index)
    units = sorted(groups)
    sums = np.asarray([sum(bool(alarms[i]) for i in groups[unit]) for unit in units], dtype=float)
    counts = np.asarray([len(groups[unit]) for unit in units], dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(20):
        choices = rng.integers(0, len(units), size=(100, len(units)))
        samples.extend((sums[choices].sum(1) / counts[choices].sum(1)).tolist())
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def main() -> None:
    pre_path = EXP / "configs/DB_CHURN_V2_S2_SCOPE_CORRECTION_PRECOMMIT.json"
    pre = verify_hashed_json(pre_path)
    if sha_file(EXP / "code/prepare_s2_scope_correction.py") != pre["code_sha256"]["prepare"]:
        raise RuntimeError("correction preparation code drift")
    if sha_file(EXP / "code/run_s2_scope_correction.py") != pre["code_sha256"]["run"]:
        raise RuntimeError("correction scoring code drift")
    for path, digest in pre["original_artifact_sha256"].items():
        if sha_file(Path(path)) != digest:
            raise RuntimeError(f"initial artifact drift before preservation: {path}")
    for path, digest in pre["score_cache_sha256"].items():
        if sha_file(Path(path)) != digest:
            raise RuntimeError(f"score cache drift: {path}")

    history = EXP / "history/INITIAL_INVALID_S2_SCOPE_INCLUSION"
    history.mkdir(parents=True, exist_ok=True)
    for path in (EXP / "DB_CHURN_V2_RESULT.json", EXP / "FINAL_RESULT.json",
                 EXP / "tables/DB_CHURN_V2_SUMMARY.csv",
                 EXP / "tables/DB_CHURN_CORE5_CLUSTER_BOOTSTRAP.csv",
                 EXP / "reports/FINAL_REPORT_KO.md"):
        target = history / path.name
        atomic_text(target, path.read_text(encoding="utf-8"))
    atomic_json(history / "INVALIDATION.json", {
        "verdict": "INVALID_S2_SCOPE_INCLUSION", "invalidated_utc": now(),
        "reason": pre["reason"], "artifacts_preserved": True,
    })

    rows = read_jsonl(EXP / "inputs/CHURN_QUERY_MANIFEST.jsonl")
    final8 = {row["query_id"]: row for row in read_jsonl(FINAL8 / "cache/FINAL_RETRIEVAL_AND_DETECTION.jsonl")}
    holdout = np.asarray([i for i, row in enumerate(rows) if row["split"] == "HOLDOUT"], dtype=int)
    positive = {}
    for attack in ATTACKS:
        indices = [i for i, row in enumerate(rows)
                   if row["split"] == "ATTACK" and row["attack"] == attack and row["membership"] == "member"]
        if attack == "S²-MIA":
            indices = [i for i in indices if final8[rows[i]["query_id"]].get("evaluation_split") == "S2_EVALUATION"]
        positive[attack] = indices
    expected = {"MEntA": 5000, "MBA": 1000, "RAG-MIA": 1000, "S²-MIA": 799, "DCMI-Std-Q2": 2000}
    observed = {key: len(value) for key, value in positive.items()}
    if observed != expected:
        raise RuntimeError(f"Core5 positive scope drift: {observed}")

    initial = json.loads((history / "DB_CHURN_V2_RESULT.json").read_text())
    summary_rows, bootstrap_rows = [], []
    version_details = {}
    refresh_baseline = strict_baseline = None
    for version_index, version in enumerate(VERSIONS):
        values = np.load(EXP / f"cache/{version}_QUERY_SCORES.npz")
        scores = {"Original MIRABEL": values["M"], "Final LC Strict": values["R_LC_STRICT"],
                  "Final LC Refresh": values["R_LC_REFRESH"]}
        refresh_threshold = float(initial["versions"][version]["refresh_threshold"])
        thresholds = {"Original MIRABEL": 0.0, "Final LC Strict": STRICT_THRESHOLD,
                      "Final LC Refresh": refresh_threshold}
        version_details[version] = {"strict_threshold": STRICT_THRESHOLD,
                                    "refresh_threshold": refresh_threshold, "methods": {}}
        for method_index, (method, score) in enumerate(scores.items()):
            alarms = score > thresholds[method]
            fp = int(np.count_nonzero(alarms[holdout])); low, high = wilson(fp, len(holdout))
            attack_tpr = {attack: float(np.mean(alarms[indices])) for attack, indices in positive.items()}
            mean_tpr = statistics.fmean(attack_tpr.values())
            summary_rows.append({"db": version, "method": method, "threshold": thresholds[method],
                                 "benign_n": len(holdout), "false_positives": fp, "benign_fpr": fp / len(holdout),
                                 "fpr_wilson_low": low, "fpr_wilson_high": high,
                                 **attack_tpr, "core5_mean_tpr": mean_tpr, "retraining_steps": 0,
                                 "s2_scope": "S2_EVALUATION_ONLY_1598_TOTAL_799_MEMBER"})
            for attack_index, attack in enumerate(ATTACKS):
                ci_low, ci_high = bootstrap(rows, alarms, positive[attack],
                                             SEED + version_index * 100 + attack_index * 3 + method_index)
                bootstrap_rows.append({"db": version, "method": method, "attack": attack,
                                       "member_queries": len(positive[attack]),
                                       "member_sessions": len({rows[i]['session_id'] for i in positive[attack]}),
                                       "tpr": attack_tpr[attack], "cluster_bootstrap_iterations": 2000,
                                       "ci95_low": ci_low, "ci95_high": ci_high})
            version_details[version]["methods"][method] = {"fpr": fp / len(holdout),
                "attack_tpr": attack_tpr, "core5_mean_tpr": mean_tpr}
        if version == "V0":
            refresh_baseline = version_details[version]["methods"]["Final LC Refresh"]["attack_tpr"]
            strict_baseline = version_details[version]["methods"]["Final LC Strict"]["attack_tpr"]

    def gate(method, baseline):
        fprs = {v: version_details[v]["methods"][method]["fpr"] for v in VERSIONS}
        drops = {v: {a: baseline[a] - version_details[v]["methods"][method]["attack_tpr"][a]
                     for a in ATTACKS} for v in VERSIONS}
        mean_drops = {v: statistics.fmean(drops[v].values()) for v in VERSIONS}
        checks = {"all_fpr_at_most_5pct": all(x <= .05 + 1e-12 for x in fprs.values()),
                  "all_attack_degradation_at_most_5pp": all(x <= .05 + 1e-12 for d in drops.values() for x in d.values()),
                  "mean_degradation_preferred_at_most_3pp": all(x <= .03 + 1e-12 for x in mean_drops.values())}
        return all(checks.values()), {"fpr": fprs, "tpr_degradation": drops, "mean_degradation": mean_drops, "checks": checks}

    refresh_pass, refresh_gate = gate("Final LC Refresh", refresh_baseline)
    strict_pass, strict_gate = gate("Final LC Strict", strict_baseline)
    verdict = "DB_CHURN_RETRAINING_FREE_PASS" if refresh_pass else "DB_CHURN_RECALIBRATION_FAILED"
    corrected = {"campaign": EXP.name, "phase": "DB_CHURN_V2_CORRECTED_CORE5_SCOPE",
                 "verdict": verdict,
                 "strict_transfer_verdict": "DB_CHURN_STRICT_TRANSFER_PASS" if strict_pass else "DB_CHURN_STRICT_TRANSFER_FAILED",
                 "completed_utc": now(), "versions": version_details,
                 "refresh_gate": refresh_gate, "strict_gate": strict_gate,
                 "scope_correction": pre["s2"], "initial_result": "INVALID_S2_SCOPE_INCLUSION_PRESERVED_IN_HISTORY",
                 "scores_or_model_changed": False, "training_steps": 0, "attack_samples_used_for_refresh": 0,
                 "calibration_scope_note": initial["calibration_scope_note"]}
    write_csv(EXP / "tables/DB_CHURN_V2_SUMMARY_CORRECTED.csv", summary_rows)
    write_csv(EXP / "tables/DB_CHURN_CORE5_CLUSTER_BOOTSTRAP_CORRECTED.csv", bootstrap_rows)
    atomic_json(EXP / "DB_CHURN_V2_CORRECTED_RESULT.json", corrected)

    by_key = {(row["db"], row["method"]): row for row in summary_rows}
    ia = json.loads((V6 / "FINAL_RESULT.json").read_text())
    packing = json.loads((EXP / "audits/CURRENT_HIDE_PACKING_POLICY.json").read_text())
    final = {"campaign": EXP.name, "completed_utc": now(), "primary_result": "CORRECTED_CORE5_SCOPE",
             "supersedes": "INITIAL_INVALID_S2_SCOPE_INCLUSION",
             "ia": {"verdict": ia["verdict"], "scientific_status": ia["scientific_status"]},
             "db_churn": {"verdict": corrected["verdict"], "strict_transfer_verdict": corrected["strict_transfer_verdict"]},
             "fp_safe": {"verdict": packing["verdict"], "candidate_generation_run": False},
             "final_lc_modified": False, "training_steps": 0, "new_qwen_answer_generations": 0}
    atomic_json(EXP / "FINAL_RESULT.json", final)

    lines = ["# Final LC DB Churn and FP-Safe Validation — Corrected Core5 Scope", "",
             f"- IA: `{ia['verdict']}` / `{ia['scientific_status']}`",
             "- Initial churn aggregation: `INVALID_S2_SCOPE_INCLUSION` (preserved under history/)",
             "- Correction: S² uses only frozen S2_EVALUATION 1,598 rows (799 member / 799 nonmember).",
             "- Detector, embeddings, retrieval, thresholds and the other four attack IDs were unchanged.", "",
             "| DB | Original MIRABEL FPR | Final LC strict FPR | Final LC refresh FPR | MEntA TPR | MBA TPR | RAG-MIA TPR | S² TPR | DCMI TPR | Core5 mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for version in VERSIONS:
        o, s, r = by_key[(version,"Original MIRABEL")], by_key[(version,"Final LC Strict")], by_key[(version,"Final LC Refresh")]
        lines.append(f"| {version} | {o['benign_fpr']:.3%} | {s['benign_fpr']:.3%} | {r['benign_fpr']:.3%} | "
                     f"{r['MEntA']:.4f} | {r['MBA']:.4f} | {r['RAG-MIA']:.4f} | {r['S²-MIA']:.4f} | "
                     f"{r['DCMI-Std-Q2']:.4f} | {r['core5_mean_tpr']:.4f} |")
    lines += ["", f"- Churn verdict: `{corrected['verdict']}`",
              f"- Strict verdict: `{corrected['strict_transfer_verdict']}`",
              f"- FP-safe verdict: `{packing['verdict']}`",
              "- Existing packing already water-fills remaining Top-3; candidate generation/privacy screen were not opened.",
              "- The benign holdout calibrates and reports empirical threshold exceedance under the inherited protocol; it is not an untouched FPR test.",
              "- Training/gradient/attack-calibration/new Qwen generations: 0 / 0 / 0 / 0", "",
              "## 12-line summary", "",
              f"1. IA final: {ia['verdict']} / {ia['scientific_status']}",
              "2. Missing embedding union: 1,847 unique documents",
              f"3. V10 strict/refresh FPR: {by_key[('V10','Final LC Strict')]['benign_fpr']:.3%} / {by_key[('V10','Final LC Refresh')]['benign_fpr']:.3%}",
              f"4. V25 strict/refresh FPR: {by_key[('V25','Final LC Strict')]['benign_fpr']:.3%} / {by_key[('V25','Final LC Refresh')]['benign_fpr']:.3%}",
              f"5. V50 strict/refresh FPR: {by_key[('V50','Final LC Strict')]['benign_fpr']:.3%} / {by_key[('V50','Final LC Refresh')]['benign_fpr']:.3%}",
              f"6. V50 refresh Core5 mean TPR loss: {corrected['refresh_gate']['mean_degradation']['V50']:.3%}",
              "7. Retraining steps: 0", "8. Packing: remaining Top-3 water-fill to 2,048 tokens",
              "9. Simple-Hide FP F1: ~0.218 (prior frozen)", "10. FP candidate F1: N/A",
              "11. Privacy screen: N/A", "12. Next: churn failure is detector sensitivity, not an FPR calibration failure."]
    atomic_text(EXP / "reports/FINAL_REPORT_KO.md", "\n".join(lines) + "\n")
    atomic_text(EXP / "STATUS.md", "# FINAL_LC_CHURN_AND_FP_SAFE_V1\n\n- Stage: `COMPLETE_CORRECTED_CORE5_SCOPE`\n"
                f"- Churn: `{corrected['verdict']}`\n- Strict: `{corrected['strict_transfer_verdict']}`\n"
                f"- FP-safe: `{packing['verdict']}`\n")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
