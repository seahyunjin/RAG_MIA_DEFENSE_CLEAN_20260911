#!/usr/bin/env python3
"""Core5 matched-benign-FPR detection after fail-closed IA exclusion."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from common import BUDGETS, EXP, FINAL8, ROOT, SEED, atomic_json, atomic_text, checkpoint, now, read_jsonl, sha_file, verify_hashed_json, write_csv, write_jsonl
from run_phase_a_detection import bootstrap_delta, score_new_queries, threshold_at_budget

CORE5 = ("MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2")
EXPECTED_MEMBER_QUERIES = {"MEntA": 5000, "MBA": 1000, "RAG-MIA": 1000, "S²-MIA": 799, "DCMI-Std-Q2": 2000}


def evaluate(new_rows: list[dict], runtime: dict) -> dict:
    old_rows = read_jsonl(FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl")
    benign = [row for row in old_rows if row["attack"] == "BENIGN"]
    if len(benign) != 1000:
        raise RuntimeError("benign holdout count drift")
    attacks = [row for row in old_rows if row["attack"] in set(CORE5[:-1])] + new_rows
    thresholds, alarm_rows = {}, []
    for budget in BUDGETS:
        for method, key in (("MIRABEL", "M"), ("Final LC", "R_LC")):
            threshold, count = threshold_at_budget([row[key] for row in benign], budget)
            thresholds[(method, budget)] = threshold
            alarm_rows.append({"method": method, "nominal_budget": budget, "benign_n": 1000,
                               "threshold": threshold, "strict_operator": ">", "benign_alarms": count,
                               "actual_benign_fpr": count / 1000})
    write_csv(EXP / "tables" / "CORE5_BENIGN_THRESHOLDS_AND_ALARMS.csv", alarm_rows)

    table, boots, primary = [], [], {}
    for attack in CORE5:
        subset = [row for row in attacks if row["attack"] == attack and row.get("membership") == "member"]
        if attack == "S²-MIA":
            subset = [row for row in subset if row.get("evaluation_split") == "S2_EVALUATION"]
        if len(subset) != EXPECTED_MEMBER_QUERIES[attack]:
            raise RuntimeError(f"{attack} member query count {len(subset)}/{EXPECTED_MEMBER_QUERIES[attack]}")
        row = {"attack": attack, "member_queries": len(subset),
               "member_sessions": len({value["session_id"] for value in subset}), "benign_queries": 1000}
        for budget in BUDGETS:
            label = "2_5" if budget == .025 else str(int(100 * budget))
            m = statistics.fmean(float(value["M"]) > thresholds[("MIRABEL", budget)] for value in subset)
            lc = statistics.fmean(float(value["R_LC"]) > thresholds[("Final LC", budget)] for value in subset)
            row[f"mirabel_tpr_at_{label}pct"] = m
            row[f"final_lc_tpr_at_{label}pct"] = lc
            if budget == .025:
                mean, low, high = bootstrap_delta(subset, thresholds[("MIRABEL", budget)],
                                                  thresholds[("Final LC", budget)], SEED + CORE5.index(attack))
                row.update({"delta_at_2_5pct": lc - m, "bootstrap_mean_delta": mean,
                            "delta_ci95_low": low, "delta_ci95_high": high})
                primary[attack] = {"MIRABEL": m, "Final LC": lc, "delta": lc - m, "ci95": [low, high]}
                boots.append({"attack": attack, "iterations": 2000, "cluster": "target/session",
                              "point_delta": lc - m, "mean_delta": mean, "ci95_low": low, "ci95_high": high})
        table.append(row)
    write_csv(EXP / "tables" / "CORE5_MATCHED_FPR_DETECTION.csv", table)
    write_csv(EXP / "tables" / "CORE5_CLUSTER_BOOTSTRAP.csv", boots)

    session_rows = []
    for attack in ("MEntA", "DCMI-Std-Q2"):
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in attacks:
            if row["attack"] == attack:
                groups[(row["membership"], row["session_id"])].append(row)
        for (membership, session_id), group in sorted(groups.items()):
            group.sort(key=lambda item: int(item["query_index"]))
            expected = 5 if attack == "MEntA" else 2
            if len(group) != expected:
                raise RuntimeError(f"{attack} session query-count drift")
            for method, key in (("MIRABEL", "M"), ("Final LC", "R_LC")):
                flags = [float(row[key]) > thresholds[(method, .025)] for row in group]
                session_rows.append({"attack": attack, "membership": membership, "session_id": session_id,
                                     "method": method, "queries": len(group), "alarms": sum(flags),
                                     "any_alarm": any(flags), "all_alarm": all(flags),
                                     "first_alarm_index": next((i + 1 for i, flag in enumerate(flags) if flag), None)})
    write_csv(EXP / "tables" / "CORE5_MULTIQUERY_SESSION_DETECTION.csv", session_rows)

    mean_delta = statistics.fmean(value["delta"] for value in primary.values())
    nondegrading = sum(value["delta"] >= -1e-12 for value in primary.values())
    worst_delta = min(value["delta"] for value in primary.values())
    checks = {"core5_mean_delta_positive": mean_delta > 0, "nondegrading_at_least_4_of_5": nondegrading >= 4,
              "no_family_drop_below_minus_3pp": worst_delta >= -.03 - 1e-12,
              "dcmi_no_catastrophic_failure": primary["DCMI-Std-Q2"]["Final LC"] >= .05}
    passed = all(checks.values())
    verdict = "CORE5_DETECTION_PASS" if passed else "CORE5_DETECTION_FAILED"
    result = {"campaign": EXP.name, "phase": "A_CORE5_DETECTION", "verdict": verdict,
              "completed_utc": now(), "primary_budget": .025, "primary": primary,
              "macro": {"mirabel_tpr": statistics.fmean(v["MIRABEL"] for v in primary.values()),
                         "final_lc_tpr": statistics.fmean(v["Final LC"] for v in primary.values()),
                         "mean_delta": mean_delta, "nondegrading_families": nondegrading, "worst_delta": worst_delta},
              "checks": checks, "benign_alarm_rows": alarm_rows, "runtime": runtime,
              "ia_status": "EXCLUDED_AFTER_PARSER_V2_GATE_FAILURE_NO_REGENERATION",
              "next_stage": "CORE5_MATCHED_BUDGET_E2E_CHARACTERIZATION"}
    atomic_json(EXP / "PHASE_A_RESULT.json", result)
    lines = ["# Final LC Core5 Detection", "", f"- Verdict: `{verdict}`",
             "- IA: parser-v2 gate failure; excluded without regeneration", "- E2E continues by the user's fail branch.", "",
             "| Attack | MIRABEL TPR | Final LC TPR | Delta | 95% CI |", "|---|---:|---:|---:|---:|"]
    for attack in CORE5:
        value = primary[attack]
        lines.append(f"| {attack} | {value['MIRABEL']:.4f} | {value['Final LC']:.4f} | {value['delta']:+.4f} | [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] |")
    atomic_text(EXP / "reports" / "PHASE_A_CORE5_DETECTION_KO.md", "\n".join(lines) + "\n")
    checkpoint(verdict, mean_delta=round(mean_delta, 6), nondegrading=f"{nondegrading}/5",
               worst_delta=round(worst_delta, 6), ia_excluded=True, next_stage=result["next_stage"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    pre = verify_hashed_json(EXP / "configs" / "CORE5_RESUME_PRECOMMIT.json")
    for relative, expected in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != expected:
            raise RuntimeError(f"Core5 code drift: {relative}")
    query_path = Path(pre["dcmi_queries"]["path"])
    if sha_file(query_path) != pre["dcmi_queries"]["sha256"]:
        raise RuntimeError("DCMI bundle drift")
    queries = read_jsonl(query_path)
    rows, runtime = score_new_queries(queries)
    output = EXP / "cache" / "CORE5_DCMI_RETRIEVAL_AND_DETECTION.jsonl"
    write_jsonl(output, rows)
    runtime["detection_cache_sha256"] = sha_file(output)
    atomic_json(EXP / "cache" / "CORE5_DETECTION_MANIFEST.json", runtime)
    evaluate(rows, runtime)


if __name__ == "__main__":
    main()
