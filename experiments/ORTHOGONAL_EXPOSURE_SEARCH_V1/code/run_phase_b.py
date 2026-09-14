#!/usr/bin/env python3
"""Phase B: the single precommitted training-free Sparse Exposure detector."""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import (ALPHAS, ATTACKS, EXP, PARENT, PRIMARY_ALPHA, SEED, V1, V2, atomic_json,
                    bootstrap_delta, checkpoint, empirical_cutoff, empirical_upper_tail,
                    matched_binary_threshold, normalize_tokens, now, percentile, read_jsonl,
                    sha_file, spearman, tpr_at_fpr, wilson, write_csv)


PRE = EXP / "configs/SPARSE_EXPOSURE_V1_PRECOMMIT.json"
PRE_SHA = PRE.with_suffix(".sha256")
FRESH_SCORES = V2 / "cache/FRESH_REPLICATION_RETRIEVAL_SCORES.jsonl"
LEGACY_CACHE = PARENT / "cache/CLEAN_CORE3_RETRIEVAL_CACHE.jsonl"
LEGACY_SCORE_ROWS = V1 / "tables/SCORE_ROWS.csv"


def preflight() -> tuple[dict, list[dict], list[dict], dict[str, str]]:
    expected = PRE_SHA.read_text().strip()
    if sha_file(PRE) != expected: raise RuntimeError("Phase B precommit hash drift")
    pre = json.loads(PRE.read_text())
    if sha_file(Path(__file__)) != pre["code_sha256"][Path(__file__).name]:
        raise RuntimeError("Phase B implementation changed after precommit")
    if not (EXP / "PHASE_A_RESULT.json").is_file(): raise RuntimeError("Phase A result missing")
    if sha_file(V2 / "PHASE1_RESULT.json") != pre["frozen_v2_result"]["sha256"]:
        raise RuntimeError("frozen V2 result drift")
    if sha_file(FRESH_SCORES) != pre["fresh_retrieval"]["sha256"] or sha_file(LEGACY_CACHE) != pre["legacy_retrieval"]["sha256"]:
        raise RuntimeError("retrieval substrate drift")
    fresh = read_jsonl(FRESH_SCORES); legacy = read_jsonl(LEGACY_CACHE)
    docs = {r["document_id"]: r["source_text"] for r in read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")}
    if len(fresh) != 2400 or len(legacy) != 1280 or len(docs) != 3000:
        raise RuntimeError("substrate count mismatch")
    checkpoint("PHASE_B_PREFLIGHT_PASS", fresh_rows=len(fresh), legacy_rows=len(legacy), documents=len(docs),
               precommit_sha256=expected, prior_phase=json.loads((EXP / "PHASE_A_RESULT.json").read_text())["verdict"])
    return pre, fresh, legacy, docs


def build_idf(docs: dict[str, str]) -> tuple[dict[str, float], dict[str, set[str]]]:
    token_sets = {}; df = Counter()
    for doc_id, text in docs.items():
        tokens = normalize_tokens(text); token_sets[doc_id] = tokens; df.update(tokens)
    n = len(docs)
    return {token: math.log((n+1)/(count+1))+1 for token, count in df.items()}, token_sets


def lexical_score(query: str, doc_id: str, idf: dict[str, float], doc_tokens: dict[str, set[str]]) -> tuple[float, bool, int, int]:
    query_tokens = normalize_tokens(query)
    denominator = sum(idf.get(t, math.log(3001)+1) for t in query_tokens)
    if denominator == 0: return 0.0, True, len(query_tokens), 0
    shared = query_tokens & doc_tokens[doc_id]
    numerator = sum(idf[t] for t in shared)
    return numerator/denominator, False, len(query_tokens), len(shared)


def legacy_rows_with_partition(legacy: list[dict]) -> list[dict]:
    scores = list(csv.DictReader(LEGACY_SCORE_ROWS.open(encoding="utf-8")))
    by_id = {r["query_id"]: r for r in scores if r["cohort"] == "BENIGN"}
    output = []
    for row in legacy:
        if row["cohort"] != "BENIGN": continue
        score = by_id.get(row["query_id"])
        if score is None: raise RuntimeError(f"legacy benign score missing: {row['query_id']}")
        output.append({**row, "M": float(score["M"]), "calibration_partition": score["calibration_partition"]})
    if Counter(r["calibration_partition"] for r in output) != Counter({"TAIL_REFERENCE": 250, "THRESHOLD_CALIBRATION": 250, "LOCKED_HOLDOUT": 500}):
        raise RuntimeError("legacy benign 250/250/500 partition drift")
    return output


def binary_overlap(member: list[dict], benign: list[dict], x_key: str, m_t: float, x_t: float) -> tuple[Counter, Counter, list[dict]]:
    mc = Counter(); bc = Counter(); cases = []
    for r, counter, is_member in [(r, mc, True) for r in member] + [(r, bc, False) for r in benign]:
        mh, xh = r["M"] > m_t, r[x_key] > x_t
        cell = "BOTH_HIT" if mh and xh else "MIRABEL_ONLY" if mh else "SPARSE_ONLY" if xh else "BOTH_MISS"
        counter[cell] += 1
        if is_member: cases.append({"query_id": r["query_id"], "session_id": r["session_id"], "query_index": r["query_index"],
                                    "target_id": r["target_id"], "cell": cell, "M": r["M"], "L": r["L"], "S_sparse": r[x_key]})
    return mc, bc, cases


def main() -> None:
    started = time.monotonic(); pre, fresh, legacy, docs = preflight()
    checkpoint("PHASE_B_IDF_STARTED", documents=len(docs), formula="log((N+1)/(df+1))+1")
    idf, doc_tokens = build_idf(docs)
    checkpoint("PHASE_B_IDF_COMPLETE", vocabulary=len(idf), documents=len(doc_tokens))
    old = legacy_rows_with_partition(legacy)
    empty_cases = []
    for row in old + fresh:
        doc_id = row["top_document_ids"][0]
        value, empty, qn, sn = lexical_score(row["query"], doc_id, idf, doc_tokens)
        row["L"] = value
        if empty: empty_cases.append({"query_id": row["query_id"], "cohort": row["cohort"], "EMPTY_TOKEN_CASE": True})
    tail = [r for r in old if r["calibration_partition"] == "TAIL_REFERENCE"]
    threshold_cal = [r for r in old if r["calibration_partition"] == "THRESHOLD_CALIBRATION"]
    ref_m, ref_l = sorted(r["M"] for r in tail), sorted(r["L"] for r in tail)
    for row in old + fresh:
        row["p_M_sparse"] = empirical_upper_tail(ref_m, row["M"])
        row["p_L"] = empirical_upper_tail(ref_l, row["L"])
        row["S_sparse"] = max(-math.log(row["p_M_sparse"]), -math.log(row["p_L"]))
    deployment = {"M": {str(a): empirical_cutoff([r["M"] for r in threshold_cal], a) for a in ALPHAS},
                  "S_sparse": {str(a): empirical_cutoff([r["S_sparse"] for r in threshold_cal], a) for a in ALPHAS}}
    atomic_json(EXP / "configs/SPARSE_EXPOSURE_FROZEN_THRESHOLDS.json", {
        "created_utc": now(), "precommit_sha256": sha_file(PRE), "thresholds": deployment,
        "source": "THRESHOLD_CALIBRATION 250 only", "operator": "strict >"})
    fresh_benign = [r for r in fresh if r["cohort"] == "BENIGN"]
    neg_m, neg_s = [r["M"] for r in fresh_benign], [r["S_sparse"] for r in fresh_benign]
    matched = []; tpr3 = {}
    for attack in ATTACKS:
        member = [r for r in fresh if r.get("attack") == attack and r.get("membership") == "member"]
        for alpha in ALPHAS:
            mt = tpr_at_fpr([r["M"] for r in member], neg_m, alpha)
            st = tpr_at_fpr([r["S_sparse"] for r in member], neg_s, alpha)
            matched.append({"attack": attack, "positive": "member_attack_query", "member_query_n": len(member),
                            "benign_query_n": len(fresh_benign), "target_query_fpr": alpha,
                            "interpolation": "LINEAR_INTERPOLATED_EMPIRICAL_ROC", "mirabel_tpr": mt,
                            "sparse_tpr": st, "delta": st-mt})
            if alpha == PRIMARY_ALPHA: tpr3[attack] = {"M": mt, "S_sparse": st, "delta": st-mt}
    m_t, m_n = matched_binary_threshold(neg_m, PRIMARY_ALPHA)
    s_t, s_n = matched_binary_threshold(neg_s, PRIMARY_ALPHA)
    recovery_rows = []; case_rows = []; menta_cases = []
    for attack in ATTACKS:
        member = [r for r in fresh if r.get("attack") == attack and r.get("membership") == "member"]
        mc, bc, cases = binary_overlap(member, fresh_benign, "S_sparse", m_t, s_t)
        recovery_rows.append({"attack": attack, "member_n": len(member), "both_hit": mc["BOTH_HIT"],
            "mirabel_only_lost": mc["MIRABEL_ONLY"], "sparse_only_recovered": mc["SPARSE_ONLY"],
            "both_miss": mc["BOTH_MISS"], "net_recovery": mc["SPARSE_ONLY"]-mc["MIRABEL_ONLY"],
            "benign_both_fp": bc["BOTH_HIT"], "benign_mirabel_only_fp": bc["MIRABEL_ONLY"],
            "benign_sparse_only_fp": bc["SPARSE_ONLY"], "benign_neither": bc["BOTH_MISS"]})
        if attack == "MEntA": menta_cases = cases
        case_rows.extend({"attack": attack, **c} for c in cases)
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in menta_cases: by_session[r["session_id"]].append(r)
    menta_sessions = []
    recovered_by_target = Counter()
    for sid, cases in sorted(by_session.items()):
        recovered = sum(r["cell"] == "SPARSE_ONLY" for r in cases); lost = sum(r["cell"] == "MIRABEL_ONLY" for r in cases)
        recovered_by_target[sid] = recovered
        menta_sessions.append({"session_id": sid, "unique_recovered_queries": recovered, "lost_queries": lost,
                               "unique_recovered_session": recovered > 0, "lost_session": lost > 0})
    total_recovered = sum(recovered_by_target.values()); shares = sorted(recovered_by_target.values(), reverse=True)
    largest_share = (shares[0]/total_recovered) if total_recovered else 0
    top2_share = (sum(shares[:2])/total_recovered) if total_recovered else 0
    concentration_warning = "HIGHLY_CONCENTRATED_GAIN" if top2_share > .5 else "TARGET_CONCENTRATED_GAIN" if largest_share > .25 else None
    bootstrap_rows = []
    for idx, attack in enumerate(ATTACKS):
        member = [r for r in fresh if r.get("attack") == attack and r.get("membership") == "member"]
        mean, low, high = bootstrap_delta(member, neg_m, neg_s, "S_sparse", 2000, SEED+idx)
        bootstrap_rows.append({"attack": attack, "unit": "target/session", "iterations": 2000,
                               "point_delta": tpr3[attack]["delta"], "bootstrap_mean_delta": mean,
                               "ci95_low": low, "ci95_high": high})
    nonmember = []
    for attack in ATTACKS:
        subset = [r for r in fresh if r.get("attack") == attack and r.get("membership") == "nonmember"]
        nonmember.append({"attack": attack, "queries": len(subset),
                          "mirabel_alarm_rate_matched_3pct": statistics.fmean(r["M"] > m_t for r in subset),
                          "sparse_alarm_rate_matched_3pct": statistics.fmean(r["S_sparse"] > s_t for r in subset)})
    correlation = []
    for group, subset in [("fresh_benign", fresh_benign)] + [(a, [r for r in fresh if r.get("attack") == a and r.get("membership") == "member"]) for a in ATTACKS]:
        correlation.append({"group": group, "n": len(subset), "spearman_rho_M_L": spearman([r["M"] for r in subset], [r["L"] for r in subset])})
    domain_fpr = []
    for eval_name, source in [("FRESH_DEVELOPMENT", fresh_benign),
                              ("LEGACY_LOCKED", [r for r in old if r["calibration_partition"] == "LOCKED_HOLDOUT"])]:
        domains = ["ALL"] + sorted({r["domain"] for r in source})
        for score in ("M", "S_sparse"):
            for alpha in ALPHAS:
                threshold = deployment[score][str(alpha)]
                for domain in domains:
                    subset = source if domain == "ALL" else [r for r in source if r["domain"] == domain]
                    fp = sum(r[score] > threshold for r in subset); lo, hi = wilson(fp, len(subset))
                    domain_fpr.append({"evaluation": eval_name, "detector": score, "nominal_fpr": alpha,
                        "domain": domain, "n": len(subset), "false_positives": fp,
                        "measured_fpr": fp/len(subset) if subset else None, "wilson95_low": lo, "wilson95_high": hi})
    macro_m = statistics.fmean(tpr3[a]["M"] for a in ATTACKS)
    macro_s = statistics.fmean(tpr3[a]["S_sparse"] for a in ATTACKS)
    menta_boot = next(r for r in bootstrap_rows if r["attack"] == "MEntA")
    checks = {"menta_delta_at_least_5pp": tpr3["MEntA"]["delta"] >= .05-1e-12,
              "menta_cluster_bootstrap_ci_low_above_zero": menta_boot["ci95_low"] > 0,
              "mba_noninferior_within_5pp": tpr3["MBA"]["delta"] >= -.05-1e-12,
              "rag_mia_noninferior_within_5pp": tpr3["RAG-MIA"]["delta"] >= -.05-1e-12,
              "core3_macro_member_tpr_improved": macro_s > macro_m+1e-12}
    passed = all(checks.values())
    verdict = "SPARSE_EXPOSURE_SCREEN_PASS" if passed else "SPARSE_EXPOSURE_SCREEN_FAILED"
    final_verdict = "SPARSE_EXPOSURE_CONFIRMATION_REQUIRED" if passed else "TRAINING_FREE_HANDCRAFTED_DETECTOR_SEARCH_CLOSED"
    write_csv(EXP / "tables/PHASE_B_MATCHED_MEMBER_QUERY_TPR.csv", matched)
    write_csv(EXP / "tables/PHASE_B_RECOVERY.csv", recovery_rows)
    write_csv(EXP / "tables/PHASE_B_MEMBER_CASES.csv", case_rows)
    write_csv(EXP / "tables/PHASE_B_MENTA_SESSION_RECOVERY.csv", menta_sessions)
    write_csv(EXP / "tables/PHASE_B_PAIRED_CLUSTER_BOOTSTRAP.csv", bootstrap_rows)
    write_csv(EXP / "tables/PHASE_B_NONMEMBER_ALARM.csv", nonmember)
    write_csv(EXP / "tables/PHASE_B_M_L_CORRELATION.csv", correlation)
    write_csv(EXP / "tables/PHASE_B_DOMAIN_FPR.csv", domain_fpr)
    write_csv(EXP / "audits/EMPTY_TOKEN_CASES.csv", empty_cases, ["query_id", "cohort", "EMPTY_TOKEN_CASE"])
    result = {"verdict": verdict, "final_verdict_if_stopped_now": final_verdict, "completed_utc": now(),
        "runtime_seconds": time.monotonic()-started, "precommit_sha256": sha_file(PRE),
        "development_notice": "THIS_COHORT_IS_NOW_DEVELOPMENT_DATA",
        "matched_member_tpr_at_3pct": tpr3, "macro": {"MIRABEL": macro_m, "SPARSE": macro_s, "delta": macro_s-macro_m},
        "bootstrap": bootstrap_rows, "checks": checks,
        "binary_threshold_diagnostic": {"M": {"threshold": m_t, "benign_alarms": m_n},
                                        "S_sparse": {"threshold": s_t, "benign_alarms": s_n}},
        "gain_concentration": {"total_unique_recovered_queries": total_recovered,
                               "largest_target_fraction": largest_share, "top2_target_fraction": top2_share,
                               "warning": concentration_warning},
        "worst_domain_fpr_at_3pct": max((r["measured_fpr"] for r in domain_fpr if r["detector"] == "S_sparse" and r["nominal_fpr"] == .03 and r["domain"] != "ALL"), default=None),
        "screen_passed": passed, "confirmation_required": passed, "e2e_allowed": False,
        "failure_closure": None if passed else "No further handcrafted detector search; no generation."}
    atomic_json(EXP / "PHASE_B_RESULT.json", result)
    atomic_json(EXP / "RESULT.json", {"campaign": "ORTHOGONAL_EXPOSURE_SEARCH_V1", "phase_a": json.loads((EXP / "PHASE_A_RESULT.json").read_text()), "phase_b": result,
                                      "final_verdict": final_verdict, "fresh_confirmation_completed": False, "e2e_completed": False})
    report = ["# Phase B — Training-free Sparse Exposure", "", f"- Screen 판정: `{verdict}`",
              f"- 현재 최종 판정: `{final_verdict}`", "- 이 100/100 cohort는 development data이다.", "",
              "| Attack | MIRABEL TPR@3% | Sparse TPR@3% | Delta |", "|---|---:|---:|---:|"]
    for attack in ATTACKS: report.append(f"| {attack} | {tpr3[attack]['M']:.3f} | {tpr3[attack]['S_sparse']:.3f} | {tpr3[attack]['delta']:+.3f} |")
    report += [f"| Macro | {macro_m:.3f} | {macro_s:.3f} | {macro_s-macro_m:+.3f} |", "", "## Gate"]
    report.extend(f"- {k}: `{'PASS' if v else 'FAIL'}`" for k, v in checks.items())
    report += ["", f"- MEntA bootstrap 95% CI: [{menta_boot['ci95_low']:+.4f}, {menta_boot['ci95_high']:+.4f}]",
               f"- gain concentration warning: `{concentration_warning}`",
               "", "Screen 실패 시 명세대로 generation과 추가 handcrafted 탐색을 실행하지 않는다." if not passed else "Screen 통과: 별도 fresh 100/100 confirmation precommit으로 이동한다."]
    (EXP / "reports/PHASE_B_REPORT_KO.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    checkpoint(verdict, final_verdict=final_verdict, confirmation_required=passed, e2e_allowed=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

