#!/usr/bin/env python3
"""Analyze the frozen CLEAN_V1 small Oracle/locator LOO mechanism gate."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ROOT, SEED, TARGET_FPRS, atomic_csv, atomic_json, atomic_text, checkpoint, sha256_file

SCORES = ROOT / "private/PHASE1_QUERY_SCORES.csv.gz"
CONDITIONS = {
    "ORACLE": "risk_oracle",
    "MIRABEL": "risk_mirabel",
    "QLL": "risk_qll",
    "UNION2": "risk_union2",
}
BOOTSTRAPS = 10000


def strict_threshold(values: np.ndarray, target: float) -> tuple[float, int, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite calibration vector required")
    allowed = int(math.floor(target * len(values) + 1e-12))
    ordered = np.sort(values)[::-1]
    threshold = float(ordered[allowed]) if allowed < len(values) else float(np.nextafter(ordered[-1], -np.inf))
    count = int(np.sum(values > threshold))
    if count > allowed:
        raise RuntimeError("strict FPR contract failed")
    return threshold, count, count / len(values)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z*z/total
    center = (rate + z*z/(2*total)) / denominator
    radius = z * math.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denominator
    return max(0.0, center-radius), min(1.0, center+radius)


def bootstrap_auc_effect(member: np.ndarray, nonmember: np.ndarray, seed: int) -> dict:
    member = np.asarray(member, dtype=float)
    nonmember = np.asarray(nonmember, dtype=float)
    labels = np.r_[np.ones(len(member), dtype=int), np.zeros(len(nonmember), dtype=int)]
    values = np.r_[member, nonmember]
    raw = float(roc_auc_score(labels, values))
    effect = float(member.mean() - nonmember.mean())
    rng = np.random.default_rng(seed)
    aucs = np.empty(BOOTSTRAPS)
    effects = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        m = member[rng.integers(0, len(member), len(member))]
        n = nonmember[rng.integers(0, len(nonmember), len(nonmember))]
        pair = m[:, None] - n[None, :]
        aucs[index] = float((pair > 0).mean() + 0.5*(pair == 0).mean())
        effects[index] = float(m.mean() - n.mean())
    return {
        "roc_auc": raw, "symmetric_auc": max(raw, 1-raw),
        "auc_ci_low": float(np.quantile(aucs, .025)), "auc_ci_high": float(np.quantile(aucs, .975)),
        "member_mean": float(member.mean()), "member_median": float(np.median(member)),
        "nonmember_mean": float(nonmember.mean()), "nonmember_median": float(np.median(nonmember)),
        "mean_effect": effect, "effect_ci_low": float(np.quantile(effects, .025)),
        "effect_ci_high": float(np.quantile(effects, .975)),
    }


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    precommit = ROOT / "configs/PRECOMMIT.json"
    expected = (ROOT / "configs/PRECOMMIT.sha256").read_text().split()[0]
    if sha256_file(precommit) != expected:
        raise RuntimeError("PRECOMMIT drift before analysis")
    protocol = json.loads(precommit.read_text())
    scores = pd.read_csv(SCORES, keep_default_na=False, low_memory=False,
                         dtype={"case_id": str, "session_id": str, "target_id": str})
    attack = pd.read_csv(ROOT / "private/PHASE1_ATTACK_SELECTION.csv.gz", keep_default_na=False,
                         dtype={"case_id": str, "session_id": str, "target_id": str})
    benign = pd.read_csv(ROOT / "private/PHASE1_BENIGN_SELECTION.csv.gz", keep_default_na=False,
                         dtype={"case_id": str, "session_id": str, "target_id": str})
    if len(scores) != 780 or scores.case_id.nunique() != 780:
        raise RuntimeError("incomplete Phase-1 query scores")
    return scores, attack, benign, protocol


def main() -> None:
    checkpoint("PHASE1_ANALYSIS_STARTED")
    scores, attack_selection, benign_selection, protocol = load()
    benign = scores[scores.kind.eq("BENIGN")].merge(
        benign_selection[["case_id", "split"]], on="case_id", validate="one_to_one")
    attack = scores[scores.kind.eq("ATTACK")].copy()
    if len(benign) != 500 or len(attack) != 280:
        raise RuntimeError(f"cohort drift benign={len(benign)} attack={len(attack)}")

    threshold_rows = []
    low_rows = []
    session_scores = {}
    for condition, column in CONDITIONS.items():
        calibration = benign[benign.split.eq("CALIBRATION")][column].to_numpy(float)
        holdout = benign[benign.split.eq("HOLDOUT")][column].to_numpy(float)
        for target in TARGET_FPRS:
            tau, calibration_fp, calibration_fpr = strict_threshold(calibration, target)
            holdout_fp = int(np.sum(holdout > tau))
            threshold_rows.append({
                "condition": condition, "target_fpr": target, "threshold": tau,
                "calibration_n": len(calibration), "calibration_fp": calibration_fp,
                "calibration_fpr": calibration_fpr, "holdout_n": len(holdout),
                "holdout_fp": holdout_fp, "holdout_fpr": holdout_fp/len(holdout),
            })
            grouped = attack.groupby(["family", "member", "session_id"], as_index=False)[column].max()
            for family in sorted(grouped.family.unique()):
                cell = grouped[grouped.family.eq(family)]
                for population, part in (
                    ("ALL_ATTACK", cell), ("MEMBER", cell[cell.member.eq(1)]),
                    ("NONMEMBER", cell[cell.member.eq(0)]),
                ):
                    detected = int(np.sum(part[column].to_numpy(float) > tau))
                    lo, hi = wilson(detected, len(part))
                    low_rows.append({
                        "condition": condition, "family": family, "population": population,
                        "target_fpr": target, "threshold": tau, "sessions": len(part),
                        "detected": detected, "tpr": detected/len(part),
                        "wilson_ci_low": lo, "wilson_ci_high": hi,
                    })
        session_scores[condition] = attack.groupby(
            ["family", "member", "session_id"], as_index=False
        )[column].max().rename(columns={column: "risk"})
    thresholds = pd.DataFrame(threshold_rows)
    low = pd.DataFrame(low_rows)
    atomic_csv(thresholds, ROOT / "tables/PHASE1_THRESHOLDS_AND_HOLDOUT_FPR.csv")
    atomic_csv(low, ROOT / "tables/PHASE1_LOW_FPR_TPR.csv")

    auc_rows = []
    for cidx, condition in enumerate(CONDITIONS):
        grouped = session_scores[condition]
        for fidx, family in enumerate(sorted(grouped.family.unique())):
            cell = grouped[grouped.family.eq(family)]
            member = cell[cell.member.eq(1)].risk.to_numpy(float)
            nonmember = cell[cell.member.eq(0)].risk.to_numpy(float)
            result = bootstrap_auc_effect(member, nonmember, SEED + 100*cidx + fidx)
            auc_rows.append({"condition": condition, "family": family,
                             "member_sessions": len(member), "nonmember_sessions": len(nonmember), **result})
    auc_table = pd.DataFrame(auc_rows)
    atomic_csv(auc_table, ROOT / "tables/PHASE1_AUC_EFFECT_BOOTSTRAP.csv")

    retrieval_rows = []
    for family in sorted(attack.family.unique()):
        for member in (0, 1):
            cell = attack[(attack.family.eq(family)) & attack.member.eq(member)]
            retrieval_rows.append({
                "family": family, "member": member, "queries": len(cell),
                "sessions": cell.session_id.nunique(),
                "target_retrieval_at4": float(cell.target_in_top4.astype(bool).mean()),
                "target_rank1": int(cell.target_rank.eq(1).sum()),
                "target_rank2": int(cell.target_rank.eq(2).sum()),
                "target_rank3": int(cell.target_rank.eq(3).sum()),
                "target_rank4": int(cell.target_rank.eq(4).sum()),
                "target_not_retrieved": int(cell.target_rank.eq(0).sum()),
            })
    retrieval = pd.DataFrame(retrieval_rows)
    atomic_csv(retrieval, ROOT / "tables/TARGET_RETRIEVAL_AT4_BY_FAMILY.csv")

    locator_rows = []
    members = attack[attack.member.eq(1)].copy()
    members["mirabel_target_hit"] = members.mirabel_source_id.astype(str).eq(members.target_id.astype(str))
    members["qll_target_hit"] = members.qll_source_id.astype(str).eq(members.target_id.astype(str))
    members["union2_target_hit"] = members.apply(
        lambda row: int(row.target_rank) in set(json.loads(row.union_candidate_ranks)), axis=1)
    for family in sorted(members.family.unique()):
        family_cell = members[members.family.eq(family)]
        for rank in (0, 1, 2, 3, 4):
            cell = family_cell[family_cell.target_rank.eq(rank)]
            if not len(cell):
                continue
            locator_rows.append({
                "family": family, "target_rank": rank, "queries": len(cell),
                "mirabel_hit1": float(cell.mirabel_target_hit.mean()),
                "qll_hit1": float(cell.qll_target_hit.mean()),
                "union2_hit2": float(cell.union2_target_hit.mean()),
            })
        retrieved = family_cell[family_cell.target_in_top4.astype(bool)]
        locator_rows.append({
            "family": family, "target_rank": "RETRIEVED_ALL", "queries": len(retrieved),
            "mirabel_hit1": float(retrieved.mirabel_target_hit.mean()) if len(retrieved) else math.nan,
            "qll_hit1": float(retrieved.qll_target_hit.mean()) if len(retrieved) else math.nan,
            "union2_hit2": float(retrieved.union2_target_hit.mean()) if len(retrieved) else math.nan,
        })
    locator = pd.DataFrame(locator_rows)
    atomic_csv(locator, ROOT / "tables/LOCATOR_HIT_BY_TARGET_RANK.csv")

    grounding_rows = []
    for split in ("CALIBRATION", "HOLDOUT"):
        cell = benign[benign.split.eq(split)]
        retrieved = cell[cell.target_in_top4.astype(bool)]
        gold_values = []
        nongold_values = []
        for row in retrieved.itertuples(index=False):
            rank = int(row.target_rank)
            gold_values.append(float(getattr(row, f"I_{rank}")))
            nongold_values.extend(float(getattr(row, f"I_{other}")) for other in range(1, 5) if other != rank)
        grounding_rows.append({
            "split": split, "queries": len(cell), "gold_retrieved_queries": len(retrieved),
            "gold_retrieval_at4": len(retrieved)/len(cell),
            "gold_source_influence_mean": float(np.mean(gold_values)) if gold_values else math.nan,
            "gold_source_influence_median": float(np.median(gold_values)) if gold_values else math.nan,
            "non_gold_source_influence_mean": float(np.mean(nongold_values)) if nongold_values else math.nan,
            "non_gold_source_influence_median": float(np.median(nongold_values)) if nongold_values else math.nan,
        })
    grounding = pd.DataFrame(grounding_rows)
    atomic_csv(grounding, ROOT / "tables/BENIGN_GOLD_VS_NONGOLD_LOO.csv")

    def auc_row(condition: str, family: str) -> pd.Series:
        return auc_table[(auc_table.condition.eq(condition)) & auc_table.family.eq(family)].iloc[0]

    def tpr(condition: str, family: str, target: float = .03, population: str = "MEMBER") -> float:
        return float(low[(low.condition.eq(condition)) & low.family.eq(family) &
                         low.target_fpr.eq(target) & low.population.eq(population)].iloc[0].tpr)

    menta_oracle = auc_row("ORACLE", "MEntA")
    secondary_oracle = [auc_row("ORACLE", family) for family in ("S²-MIA", "MBA")]
    menta_oracle_pass = bool(
        menta_oracle.roc_auc >= .60 and menta_oracle.auc_ci_low > .50 and
        menta_oracle.mean_effect > 0 and menta_oracle.effect_ci_low > 0
    )
    secondary_oracle_pass = any(
        row.roc_auc >= .60 and row.auc_ci_low > .50 and row.mean_effect > 0 and row.effect_ci_low > 0
        for row in secondary_oracle
    )
    oracle_pass = menta_oracle_pass and secondary_oracle_pass

    menta_locator = locator[(locator.family.eq("MEntA")) & locator.target_rank.eq("RETRIEVED_ALL")]
    union_hit = float(menta_locator.iloc[0].union2_hit2) if len(menta_locator) else math.nan
    oracle_tpr = tpr("ORACLE", "MEntA")
    union_tpr = tpr("UNION2", "MEntA")
    mirabel_tpr = tpr("MIRABEL", "MEntA")
    retention = union_tpr/oracle_tpr if oracle_tpr > 0 else 0.0
    locator_checks = {
        "union2_retrieved_member_target_hit2_ge_0_90": bool(np.isfinite(union_hit) and union_hit >= .90),
        "menta_union2_tpr3_gt_mirabel": bool(union_tpr > mirabel_tpr),
        "menta_union2_oracle_tpr3_retention_ge_0_80": bool(retention >= .80),
        "s2_union2_not_worse_than_mirabel_by_0_05": bool(tpr("UNION2", "S²-MIA") >= tpr("MIRABEL", "S²-MIA") - .05),
        "mba_union2_not_worse_than_mirabel_by_0_05": bool(tpr("UNION2", "MBA") >= tpr("MIRABEL", "MBA") - .05),
    }
    locator_pass = bool(all(locator_checks.values()))
    if not oracle_pass:
        verdict = "LOO_MECHANISM_REJECTED"
    elif not locator_pass:
        verdict = "LOO_LOCATOR_LIMITED"
    else:
        verdict = "LOO_PHASE1_PASS"

    result = {
        "campaign": protocol["campaign"], "phase": 1, "verdict": verdict,
        "phase2_opened": False, "protected_generation_count": 0,
        "oracle_gate": {
            "pass": oracle_pass, "menta_pass": menta_oracle_pass,
            "s2_or_mba_pass": secondary_oracle_pass,
        },
        "locator_gate": {"evaluated": oracle_pass, "pass": locator_pass if oracle_pass else False,
                         "checks": locator_checks, "menta_union2_hit2": union_hit,
                         "menta_oracle_tpr3": oracle_tpr, "menta_union2_tpr3": union_tpr,
                         "menta_mirabel_tpr3": mirabel_tpr, "menta_oracle_retention": retention},
        "files": {
            "scores_sha256": sha256_file(SCORES),
            "precommit_sha256": sha256_file(ROOT / "configs/PRECOMMIT.json"),
        },
        "interpretation": (
            "Phase 1 measures LOO mechanism feasibility only. It is not post-generation privacy evidence. "
            "Oracle overall scores assign zero when the target is absent; target-retrieved-only membership AUC "
            "is generally undefined because nonmember targets are intentionally outside the corpus."
        ),
    }
    atomic_json(ROOT / "FINAL_RESULT.json", result)

    key = auc_table[auc_table.condition.isin(["ORACLE", "MIRABEL", "QLL", "UNION2"])].copy()
    report = [
        "# CLEAN_V1 Phase 1 결과", "", f"- Verdict: **{verdict}**",
        f"- New retrieval/query scores: **{len(scores)}**", "- Protected-answer generation: **0**",
        "- Legacy numeric artifact mixed: **NO**", "",
        "## Oracle mechanism", "",
        "| Attack | AUC | 95% CI | Member mean | Nonmember mean | Effect 95% CI | TPR@3% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("MEntA", "S²-MIA", "MBA"):
        row = auc_row("ORACLE", family)
        report.append(
            f"| {family} | {row.roc_auc:.3f} | [{row.auc_ci_low:.3f}, {row.auc_ci_high:.3f}] | "
            f"{row.member_mean:.5f} | {row.nonmember_mean:.5f} | "
            f"[{row.effect_ci_low:.5f}, {row.effect_ci_high:.5f}] | {tpr('ORACLE', family):.3f} |"
        )
    report += ["", "## Locator comparison at benign-calibrated 3% FPR", "",
               "| Attack | Oracle | MIRABEL | QLL | UNION-2 |", "|---|---:|---:|---:|---:|"]
    for family in ("MEntA", "S²-MIA", "MBA"):
        report.append(f"| {family} | {tpr('ORACLE', family):.3f} | {tpr('MIRABEL', family):.3f} | "
                      f"{tpr('QLL', family):.3f} | {tpr('UNION2', family):.3f} |")
    report += ["", "## Gate", "", f"- Oracle gate: **{oracle_pass}**",
               f"- Locator gate evaluated: **{oracle_pass}**", f"- Locator gate pass: **{locator_pass if oracle_pass else False}**",
               "- Phase 2 was not started automatically by this script.", "",
               "Oracle의 target-not-retrieved row는 overall 분석에서 risk=0으로 기록했다. "
               "target이 Top-4에 있는 subset은 별도 retrieval/분포 표로 보존했으며, nonmember target은 corpus 밖이므로 "
               "그 subset의 membership AUC는 일반적으로 정의되지 않는다.", ""]
    atomic_text(ROOT / "reports/PHASE1_REPORT_KO.md", "\n".join(report))
    checkpoint(verdict, oracle_pass=oracle_pass, locator_pass=locator_pass if oracle_pass else False,
               phase2_opened=False, final_result_sha256=sha256_file(ROOT / "FINAL_RESULT.json"))


if __name__ == "__main__":
    main()
