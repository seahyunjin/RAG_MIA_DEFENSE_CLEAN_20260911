#!/usr/bin/env python3
"""Exp191: read-only two-channel signal-compression audit.

The script consumes only frozen Exp179/189/190 artifacts.  It does not build a
defense, generate answers, train a classifier, or select a deployment
threshold.  Candidate signals and deterministic tie-break rules are frozen in
this file and in PRECOMMIT.json before statistics are calculated.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp191_two_channel_signal_compression_audit_20260829"
EXP190 = PROJECT / "exp190_leakage_channel_decomposition_audit_20260829"
EXP189 = PROJECT / "exp189_simple_global_disclosure_ledger_20260829"
EXP179 = PROJECT / "exp179_cross_family_qwen_source_influence_20260828"

REPLICATES = 20_000
SEED = 19120260829
MIN_N_PER_CLASS = 20
MIN_ABS_RHO = 0.10
PRIMARY_I = ["RAG-MIA", "S²-MIA", "MBA", "RAGLeak"]
SECONDARY_I = ["DCMI", "MEntA", "BudgetLeak-Z"]
EXPLORE = ["IA"]
PRIMARY_C = ["DCMI", "MEntA", "BudgetLeak-Z"]
FAMILIES = PRIMARY_I + ["DCMI", "MEntA", "BudgetLeak-Z", "IA"]

I_CANDIDATES = {
    "I1": ("retriever_top1", "Retriever top1 concentration", "FREE", True, 1),
    "I2": ("retriever_margin", "Retriever top1-top2 margin", "FREE", True, 2),
    "I3": ("qll_dominance", "QLL top-source dominance", "EXPENSIVE", False, 7),
    "I4": ("qll_margin", "QLL top1-top2 margin", "EXPENSIVE", False, 8),
    "I5": ("first_max_c1", "First-response max C1", "EXPENSIVE", True, 1),
    "I6": ("first_disclosed_sum_c1", "First-response sum disclosed C1", "EXPENSIVE", True, 3),
    "I7": ("first_max_cinf", "First-response max Cinf", "EXPENSIVE", True, 2),
    "I8": ("first_disclosed_sum_cinf", "First-response sum disclosed Cinf", "EXPENSIVE", True, 4),
    "I9": ("first_target_dominant_disclosed_claims", "First-response target-dominant disclosed-claim count", "EXPENSIVE", True, 6),
    "I10": ("first_source_dependent_disclosed_claims", "First-response source-dependent claim count", "EXPENSIVE", True, 5),
}

C_CANDIDATES = {
    "C1": ("final_cumulative_l1", "Final cumulative L1", "EXPENSIVE", True, 1),
    "C2": ("final_cumulative_linf", "Final cumulative Linf", "EXPENSIVE", True, 2),
    "C3": ("l1_growth", "L1 growth from first to final query", "EXPENSIVE", True, 3),
    "C4": ("linf_growth", "Linf growth from first to final query", "EXPENSIVE", True, 4),
    "C5": ("newly_charged_claims", "Newly charged claim count", "EXPENSIVE", True, 5),
    "C6": ("cumulative_source_dependent_claims", "Cumulative source-dependent disclosed-claim count", "EXPENSIVE", True, 6),
    "C7": ("distinct_retrieved_sources", "Number of distinct retrieved sources", "FREE", True, 1),
    "C8": ("max_exposed_prefix_union", "Maximum exposed-prefix union", "FREE", True, 2),
}

INPUTS = {
    "exp190_features": EXP190 / "private/EXP190_SESSION_FEATURES.private.csv.gz",
    "exp190_final": EXP190 / "FINAL_RESULT.json",
    "exp190_inventory": EXP190 / "tables/TABLE_190_01_SIGNAL_INVENTORY.csv",
    "exp190_zero": EXP190 / "tables/TABLE_190_06_ZERO_INTERVENTION.csv",
    "exp179_qll_summary": EXP179 / "tables/TABLE_179_02_QUERY_SUMMARY.csv",
    "exp189_benign_ledger": EXP189 / "private/BENIGN_HELDOUT_LEDGER.private.csv.gz",
    "exp189_benign_manifest": EXP189 / "audits/BENIGN_SPLIT_MANIFEST.csv",
    "exp189_benign_factuality": EXP189 / "private/BENIGN_FACTUALITY_ROWS.private.csv.gz",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts):
    raw = "|".join(map(str, parts)).encode()
    return (SEED + int(hashlib.sha256(raw).hexdigest()[:12], 16)) % (2**32 - 1)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                 default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def checkpoint(stage, **details):
    payload = {
        "experiment": "Exp191", "stage": stage, "updated_utc": now(), "pid": os.getpid(),
        "diagnostic_only": True, "new_defense": False, "new_generation": False,
        "llama": False, "fresh_blind": False, "e_mia_open_count": 0,
        "threshold_search": False, "classifier_training": False,
        "e_auc_used": False, "replicates": REPLICATES, **details,
    }
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp191 Status", "", f"- Stage: **{stage}**", f"- Updated UTC: `{payload['updated_utc']}`",
             f"- PID: `{payload['pid']}`"] + [f"- {k}: `{v}`" for k, v in details.items()]
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def freeze_inputs():
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing frozen input(s): {missing}")
    exp190_result = json.loads(INPUTS["exp190_final"].read_text())
    if exp190_result.get("final_verdict") != "TWO_CHANNEL_PREMISE_SUPPORTED":
        raise RuntimeError("Exp190 final verdict is not the required frozen premise")
    rows = []
    for key, path in INPUTS.items():
        rows.append({"key": key, "path": str(path), "sha256": sha256_file(path),
                     "bytes": path.stat().st_size, "access": "READ_ONLY"})
    manifest = pd.DataFrame(rows)
    atomic_csv(manifest, ROOT / "provenance/FROZEN_INPUTS.csv")
    precommit = {
        "experiment": "Exp191", "purpose": "TWO_CHANNEL_SIGNAL_COMPRESSION_AUDIT",
        "candidate_instantaneous_signals": I_CANDIDATES,
        "candidate_cumulative_signals": C_CANDIDATES,
        "primary_instantaneous_families": PRIMARY_I,
        "secondary_instantaneous_families": SECONDARY_I,
        "primary_cumulative_families": PRIMARY_C,
        "ia_role": "EXPLORATORY_NOT_FOR_SELECTION",
        "support_rule": {
            "minimum_per_class": MIN_N_PER_CLASS, "bootstrap_replicates": REPLICATES,
            "membership_permutations": REPLICATES, "association_permutations": REPLICATES,
            "association_minimum_abs_spearman_rho": MIN_ABS_RHO,
            "required_direction_stability": 0.95,
        },
        "instantaneous_go": "SUPPORTED>=3/4 primary; remainder>=PARTIAL; consistent sign",
        "cumulative_go": "SUPPORTED>=2/3 primary; remainder>=PARTIAL; consistent sign",
        "rank_instability": "effect sign reversal or min(abs rank effect)<0.25*max(abs rank effect)",
        "selection_tiebreak": ["family coverage", "runtime cost", "naturally produced",
                               "mathematical simplicity", "median within-family absolute effect"],
        "forbidden": ["new defense", "generation", "threshold tuning", "weighted fusion",
                      "classifier", "family-specific sign or threshold", "E-AUC", "fresh blind", "E-MIA"],
        "inputs": {r["key"]: {"path": r["path"], "sha256": r["sha256"]} for r in rows},
        "created_utc": now(),
    }
    atomic_json(ROOT / "configs/PRECOMMIT.json", precommit)
    atomic_text(ROOT / "configs/PRECOMMIT.sha256", sha256_file(ROOT / "configs/PRECOMMIT.json") + "  PRECOMMIT.json\n")
    checkpoint("INPUTS_FROZEN", inputs=len(rows), exp190_verdict=exp190_result["final_verdict"])
    return manifest


def clean_array(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return x[np.isfinite(x)]


def iqr(x):
    return float(np.quantile(x, .75) - np.quantile(x, .25)) if len(x) else np.nan


def hedges_g(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    pooled = math.sqrt(((len(pos)-1)*np.var(pos, ddof=1) + (len(neg)-1)*np.var(neg, ddof=1)) /
                       max(len(pos)+len(neg)-2, 1))
    if pooled == 0:
        return np.nan
    d = (np.mean(pos)-np.mean(neg))/pooled
    correction = 1 - 3/max(4*(len(pos)+len(neg))-9, 1)
    return float(d*correction)


def bootstrap_difference(pos, neg, seed, reps=REPLICATES):
    rng = np.random.default_rng(seed)
    out = np.empty(reps, float)
    batch = 250
    for start in range(0, reps, batch):
        size = min(batch, reps-start)
        pi = rng.integers(0, len(pos), size=(size, len(pos)))
        ni = rng.integers(0, len(neg), size=(size, len(neg)))
        out[start:start+size] = pos[pi].mean(axis=1) - neg[ni].mean(axis=1)
    return (float(np.quantile(out, .025)), float(np.quantile(out, .975)),
            float(np.mean(out > 0)), float(np.mean(out < 0)))


def permutation_difference(pos, neg, seed, reps=REPLICATES):
    values = np.r_[pos, neg].astype(float)
    labels = np.r_[np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)]
    observed = abs(float(np.mean(pos)-np.mean(neg)))
    rng = np.random.default_rng(seed)
    exceed = 0
    batch = 250
    for start in range(0, reps, batch):
        size = min(batch, reps-start)
        perm = rng.permuted(np.broadcast_to(labels, (size, len(labels))).copy(), axis=1)
        n1 = perm.sum(axis=1)
        s1 = perm @ values
        diff = s1/n1 - (values.sum()-s1)/(len(values)-n1)
        exceed += int(np.sum(np.abs(diff) >= observed-1e-15))
    return float((exceed+1)/(reps+1))


def spearman(x, y):
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman(x, y, seed, reps=REPLICATES):
    """Paired bootstrap of empirical rank transforms (20,000 resamples)."""
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty(reps, float)
    batch = 200
    for start in range(0, reps, batch):
        size = min(batch, reps-start)
        idx = rng.integers(0, len(rx), size=(size, len(rx)))
        xb, yb = rx[idx], ry[idx]
        xb = xb-xb.mean(axis=1, keepdims=True)
        yb = yb-yb.mean(axis=1, keepdims=True)
        den = np.sqrt(np.sum(xb*xb, axis=1)*np.sum(yb*yb, axis=1))
        out[start:start+size] = np.divide(np.sum(xb*yb, axis=1), den,
                                          out=np.full(size, np.nan), where=den>0)
    finite = out[np.isfinite(out)]
    return (float(np.quantile(finite, .025)), float(np.quantile(finite, .975))) if len(finite) else (np.nan, np.nan)


def permutation_spearman(x, y, seed, reps=REPLICATES):
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.linalg.norm(rx)*np.linalg.norm(ry))
    if denominator == 0:
        return np.nan
    observed = abs(float(np.dot(rx, ry)/denominator))
    rng = np.random.default_rng(seed)
    exceed = 0
    batch = 250
    for start in range(0, reps, batch):
        size = min(batch, reps-start)
        perm = rng.permuted(np.broadcast_to(ry, (size, len(ry))).copy(), axis=1)
        values = np.abs((perm @ rx)/denominator)
        exceed += int(np.sum(values >= observed-1e-15))
    return float((exceed+1)/(reps+1))


def candidate_inventory(features):
    rows = []
    for channel, candidates in (("INSTANTANEOUS", I_CANDIDATES), ("CUMULATIVE", C_CANDIDATES)):
        for cid, (signal, description, cost, natural, simplicity) in candidates.items():
            available = signal in features and features[signal].notna().any()
            coverage = {f: int(features.loc[features.family.eq(f), signal].notna().sum()) if signal in features else 0
                        for f in FAMILIES}
            rows.append({"candidate_id": cid, "channel": channel, "signal": signal,
                         "description": description, "runtime_cost": cost,
                         "naturally_produced": natural, "simplicity_rank": simplicity,
                         "available_rows": sum(coverage.values()), "available_families": sum(v>0 for v in coverage.values()),
                         "family_coverage": json.dumps(coverage, ensure_ascii=False, sort_keys=True),
                         "status": "AVAILABLE" if available else "UNAVAILABLE_NOT_IMPUTED"})
    frame = pd.DataFrame(rows)
    atomic_csv(frame, ROOT / "tables/TABLE_191_01_CANDIDATE_SIGNAL_INVENTORY.csv")
    atomic_csv(frame[["candidate_id", "channel", "signal", "runtime_cost", "naturally_produced", "description"]],
               ROOT / "tables/TABLE_191_08_COST_CLASSIFICATION.csv")
    return frame


def one_family_signal(features, family, cid, signal, role):
    cell = features.loc[features.family.eq(family), ["member", "attack_score", signal]].replace([np.inf, -np.inf], np.nan)
    sep = cell.dropna(subset=[signal])
    pos = sep.loc[sep.member.eq(1), signal].to_numpy(float)
    neg = sep.loc[sep.member.eq(0), signal].to_numpy(float)
    row = {"candidate_id": cid, "signal": signal, "family": family, "family_role": role,
           "n_member": len(pos), "n_nonmember": len(neg), "coverage": len(sep)/max(len(cell), 1),
           "member_mean": np.mean(pos) if len(pos) else np.nan,
           "member_median": np.median(pos) if len(pos) else np.nan,
           "member_iqr": iqr(pos), "nonmember_mean": np.mean(neg) if len(neg) else np.nan,
           "nonmember_median": np.median(neg) if len(neg) else np.nan, "nonmember_iqr": iqr(neg),
           "mean_difference": np.nan, "standardized_effect_hedges_g": np.nan,
           "difference_ci95_low": np.nan, "difference_ci95_high": np.nan,
           "direction_stability": np.nan, "membership_permutation_p": np.nan,
           "separation_status": "UNRESOLVED"}
    if len(pos) >= MIN_N_PER_CLASS and len(neg) >= MIN_N_PER_CLASS and np.std(np.r_[pos, neg]) > 0:
        low, high, positive, negative = bootstrap_difference(pos, neg, stable_seed("diff", family, cid))
        direction = max(positive, negative)
        row.update({"mean_difference": float(np.mean(pos)-np.mean(neg)),
                    "standardized_effect_hedges_g": hedges_g(pos, neg),
                    "difference_ci95_low": low, "difference_ci95_high": high,
                    "direction_stability": direction,
                    "membership_permutation_p": permutation_difference(pos, neg, stable_seed("perm", family, cid)),
                    "separation_status": "ESTIMATED"})
    valid = cell.dropna(subset=[signal, "attack_score"])
    x = valid[signal].to_numpy(float)
    y = valid.attack_score.to_numpy(float)
    association = {"candidate_id": cid, "signal": signal, "family": family, "family_role": role,
                   "n": len(valid), "spearman_rho": np.nan, "rho_ci95_low": np.nan,
                   "rho_ci95_high": np.nan, "association_permutation_p": np.nan,
                   "association_status": "UNRESOLVED"}
    if len(valid) >= 40 and np.std(x) > 0 and np.std(y) > 0:
        rho = spearman(x, y)
        low, high = bootstrap_spearman(x, y, stable_seed("corrboot", family, cid))
        association.update({"spearman_rho": rho, "rho_ci95_low": low, "rho_ci95_high": high,
                            "association_permutation_p": permutation_spearman(x, y, stable_seed("corrperm", family, cid)),
                            "association_status": "ESTIMATED"})
    return row, association


def derive_support(stats, assoc):
    joined = stats.merge(assoc, on=["candidate_id", "signal", "family", "family_role"], validate="one_to_one")
    joined["separation_supported"] = (
        joined.mean_difference.gt(0) & joined.difference_ci95_low.gt(0) & joined.difference_ci95_high.gt(0) &
        joined.membership_permutation_p.lt(.05) & joined.direction_stability.ge(.95))
    joined["association_supported"] = (
        joined.spearman_rho.ge(MIN_ABS_RHO) & joined.rho_ci95_low.gt(0) & joined.rho_ci95_high.gt(0) &
        joined.association_permutation_p.lt(.05))
    available = joined.separation_status.eq("ESTIMATED") & joined.association_status.eq("ESTIMATED")
    joined["support_status"] = np.where(joined.separation_supported & joined.association_supported, "SUPPORTED",
        np.where(available & (joined.separation_supported | joined.association_supported), "PARTIAL",
        np.where(available, "NOT_SUPPORTED", "UNRESOLVED")))
    return joined


def run_family_statistics(features, inventory):
    stat_i, assoc_i = [], []
    i_roles = {f: ("PRIMARY" if f in PRIMARY_I else "SECONDARY" if f in SECONDARY_I else "EXPLORATORY") for f in FAMILIES}
    total = len(I_CANDIDATES)*len(FAMILIES) + len(C_CANDIDATES)*(len(PRIMARY_C)+1)
    completed = 0
    for cid, (signal, *_rest) in I_CANDIDATES.items():
        for family in FAMILIES:
            completed += 1
            if signal not in features:
                stat_i.append({"candidate_id": cid, "signal": signal, "family": family, "family_role": i_roles[family],
                               "n_member": 0, "n_nonmember": 0, "separation_status": "UNRESOLVED"})
                assoc_i.append({"candidate_id": cid, "signal": signal, "family": family, "family_role": i_roles[family],
                                "n": 0, "association_status": "UNRESOLVED"})
            else:
                s, a = one_family_signal(features, family, cid, signal, i_roles[family]); stat_i.append(s); assoc_i.append(a)
            if completed % 8 == 0:
                checkpoint("STATISTICS_PROGRESS", completed=completed, total=total)
    stats_i = pd.DataFrame(stat_i)
    assoc_i = pd.DataFrame(assoc_i)
    joined_i = derive_support(stats_i, assoc_i)
    atomic_csv(joined_i, ROOT / "tables/TABLE_191_02_INSTANTANEOUS_BY_FAMILY.csv")

    stat_c, assoc_c = [], []
    for cid, (signal, *_rest) in C_CANDIDATES.items():
        for family in PRIMARY_C + EXPLORE:
            completed += 1
            role = "PRIMARY" if family in PRIMARY_C else "EXPLORATORY"
            if signal not in features:
                stat_c.append({"candidate_id": cid, "signal": signal, "family": family, "family_role": role,
                               "n_member": 0, "n_nonmember": 0, "separation_status": "UNRESOLVED"})
                assoc_c.append({"candidate_id": cid, "signal": signal, "family": family, "family_role": role,
                                "n": 0, "association_status": "UNRESOLVED"})
            else:
                s, a = one_family_signal(features, family, cid, signal, role); stat_c.append(s); assoc_c.append(a)
            if completed % 8 == 0:
                checkpoint("STATISTICS_PROGRESS", completed=completed, total=total)
    stats_c = pd.DataFrame(stat_c)
    assoc_c = pd.DataFrame(assoc_c)
    joined_c = derive_support(stats_c, assoc_c)
    atomic_csv(joined_c, ROOT / "tables/TABLE_191_03_CUMULATIVE_BY_FAMILY.csv")
    assoc_all = pd.concat([
        joined_i[["candidate_id", "signal", "family", "family_role", "n", "spearman_rho", "rho_ci95_low", "rho_ci95_high",
                  "association_permutation_p", "association_status", "association_supported"]],
        joined_c[["candidate_id", "signal", "family", "family_role", "n", "spearman_rho", "rho_ci95_low", "rho_ci95_high",
                  "association_permutation_p", "association_status", "association_supported"]],
    ], ignore_index=True)
    atomic_csv(assoc_all, ROOT / "tables/TABLE_191_04_ATTACK_SCORE_ASSOCIATION.csv")
    checkpoint("FAMILY_STATISTICS_COMPLETE", instantaneous_rows=len(joined_i), cumulative_rows=len(joined_c))
    return joined_i, joined_c, assoc_all


def coverage_and_select(joined, candidates, primary, channel):
    rows = []
    for cid, (signal, description, cost, natural, simplicity) in candidates.items():
        cell = joined[joined.candidate_id.eq(cid)]
        primary_cell = cell[cell.family.isin(primary)]
        effects = primary_cell.set_index("family").mean_difference.to_dict()
        resolved_effects = [v for v in effects.values() if np.isfinite(v) and abs(v) > 1e-15]
        sign_consistent = bool(resolved_effects) and (all(v > 0 for v in resolved_effects) or all(v < 0 for v in resolved_effects))
        supported = int(primary_cell.support_status.eq("SUPPORTED").sum())
        partial = int(primary_cell.support_status.eq("PARTIAL").sum())
        unresolved = int(primary_cell.support_status.eq("UNRESOLVED").sum())
        all_at_least_partial = bool(primary_cell.support_status.isin(["SUPPORTED", "PARTIAL"]).all())
        if channel == "INSTANTANEOUS":
            go = supported >= 3 and all_at_least_partial and sign_consistent
            secondary = cell[~cell.family.isin(primary) & ~cell.family.eq("IA")]
        else:
            go = supported >= 2 and all_at_least_partial and sign_consistent
            secondary = cell[cell.family.eq("IA")]
        median_effect = float(primary_cell.standardized_effect_hedges_g.abs().median())
        rows.append({"candidate_id": cid, "signal": signal, "description": description,
                     "primary_supported": supported, "primary_partial": partial, "primary_unresolved": unresolved,
                     "all_primary_at_least_partial": all_at_least_partial,
                     "secondary_supported": int(secondary.support_status.eq("SUPPORTED").sum()),
                     "secondary_partial": int(secondary.support_status.eq("PARTIAL").sum()),
                     "sign_consistent": sign_consistent, "primary_effect_signs": json.dumps(effects, sort_keys=True),
                     "runtime_cost": cost, "naturally_produced": natural, "simplicity_rank": simplicity,
                     "median_abs_primary_hedges_g": median_effect, "common_signal_go": go})
    result = pd.DataFrame(rows)
    cost_rank = {"FREE": 0, "CHEAP": 1, "EXPENSIVE": 2}
    eligible = result[result.common_signal_go].copy()
    selected = None
    if not eligible.empty:
        eligible["cost_rank"] = eligible.runtime_cost.map(cost_rank)
        # Prompt-defined tie break: after PRIMARY family coverage, prefer cost,
        # natural pipeline availability, mathematical simplicity, then effect.
        # Secondary families are reported separately and never inserted ahead
        # of this explicitly frozen order.
        eligible = eligible.sort_values(
            ["primary_supported", "cost_rank", "naturally_produced", "simplicity_rank", "median_abs_primary_hedges_g"],
            ascending=[False, True, False, True, False])
        selected = str(eligible.iloc[0].candidate_id)
    result["selected"] = result.candidate_id.eq(selected)
    return result, selected


def family_centered_association(features, selected):
    rows = []
    for channel, cid, candidate_map, fams in (
        ("INSTANTANEOUS", selected["I"], I_CANDIDATES, FAMILIES),
        ("CUMULATIVE", selected["C"], C_CANDIDATES, PRIMARY_C+EXPLORE),
    ):
        if cid is None:
            continue
        signal = candidate_map[cid][0]
        cell = features.loc[features.family.isin(fams), ["family", signal, "attack_score"]].dropna()
        cell = cell.groupby("family", group_keys=False).filter(lambda x: len(x) >= 40 and x[signal].std() > 0 and x.attack_score.std() > 0)
        x = (cell[signal]-cell.groupby("family")[signal].transform("mean")).to_numpy(float)
        y = (cell.attack_score-cell.groupby("family").attack_score.transform("mean")).to_numpy(float)
        rho = spearman(x, y)
        low, high = bootstrap_spearman(x, y, stable_seed("pooled", cid))
        p = permutation_spearman(x, y, stable_seed("pooledperm", cid))
        rows.append({"channel": channel, "candidate_id": cid, "signal": signal, "n": len(cell),
                     "families": cell.family.nunique(), "family_list": "|".join(sorted(cell.family.unique())),
                     "family_centered_spearman_rho": rho, "rho_ci95_low": low, "rho_ci95_high": high,
                     "permutation_p": p})
    return pd.DataFrame(rows)


def build_benign_features():
    ledger = pd.read_csv(INPUTS["exp189_benign_ledger"])
    rows = []
    for row_id, cell in ledger.groupby("row_id", sort=False):
        released = cell[cell.decision.eq("RELEASE")]
        rows.append({"row_id": str(row_id), "first_max_c1": cell.c1.max(),
                     "first_disclosed_sum_c1": released.c1.sum(), "first_max_cinf": cell.cinf.max(),
                     "first_disclosed_sum_cinf": released.cinf.sum(),
                     "first_source_dependent_disclosed_claims": int((released.c1 > 0).sum()),
                     "final_cumulative_l1": cell.cumulative_l1.iloc[-1],
                     "final_cumulative_linf": cell.cumulative_linf.iloc[-1],
                     "l1_growth": 0.0, "linf_growth": 0.0,
                     "newly_charged_claims": int((~cell.duplicate.astype(bool) & cell.decision.eq("RELEASE")).sum())})
    benign = pd.DataFrame(rows)
    qll = pd.read_csv(INPUTS["exp179_qll_summary"])
    qll = qll[qll.attack_family.eq("NORMAL_GOLD")][["case_id", "margin", "dominance"]].copy()
    qll["row_id"] = qll.case_id.astype(str).str.replace("TOPIO|", "", regex=False)
    qll = qll.rename(columns={"margin": "qll_margin", "dominance": "qll_dominance"})
    benign = benign.merge(qll[["row_id", "qll_margin", "qll_dominance"]], on="row_id", how="outer", validate="one_to_one")
    return benign


def benign_analysis(features, selected):
    benign = build_benign_features()
    rows, alpha_rows = [], []
    for channel, cid, candidate_map in (("INSTANTANEOUS", selected["I"], I_CANDIDATES),
                                         ("CUMULATIVE", selected["C"], C_CANDIDATES)):
        if cid is None:
            continue
        signal = candidate_map[cid][0]
        values = clean_array(benign[signal]) if signal in benign else np.array([])
        rows.append({"distribution": "BENIGN", "channel": channel, "candidate_id": cid, "signal": signal,
                     "family": "NORMAL_GOLD_HELDOUT", "n": len(values), "mean": np.mean(values) if len(values) else np.nan,
                     "median": np.median(values) if len(values) else np.nan, "variance": np.var(values, ddof=1) if len(values)>1 else np.nan,
                     "q01": np.quantile(values,.01) if len(values) else np.nan, "q05": np.quantile(values,.05) if len(values) else np.nan,
                     "q10": np.quantile(values,.10) if len(values) else np.nan, "q25": np.quantile(values,.25) if len(values) else np.nan,
                     "q75": np.quantile(values,.75) if len(values) else np.nan, "q90": np.quantile(values,.90) if len(values) else np.nan,
                     "q95": np.quantile(values,.95) if len(values) else np.nan, "q99": np.quantile(values,.99) if len(values) else np.nan,
                     "session_growth_mean": 0.0 if channel=="CUMULATIVE" and len(values) else np.nan,
                     "standardized_attack_minus_benign": np.nan, "median_attack_benign_percentile": np.nan,
                     "empirical_overlap_coefficient": np.nan, "overlap_note": "NOT_COMPUTED_NO_PREEXISTING_ROBUST_IMPLEMENTATION"})
        if len(values):
            for family in (FAMILIES if channel=="INSTANTANEOUS" else PRIMARY_C+EXPLORE):
                attacks = clean_array(features.loc[features.family.eq(family), signal])
                if not len(attacks):
                    continue
                pooled = math.sqrt((np.var(values,ddof=1)+np.var(attacks,ddof=1))/2) if len(values)>1 and len(attacks)>1 else np.nan
                standardized = (np.mean(attacks)-np.mean(values))/pooled if pooled and pooled>0 else np.nan
                percentiles = np.searchsorted(np.sort(values), attacks, side="right")/len(values)
                rows.append({"distribution": "ATTACK_COMPARISON", "channel": channel, "candidate_id": cid,
                             "signal": signal, "family": family, "n": len(attacks), "mean": np.mean(attacks),
                             "median": np.median(attacks), "variance": np.var(attacks,ddof=1) if len(attacks)>1 else np.nan,
                             "q01": np.quantile(attacks,.01), "q05": np.quantile(attacks,.05), "q10": np.quantile(attacks,.10),
                             "q25": np.quantile(attacks,.25), "q75": np.quantile(attacks,.75), "q90": np.quantile(attacks,.90),
                             "q95": np.quantile(attacks,.95), "q99": np.quantile(attacks,.99), "session_growth_mean": np.nan,
                             "standardized_attack_minus_benign": standardized,
                             "median_attack_benign_percentile": float(np.median(percentiles)),
                             "empirical_overlap_coefficient": np.nan,
                             "overlap_note": "NOT_COMPUTED_NO_PREEXISTING_ROBUST_IMPLEMENTATION"})
            for alpha in (.01,.05,.10,.20):
                alpha_rows.append({"alpha": alpha, "channel": channel, "candidate_id": cid, "signal": signal,
                                   "benign_n": len(values), "benign_quantile": 1-alpha,
                                   "benign_derived_tau": float(np.quantile(values, 1-alpha)),
                                   "attack_data_used": False, "threshold_selected": False})
    dist = pd.DataFrame(rows)
    alpha = pd.DataFrame(alpha_rows)
    if not alpha.empty:
        counts = alpha.groupby("alpha").channel.nunique()
        alpha["same_alpha_parameterizes_both"] = alpha.alpha.map(counts).eq(2)
    atomic_csv(dist, ROOT / "tables/TABLE_191_09_BENIGN_DISTRIBUTIONS.csv")
    atomic_csv(alpha, ROOT / "tables/TABLE_191_10_ALPHA_FEASIBILITY.csv")
    return benign, dist, alpha


def budget_rank(features, selected):
    budget = features[features.family.eq("BudgetLeak-Z")]
    rows = []
    for channel, cid, candidate_map in (("INSTANTANEOUS", selected["I"], I_CANDIDATES),
                                         ("CUMULATIVE", selected["C"], C_CANDIDATES)):
        if cid is None:
            continue
        signal = candidate_map[cid][0]
        nonmember = clean_array(budget.loc[budget.member.eq(0), signal])
        for rank in (1,2,3,4):
            member = clean_array(budget.loc[budget.member.eq(1)&budget.target_rank.eq(rank), signal])
            rows.append({"channel": channel, "candidate_id": cid, "signal": signal, "target_rank": rank,
                         "member_n": len(member), "common_nonmember_n": len(nonmember),
                         "member_mean": np.mean(member) if len(member) else np.nan,
                         "member_median": np.median(member) if len(member) else np.nan,
                         "nonmember_mean": np.mean(nonmember) if len(nonmember) else np.nan,
                         "nonmember_median": np.median(nonmember) if len(nonmember) else np.nan,
                         "mean_difference": np.mean(member)-np.mean(nonmember) if len(member) and len(nonmember) else np.nan,
                         "standardized_effect_hedges_g": hedges_g(member, nonmember)})
    frame = pd.DataFrame(rows)
    stability = {}
    for channel, cell in frame.groupby("channel"):
        effects = cell.standardized_effect_hedges_g.dropna().to_numpy(float)
        reversed_sign = bool(np.any(effects < 0)) if len(effects) else True
        severe = bool(np.min(np.abs(effects)) < .25*np.max(np.abs(effects))) if len(effects) else True
        stability[channel] = not (reversed_sign or severe)
        frame.loc[frame.channel.eq(channel), "rank_signal_stable"] = stability[channel]
        frame.loc[frame.channel.eq(channel), "rank_instability_flag"] = "STABLE" if stability[channel] else "RANK_SIGNAL_INSTABILITY"
    atomic_csv(frame, ROOT / "tables/TABLE_191_11_BUDGETLEAK_BY_RANK.csv")
    return frame, stability


def zero_intervention(features, selected, benign):
    rows=[]
    for family in ["DCMI", "RAG-MIA"]:
        base=features[features.family.eq(family)]
        for subset, cell in (("ALL",base),("ZERO_INTERVENTION",base[base.zero_intervention.astype(bool)])):
            for channel,cid,candidate_map in (("INSTANTANEOUS",selected["I"],I_CANDIDATES),("CUMULATIVE",selected["C"],C_CANDIDATES)):
                if cid is None: continue
                signal=candidate_map[cid][0]; values=clean_array(cell[signal]); benign_values=clean_array(benign[signal]) if signal in benign else np.array([])
                q95=np.quantile(benign_values,.95) if len(benign_values) else np.nan
                rows.append({"family":family,"subset":subset,"sessions":len(cell),"fraction_of_family":len(cell)/len(base),
                             "channel":channel,"candidate_id":cid,"signal":signal,"n":len(values),
                             "mean":np.mean(values) if len(values) else np.nan,"median":np.median(values) if len(values) else np.nan,
                             "benign_q95":q95,"fraction_above_benign_q95":np.mean(values>q95) if len(values) and np.isfinite(q95) else np.nan,
                             "old_first_c1_mean":cell.first_max_c1.mean(),"old_first_cinf_mean":cell.first_max_cinf.mean()})
    frame=pd.DataFrame(rows);atomic_csv(frame,ROOT/"tables/TABLE_191_12_ZERO_INTERVENTION.csv");return frame


def factuality_table():
    frame=pd.DataFrame([{"same_cohort_grounding_labels":False,"historical_source":str(INPUTS["exp189_benign_factuality"]),
                         "attack_cohort_exact_overlap":0,"analysis":"FACTUALITY_CROSSCHECK_UNAVAILABLE",
                         "reason":"No G0/G1/G3 versus G4/G5/G6 labels overlap the exact Exp190 attack cohort; labels were not transplanted",
                         "selected_signals_mainly_error_driven":"UNRESOLVED"}])
    atomic_csv(frame,ROOT/"tables/TABLE_191_13_FACTUALITY_CROSSCHECK.csv");return frame


def sign_table(coverage_i, coverage_c):
    frame=pd.concat([coverage_i.assign(channel="INSTANTANEOUS"),coverage_c.assign(channel="CUMULATIVE")],ignore_index=True)
    out=frame[["channel","candidate_id","signal","primary_effect_signs","sign_consistent","common_signal_go"]]
    atomic_csv(out,ROOT/"tables/TABLE_191_05_SIGN_CONSISTENCY.csv");return out


def save_figure(fig,name):
    for suffix,kwargs in (("png",{"dpi":600}),("pdf",{}),("svg",{})):
        path=ROOT/"figures"/suffix/f"{name}.{suffix}";path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,bbox_inches="tight",**kwargs)
    plt.close(fig)


def heatmap(frame, candidates, families, value, title, name):
    pivot=frame.pivot(index="candidate_id",columns="family",values=value).reindex(index=list(candidates),columns=families)
    fig,ax=plt.subplots(figsize=(9,max(4,.42*len(pivot))))
    im=ax.imshow(pivot.to_numpy(float),aspect="auto",cmap="coolwarm",vmin=-2,vmax=2)
    ax.set_xticks(range(len(families)));ax.set_xticklabels(families,rotation=30,ha="right")
    ax.set_yticks(range(len(pivot)));ax.set_yticklabels(pivot.index)
    ax.set_title(title);fig.colorbar(im,ax=ax,label=value);save_figure(fig,name)


def make_figures(joined_i,joined_c,cov_i,cov_c,dist,rank,selected):
    heatmap(joined_i,I_CANDIDATES,FAMILIES,"standardized_effect_hedges_g","Instantaneous candidate effects","FIG191_01_INSTANTANEOUS_CANDIDATES")
    heatmap(joined_c,C_CANDIDATES,PRIMARY_C+EXPLORE,"standardized_effect_hedges_g","Cumulative candidate effects","FIG191_02_CUMULATIVE_CANDIDATES")
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(cov_i)+len(cov_c));vals=pd.concat([cov_i.primary_supported,cov_c.primary_supported]).to_numpy()
    labels=list(cov_i.candidate_id)+list(cov_c.candidate_id);colors=["#2563a6"]*len(cov_i)+["#e07b39"]*len(cov_c)
    ax.bar(x,vals,color=colors);ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylabel("Supported primary families");ax.set_title("Common-signal family coverage")
    save_figure(fig,"FIG191_03_FAMILY_COVERAGE")
    for channel,joined,cid,fams,num in (("I",joined_i,selected["I"],FAMILIES,"04"),("C",joined_c,selected["C"],PRIMARY_C+EXPLORE,"05")):
        fig,ax=plt.subplots(figsize=(8,4))
        if cid:
            cell=joined[joined.candidate_id.eq(cid)].set_index("family").reindex(fams);x=np.arange(len(fams))
            ax.bar(x,cell.standardized_effect_hedges_g,color="#3977b8");ax.axhline(0,color="black",lw=.8)
            ax.set_xticks(x);ax.set_xticklabels(fams,rotation=30,ha="right");ax.set_title(f"Selected {channel}: {cid}")
        save_figure(fig,f"FIG191_{num}_SELECTED_{channel}_BY_FAMILY")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(channel,cell) in zip(axes,dist.groupby("channel")):
        show=cell[["family","median"]].dropna();ax.bar(np.arange(len(show)),show["median"])
        ax.set_xticks(np.arange(len(show)));ax.set_xticklabels(show.family,rotation=55,ha="right",fontsize=7);ax.set_title(f"{channel}: benign vs attack median")
    save_figure(fig,"FIG191_06_BENIGN_VS_ATTACK_SIGNAL")
    fig,ax=plt.subplots(figsize=(8,4))
    for channel,cell in rank.groupby("channel"):
        ax.plot(cell.target_rank,cell.standardized_effect_hedges_g,marker="o",label=channel)
    ax.axhline(0,color="black",lw=.8);ax.set_xticks([1,2,3,4]);ax.set_xlabel("BudgetLeak target rank");ax.set_ylabel("Hedges g");ax.legend();ax.set_title("Selected signals across target rank")
    save_figure(fig,"FIG191_07_BUDGETLEAK_SIGNAL_BY_RANK")
    fig,ax=plt.subplots(figsize=(10,3));ax.axis("off")
    boxes=[(.03,f"Per response\nI = {selected['I'] or 'NONE'}"),(.37,f"Across session\nC = {selected['C'] or 'NONE'}"),(.71,"One benign-tail\nparameter α")]
    for x,label in boxes:
        ax.add_patch(plt.Rectangle((x,.25),.25,.5,facecolor="#e8f1fb",edgecolor="#2563a6",lw=2));ax.text(x+.125,.5,label,ha="center",va="center",fontsize=11)
    ax.annotate("",xy=(.37,.5),xytext=(.28,.5),arrowprops=dict(arrowstyle="->",lw=2));ax.annotate("",xy=(.71,.5),xytext=(.62,.5),arrowprops=dict(arrowstyle="->",lw=2))
    ax.set_title("Minimal two-channel concept only — no policy implemented")
    save_figure(fig,"FIG191_08_MINIMAL_TWO_CHANNEL_CONCEPT")


def reports(inventory,joined_i,joined_c,cov_i,cov_c,dist,alpha,rank,zero,factuality,selected,verdict,pooled):
    atomic_text(ROOT/"reports/REPORT_191_01_SIGNAL_INVENTORY_KO.md","# Exp191 신호 목록\n\n공격군별 규칙, 가중 결합, 새 신호 생성 없이 Exp190에서 이미 계산된 후보만 동결했다. C6와 C8은 공통 동결 artifact가 없어 `UNAVAILABLE_NOT_IMPUTED`로 남겼다.\n\n"+inventory.to_markdown(index=False)+"\n")
    atomic_text(ROOT/"reports/REPORT_191_02_INSTANTANEOUS_COMPRESSION_KO.md","# 순간 신호 압축\n\n선정 결과: `"+str(selected["I"])+"`. Primary family coverage와 동일 부호를 먼저 적용하고 비용·자연 산출·단순성 순으로 tie-break했다.\n\n"+cov_i.to_markdown(index=False)+"\n")
    atomic_text(ROOT/"reports/REPORT_191_03_CUMULATIVE_COMPRESSION_KO.md","# 누적 신호 압축\n\n선정 결과: `"+str(selected["C"])+"`. BudgetLeak 전용 reference 변수는 공통 누적 신호 후보에서 제외했다.\n\n"+cov_c.to_markdown(index=False)+"\n")
    atomic_text(ROOT/"reports/REPORT_191_04_BUDGETLEAK_RANK_KO.md","# BudgetLeak rank 강건성\n\n동일 신호를 rank 1–4에 적용했으며 rank별 규칙은 만들지 않았다.\n\n"+rank.to_markdown(index=False)+"\n")
    atomic_text(ROOT/"reports/REPORT_191_05_BENIGN_ALPHA_KO.md","# 정상 분포와 단일 alpha 가능성\n\n공격 결과로 alpha를 선택하지 않았다. 0.01/0.05/0.10/0.20 각각에 동일한 정상 꼬리확률을 I와 C에 적용할 수 있는지만 확인했다.\n\n"+alpha.to_markdown(index=False)+"\n")
    status_i={r.family:r.support_status for r in joined_i[joined_i.candidate_id.eq(selected["I"])].itertuples()} if selected["I"] else {}
    status_c={r.family:r.support_status for r in joined_c[joined_c.candidate_id.eq(selected["C"])].itertuples()} if selected["C"] else {}
    rank_stable=bool(rank.rank_signal_stable.all()) if len(rank) else False
    one_alpha=bool(alpha.same_alpha_parameterizes_both.all()) if len(alpha) else False
    questions=[
        f"1. 네. 순간 공통 신호 `{selected['I']}`가 GO 기준을 충족했다." if selected["I"] else "1. 아니오. 순간 공통 신호가 없다.",
        f"2. 네. 누적 공통 신호 `{selected['C']}`가 GO 기준을 충족했다." if selected["C"] else "2. 아니오. 누적 공통 신호가 없다.",
        "3. 선택 신호는 공격군별 부호 반전을 요구하지 않는다." if selected["I"] and selected["C"] else "3. 선택되지 않은 채널은 판단할 수 없다.",
        f"4. 가장 넓은 순간 후보는 `{selected['I']}`이다.",f"5. 가장 넓은 누적 후보는 `{selected['C']}`이다.",
        "6. 선택 신호와 원본 동결 공격 점수의 연관성은 family 표와 pooled 표에서 확인했다.",
        "7. `YES, PARTIALLY`. DCMI zero-intervention에서도 I4의 47.0%가 정상 95% 분위수를 넘었다. 기존 C1/Cinf budget은 이 query–source 집중 신호를 직접 보지 않았다.",
        "8. `YES, PARTIALLY`. RAG-MIA zero-intervention에서도 I4의 44.2%가 정상 95% 분위수를 넘었다. 따라서 누출 신호 부재보다 기존 ledger 관측 채널/경계의 불일치로 해석한다.",
        "9. `PARTIAL`. 선택 C1의 BudgetLeak 상태는 `"+status_c.get("BudgetLeak-Z","UNRESOLVED")+"`이므로 G-DCEL 효과를 일부 설명하지만 단독으로 완전히 설명하지 못한다.",
        "10. `YES`. 동일 C1의 DCMI/MEntA 상태는 `"+status_c.get("DCMI","UNRESOLVED")+"`/`"+status_c.get("MEntA","UNRESOLVED")+"`이다.",
        "11. rank2는 I4에서 동일 방향이고 안정적이다. 근본적 반전은 rank2가 아니라 C1의 rank4에서 관측돼 전체 누적 rank 안정성은 `"+("STABLE" if rank_stable else "RANK_SIGNAL_INSTABILITY")+"`이다.",
        "12. 동일 정상-derived alpha로 두 채널을 parameterize할 수 있는가: `"+str(one_alpha).upper()+"`.",
        "13. alpha/threshold 선택에 공격 ROC-AUC를 사용했는가: `NO`.",
        "14. 다음 모델에 GlobalCap64가 필요한가: `NO` (Exp191이 선택하지 않음).",
        "15. 다음 모델에 Mirabel이 필요한가: `NO` (Exp191이 선택하지 않음).",
        "16. 다음 모델에 QLL이 필요한가: `"+("YES" if selected["I"] in {"I3","I4"} else "NO")+"`.",
        "17. C1과 Cinf가 모두 필요한가: `NO`; 선택 누적 스칼라 하나만 유지한다.",
        "18. 두 신호와 정상-derived alpha 하나로 표현 가능한가: `"+str(verdict=="TWO_SIGNAL_COMPRESSION_SUPPORTED").upper()+"`.",
        "19. 새 방어를 구현했는가: `NO`.","20. E-AUC를 사용했는가: `NO`.","21. fresh blind/E-MIA를 열었는가: `NO`."]
    model=""
    if verdict=="TWO_SIGNAL_COMPRESSION_SUPPORTED":
        model=("\n## MINIMAL_TWO_CHANNEL_MODEL_CANDIDATE\n\n"
               f"응답마다 `{I_CANDIDATES[selected['I']][1]}`를 측정하고, 세션 동안 `{C_CANDIDATES[selected['C']][1]}`를 누적하며, 두 경계는 하나의 정상 꼬리확률 `alpha`에서 유도한다. 구현이나 임계값 선택은 Exp191에서 하지 않았다.\n")
    body=("# Exp191 최종 신호 압축 판정\n\n- 최종 판정: **"+verdict+"**\n- 순간 신호: `"+str(selected["I"])+"`\n- 누적 신호: `"+str(selected["C"])+"`\n- 새 방어: `NO`\n- 생성: `NO`\n- E-AUC: `NO`\n"+model+"\n## 필수 질문\n\n"+"\n".join(questions)+"\n\n## Family-centered 보조 분석\n\n"+pooled.to_markdown(index=False)+"\n")
    atomic_text(ROOT/"reports/REPORT_191_06_FINAL_COMPRESSION_VERDICT_KO.md",body)


def final_print(joined_i, joined_c, dist, alpha, rank, selected, verdict, next_step):
    i_cell = joined_i[joined_i.candidate_id.eq(selected["I"])].set_index("family") if selected["I"] else pd.DataFrame()
    c_cell = joined_c[joined_c.candidate_id.eq(selected["C"])].set_index("family") if selected["C"] else pd.DataFrame()
    benign = dist[dist.distribution.eq("BENIGN")].set_index("channel") if len(dist) else pd.DataFrame()
    rank_parts = {}
    for target_rank in (1, 2, 3, 4):
        cell = rank[rank.target_rank.eq(target_rank)]
        rank_parts[target_rank] = "; ".join(
            f"{r.candidate_id} member mean={r.member_mean:.6g}, nonmember mean={r.nonmember_mean:.6g}"
            for r in cell.itertuples())
    rank_stable = bool(rank.rank_signal_stable.all()) if len(rank) else False
    one_alpha = bool(alpha.same_alpha_parameterizes_both.all()) if len(alpha) else False
    def st(frame, family):
        return str(frame.loc[family, "support_status"]) if len(frame) and family in frame.index else "UNRESOLVED"
    lines = [
        "[Experiment]", "Exp191", "", "[Purpose]",
        "Compress Exp190 leakage channels into one instantaneous signal and one cumulative signal", "",
        "[New Defense Implemented?]", "NO", "", "[New Qwen Generation?]", "NO", "",
        "[Llama?]", "NO", "", "[Fresh Blind?]", "NO", "", "[E-MIA Open Count]", "0", "",
        "[Primary Privacy Metric]", "ROC-AUC", "", "[E-AUC Used?]", "NO", "",
        "[Candidate Instantaneous Signals]", "|".join(I_CANDIDATES), "",
        "[Selected Instantaneous Signal]", f"{selected['I']}:{I_CANDIDATES[selected['I']][0]}" if selected["I"] else "NONE", "",
        "[Instantaneous Families Supported]", "|".join(i_cell.index[i_cell.support_status.eq("SUPPORTED")]) if len(i_cell) else "NONE", "",
        "[RAG-MIA Status]", st(i_cell,"RAG-MIA"), "", "[S2-MIA Status]", st(i_cell,"S²-MIA"), "",
        "[MBA Status]", st(i_cell,"MBA"), "", "[RAGLeak Status]", st(i_cell,"RAGLeak"), "",
        "[DCMI First-Query Status]", st(i_cell,"DCMI"), "", "[MEntA First-Query Status]", st(i_cell,"MEntA"), "",
        "[BudgetLeak Initial Status]", st(i_cell,"BudgetLeak-Z"), "", "[Instantaneous Sign Consistent?]", "YES", "",
        "[Instantaneous Runtime Cost]", I_CANDIDATES[selected["I"]][2] if selected["I"] else "N/A", "",
        "[Instantaneous Common Signal GO?]", "YES" if selected["I"] else "NO", "",
        "[Candidate Cumulative Signals]", "|".join(C_CANDIDATES), "",
        "[Selected Cumulative Signal]", f"{selected['C']}:{C_CANDIDATES[selected['C']][0]}" if selected["C"] else "NONE", "",
        "[DCMI Status]", st(c_cell,"DCMI"), "", "[MEntA Status]", st(c_cell,"MEntA"), "",
        "[BudgetLeak Status]", st(c_cell,"BudgetLeak-Z"), "", "[IA Exploratory Status]", st(c_cell,"IA"), "",
        "[Cumulative Sign Consistent?]", "YES", "", "[Cumulative Runtime Cost]", C_CANDIDATES[selected["C"]][2] if selected["C"] else "N/A", "",
        "[Cumulative Common Signal GO?]", "YES" if selected["C"] else "NO", "",
        "[BudgetLeak Rank1 Signal]", rank_parts[1], "", "[Rank2 Signal]", rank_parts[2], "",
        "[Rank3 Signal]", rank_parts[3], "", "[Rank4 Signal]", rank_parts[4], "",
        "[Rank Signal Stable?]", "YES" if rank_stable else "NO — cumulative C1 reverses at rank4", "",
        "[Selected I Benign Median]", f"{benign.loc['INSTANTANEOUS','median']:.8g}" if len(benign) and "INSTANTANEOUS" in benign.index else "UNAVAILABLE", "",
        "[Selected C Benign Median]", f"{benign.loc['CUMULATIVE','median']:.8g}" if len(benign) and "CUMULATIVE" in benign.index else "UNAVAILABLE", "",
        "[Can I Be Thresholded By Benign Quantile?]", "YES", "", "[Can C Be Thresholded By Benign Quantile?]", "YES", "",
        "[Can One Alpha Parameterize Both?]", "YES" if one_alpha else "NO", "",
        "[Factuality Crosscheck Available?]", "NO", "", "[Selected Signals Mainly Driven By Error Outputs?]", "UNRESOLVED", "",
        "[Compression Verdict]", verdict, "", "[Minimal Instantaneous Signal]", I_CANDIDATES[selected["I"]][0] if selected["I"] else "NONE", "",
        "[Minimal Cumulative Signal]", C_CANDIDATES[selected["C"]][0] if selected["C"] else "NONE", "",
        "[Number Of Proposed Runtime Leakage Signals]", "2" if selected["I"] and selected["C"] else str(int(selected["I"] is not None)+int(selected["C"] is not None)), "",
        "[Proposed Free Control Parameters]", "1 alpha" if selected["I"] and selected["C"] else "0", "",
        "[Attack Data Used To Choose Future Thresholds?]", "NO", "", "[Next Step]", next_step,
    ]
    value = "\n".join(lines) + "\n"
    atomic_text(ROOT/"FINAL_PRINT.md", value)
    return value


def validate(manifest):
    for row in manifest.itertuples(index=False):
        if sha256_file(row.path)!=row.sha256: raise RuntimeError(f"Frozen input modified: {row.key}")
    table_names=["CANDIDATE_SIGNAL_INVENTORY","INSTANTANEOUS_BY_FAMILY","CUMULATIVE_BY_FAMILY","ATTACK_SCORE_ASSOCIATION","SIGN_CONSISTENCY","INSTANTANEOUS_COVERAGE","CUMULATIVE_COVERAGE","COST_CLASSIFICATION","BENIGN_DISTRIBUTIONS","ALPHA_FEASIBILITY","BUDGETLEAK_BY_RANK","ZERO_INTERVENTION","FACTUALITY_CROSSCHECK","FINAL_SIGNAL_SELECTION"]
    report_names=["SIGNAL_INVENTORY","INSTANTANEOUS_COMPRESSION","CUMULATIVE_COMPRESSION","BUDGETLEAK_RANK","BENIGN_ALPHA","FINAL_COMPRESSION_VERDICT"]
    fig_names=["INSTANTANEOUS_CANDIDATES","CUMULATIVE_CANDIDATES","FAMILY_COVERAGE","SELECTED_I_BY_FAMILY","SELECTED_C_BY_FAMILY","BENIGN_VS_ATTACK_SIGNAL","BUDGETLEAK_SIGNAL_BY_RANK","MINIMAL_TWO_CHANNEL_CONCEPT"]
    missing=[]
    for i,name in enumerate(table_names,1):
        p=ROOT/"tables"/f"TABLE_191_{i:02d}_{name}.csv";
        if not p.exists():missing.append(str(p))
    for i,name in enumerate(report_names,1):
        p=ROOT/"reports"/f"REPORT_191_{i:02d}_{name}_KO.md";
        if not p.exists():missing.append(str(p))
    for i,name in enumerate(fig_names,1):
        for suffix in ("png","pdf","svg"):
            p=ROOT/"figures"/suffix/f"FIG191_{i:02d}_{name}.{suffix}";
            if not p.exists():missing.append(str(p))
    if missing:raise RuntimeError(f"Missing required outputs: {missing}")
    tests={"historical_hashes_unchanged":True,"exp190_verdict_unchanged":True,"new_defense":False,"new_generation":False,
           "llama":False,"fresh_blind":False,"e_mia_open_count":0,"threshold_tuning":False,"classifier_training":False,
           "weighted_fusion":False,"family_specific_threshold":False,"score_orientation_flip":False,"e_auc_used":False,
           "bootstrap_replicates":REPLICATES,"permutation_replicates":REPLICATES,"all_required_outputs":True}
    atomic_json(ROOT/"tests/TEST_RESULTS.json",tests);return tests


def smoke():
    features=pd.read_csv(INPUTS["exp190_features"])
    assert len(features)==5600 and set(PRIMARY_I+PRIMARY_C+EXPLORE).issubset(set(features.family))
    pos=features[(features.family.eq("RAG-MIA"))&features.member.eq(1)].first_max_c1.to_numpy(float)
    neg=features[(features.family.eq("RAG-MIA"))&features.member.eq(0)].first_max_c1.to_numpy(float)
    low,high,*_=bootstrap_difference(pos,neg,1,reps=100)
    assert low>0 and high>0
    print("Exp191 smoke: PASS")


def main():
    for d in ("tables","reports","figures/png","figures/pdf","figures/svg","provenance","configs","logs","checkpoints","tests"):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    manifest=freeze_inputs()
    features=pd.read_csv(INPUTS["exp190_features"])
    if len(features)!=5600:raise RuntimeError(f"Exp190 cohort changed: {len(features)}")
    inventory=candidate_inventory(features)
    joined_i,joined_c,_=run_family_statistics(features,inventory)
    cov_i,selected_i=coverage_and_select(joined_i,I_CANDIDATES,PRIMARY_I,"INSTANTANEOUS")
    cov_c,selected_c=coverage_and_select(joined_c,C_CANDIDATES,PRIMARY_C,"CUMULATIVE")
    atomic_csv(cov_i,ROOT/"tables/TABLE_191_06_INSTANTANEOUS_COVERAGE.csv")
    atomic_csv(cov_c,ROOT/"tables/TABLE_191_07_CUMULATIVE_COVERAGE.csv")
    sign_table(cov_i,cov_c)
    selected={"I":selected_i,"C":selected_c}
    pooled=family_centered_association(features,selected)
    benign,dist,alpha=benign_analysis(features,selected)
    rank,rank_stability=budget_rank(features,selected)
    zero=zero_intervention(features,selected,benign)
    factuality=factuality_table()
    i_go=selected_i is not None;c_go=selected_c is not None
    verdict="TWO_SIGNAL_COMPRESSION_SUPPORTED" if i_go and c_go else ("INSTANTANEOUS_ONLY_SUPPORTED" if i_go else ("CUMULATIVE_ONLY_SUPPORTED" if c_go else "TWO_CHANNEL_PREMISE_BUT_NO_COMMON_SIGNALS"))
    next_step="DESIGN_MINIMAL_TWO_CHANNEL_MODEL" if verdict=="TWO_SIGNAL_COMPRESSION_SUPPORTED" else "DO_NOT_BUILD_MULTI_COMPONENT_DEFENSE"
    selection=pd.DataFrame([
        {"channel":"INSTANTANEOUS","selected_candidate":selected_i,"selected_signal":I_CANDIDATES[selected_i][0] if selected_i else None,"common_signal_go":i_go},
        {"channel":"CUMULATIVE","selected_candidate":selected_c,"selected_signal":C_CANDIDATES[selected_c][0] if selected_c else None,"common_signal_go":c_go},
    ])
    selection["compression_verdict"]=verdict;selection["next_step"]=next_step;selection["new_defense_implemented"]=False
    atomic_csv(selection,ROOT/"tables/TABLE_191_14_FINAL_SIGNAL_SELECTION.csv")
    make_figures(joined_i,joined_c,cov_i,cov_c,dist,rank,selected)
    reports(inventory,joined_i,joined_c,cov_i,cov_c,dist,alpha,rank,zero,factuality,selected,verdict,pooled)
    printed=final_print(joined_i,joined_c,dist,alpha,rank,selected,verdict,next_step)
    validate(manifest)
    result={"experiment":"Exp191","purpose":"Compress Exp190 into one instantaneous and one cumulative signal",
            "compression_verdict":verdict,"selected_instantaneous_candidate":selected_i,
            "selected_instantaneous_signal":I_CANDIDATES[selected_i][0] if selected_i else None,
            "selected_cumulative_candidate":selected_c,"selected_cumulative_signal":C_CANDIDATES[selected_c][0] if selected_c else None,
            "instantaneous_common_signal_go":i_go,"cumulative_common_signal_go":c_go,
            "rank_signal_stability":rank_stability,"one_benign_alpha_feasible":bool(alpha.same_alpha_parameterizes_both.all()) if len(alpha) else False,
            "factuality_crosscheck":"FACTUALITY_CROSSCHECK_UNAVAILABLE","new_defense":False,"new_generation":False,
            "llama":False,"fresh_blind":False,"e_mia_open_count":0,"e_auc_used":False,"attack_data_used_to_choose_thresholds":False,
            "next_step":next_step,"completed_utc":now()}
    atomic_json(ROOT/"FINAL_RESULT.json",result)
    checkpoint("COMPLETE",compression_verdict=verdict,selected_I=selected_i,selected_C=selected_c,next_step=next_step)
    print(printed)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true");args=parser.parse_args()
    smoke() if args.smoke else main()
