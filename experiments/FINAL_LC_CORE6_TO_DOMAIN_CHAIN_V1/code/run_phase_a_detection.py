#!/usr/bin/env python3
"""Frozen Core6 same-benign-FPR detection comparison with clustered uncertainty."""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import (BGE, BUDGETS, CORE6, EXP, FINAL8, K_LOCAL, LC, ROOT, SEED,
                    atomic_json, atomic_text, checkpoint, now, read_jsonl, sha_file,
                    sha_text, verify_hashed_json, write_csv, write_jsonl)

sys.path.insert(0, str(ROOT / "code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments, MIRABEL_FORMULA_VERSION


def threshold_at_budget(values: list[float], alpha: float) -> tuple[float, int]:
    """Most permissive observed threshold whose strict-'>' count is <= floor(alpha*N)."""
    ordered = sorted(map(float, values), reverse=True)
    allowance = math.floor(alpha * len(ordered) + 1e-12)
    threshold = ordered[allowance] if allowance < len(ordered) else -math.inf
    alarms = sum(value > threshold for value in ordered)
    if alarms > allowance:
        raise RuntimeError("threshold construction exceeded benign budget")
    return threshold, alarms


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def bootstrap_delta(rows: list[dict], m_threshold: float, lc_threshold: float, seed: int) -> tuple[float, float, float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["session_id"]].append(row)
    units = sorted(grouped)
    m_hits = np.asarray([sum(float(row["M"]) > m_threshold for row in grouped[key]) for key in units], dtype=float)
    lc_hits = np.asarray([sum(float(row["R_LC"]) > lc_threshold for row in grouped[key]) for key in units], dtype=float)
    sizes = np.asarray([len(grouped[key]) for key in units], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(2000, dtype=float)
    for start in range(0, 2000, 100):
        index = rng.integers(0, len(units), size=(min(100, 2000-start), len(units)))
        denominator = sizes[index].sum(axis=1)
        deltas[start:start+len(index)] = (lc_hits[index].sum(axis=1)-m_hits[index].sum(axis=1))/denominator
    return float(deltas.mean()), quantile(deltas.tolist(), .025), quantile(deltas.tolist(), .975)


def score_new_queries(query_rows: list[dict]) -> tuple[list[dict], dict]:
    prior = read_jsonl(LC / "cache" / "LARGE_DETECTION_SCORES.jsonl")
    prior_embeddings = np.asarray(np.load(LC / "cache" / "QUERY_EMBEDDINGS.float16.npy"), dtype=np.float32)
    if len(prior) != prior_embeddings.shape[0]:
        raise RuntimeError("LC row/embedding alignment drift")
    reference_indices = [index for index, row in enumerate(prior) if row["split"] == "REFERENCE"]
    if len(reference_indices) != 1000:
        raise RuntimeError("expected 1000 frozen benign reference rows")
    ref_embeddings = prior_embeddings[reference_indices]
    ref_margins = [float(prior[index]["M"]) for index in reference_indices]
    documents = read_jsonl(__import__("common").CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")
    document_embeddings = np.asarray(np.load(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy"), dtype=np.float32)
    if len(documents) != document_embeddings.shape[0]:
        raise RuntimeError("document embedding alignment drift")
    doc_ids = [row["document_id"] for row in documents]
    checkpoint("PHASE_A_BGE_LOADING", new_queries=len(query_rows), gpu_free_bytes=torch.cuda.mem_get_info()[0])
    started = time.monotonic()
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    query_embeddings = np.asarray(model.encode([row["query"] for row in query_rows], batch_size=48,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    del model
    torch.cuda.empty_cache()
    np.save(EXP / "cache" / "STANDARDIZED_QUERY_EMBEDDINGS.float16.npy", query_embeddings.astype(np.float16))
    global_sorted = sorted(ref_margins)
    output = []
    retrieval_started = time.monotonic()
    for start in range(0, len(query_rows), 128):
        query_batch = query_embeddings[start:start+128]
        score_batch = query_batch @ document_embeddings.T
        similarity_batch = query_batch @ ref_embeddings.T
        for offset, scores in enumerate(score_batch):
            source = query_rows[start+offset]
            top_idx = np.argpartition(-scores, 4)[:4]
            top_idx = top_idx[np.argsort(-scores[top_idx], kind="stable")]
            top_ids = [doc_ids[int(index)] for index in top_idx]
            top_scores = [float(scores[int(index)]) for index in top_idx]
            stat = canonical_mirabel_from_moments(top1=top_scores[0], sum_all=float(scores.sum(dtype=np.float64)),
                sumsq_all=float(np.square(scores, dtype=np.float64).sum(dtype=np.float64)),
                corpus_size=len(doc_ids), confidence=.95)
            similarities = similarity_batch[offset]
            neighbors = np.argpartition(-similarities, K_LOCAL)[:K_LOCAL]
            neighbors = neighbors[np.argsort(-similarities[neighbors], kind="stable")]
            local_sorted = sorted(ref_margins[int(index)] for index in neighbors)
            p_local = (1 + len(local_sorted) - np.searchsorted(local_sorted, float(stat.margin), side="left"))/(len(local_sorted)+1)
            p_global = (1 + len(global_sorted) - np.searchsorted(global_sorted, float(stat.margin), side="left"))/(len(global_sorted)+1)
            target_rank = top_ids.index(source["target_id"])+1 if source["target_id"] in top_ids else 0
            output.append({**source, "top_document_ids": top_ids, "top_scores": top_scores,
                           "selected_source_id": top_ids[0], "target_rank": target_rank,
                           "target_retrieved_at1": target_rank == 1, "target_retrieved_at4": target_rank > 0,
                           "M": float(stat.margin), "mirabel_threshold": float(stat.threshold),
                           "mirabel_background_mean": float(stat.background_mean),
                           "mirabel_background_std": float(stat.background_std),
                           "mirabel_formula_version": MIRABEL_FORMULA_VERSION,
                           "p_local": float(p_local), "R_LC": float(-math.log(p_local)),
                           "p_global": float(p_global), "R_GLOBAL": float(-math.log(p_global)),
                           "local_neighbor_ids_sha256": sha_text("\n".join(prior[reference_indices[int(i)]]["query_id"] for i in neighbors)),
                           "local_neighbor_similarity_max": float(similarities[neighbors[0]]),
                           "local_neighbor_similarity_min": float(similarities[neighbors[-1]])})
        checkpoint("PHASE_A_STANDARDIZED_SCORING_PROGRESS", completed=min(start+128, len(query_rows)),
                   total=len(query_rows), percent=round(100*min(start+128, len(query_rows))/len(query_rows), 2))
    return output, {"new_queries": len(query_rows), "encoding_and_scoring_seconds": time.monotonic()-started,
                    "retrieval_seconds": time.monotonic()-retrieval_started,
                    "embedding_sha256": sha_file(EXP / "cache" / "STANDARDIZED_QUERY_EMBEDDINGS.float16.npy")}


def evaluate(new_rows: list[dict], runtime: dict) -> dict:
    old_rows = read_jsonl(FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl")
    benign = [row for row in old_rows if row["attack"] == "BENIGN"]
    if len(benign) != 1000:
        raise RuntimeError("benign holdout count drift")
    old_attacks = [row for row in old_rows if row["attack"] in {"MEntA", "MBA", "RAG-MIA", "S²-MIA"}]
    attacks = old_attacks + new_rows
    thresholds = {}
    alarm_rows = []
    for budget in BUDGETS:
        for method, key in (("MIRABEL", "M"), ("Final LC", "R_LC")):
            threshold, count = threshold_at_budget([row[key] for row in benign], budget)
            thresholds[(method, budget)] = threshold
            alarm_rows.append({"method": method, "nominal_budget": budget, "benign_n": len(benign),
                               "threshold": threshold, "strict_operator": ">", "benign_alarms": count,
                               "actual_benign_fpr": count/len(benign)})
    write_csv(EXP / "tables" / "BENIGN_THRESHOLDS_AND_ALARMS.csv", alarm_rows)

    table, boots = [], []
    primary = {}
    attack_member_rows = {}
    for attack in CORE6:
        subset = [row for row in attacks if row["attack"] == attack and row.get("membership") == "member"]
        if attack == "S²-MIA":
            subset = [row for row in subset if row.get("evaluation_split") == "S2_EVALUATION"]
        expected = {"MEntA": 5000, "MBA": 1000, "RAG-MIA": 1000, "S²-MIA": 799,
                    "DCMI-Std-Q2": 2000, "IA-Std-Q15": 15000}[attack]
        if len(subset) != expected:
            raise RuntimeError(f"{attack} member query count {len(subset)}/{expected}")
        attack_member_rows[attack] = subset
        row = {"attack": attack, "positive_unit": "member-target attack query", "member_queries": len(subset),
               "member_sessions": len({value["session_id"] for value in subset}), "benign_queries": len(benign)}
        for budget in BUDGETS:
            label = "2_5" if budget == .025 else str(int(100*budget))
            m = statistics.fmean(float(value["M"]) > thresholds[("MIRABEL", budget)] for value in subset)
            lc = statistics.fmean(float(value["R_LC"]) > thresholds[("Final LC", budget)] for value in subset)
            row[f"mirabel_tpr_at_{label}pct"] = m
            row[f"final_lc_tpr_at_{label}pct"] = lc
            if budget == .025:
                mean, low, high = bootstrap_delta(subset, thresholds[("MIRABEL", budget)], thresholds[("Final LC", budget)], SEED+CORE6.index(attack))
                row.update({"delta_at_2_5pct": lc-m, "bootstrap_mean_delta": mean,
                            "delta_ci95_low": low, "delta_ci95_high": high})
                primary[attack] = {"MIRABEL": m, "Final LC": lc, "delta": lc-m, "ci95": [low, high]}
                boots.append({"attack": attack, "iterations": 2000, "cluster": "target/session",
                              "point_delta": lc-m, "mean_delta": mean, "ci95_low": low, "ci95_high": high})
        table.append(row)
    write_csv(EXP / "tables" / "CORE6_MATCHED_FPR_DETECTION.csv", table)
    write_csv(EXP / "tables" / "CORE6_CLUSTER_BOOTSTRAP.csv", boots)

    session_rows = []
    for attack in ("MEntA", "DCMI-Std-Q2", "IA-Std-Q15"):
        relevant = [row for row in attacks if row["attack"] == attack]
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in relevant:
            groups[(row["membership"], row["session_id"])].append(row)
        for (membership, session_id), group in sorted(groups.items()):
            group.sort(key=lambda item: int(item["query_index"]))
            for method, key in (("MIRABEL", "M"), ("Final LC", "R_LC")):
                flags = [float(row[key]) > thresholds[(method, .025)] for row in group]
                removed = [flag and row["selected_source_id"] == row["target_id"] for flag, row in zip(flags, group)]
                first = next((index+1 for index, flag in enumerate(flags) if flag), None)
                session_rows.append({"attack": attack, "membership": membership, "session_id": session_id,
                                     "method": method, "queries": len(group), "alarms": sum(flags),
                                     "protected_queries": sum(flags), "any_alarm": any(flags), "all_alarm": all(flags),
                                     "first_alarm_index": first, "target_removals": sum(removed),
                                     "any_target_removal": any(removed), "all_target_removal": all(removed)})
    write_csv(EXP / "tables" / "MULTIQUERY_SESSION_DETECTION.csv", session_rows)

    mean_delta = statistics.fmean(item["delta"] for item in primary.values())
    nondegrading = sum(item["delta"] >= -1e-12 for item in primary.values())
    worst_delta = min(item["delta"] for item in primary.values())
    checks = {
        "core6_mean_delta_positive": mean_delta > 0,
        "nondegrading_at_least_5_of_6": nondegrading >= 5,
        "no_family_drop_below_minus_3pp": worst_delta >= -.03-1e-12,
        "dcmi_no_catastrophic_failure": primary["DCMI-Std-Q2"]["Final LC"] >= .05,
        "ia_no_catastrophic_failure": primary["IA-Std-Q15"]["Final LC"] >= .05,
    }
    passed = all(checks.values())
    verdict = "CORE6_DETECTION_PASS" if passed else "CORE6_DETECTION_FAILED"
    result = {"campaign": EXP.name, "phase": "A_CORE6_DETECTION", "verdict": verdict,
              "completed_utc": now(), "primary_budget": .025, "primary": primary,
              "macro": {"mirabel_tpr": statistics.fmean(item["MIRABEL"] for item in primary.values()),
                         "final_lc_tpr": statistics.fmean(item["Final LC"] for item in primary.values()),
                         "mean_delta": mean_delta, "nondegrading_families": nondegrading, "worst_delta": worst_delta},
              "checks": checks, "benign_alarm_rows": alarm_rows, "runtime": runtime,
              "next_stage": "PHASE_B_CORE6_MATCHED_BUDGET_E2E" if passed else "STOP_NO_PHASE_B_GOLD_OR_DOMAIN"}
    atomic_json(EXP / "PHASE_A_RESULT.json", result)
    lines = ["# Final LC Core6 Detection", "", f"- Verdict: `{verdict}`",
             "- Primary positive: member-target attack query", "- Negative: frozen benign holdout 1,000 queries",
             "- Primary: strict threshold under 2.5% benign FPR budget", "",
             "| Attack | MIRABEL TPR | Final LC TPR | Delta | Cluster bootstrap 95% CI |",
             "|---|---:|---:|---:|---:|"]
    for attack in CORE6:
        value = primary[attack]
        lines.append(f"| {attack} | {value['MIRABEL']:.4f} | {value['Final LC']:.4f} | {value['delta']:+.4f} | [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] |")
    lines.extend(["", f"- Mean delta: `{mean_delta:+.4f}`", f"- Non-degrading families: `{nondegrading}/6`",
                  f"- Worst delta: `{worst_delta:+.4f}`", "", "## Gate",
                  *[f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in checks.items()]])
    atomic_text(EXP / "reports" / "PHASE_A_CORE6_DETECTION_KO.md", "\n".join(lines) + "\n")
    checkpoint(verdict, phase_a_pass=passed, mean_delta=round(mean_delta, 6), nondegrading=f"{nondegrading}/6",
               worst_delta=round(worst_delta, 6), next_stage=result["next_stage"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    pre = verify_hashed_json(EXP / "configs" / "CORE6_DETECTION_PRECOMMIT.json")
    for relative, expected in pre["code_sha256"].items():
        if sha_file(ROOT / relative) != expected:
            raise RuntimeError(f"Phase-A code drift: {relative}")
    bundle = verify_hashed_json(EXP / "configs" / "STANDARDIZED_QUERY_BUNDLE_MANIFEST.json")
    for key in ("dcmi", "ia", "provenance"):
        if sha_file(Path(bundle[key]["path"])) != bundle[key]["sha256"]:
            raise RuntimeError(f"standardized query bundle drift: {key}")
    new_queries = read_jsonl(Path(bundle["dcmi"]["path"])) + read_jsonl(Path(bundle["ia"]["path"]))
    if len(new_queries) != 34000 or len({row["query_id"] for row in new_queries}) != 34000:
        raise RuntimeError("standardized query count/ID drift")
    rows, runtime = score_new_queries(new_queries)
    write_jsonl(EXP / "cache" / "STANDARDIZED_RETRIEVAL_AND_DETECTION.jsonl", rows)
    runtime["detection_cache_sha256"] = sha_file(EXP / "cache" / "STANDARDIZED_RETRIEVAL_AND_DETECTION.jsonl")
    atomic_json(EXP / "cache" / "STANDARDIZED_DETECTION_MANIFEST.json", runtime)
    evaluate(rows, runtime)


if __name__ == "__main__":
    main()

