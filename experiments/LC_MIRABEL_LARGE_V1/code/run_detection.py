#!/usr/bin/env python3
"""Large score-only evaluation of frozen Raw/Global/Local calibrated MIRABEL."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import (ALPHAS, ATTACKS, BGE, EXP, K_LOCAL, PARENT, SEED, atomic_json,
                    atomic_text, checkpoint, empirical_upper_tail, matched_threshold,
                    now, percentile, read_jsonl, sha_file, tpr_at_fpr, wilson,
                    write_csv, write_jsonl)

sys.path.insert(0, str(PARENT.parents[1] / "code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments, MIRABEL_FORMULA_VERSION  # noqa: E402


PRECOMMIT = EXP / "configs/LC_MIRABEL_LARGE_V1_PRECOMMIT.json"


def verify() -> dict:
    expected = PRECOMMIT.with_suffix(".sha256").read_text().split()[0]
    if sha_file(PRECOMMIT) != expected:
        raise RuntimeError("precommit checksum mismatch")
    pre = json.loads(PRECOMMIT.read_text())
    if pre["lineage"]["code_sha256"].get(Path(__file__).name) != sha_file(Path(__file__)):
        raise RuntimeError("detection code changed after precommit")
    manifest = EXP / "manifests/LARGE_QUERY_MANIFEST.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("verdict") != "LARGE_QUERY_MANIFEST_PASS":
        raise RuntimeError("frozen query manifest unavailable")
    for key in ("targets", "benign_reference", "benign_holdout"):
        item = pre["substrate"][key]
        if sha_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"substrate drift: {key}")
    return pre


def load_queries() -> tuple[list[dict], list[dict], list[dict]]:
    reference = [{**row, "cohort": "BENIGN", "split": "REFERENCE", "attack": None,
                  "membership": None, "target_id": None, "session_id": row["query_id"], "query_index": 1}
                 for row in read_jsonl(EXP / "inputs/BENIGN_REFERENCE.jsonl")]
    holdout = [{**row, "cohort": "BENIGN", "split": "HOLDOUT", "attack": None,
                "membership": None, "target_id": None, "session_id": row["query_id"], "query_index": 1}
               for row in read_jsonl(EXP / "inputs/BENIGN_DEPLOYMENT_HOLDOUT.jsonl")]
    attacks = []
    for attack, filename in (("MEntA", "LARGE_MENTA_ATTACK_QUERIES.jsonl"),
                             ("MBA", "LARGE_MBA_ATTACK_QUERIES.jsonl"),
                             ("RAG-MIA", "LARGE_RAG_MIA_ATTACK_QUERIES.jsonl")):
        attacks.extend({**row, "cohort": "ATTACK", "split": "LARGE_FRESH_DEVELOPMENT", "attack": attack}
                       for row in read_jsonl(EXP / "inputs" / filename))
    if len(reference) != 1000 or len(holdout) != 1000 or len(attacks) != 14000:
        raise RuntimeError(f"query count mismatch ref={len(reference)} hold={len(holdout)} attack={len(attacks)}")
    all_rows = reference + holdout + attacks
    if len({row["query_id"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("query IDs are not unique")
    return reference, holdout, attacks


def encode_and_score(pre: dict, query_rows: list[dict]) -> tuple[list[dict], dict]:
    docs = read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")
    if len(docs) != 3000 or sha_file(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl") != pre["substrate"]["protected_db"]["sha256"]:
        raise RuntimeError("protected DB drift")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for frozen BGE-M3")
    checkpoint("BGE_LARGE_LOADING", queries=len(query_rows), documents=len(docs))
    started = time.monotonic()
    model = SentenceTransformer(str(BGE), device="cuda")
    model.max_seq_length = 512
    parent_manifest = json.loads((PARENT / "cache/CLEAN_CORE3_RETRIEVAL_CACHE_MANIFEST.json").read_text())
    parent_doc_cache = PARENT / "cache/CORPUS_EMBEDDINGS.float16.npy"
    if (parent_manifest["corpus_sha256"] == pre["substrate"]["protected_db"]["sha256"]
            and parent_doc_cache.is_file()):
        document_embeddings = np.asarray(np.load(parent_doc_cache), dtype=np.float32)
        document_cache_mode = "VERIFIED_PARENT_CACHE_REUSE"
    else:
        document_embeddings = np.asarray(model.encode([row["source_text"] for row in docs], batch_size=16,
            show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
        document_cache_mode = "FRESH_ENCODING"
    query_started = time.monotonic()
    query_embeddings = np.asarray(model.encode([row["query"] for row in query_rows], batch_size=32,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    query_seconds = time.monotonic() - query_started
    del model
    torch.cuda.empty_cache()
    np.save(EXP / "cache/QUERY_EMBEDDINGS.float16.npy", query_embeddings.astype(np.float16))
    np.save(EXP / "cache/CORPUS_EMBEDDINGS.float16.npy", document_embeddings.astype(np.float16))

    doc_ids = [row["document_id"] for row in docs]
    output = []
    retrieval_started = time.monotonic()
    for start in range(0, len(query_rows), 128):
        scores_batch = query_embeddings[start:start+128] @ document_embeddings.T
        for offset, scores in enumerate(scores_batch):
            source = query_rows[start+offset]
            top_index = np.argpartition(-scores, 4)[:4]
            top_index = top_index[np.argsort(-scores[top_index], kind="stable")]
            top_ids = [doc_ids[int(index)] for index in top_index]
            top_scores = [float(scores[int(index)]) for index in top_index]
            stats = canonical_mirabel_from_moments(
                top1=top_scores[0], sum_all=float(scores.sum(dtype=np.float64)),
                sumsq_all=float(np.square(scores, dtype=np.float64).sum(dtype=np.float64)),
                corpus_size=len(doc_ids), confidence=0.95)
            target = source.get("target_id")
            target_rank = top_ids.index(target)+1 if target in top_ids else 0
            output.append({**source, "top_document_ids": top_ids, "top_scores": top_scores,
                           "selected_source_id": top_ids[0], "target_rank": target_rank,
                           "target_retrieved_at1": target_rank == 1, "target_retrieved_at4": target_rank > 0,
                           "M": float(stats.margin), "mirabel_threshold": float(stats.threshold),
                           "mirabel_background_mean": float(stats.background_mean),
                           "mirabel_background_std": float(stats.background_std),
                           "mirabel_formula_version": MIRABEL_FORMULA_VERSION})
        checkpoint("MIRABEL_LARGE_SCORING_PROGRESS", completed_queries=min(start+128, len(query_rows)),
                   total_queries=len(query_rows), percent=round(100*min(start+128, len(query_rows))/len(query_rows), 2))
    retrieval_seconds = time.monotonic() - retrieval_started
    metadata = {"document_cache_mode": document_cache_mode, "query_encoding_seconds": query_seconds,
                "query_encoding_per_query_ms": 1000*query_seconds/len(query_rows),
                "retrieval_mirabel_seconds": retrieval_seconds,
                "retrieval_mirabel_per_query_ms": 1000*retrieval_seconds/len(query_rows),
                "total_seconds": time.monotonic()-started,
                "query_embedding_bytes": (EXP / "cache/QUERY_EMBEDDINGS.float16.npy").stat().st_size}
    return output, {**metadata, "query_embeddings": query_embeddings}


def add_local_scores(rows: list[dict], query_embeddings: np.ndarray) -> dict:
    reference_indices = [index for index, row in enumerate(rows) if row["split"] == "REFERENCE"]
    evaluation_indices = [index for index, row in enumerate(rows) if row["split"] != "REFERENCE"]
    if len(reference_indices) != 1000:
        raise RuntimeError("reference size drift")
    reference_embeddings = query_embeddings[reference_indices]
    reference_margins = [rows[index]["M"] for index in reference_indices]
    global_sorted = sorted(reference_margins)
    started = time.monotonic()
    neighbor_audit = []
    for batch_start in range(0, len(evaluation_indices), 256):
        indices = evaluation_indices[batch_start:batch_start+256]
        similarities = query_embeddings[indices] @ reference_embeddings.T
        for local_index, row_index in enumerate(indices):
            similarity = similarities[local_index]
            selected = np.argpartition(-similarity, K_LOCAL)[:K_LOCAL]
            selected = selected[np.argsort(-similarity[selected], kind="stable")]
            local_margins = sorted(reference_margins[int(index)] for index in selected)
            p_local = empirical_upper_tail(local_margins, rows[row_index]["M"])
            p_global = empirical_upper_tail(global_sorted, rows[row_index]["M"])
            rows[row_index]["p_local"] = p_local
            rows[row_index]["R_LC"] = -math.log(p_local)
            rows[row_index]["p_global"] = p_global
            rows[row_index]["R_GLOBAL"] = -math.log(p_global)
            neighbor_ids = [rows[reference_indices[int(index)]]["query_id"] for index in selected]
            rows[row_index]["local_neighbor_ids_sha256"] = __import__("hashlib").sha256(
                "\n".join(neighbor_ids).encode("utf-8")).hexdigest()
            rows[row_index]["local_neighbor_similarity_max"] = float(similarity[selected[0]])
            rows[row_index]["local_neighbor_similarity_min"] = float(similarity[selected[-1]])
        checkpoint("LC_SCORE_PROGRESS", completed_queries=min(batch_start+len(indices), len(evaluation_indices)),
                   total_queries=len(evaluation_indices), percent=round(100*min(batch_start+len(indices), len(evaluation_indices))/len(evaluation_indices), 2))
    elapsed = time.monotonic()-started
    return {"knn_seconds": elapsed, "knn_per_query_ms": 1000*elapsed/len(evaluation_indices),
            "reference_embedding_bytes": int(reference_embeddings.nbytes)}


def bootstrap_delta(member_rows: list[dict], benign: list[dict], iterations: int, seed: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in member_rows:
        groups[row["session_id"]].append(row)
    units = sorted(groups)
    rng = random.Random(seed)
    deltas = []
    negative_m = [row["M"] for row in benign]
    negative_lc = [row["R_LC"] for row in benign]
    for _ in range(iterations):
        sampled = [row for _ in units for row in groups[rng.choice(units)]]
        deltas.append(tpr_at_fpr([row["R_LC"] for row in sampled], negative_lc, .03)
                      - tpr_at_fpr([row["M"] for row in sampled], negative_m, .03))
    return {"iterations": iterations, "unit": "target/session", "bootstrap_mean_delta": statistics.fmean(deltas),
            "ci95_low": percentile(deltas, .025), "ci95_high": percentile(deltas, .975)}


def gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    return sum((2*i-n-1)*value for i, value in enumerate(ordered, 1))/(n*sum(ordered))


def evaluate(rows: list[dict], pre: dict, costs: dict) -> dict:
    holdout = [row for row in rows if row["split"] == "HOLDOUT"]
    attacks = [row for row in rows if row["cohort"] == "ATTACK"]
    negatives = {"M": [row["M"] for row in holdout], "R_GLOBAL": [row["R_GLOBAL"] for row in holdout],
                 "R_LC": [row["R_LC"] for row in holdout]}
    matched_rows = []
    tpr3: dict[str, dict] = {}
    for attack in ATTACKS:
        member = [row for row in attacks if row["attack"] == attack and row["membership"] == "member"]
        for alpha in ALPHAS:
            values = {score: tpr_at_fpr([row[score] for row in member], negatives[score], alpha)
                      for score in ("M", "R_GLOBAL", "R_LC")}
            matched_rows.append({"attack": attack, "positive": "member-target attack query",
                                 "member_queries": len(member), "benign_queries": len(holdout),
                                 "fpr": alpha, "interpolation": "LINEAR_INTERPOLATED_EMPIRICAL_ROC",
                                 "mirabel_tpr": values["M"], "global_bc_tpr": values["R_GLOBAL"],
                                 "lc_mirabel_tpr": values["R_LC"], "lc_minus_mirabel": values["R_LC"]-values["M"],
                                 "miss_reduction": ((1-values["M"])-(1-values["R_LC"]))/max(1-values["M"], 1e-12)})
            if alpha == .03:
                tpr3[attack] = {"M": values["M"], "GLOBAL": values["R_GLOBAL"], "LC": values["R_LC"],
                                "delta": values["R_LC"]-values["M"]}
    write_csv(EXP / "tables/MATCHED_FPR_DETECTION.csv", matched_rows)

    thresholds = {score: matched_threshold(values, .03) for score, values in negatives.items()}
    bootstrap_rows = []
    recovery_rows = []
    recovery_summary = []
    for attack_index, attack in enumerate(ATTACKS):
        member = [row for row in attacks if row["attack"] == attack and row["membership"] == "member"]
        boot = bootstrap_delta(member, holdout, 2000, SEED+attack_index)
        bootstrap_rows.append({"attack": attack, "point_delta": tpr3[attack]["delta"], **boot})
        categories = Counter()
        recovered_by_target = Counter()
        lost_by_target = Counter()
        for row in member:
            m_alarm = row["M"] > thresholds["M"][0]
            lc_alarm = row["R_LC"] > thresholds["R_LC"][0]
            category = "BOTH_HIT" if m_alarm and lc_alarm else "LC_UNIQUE_RECOVERY" if lc_alarm else "MIRABEL_ONLY_LOST" if m_alarm else "BOTH_MISS"
            categories[category] += 1
            if category == "LC_UNIQUE_RECOVERY": recovered_by_target[row["session_id"]] += 1
            if category == "MIRABEL_ONLY_LOST": lost_by_target[row["session_id"]] += 1
            recovery_rows.append({"attack": attack, "session_id": row["session_id"], "query_id": row["query_id"],
                                  "query_index": row["query_index"], "target_id": row["target_id"], "category": category,
                                  "M": row["M"], "R_LC": row["R_LC"], "p_local": row["p_local"]})
        values = list(recovered_by_target.values())
        total = sum(values)
        recovery_summary.append({"attack": attack, **dict(categories), "recovered_queries": total,
                                 "recovered_targets": len(recovered_by_target), "lost_queries": sum(lost_by_target.values()),
                                 "lost_targets": len(lost_by_target), "top1_target_share": max(values, default=0)/total if total else 0,
                                 "top5_target_share": sum(sorted(values, reverse=True)[:5])/total if total else 0,
                                 "gini_recovery": gini(values)})
    write_csv(EXP / "tables/PAIRED_CLUSTER_BOOTSTRAP.csv", bootstrap_rows)
    write_csv(EXP / "tables/RECOVERY_CASES.csv", recovery_rows)
    write_csv(EXP / "tables/RECOVERY_CONCENTRATION.csv", recovery_summary)

    nonmember_rows = []
    for attack in ATTACKS:
        subset = [row for row in attacks if row["attack"] == attack and row["membership"] == "nonmember"]
        for score, detector in (("M", "Raw MIRABEL"), ("R_GLOBAL", "Global BC-MIRABEL"), ("R_LC", "LC-MIRABEL")):
            nonmember_rows.append({"attack": attack, "detector": detector, "queries": len(subset),
                                   "alarm_rate_at_matched_3pct": statistics.fmean(row[score] > thresholds[score][0] for row in subset)})
    write_csv(EXP / "tables/NONMEMBER_ATTACK_ALARM_RATE.csv", nonmember_rows)

    deployment_rows = []
    rules = (("Original MIRABEL", "M>0", lambda row: row["M"] > 0),
             ("Global BC-MIRABEL", "p_global<=0.03", lambda row: row["p_global"] <= .03),
             ("LC-MIRABEL", "p_local<=0.03", lambda row: row["p_local"] <= .03))
    for method, rule, alarm in rules:
        for domain in ("ALL", *sorted({row["domain"] for row in holdout})):
            subset = holdout if domain == "ALL" else [row for row in holdout if row["domain"] == domain]
            fp = sum(alarm(row) for row in subset)
            low, high = wilson(fp, len(subset))
            deployment_rows.append({"method": method, "nominal_rule": rule, "domain": domain, "n": len(subset),
                                    "false_positives": fp, "actual_fpr": fp/len(subset) if subset else None,
                                    "wilson95_low": low, "wilson95_high": high})
    write_csv(EXP / "tables/DEPLOYMENT_FPR.csv", deployment_rows)

    session_rows = []
    for attack in ATTACKS:
        subset = [row for row in attacks if row["attack"] == attack]
        for membership in ("member", "nonmember"):
            groups: dict[str, list[dict]] = defaultdict(list)
            for row in subset:
                if row["membership"] == membership:
                    groups[row["session_id"]].append(row)
            for session_id, group in sorted(groups.items()):
                group.sort(key=lambda row: row["query_index"])
                alarms = [row["R_LC"] > thresholds["R_LC"][0] for row in group]
                removed = [flag and row["selected_source_id"] == row["target_id"] for flag, row in zip(alarms, group)]
                session_rows.append({"attack": attack, "session_id": session_id, "membership": membership,
                                     "queries": len(group), "alarms": sum(alarms), "any_alarm": any(alarms),
                                     "all_alarm": all(alarms), "target_removed": sum(removed),
                                     "any_target_removed": any(removed), "all_target_removed": all(removed)})
    write_csv(EXP / "tables/SESSION_DETECTION.csv", session_rows)

    macro_m = statistics.fmean(tpr3[attack]["M"] for attack in ATTACKS)
    macro_lc = statistics.fmean(tpr3[attack]["LC"] for attack in ATTACKS)
    menta_boot = next(row for row in bootstrap_rows if row["attack"] == "MEntA")
    checks = {"menta_delta_at_least_5pp": tpr3["MEntA"]["delta"] >= .05-1e-12,
              "menta_bootstrap_ci_low_above_zero": menta_boot["ci95_low"] > 0,
              "mba_noninferior_within_2pp": tpr3["MBA"]["delta"] >= -.02-1e-12,
              "rag_mia_noninferior_within_2pp": tpr3["RAG-MIA"]["delta"] >= -.02-1e-12,
              "core3_macro_delta_at_least_2pp": macro_lc-macro_m >= .02-1e-12}
    passed = all(checks.values())
    overall_deploy = {row["method"]: row["actual_fpr"] for row in deployment_rows if row["domain"] == "ALL"}
    worst_deploy = {method: max(row["actual_fpr"] for row in deployment_rows if row["method"] == method and row["domain"] != "ALL")
                    for method in overall_deploy}
    result = {"campaign": "LC_MIRABEL_LARGE_V1", "phase": "SCORE_ONLY_DETECTION",
              "verdict": "LC_MIRABEL_DETECTION_PHASE1_PASS" if passed else "LC_MIRABEL_DETECTION_NOT_SUPPORTED",
              "completed_utc": now(), "development_notice": "THIS_LARGE_COHORT_IS_DEVELOPMENT_DATA",
              "query_counts": {"benign_reference": 1000, "benign_holdout": 1000, "attack": 14000},
              "matched_tpr_at_3pct": tpr3,
              "macro": {"MIRABEL": macro_m, "LC_MIRABEL": macro_lc, "delta": macro_lc-macro_m},
              "bootstrap": bootstrap_rows, "checks": checks,
              "binary_thresholds": {key: {"threshold": value[0], "benign_alarms": value[1]} for key, value in thresholds.items()},
              "deployment_fpr": overall_deploy, "worst_domain_fpr": worst_deploy,
              "cost": {key: value for key, value in costs.items() if key != "query_embeddings"},
              "training_steps": 0, "trainable_parameters_updated": 0,
              "e2e_allowed": passed, "next_step": "LARGE_E2E" if passed else "STOP_NO_GENERATION"}
    atomic_json(EXP / "PHASE1_RESULT.json", result)
    lines = ["# LC-MIRABEL Large Score-only Report", "", f"- 판정: `{result['verdict']}`",
             "- Positive: member-target attack query", "- Negative: 동일 benign holdout 1,000 query", "",
             "| Attack | MIRABEL TPR@3% | LC TPR@3% | Delta | Bootstrap 95% CI |",
             "|---|---:|---:|---:|---:|"]
    for attack in ATTACKS:
        boot = next(row for row in bootstrap_rows if row["attack"] == attack)
        lines.append(f"| {attack} | {tpr3[attack]['M']:.4f} | {tpr3[attack]['LC']:.4f} | {tpr3[attack]['delta']:+.4f} | [{boot['ci95_low']:.4f}, {boot['ci95_high']:.4f}] |")
    lines += [f"| **Macro** | **{macro_m:.4f}** | **{macro_lc:.4f}** | **{macro_lc-macro_m:+.4f}** | — |", "",
              "## Gate", *[f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in checks.items()], "",
              "실패 시 명세대로 Qwen E2E와 추가 local-feature 탐색을 실행하지 않는다."]
    atomic_text(EXP / "reports/LC_MIRABEL_PHASE1_REPORT_KO.md", "\n".join(lines)+"\n")
    checkpoint(result["verdict"], e2e_allowed=passed, menta_delta=tpr3["MEntA"]["delta"], macro_delta=macro_lc-macro_m)
    return result


def main() -> None:
    pre = verify()
    reference, holdout, attacks = load_queries()
    rows, costs = encode_and_score(pre, reference+holdout+attacks)
    costs.update(add_local_scores(rows, costs.pop("query_embeddings")))
    cache_path = EXP / "cache/LARGE_DETECTION_SCORES.jsonl"
    write_jsonl(cache_path, rows)
    atomic_json(EXP / "cache/LARGE_DETECTION_CACHE_MANIFEST.json", {
        "rows": len(rows), "reference": 1000, "holdout": 1000, "attack": 14000,
        "sha256": sha_file(cache_path), "created_utc": now(), "k": K_LOCAL,
        "cost": costs})
    result = evaluate(rows, pre, costs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
