#!/usr/bin/env python3
"""CPU exact-index scoring for frozen Final LC under V0/V10/V25/V50 churn."""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (ATTACKS, CLEAN, CORE6, EXP, FINAL8, K_LOCAL, LC, OLD_SIDECAR,
                    OUTER_BUDGET, ROOT, SEED, STRICT_THRESHOLD, VERSIONS,
                    atomic_json, atomic_text, checkpoint, matched_threshold, now,
                    read_jsonl, sha_file, sha_text, verify_hashed_json, wilson,
                    write_csv)

sys.path.insert(0, str(ROOT / "code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments  # noqa: E402


def verify(pre: dict) -> None:
    for section in ("input_sha256", "derived_input_sha256"):
        for path, digest in pre[section].items():
            if sha_file(Path(path)) != digest:
                raise RuntimeError(f"input drift: {path}")
    embedding_manifest = json.loads((EXP / "cache/BGE_EMBEDDING_UNION_MANIFEST.json").read_text())
    for key in ("document_cache", "query_cache"):
        if sha_file(Path(embedding_manifest[key])) != embedding_manifest[key + "_sha256"]:
            raise RuntimeError(f"embedding cache drift: {key}")


def assemble_queries(rows: list[dict]) -> np.ndarray:
    large = np.load(LC / "cache/QUERY_EMBEDDINGS.float16.npy", mmap_mode="r")
    dcmi = np.load(CORE6 / "cache/STANDARDIZED_QUERY_EMBEDDINGS.float16.npy", mmap_mode="r")
    s2 = np.load(EXP / "cache/S2_QUERY_EMBEDDINGS.float16.npy", mmap_mode="r")
    s2_rows = [row for row in rows if row["embedding_source"] == "NEW_BGE_S2"]
    s2_index = {row["query_id"]: index for index, row in enumerate(s2_rows)}
    output = np.empty((len(rows), large.shape[1]), dtype=np.float32)
    for index, row in enumerate(rows):
        source = row["embedding_source"]
        if source == "LC_LARGE_FROZEN":
            output[index] = large[int(row["embedding_index"])]
        elif source == "DCMI_FROZEN":
            output[index] = dcmi[int(row["embedding_index"])]
        elif source == "NEW_BGE_S2":
            output[index] = s2[s2_index[row["query_id"]]]
        else:
            raise RuntimeError(f"unknown embedding source: {source}")
    norms = np.linalg.norm(output, axis=1)
    if np.max(np.abs(norms - 1)) > .01:
        raise RuntimeError("query embedding normalization drift")
    return output


def assemble_documents(version: str, base_rows: list[dict], base_embeddings: np.ndarray,
                       added_rows: list[dict], added_embeddings: np.ndarray) -> tuple[list[str], np.ndarray]:
    base_map = {row["document_id"]: base_embeddings[index] for index, row in enumerate(base_rows)}
    added_map = {row["document_id"]: added_embeddings[index] for index, row in enumerate(added_rows)}
    manifest = json.loads((OLD_SIDECAR / f"manifests/DB_CHURN_{version}.json").read_text())
    ids = manifest["ordered_document_ids"]
    output = np.asarray([base_map[item] if item in base_map else added_map[item] for item in ids], dtype=np.float32)
    if output.shape != (3000, 1024):
        raise RuntimeError(f"document embedding shape drift: {version}: {output.shape}")
    return ids, output


def margins_for_queries(query_embeddings: np.ndarray, document_embeddings: np.ndarray,
                        version: str) -> np.ndarray:
    margins = np.empty(len(query_embeddings), dtype=np.float64)
    for start in range(0, len(query_embeddings), 128):
        matrix = query_embeddings[start:start + 128] @ document_embeddings.T
        for offset, scores in enumerate(matrix):
            stat = canonical_mirabel_from_moments(
                top1=float(scores.max()), sum_all=float(scores.sum(dtype=np.float64)),
                sumsq_all=float(np.square(scores, dtype=np.float64).sum(dtype=np.float64)),
                corpus_size=len(document_embeddings), confidence=.95)
            margins[start + offset] = float(stat.margin)
        checkpoint("CHURN_MARGIN_SCORING", version=version,
                   completed=min(start + 128, len(query_embeddings)), total=len(query_embeddings),
                   cpu_exact_index=True)
    return margins


def local_risk(margins: np.ndarray, reference_margins: np.ndarray,
               eval_indices: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    risk = np.full(len(margins), np.nan, dtype=np.float64)
    for position, row_index in enumerate(eval_indices):
        values = reference_margins[neighbors[position]]
        p_value = (1 + int(np.count_nonzero(values >= margins[row_index]))) / (K_LOCAL + 1)
        risk[row_index] = -math.log(p_value)
    return risk


def cluster_bootstrap(rows: list[dict], alarms: np.ndarray, indices: list[int], seed: int) -> tuple[float, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[rows[index]["session_id"]].append(index)
    units = sorted(groups)
    unit_sum = np.asarray([sum(bool(alarms[index]) for index in groups[unit]) for unit in units], dtype=float)
    unit_n = np.asarray([len(groups[unit]) for unit in units], dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(2000, dtype=float)
    for start in range(0, 2000, 200):
        choices = rng.integers(0, len(units), size=(min(200, 2000 - start), len(units)))
        samples[start:start + len(choices)] = unit_sum[choices].sum(axis=1) / unit_n[choices].sum(axis=1)
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def main() -> None:
    pre = verify_hashed_json(EXP / "configs/DB_CHURN_V2_PRECOMMIT.json")
    verify(pre)
    started = time.monotonic()
    rows = read_jsonl(EXP / "inputs/CHURN_QUERY_MANIFEST.jsonl")
    query_embeddings = assemble_queries(rows)
    reference_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "REFERENCE"], dtype=int)
    holdout_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "HOLDOUT"], dtype=int)
    eval_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] != "REFERENCE"], dtype=int)
    if len(reference_indices) != 1000 or len(holdout_indices) != 1000:
        raise RuntimeError("benign reference/holdout count drift")

    checkpoint("CPU_QUERY_NEIGHBOR_INDEX_STARTED", rows=len(rows), reference=1000)
    neighbor_matrix = query_embeddings[eval_indices] @ query_embeddings[reference_indices].T
    neighbors = np.argpartition(-neighbor_matrix, K_LOCAL, axis=1)[:, :K_LOCAL]
    row_order = np.arange(len(neighbors))[:, None]
    neighbor_scores = neighbor_matrix[row_order, neighbors]
    order = np.argsort(-neighbor_scores, axis=1, kind="stable")
    neighbors = neighbors[row_order, order]
    del neighbor_matrix, neighbor_scores, order, row_order

    base_rows = read_jsonl(CLEAN / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")
    base_embeddings = np.asarray(np.load(LC / "cache/CORPUS_EMBEDDINGS.float16.npy"), dtype=np.float32)
    added_rows = read_jsonl(EXP / "inputs/MISSING_EMBEDDING_UNION.jsonl")
    added_embeddings = np.asarray(np.load(EXP / "cache/CHURN_ADDED_DOCUMENT_EMBEDDINGS.float16.npy"), dtype=np.float32)
    if len(base_rows) != len(base_embeddings) or len(added_rows) != len(added_embeddings):
        raise RuntimeError("document row/embedding mismatch")

    margins_by_version = {}
    for version in VERSIONS:
        _, document_embeddings = assemble_documents(version, base_rows, base_embeddings, added_rows, added_embeddings)
        margins_by_version[version] = margins_for_queries(query_embeddings, document_embeddings, version)
        del document_embeddings
    strict_reference = margins_by_version["V0"][reference_indices]

    summary_rows = []
    domain_rows = []
    bootstrap_rows = []
    score_manifest = []
    refresh_baseline: dict[str, float] = {}
    strict_baseline: dict[str, float] = {}
    version_details = {}
    for version_index, version in enumerate(VERSIONS):
        margins = margins_by_version[version]
        strict_risk = local_risk(margins, strict_reference, eval_indices, neighbors)
        refresh_reference = margins[reference_indices]
        refresh_risk = local_risk(margins, refresh_reference, eval_indices, neighbors)
        refresh_threshold, refresh_calibration_alarms = matched_threshold(refresh_risk[holdout_indices], OUTER_BUDGET)
        npz = EXP / f"cache/{version}_QUERY_SCORES.npz"
        np.savez_compressed(npz, M=margins, R_LC_STRICT=strict_risk,
                            R_LC_REFRESH=refresh_risk, query_id_sha256=np.asarray(pre["evaluation"]["ordered_query_id_sha256"]))
        score_manifest.append({"version": version, "path": str(npz), "sha256": sha_file(npz),
                               "refresh_threshold": refresh_threshold})

        methods = {
            "Original MIRABEL": margins > 0,
            "Final LC Strict": strict_risk > STRICT_THRESHOLD,
            "Final LC Refresh": refresh_risk > refresh_threshold,
        }
        details = {"strict_threshold": STRICT_THRESHOLD, "refresh_threshold": refresh_threshold,
                   "refresh_calibration_alarms": refresh_calibration_alarms, "methods": {}}
        for method, alarms in methods.items():
            fp = int(np.count_nonzero(alarms[holdout_indices]))
            low, high = wilson(fp, len(holdout_indices))
            attack_tpr = {}
            for attack_index, attack in enumerate(ATTACKS):
                positive = [i for i, row in enumerate(rows)
                            if row["split"] == "ATTACK" and row["attack"] == attack and row["membership"] == "member"]
                tpr = float(np.mean(alarms[positive]))
                attack_tpr[attack] = tpr
                ci_low, ci_high = cluster_bootstrap(rows, alarms, positive,
                                                     SEED + version_index * 100 + attack_index * 3 + list(methods).index(method))
                bootstrap_rows.append({"db": version, "method": method, "attack": attack,
                                       "member_queries": len(positive), "member_sessions": len({rows[i]['session_id'] for i in positive}),
                                       "tpr": tpr, "cluster_bootstrap_iterations": 2000,
                                       "ci95_low": ci_low, "ci95_high": ci_high})
            mean_tpr = statistics.fmean(attack_tpr.values())
            summary_rows.append({"db": version, "method": method,
                                 "threshold": 0.0 if method == "Original MIRABEL" else (STRICT_THRESHOLD if method.endswith("Strict") else refresh_threshold),
                                 "benign_n": len(holdout_indices), "false_positives": fp, "benign_fpr": fp / len(holdout_indices),
                                 "fpr_wilson_low": low, "fpr_wilson_high": high,
                                 **{attack: attack_tpr[attack] for attack in ATTACKS}, "core5_mean_tpr": mean_tpr,
                                 "retraining_steps": 0})
            for domain in sorted({rows[i]["domain"] for i in holdout_indices}):
                subset = [i for i in holdout_indices if rows[i]["domain"] == domain]
                domain_fp = int(np.count_nonzero(alarms[subset]))
                dlow, dhigh = wilson(domain_fp, len(subset))
                domain_rows.append({"db": version, "method": method, "domain": domain,
                                    "n": len(subset), "false_positives": domain_fp,
                                    "fpr": domain_fp / len(subset), "wilson95_low": dlow, "wilson95_high": dhigh})
            details["methods"][method] = {"fpr": fp / len(holdout_indices), "attack_tpr": attack_tpr,
                                           "core5_mean_tpr": mean_tpr}
        version_details[version] = details
        if version == "V0":
            refresh_baseline = details["methods"]["Final LC Refresh"]["attack_tpr"]
            strict_baseline = details["methods"]["Final LC Strict"]["attack_tpr"]
        checkpoint("CHURN_VERSION_COMPLETE", version=version,
                   strict_fpr=details["methods"]["Final LC Strict"]["fpr"],
                   refresh_fpr=details["methods"]["Final LC Refresh"]["fpr"])

    write_csv(EXP / "tables/DB_CHURN_V2_SUMMARY.csv", summary_rows)
    write_csv(EXP / "tables/DB_CHURN_DOMAIN_FPR_WILSON.csv", domain_rows)
    write_csv(EXP / "tables/DB_CHURN_CORE5_CLUSTER_BOOTSTRAP.csv", bootstrap_rows)
    atomic_json(EXP / "cache/CHURN_SCORE_CACHE_MANIFEST.json", {"scores": score_manifest,
                "cpu_index": "exact normalized inner-product matrix; mathematically equivalent to FAISS IndexFlatIP",
                "faiss_python_available": False, "query_id_sha256": pre["evaluation"]["ordered_query_id_sha256"]})

    def gate(method: str, baseline: dict[str, float]) -> tuple[bool, dict]:
        fprs = {v: version_details[v]["methods"][method]["fpr"] for v in VERSIONS}
        drops = {v: {a: baseline[a] - version_details[v]["methods"][method]["attack_tpr"][a]
                     for a in ATTACKS} for v in VERSIONS}
        mean_drops = {v: statistics.fmean(drops[v].values()) for v in VERSIONS}
        checks = {
            "all_fpr_at_most_5pct": all(value <= .05 + 1e-12 for value in fprs.values()),
            "all_attack_degradation_at_most_5pp": all(value <= .05 + 1e-12 for d in drops.values() for value in d.values()),
            "mean_degradation_preferred_at_most_3pp": all(value <= .03 + 1e-12 for value in mean_drops.values()),
        }
        return all(checks.values()), {"fpr": fprs, "tpr_degradation": drops,
                                      "mean_degradation": mean_drops, "checks": checks}

    refresh_pass, refresh_gate = gate("Final LC Refresh", refresh_baseline)
    strict_pass, strict_gate = gate("Final LC Strict", strict_baseline)
    verdict = "DB_CHURN_RETRAINING_FREE_PASS" if refresh_pass else "DB_CHURN_RECALIBRATION_FAILED"
    result = {
        "campaign": EXP.name, "phase": "DB_CHURN_V2", "verdict": verdict,
        "strict_transfer_verdict": "DB_CHURN_STRICT_TRANSFER_PASS" if strict_pass else "DB_CHURN_STRICT_TRANSFER_FAILED",
        "completed_utc": now(), "versions": version_details,
        "refresh_gate": refresh_gate, "strict_gate": strict_gate,
        "runtime_seconds": time.monotonic() - started,
        "training_steps": 0, "gradient_updates": 0, "attack_samples_used_for_refresh": 0,
        "model_or_formula_modified": False,
        "calibration_scope_note": "The frozen 1,000-query benign deployment holdout defines the outer operating threshold and reports its empirical exceedance rate, matching the inherited Final LC protocol; it is not a separate untouched FPR test.",
    }
    atomic_json(EXP / "DB_CHURN_V2_RESULT.json", result)
    atomic_json(EXP / f"checkpoints/{verdict}.json", result)
    checkpoint(verdict, strict_transfer=result["strict_transfer_verdict"],
               runtime_seconds=round(result["runtime_seconds"], 2), training_steps=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
