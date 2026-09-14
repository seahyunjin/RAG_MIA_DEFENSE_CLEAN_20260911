#!/usr/bin/env python3
"""Frozen Final-LC versus MIRABEL same-FPR evaluation on IA-Std-Q15-ST1."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1"
POST = EXP / "post_ready"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
CORE3 = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
PRECOMMIT = POST / "configs" / "IA_ST1_DETECTION_PRECOMMIT.json"
SEED = 20260913
K_LOCAL = 200
BUDGETS = (0.01, 0.025, 0.03, 0.05)

sys.path.insert(0, str(ROOT / "code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments, MIRABEL_FORMULA_VERSION


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, "")
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **extra: object) -> None:
    value = {"campaign": "IA_STEALTH_CONFIRMATION_V1", "phase": "IA_SAME_FPR_DETECTION", "stage": stage, "updated_utc": now(), **extra}
    atomic_json(POST / "HEARTBEAT.json", value)
    lines = ["# IA-ST1 post-ready evaluation", "", f"- stage: `{stage}`", f"- updated_utc: `{value['updated_utc']}`"]
    lines.extend(f"- {key}: `{item}`" for key, item in extra.items())
    atomic_text(POST / "STATUS.md", "\n".join(lines) + "\n")


def verify_precommit() -> dict:
    expected = PRECOMMIT.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if sha(PRECOMMIT) != expected:
        raise RuntimeError("IA detection precommit hash mismatch")
    pre = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    if sha(Path(pre["code"]["path"])) != pre["code"]["sha256"]:
        raise RuntimeError("IA detection code drift")
    for key in ("queries", "gt", "sessions", "final_lc_manifest", "frozen_reference_scores", "frozen_reference_embeddings", "frozen_corpus_embeddings", "frozen_corpus", "frozen_benign_detection"):
        if sha(Path(pre[key]["path"])) != pre[key]["sha256"]:
            raise RuntimeError(f"frozen input drift: {key}")
    return pre


def threshold_at_budget(values: list[float], alpha: float) -> tuple[float, int]:
    ordered = sorted(map(float, values), reverse=True)
    allowance = math.floor(alpha * len(ordered) + 1e-12)
    threshold = ordered[allowance] if allowance < len(ordered) else -math.inf
    alarms = sum(value > threshold for value in ordered)
    if alarms > allowance:
        raise RuntimeError("benign threshold construction exceeded budget")
    return threshold, alarms


def bootstrap_query_delta(rows: list[dict], m_threshold: float, lc_threshold: float) -> tuple[float, float, float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["session_id"]].append(row)
    units = sorted(grouped)
    sizes = np.asarray([len(grouped[key]) for key in units], dtype=float)
    m_hits = np.asarray([sum(float(row["M"]) > m_threshold for row in grouped[key]) for key in units], dtype=float)
    lc_hits = np.asarray([sum(float(row["R_LC"]) > lc_threshold for row in grouped[key]) for key in units], dtype=float)
    rng = np.random.default_rng(SEED)
    boot = np.empty(2000, dtype=float)
    for start in range(0, 2000, 100):
        idx = rng.integers(0, len(units), size=(min(100, 2000 - start), len(units)))
        denominator = sizes[idx].sum(axis=1)
        boot[start:start + len(idx)] = (lc_hits[idx].sum(axis=1) - m_hits[idx].sum(axis=1)) / denominator
    return float(boot.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def normalize_queries(rows: list[dict]) -> list[dict]:
    output = []
    sessions: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        value = {
            "query_id": row["_id"],
            "query": row["text"],
            "target_id": row["target_doc_id"],
            "canonical_target_id": row["canonical_target_id"],
            "membership": row["_membership"],
            "session_id": row["session_id"],
            "query_index": int(row["turn"]),
            "attack": "IA-Std-Q15-ST1",
        }
        output.append(value)
        sessions[value["session_id"]].append(value["query_index"])
    if len(output) != len({row["query_id"] for row in output}):
        raise RuntimeError("duplicate IA query IDs")
    if not sessions or any(sorted(turns) != list(range(1, 16)) for turns in sessions.values()):
        raise RuntimeError("IA session Q15 alignment drift")
    return output


def score_queries(query_rows: list[dict]) -> tuple[list[dict], dict]:
    prior = read_jsonl(LC / "cache" / "LARGE_DETECTION_SCORES.jsonl")
    prior_embeddings = np.asarray(np.load(LC / "cache" / "QUERY_EMBEDDINGS.float16.npy"), dtype=np.float32)
    if len(prior) != prior_embeddings.shape[0]:
        raise RuntimeError("reference score/embedding alignment drift")
    reference_indices = [index for index, row in enumerate(prior) if row["split"] == "REFERENCE"]
    if len(reference_indices) != 1000:
        raise RuntimeError("frozen benign reference count drift")
    ref_embeddings = prior_embeddings[reference_indices]
    ref_margins = [float(prior[index]["M"]) for index in reference_indices]
    documents = read_jsonl(CORE3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")
    document_embeddings = np.asarray(np.load(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy"), dtype=np.float32)
    if len(documents) != document_embeddings.shape[0]:
        raise RuntimeError("corpus score/embedding alignment drift")
    doc_ids = [row["document_id"] for row in documents]
    doc_id_set = set(doc_ids)
    if any(row["target_id"] not in doc_id_set for row in query_rows):
        raise RuntimeError("IA target absent from frozen protected corpus")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for frozen BGE scoring")
    checkpoint("BGE_LOADING", queries=len(query_rows), gpu_free_bytes=torch.cuda.mem_get_info()[0])
    started = time.monotonic()
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    embeddings = np.asarray(model.encode([row["query"] for row in query_rows], batch_size=48, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    del model
    torch.cuda.empty_cache()
    np.save(POST / "cache" / "IA_ST1_QUERY_EMBEDDINGS.float16.npy", embeddings.astype(np.float16))
    global_sorted = sorted(ref_margins)
    output: list[dict] = []
    for start in range(0, len(query_rows), 128):
        qbatch = embeddings[start:start + 128]
        score_batch = qbatch @ document_embeddings.T
        similarity_batch = qbatch @ ref_embeddings.T
        for offset, scores in enumerate(score_batch):
            source = query_rows[start + offset]
            top_idx = np.argpartition(-scores, 4)[:4]
            top_idx = top_idx[np.argsort(-scores[top_idx], kind="stable")]
            top_ids = [doc_ids[int(index)] for index in top_idx]
            top_scores = [float(scores[int(index)]) for index in top_idx]
            stat = canonical_mirabel_from_moments(top1=top_scores[0], sum_all=float(scores.sum(dtype=np.float64)), sumsq_all=float(np.square(scores, dtype=np.float64).sum(dtype=np.float64)), corpus_size=len(doc_ids), confidence=0.95)
            similarities = similarity_batch[offset]
            neighbors = np.argpartition(-similarities, K_LOCAL)[:K_LOCAL]
            neighbors = neighbors[np.argsort(-similarities[neighbors], kind="stable")]
            local_sorted = sorted(ref_margins[int(index)] for index in neighbors)
            p_local = (1 + len(local_sorted) - np.searchsorted(local_sorted, float(stat.margin), side="left")) / (len(local_sorted) + 1)
            p_global = (1 + len(global_sorted) - np.searchsorted(global_sorted, float(stat.margin), side="left")) / (len(global_sorted) + 1)
            target_rank = top_ids.index(source["target_id"]) + 1 if source["target_id"] in top_ids else 0
            output.append({**source, "top_document_ids": top_ids, "top_scores": top_scores, "selected_source_id": top_ids[0], "target_rank": target_rank, "M": float(stat.margin), "mirabel_threshold": float(stat.threshold), "mirabel_background_mean": float(stat.background_mean), "mirabel_background_std": float(stat.background_std), "mirabel_formula_version": MIRABEL_FORMULA_VERSION, "p_local": float(p_local), "R_LC": float(-math.log(p_local)), "p_global": float(p_global), "R_GLOBAL": float(-math.log(p_global))})
        checkpoint("SCORING_PROGRESS", completed=min(start + 128, len(query_rows)), total=len(query_rows))
    cache = POST / "cache" / "IA_ST1_RETRIEVAL_AND_DETECTION.jsonl"
    write_jsonl(cache, output)
    return output, {"seconds": time.monotonic() - started, "rows": len(output), "cache_sha256": sha(cache), "embedding_sha256": sha(POST / "cache" / "IA_ST1_QUERY_EMBEDDINGS.float16.npy")}


def evaluate(rows: list[dict], runtime: dict, pre: dict) -> dict:
    benign_all = read_jsonl(Path(pre["frozen_benign_detection"]["path"]))
    benign = [row for row in benign_all if row.get("attack") == "BENIGN"]
    if len(benign) != 1000:
        raise RuntimeError(f"frozen benign holdout count drift: {len(benign)}")
    member = [row for row in rows if row["membership"] == "member"]
    nonmember = [row for row in rows if row["membership"] == "nonmember"]
    if not member or not nonmember:
        raise RuntimeError("member/nonmember IA rows required")

    threshold_rows, performance_rows = [], []
    thresholds: dict[tuple[str, float], float] = {}
    for budget in BUDGETS:
        for method, key in (("MIRABEL", "M"), ("Final LC", "R_LC")):
            threshold, alarms = threshold_at_budget([float(row[key]) for row in benign], budget)
            thresholds[(method, budget)] = threshold
            threshold_rows.append({"method": method, "nominal_fpr": budget, "threshold": threshold, "operator": ">", "benign_n": len(benign), "benign_alarms": alarms, "actual_fpr": alarms / len(benign)})
    write_csv(POST / "tables" / "IA_ST1_BENIGN_THRESHOLDS.csv", threshold_rows)

    for budget in BUDGETS:
        record = {"attack": "IA-Std-Q15-ST1", "nominal_fpr": budget, "member_queries": len(member), "member_sessions": len({row['session_id'] for row in member})}
        for method, key in (("MIRABEL", "M"), ("Final LC", "R_LC")):
            flags = [float(row[key]) > thresholds[(method, budget)] for row in member]
            record[f"{method}_tpr"] = statistics.fmean(flags)
            record[f"{method}_actual_benign_fpr"] = next(item["actual_fpr"] for item in threshold_rows if item["method"] == method and item["nominal_fpr"] == budget)
        record["delta_tpr"] = record["Final LC_tpr"] - record["MIRABEL_tpr"]
        performance_rows.append(record)
    primary = next(row for row in performance_rows if row["nominal_fpr"] == 0.025)
    boot_mean, boot_low, boot_high = bootstrap_query_delta(member, thresholds[("MIRABEL", 0.025)], thresholds[("Final LC", 0.025)])
    primary.update({"cluster_bootstrap_mean_delta": boot_mean, "delta_ci95_low": boot_low, "delta_ci95_high": boot_high})
    write_csv(POST / "tables" / "IA_ST1_TPR_LOW_FPR.csv", performance_rows)

    session_rows = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["membership"], row["session_id"])].append(row)
    for (membership, session_id), group in sorted(groups.items()):
        group.sort(key=lambda item: item["query_index"])
        if len(group) != 15:
            raise RuntimeError(f"session Q15 drift: {session_id}")
        for method, key in (("MIRABEL", "M"), ("Final LC", "R_LC")):
            flags = [float(row[key]) > thresholds[(method, 0.025)] for row in group]
            removals = [flag and row["selected_source_id"] == row["target_id"] for flag, row in zip(flags, group)]
            session_rows.append({"attack": "IA-Std-Q15-ST1", "membership": membership, "session_id": session_id, "method": method, "queries": 15, "alarms": sum(flags), "protected": sum(flags), "alarm_fraction": sum(flags) / 15, "any_alarm": any(flags), "all_15_alarm": all(flags), "first_alarm_index": next((index + 1 for index, flag in enumerate(flags) if flag), None), "target_removals": sum(removals), "any_target_removal": any(removals)})
    write_csv(POST / "tables" / "IA_ST1_SESSION_METRICS.csv", session_rows)

    checks = {
        "final_lc_noninferior_within_3pp": primary["Final LC_tpr"] >= primary["MIRABEL_tpr"] - 0.03 - 1e-12,
        "no_catastrophic_stealth_miss": primary["Final LC_tpr"] >= 0.05 - 1e-12,
    }
    verdict = "IA_STEALTH_DETECTION_PASS" if all(checks.values()) else "IA_STEALTH_DETECTION_FAILED"
    result = {"campaign": "IA_STEALTH_CONFIRMATION_V1", "phase": "IA_SAME_FPR_DETECTION", "completed_utc": now(), "verdict": verdict, "attack_claim": "standardized IA stealth stress test, not Original or paper-exact IA", "primary": primary, "low_fpr": performance_rows, "checks": checks, "preferred_final_lc_beats_mirabel": primary["Final LC_tpr"] > primary["MIRABEL_tpr"], "valid_member_queries": len(member), "valid_nonmember_queries": len(nonmember), "valid_sessions": len(groups), "bootstrap": {"iterations": 2000, "cluster": "target/session", "delta_mean": boot_mean, "ci95": [boot_low, boot_high]}, "runtime": runtime, "final_lc_modified": False, "next": "IA_MATCHED_BUDGET_E2E" if verdict == "IA_STEALTH_DETECTION_PASS" else "CROSS_DOMAIN_PRIVACY_GENERALIZATION"}
    atomic_json(POST / "IA_ST1_DETECTION_RESULT.json", result)
    lines = ["# IA-Std-Q15-ST1 matched-FPR detection", "", f"- Verdict: `{verdict}`", f"- Valid sessions: `{len(groups)}`", f"- Primary actual benign FPR: MIRABEL `{primary['MIRABEL_actual_benign_fpr']:.4f}`, Final LC `{primary['Final LC_actual_benign_fpr']:.4f}`", "", "| Nominal benign FPR | MIRABEL TPR | Final LC TPR | Delta |", "|---:|---:|---:|---:|"]
    for row in performance_rows:
        lines.append(f"| {100 * row['nominal_fpr']:.1f}% | {row['MIRABEL_tpr']:.4f} | {row['Final LC_tpr']:.4f} | {row['delta_tpr']:+.4f} |")
    lines.extend(["", f"- Session-cluster bootstrap delta 95% CI: `[{boot_low:.4f}, {boot_high:.4f}]`", "- Claim boundary: IA-ST1 is a standardized stealth stress test, not Original/paper-exact IA."])
    atomic_text(POST / "reports" / "IA_ST1_DETECTION_KO.md", "\n".join(lines) + "\n")
    checkpoint(verdict, verdict=verdict, mirabel_tpr=round(primary["MIRABEL_tpr"], 6), final_lc_tpr=round(primary["Final LC_tpr"], 6), delta=round(primary["delta_tpr"], 6), next=result["next"])
    return result


def main() -> None:
    pre = verify_precommit()
    full = json.loads(Path(pre["full_result"]["path"]).read_text(encoding="utf-8"))
    if full.get("verdict") != "IA_STD_Q15_ST1_READY":
        raise RuntimeError("full artifact no longer READY")
    query_rows = normalize_queries(read_jsonl(Path(pre["queries"]["path"])))
    checkpoint("INPUT_AUDIT_PASS", queries=len(query_rows), sessions=len({row['session_id'] for row in query_rows}))
    rows, runtime = score_queries(query_rows)
    result = evaluate(rows, runtime, pre)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
