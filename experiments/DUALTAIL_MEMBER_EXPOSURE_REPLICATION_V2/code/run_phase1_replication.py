#!/usr/bin/env python3
"""Fresh 100/100 DualTail member-exposure replication (Phase 1 only)."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
V1 = ROOT / "experiments" / "BC_DUALTAIL_DETECTOR_V1_STRICT"
EXP = ROOT / "experiments" / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2"
INPUTS = EXP / "inputs"
CACHE = EXP / "cache"
TABLES = EXP / "tables"
AUDITS = EXP / "audits"
REPORTS = EXP / "reports"
CONFIGS = EXP / "configs"
RETRIEVER = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
PRECOMMIT = CONFIGS / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2_PRECOMMIT.json"
PRECOMMIT_SHA = CONFIGS / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2_PRECOMMIT.sha256"
ATTACKS = ("MEntA", "MBA", "RAG-MIA")
ALPHAS = (0.01, 0.03, 0.05)

sys.path.insert(0, str(ROOT / "code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments, MIRABEL_FORMULA_VERSION  # noqa: E402


def load_v1_helpers():
    path = V1 / "code" / "run_phase1.py"
    spec = importlib.util.spec_from_file_location("frozen_dualtail_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen V1 implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def verify_precommit() -> dict:
    if not PRECOMMIT.is_file() or not PRECOMMIT_SHA.is_file():
        raise RuntimeError("precommit missing")
    expected = PRECOMMIT_SHA.read_text(encoding="utf-8").split()[0]
    if sha_file(PRECOMMIT) != expected:
        raise RuntimeError("precommit checksum mismatch")
    pre = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    if sha_file(Path(__file__)) != pre["code_sha256_at_precommit"][Path(__file__).name]:
        raise RuntimeError("replication code changed after precommit")
    if sha_file(V1 / "PHASE1_RESULT.json") != pre["old_v1_result_sha256"]:
        raise RuntimeError("old V1 result changed")
    if sha_file(V1 / "code" / "run_phase1.py") != pre["detector_freeze"]["v1_implementation_sha256"]:
        raise RuntimeError("frozen V1 detector implementation changed")
    if sha_file(INPUTS / "FRESH_SHARED_TARGETS.csv") != pre["fresh_targets"]["sha256"]:
        raise RuntimeError("fresh targets changed")
    if sha_file(INPUTS / "FRESH_BENIGN_1000.jsonl") != pre["fresh_benign"]["sha256"]:
        raise RuntimeError("fresh benign changed")
    return pre


def query_input_audit(pre: dict) -> tuple[list[dict], dict]:
    paths = {
        "MEntA": INPUTS / "FRESH_MENTA_ATTACK_QUERIES.jsonl",
        "MBA": INPUTS / "FRESH_MBA_ATTACK_QUERIES.jsonl",
        "RAG-MIA": INPUTS / "FRESH_RAG_MIA_ATTACK_QUERIES.jsonl",
    }
    if not all(p.is_file() for p in paths.values()):
        raise RuntimeError("fresh attack input missing")
    targets = list(csv.DictReader((INPUTS / "FRESH_SHARED_TARGETS.csv").open(encoding="utf-8")))
    target_label = {r["document_id"]: r["membership"] for r in targets}
    old_paths = {
        "MEntA": PARENT / "inputs" / "MENTA_ATTACK_QUERIES.jsonl",
        "MBA": PARENT / "inputs" / "MBA_ATTACK_QUERIES.jsonl",
        "RAG-MIA": PARENT / "inputs" / "RAG_MIA_ATTACK_QUERIES.jsonl",
    }
    rows = []
    details = {}
    expected_counts = {"MEntA": 1000, "MBA": 200, "RAG-MIA": 200}
    for attack in ATTACKS:
        current = read_jsonl(paths[attack])
        old = read_jsonl(old_paths[attack])
        current_hashes = {sha_text(r["query"].strip().lower()) for r in current}
        old_hashes = {sha_text(r["query"].strip().lower()) for r in old}
        target_set = {r["target_id"] for r in current}
        labels_ok = all(target_label.get(r["target_id"]) == r["membership"] for r in current)
        q5_ok = True
        if attack == "MEntA":
            by_session: dict[str, list[int]] = defaultdict(list)
            for r in current:
                by_session[r["session_id"]].append(int(r["query_index"]))
            q5_ok = len(by_session) == 200 and all(sorted(v) == [1, 2, 3, 4, 5] for v in by_session.values())
        details[attack] = {
            "queries": len(current), "expected_queries": expected_counts[attack],
            "targets": len(target_set), "expected_targets": 200,
            "labels_exact": labels_ok, "old_exact_query_text_overlap": len(current_hashes & old_hashes),
            "q5_structure_valid": q5_ok,
            "sha256": sha_file(paths[attack]),
        }
        if len(current) != expected_counts[attack] or len(target_set) != 200 or not labels_ok or not q5_ok or current_hashes & old_hashes:
            write_json(AUDITS / "FRESH_ATTACK_INPUT_AUDIT.json", {"verdict": "FRESH_ATTACK_INPUT_AUDIT_FAILED", "details": details})
            raise RuntimeError(f"fresh attack input audit failed: {attack}")
        for r in current:
            rows.append({**r, "cohort": "ATTACK", "split": "FRESH_REPLICATION", "attack": attack})
    audit = {"verdict": "FRESH_ATTACK_INPUT_AUDIT_PASS", "details": details, "queries": len(rows), "unique_query_ids": len({r["query_id"] for r in rows})}
    if audit["queries"] != 1400 or audit["unique_query_ids"] != 1400:
        raise RuntimeError("attack query IDs are not globally unique")
    write_json(AUDITS / "FRESH_ATTACK_INPUT_AUDIT.json", audit)
    return rows, audit


def retrieve(query_rows: list[dict], pre: dict) -> list[dict]:
    docs = read_jsonl(PARENT / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")
    if len(docs) != 3000 or sha_file(PARENT / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl") != pre["protected_db"]["sha256"]:
        raise RuntimeError("protected DB mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for frozen BGE-M3 retrieval")
    model = SentenceTransformer(str(RETRIEVER), device="cuda")
    model.max_seq_length = 512
    doc_embeddings = np.asarray(model.encode(
        [r["source_text"] for r in docs], batch_size=16, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ), dtype=np.float32)
    query_embeddings = np.asarray(model.encode(
        [r["query"] for r in query_rows], batch_size=16, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ), dtype=np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "CORPUS_EMBEDDINGS.float16.npy", doc_embeddings.astype(np.float16))
    np.save(CACHE / "QUERY_EMBEDDINGS.float16.npy", query_embeddings.astype(np.float16))
    doc_ids = [r["document_id"] for r in docs]
    output = []
    for start in range(0, len(query_rows), 128):
        scores_batch = query_embeddings[start:start + 128] @ doc_embeddings.T
        for offset, scores in enumerate(scores_batch):
            source = query_rows[start + offset]
            order = np.argsort(-scores, kind="stable")
            top = order[:4]
            top_ids = [doc_ids[int(i)] for i in top]
            top_scores = [float(scores[int(i)]) for i in top]
            stats = canonical_mirabel_from_moments(
                top1=top_scores[0], sum_all=float(scores.sum(dtype=np.float64)),
                sumsq_all=float(np.square(scores, dtype=np.float64).sum(dtype=np.float64)),
                corpus_size=len(doc_ids), confidence=0.95,
            )
            target_id = source.get("target_id")
            target_rank = top_ids.index(target_id) + 1 if target_id in top_ids else 0
            output.append({
                "query_id": source["query_id"], "session_id": source.get("session_id", source["query_id"]),
                "query_index": int(source.get("query_index", 1)), "cohort": source["cohort"], "split": source["split"],
                "attack": source.get("attack"), "domain": source["domain"], "membership": source.get("membership"),
                "target_id": target_id, "query": source["query"], "top_document_ids": top_ids, "top_scores": top_scores,
                "target_rank": target_rank, "target_retrieved_at1": target_rank == 1, "target_retrieved_at4": target_rank > 0,
                "selected_source_id": top_ids[0], "mirabel_margin": float(stats.margin), "mirabel_threshold": float(stats.threshold),
                "mirabel_background_mean": float(stats.background_mean), "mirabel_background_std": float(stats.background_std),
                "mirabel_corpus_n": int(stats.corpus_size), "mirabel_formula_version": MIRABEL_FORMULA_VERSION,
            })
        write_json(EXP / "HEARTBEAT.json", {"stage": "FRESH_RETRIEVAL", "completed_queries": min(start+128, len(query_rows)), "total_queries": len(query_rows), "updated_utc": utcnow()})
    return output


def main() -> None:
    started = time.monotonic()
    pre = verify_precommit()
    helpers = load_v1_helpers()
    attack_rows, attack_audit = query_input_audit(pre)
    benign_rows = [{**r, "cohort": "BENIGN", "split": "FRESH_HOLDOUT", "attack": None, "membership": None, "target_id": None, "session_id": r["query_id"], "query_index": 1} for r in read_jsonl(INPUTS / "FRESH_BENIGN_1000.jsonl")]
    query_rows = benign_rows + attack_rows
    if len(query_rows) != 2400 or len({r["query_id"] for r in query_rows}) != 2400:
        raise RuntimeError("combined query count/uniqueness mismatch")
    rows = retrieve(query_rows, pre)

    v1_scores = list(csv.DictReader((V1 / "tables" / "SCORE_ROWS.csv").open(encoding="utf-8")))
    tail = [r for r in v1_scores if r["calibration_partition"] == "TAIL_REFERENCE"]
    if len(tail) != 250 or {r["query_id"] for r in tail} != set(pre["detector_freeze"]["v1_tail_reference_ids"]):
        raise RuntimeError("frozen V1 tail reference mismatch")
    ref_m = sorted(float(r["M"]) for r in tail)
    ref_c = sorted(float(r["C"]) for r in tail)
    for row in rows:
        scores = row["top_scores"]
        row["M"] = float(row["mirabel_margin"])
        row["C"] = scores[0] - statistics.fmean(scores[1:4])
        row["p_M"] = helpers.empirical_upper_tail(ref_m, row["M"])
        row["p_C"] = helpers.empirical_upper_tail(ref_c, row["C"])
        row["S"] = max(-math.log(row["p_M"]), -math.log(row["p_C"]))

    cache_path = CACHE / "FRESH_REPLICATION_RETRIEVAL_SCORES.jsonl"
    with cache_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(CACHE / "FRESH_REPLICATION_CACHE_MANIFEST.json", {"rows": len(rows), "benign": 1000, "attack": 1400, "sha256": sha_file(cache_path), "created_utc": utcnow()})

    benign = [r for r in rows if r["cohort"] == "BENIGN"]
    negatives = {"M": [r["M"] for r in benign], "S": [r["S"] for r in benign]}
    matched = []
    tpr3 = {}
    for attack in ATTACKS:
        member = [r for r in rows if r.get("attack") == attack and r.get("membership") == "member"]
        for alpha in ALPHAS:
            mt = helpers.tpr_at_fpr([r["M"] for r in member], negatives["M"], alpha)
            st = helpers.tpr_at_fpr([r["S"] for r in member], negatives["S"], alpha)
            matched.append({"attack": attack, "positive": "member_attack_query", "member_query_n": len(member), "benign_query_n": len(benign), "target_query_fpr": alpha, "interpolation": "LINEAR_INTERPOLATED_EMPIRICAL_ROC", "mirabel_tpr": mt, "dualtail_tpr": st, "delta": st-mt})
            if alpha == 0.03:
                tpr3[attack] = {"M": mt, "S": st, "delta": st-mt}

    binary_thresholds = {}
    for score in ("M", "S"):
        threshold, alarms = helpers.matched_binary_threshold(negatives[score], 0.03)
        binary_thresholds[score] = {"threshold": threshold, "benign_alarms": alarms, "benign_fpr": alarms/len(benign)}

    nonmember_rows = []
    for attack in ATTACKS:
        subset = [r for r in rows if r.get("attack") == attack and r.get("membership") == "nonmember"]
        nonmember_rows.append({
            "attack": attack, "queries": len(subset),
            "mirabel_alarm_rate_matched_3pct": statistics.fmean(r["M"] > binary_thresholds["M"]["threshold"] for r in subset),
            "dualtail_alarm_rate_matched_3pct": statistics.fmean(r["S"] > binary_thresholds["S"]["threshold"] for r in subset),
        })

    frozen_thresholds = pre["detector_freeze"]["v1_frozen_deployment_thresholds"]
    fpr_rows = []
    for score, detector in (("M", "MIRABEL"), ("S", "DUALTAIL")):
        for alpha in ALPHAS:
            threshold = float(frozen_thresholds[score][str(alpha)])
            for domain in ("ALL", "nfcorpus", "scidocs", "trec-covid"):
                subset = benign if domain == "ALL" else [r for r in benign if r["domain"] == domain]
                fp = sum(r[score] > threshold for r in subset)
                lo, hi = helpers.wilson(fp, len(subset))
                fpr_rows.append({"detector": detector, "nominal_fpr": alpha, "frozen_threshold": threshold, "domain": domain, "n": len(subset), "false_positives": fp, "measured_fpr": fp/len(subset) if subset else None, "wilson95_low": lo, "wilson95_high": hi, "status": "NO_FRESH_QUERY_AVAILABLE" if not subset else "MEASURED"})

    correlation_rows = []
    for name, subset in [("FRESH_BENIGN", benign)] + [(a, [r for r in rows if r.get("attack") == a]) for a in ATTACKS]:
        correlation_rows.append({"group": name, "n": len(subset), "spearman_rho_M_C": helpers.spearman([r["M"] for r in subset], [r["C"] for r in subset])})

    bootstrap_rows = []
    for idx, attack in enumerate(ATTACKS):
        subset = [r for r in rows if r.get("attack") == attack and r.get("membership") == "member"]
        mean_delta, low, high = helpers.bootstrap_delta(subset, negatives["M"], negatives["S"], 2000, 20260912+idx)
        bootstrap_rows.append({"attack": attack, "unit": "target/session", "iterations": 2000, "point_delta": tpr3[attack]["delta"], "bootstrap_mean_delta": mean_delta, "ci95_low": low, "ci95_high": high})

    menta_member = [r for r in rows if r.get("attack") == "MEntA" and r.get("membership") == "member"]
    recovery = []
    categories = Counter()
    recovery_by_target = Counter()
    lost_by_target = Counter()
    for r in menta_member:
        ma = r["M"] > binary_thresholds["M"]["threshold"]
        sa = r["S"] > binary_thresholds["S"]["threshold"]
        category = "BOTH_HIT" if ma and sa else "MIRABEL_MISS_DUALTAIL_HIT" if not ma and sa else "MIRABEL_HIT_DUALTAIL_MISS" if ma and not sa else "BOTH_MISS"
        categories[category] += 1
        if category == "MIRABEL_MISS_DUALTAIL_HIT": recovery_by_target[r["session_id"]] += 1
        if category == "MIRABEL_HIT_DUALTAIL_MISS": lost_by_target[r["session_id"]] += 1
        recovery.append({"category": category, "session_id": r["session_id"], "query_id": r["query_id"], "query_index": r["query_index"], "target_id": r["target_id"], "target_rank": r["target_rank"], "M": r["M"], "C": r["C"], "p_M": r["p_M"], "p_C": r["p_C"], "S": r["S"], "s1": r["top_scores"][0], "s2": r["top_scores"][1], "s3": r["top_scores"][2], "s4": r["top_scores"][3]})
    total_recovered = sum(recovery_by_target.values())
    largest = max(recovery_by_target.values(), default=0)
    concentration = largest/total_recovered if total_recovered else 0.0
    recovery_summary = {"categories": dict(categories), "recovered_queries": total_recovered, "recovered_targets": len(recovery_by_target), "lost_queries": sum(lost_by_target.values()), "lost_targets": len(lost_by_target), "largest_single_target_gain": largest, "largest_target_gain_fraction": concentration, "warning": "TARGET_CONCENTRATED_GAIN" if concentration >= 0.5 else None}

    sessions = []
    for membership in ("member", "nonmember"):
        by_session: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r.get("attack") == "MEntA" and r.get("membership") == membership:
                by_session[r["session_id"]].append(r)
        for sid in sorted(by_session):
            rr = sorted(by_session[sid], key=lambda x: x["query_index"])
            ma = [r["M"] > binary_thresholds["M"]["threshold"] for r in rr]
            sa = [r["S"] > binary_thresholds["S"]["threshold"] for r in rr]
            mr = [a and r["target_rank"] == 1 for a, r in zip(ma, rr)]
            sr = [a and r["target_rank"] == 1 for a, r in zip(sa, rr)]
            sessions.append({"session_id": sid, "membership": membership, "mirabel_alarms": sum(ma), "dualtail_alarms": sum(sa), "mirabel_target_removed": sum(mr), "dualtail_target_removed": sum(sr), "mirabel_any_alarm": any(ma), "dualtail_any_alarm": any(sa), "mirabel_all5_alarm": all(ma), "dualtail_all5_alarm": all(sa), "mirabel_any_target_removed": any(mr), "dualtail_any_target_removed": any(sr), "mirabel_all5_target_removed": all(mr), "dualtail_all5_target_removed": all(sr)})

    macro_m = statistics.fmean(tpr3[a]["M"] for a in ATTACKS)
    macro_s = statistics.fmean(tpr3[a]["S"] for a in ATTACKS)
    menta_boot = next(r for r in bootstrap_rows if r["attack"] == "MEntA")
    checks = {
        "menta_delta_at_least_5pp": tpr3["MEntA"]["delta"] >= 0.05-1e-12,
        "menta_cluster_bootstrap_ci_low_above_zero": menta_boot["ci95_low"] > 0,
        "mba_noninferior_within_5pp": tpr3["MBA"]["delta"] >= -0.05-1e-12,
        "rag_mia_noninferior_within_5pp": tpr3["RAG-MIA"]["delta"] >= -0.05-1e-12,
        "core3_macro_member_tpr_improved": macro_s > macro_m+1e-12,
    }
    passed = all(checks.values())
    verdict = "DUALTAIL_MEMBER_EXPOSURE_PHASE1_PASS" if passed else "DUALTAIL_MEMBER_EXPOSURE_NOT_REPLICATED"

    write_csv(TABLES / "MATCHED_MEMBER_QUERY_TPR.csv", matched)
    write_csv(TABLES / "FRESH_BENIGN_FPR_TRANSFER.csv", fpr_rows)
    write_csv(TABLES / "NONMEMBER_ATTACK_ALARM_RATE.csv", nonmember_rows)
    write_csv(TABLES / "PAIRED_CLUSTER_BOOTSTRAP.csv", bootstrap_rows)
    write_csv(TABLES / "M_C_CORRELATION.csv", correlation_rows)
    write_csv(TABLES / "MENTA_MEMBER_RECOVERY_CASES.csv", recovery)
    write_csv(TABLES / "MENTA_SESSION_COVERAGE.csv", sessions)
    retrieval_rows = []
    for attack in ATTACKS:
        for membership in ("member", "nonmember"):
            subset = [r for r in rows if r.get("attack") == attack and r.get("membership") == membership]
            retrieval_rows.append({"attack": attack, "membership": membership, "queries": len(subset), "target_retrieval_at1": statistics.fmean(r["target_retrieved_at1"] for r in subset), "target_retrieval_at4": statistics.fmean(r["target_retrieved_at4"] for r in subset)})
    write_csv(TABLES / "TARGET_RETRIEVAL.csv", retrieval_rows)

    result = {
        "verdict": verdict, "completed_utc": utcnow(), "runtime_seconds": time.monotonic()-started,
        "precommit_sha256": sha_file(PRECOMMIT), "old_v1_verdict_preserved": "BC_DUALTAIL_V1_NO_TPR_GAIN",
        "fresh_cohort": {"member_targets": 100, "nonmember_targets": 100, "benign_queries": 1000, "attack_queries": 1400},
        "attack_input_audit": attack_audit, "matched_member_tpr_at_3pct": tpr3,
        "macro": {"MIRABEL": macro_m, "DUALTAIL": macro_s, "delta": macro_s-macro_m},
        "bootstrap": bootstrap_rows, "checks": checks, "binary_matched_thresholds": binary_thresholds,
        "recovery": recovery_summary, "nonmember_alarm": nonmember_rows, "correlations": correlation_rows,
        "phase2_allowed": passed, "next_step": "PHASE2_E2E" if passed else "ORTHOGONAL_DETECTOR_SIGNAL_DESIGN",
    }
    write_json(EXP / "PHASE1_RESULT.json", result)
    write_json(EXP / "checkpoints" / f"{verdict}.json", {"stage": verdict, "created_utc": utcnow(), "precommit_sha256": sha_file(PRECOMMIT)})
    report = ["# DualTail Member Exposure Replication V2 — Phase 1", "", f"- 판정: `{verdict}`", "- 기존 V1 판정: `BC_DUALTAIL_V1_NO_TPR_GAIN` (변경 없음)", "", "| Attack | MIRABEL member TPR@3% | DualTail | Delta |", "|---|---:|---:|---:|"]
    for attack in ATTACKS: report.append(f"| {attack} | {tpr3[attack]['M']:.3f} | {tpr3[attack]['S']:.3f} | {tpr3[attack]['delta']:+.3f} |")
    report += [f"| Macro | {macro_m:.3f} | {macro_s:.3f} | {macro_s-macro_m:+.3f} |", "", "## Gate"]
    report.extend(f"- {k}: `{'PASS' if v else 'FAIL'}`" for k,v in checks.items())
    report += ["", "## Recovery", f"- recovered queries/targets: {total_recovered}/{len(recovery_by_target)}", f"- lost queries/targets: {sum(lost_by_target.values())}/{len(lost_by_target)}", f"- largest target fraction: {concentration:.3f}", "", f"Phase 2 generation allowed: `{passed}`"]
    (REPORTS / "PHASE1_REPORT_KO.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    (EXP / "STATUS.md").write_text(f"# DualTail Member Exposure Replication V2\n\n- 상태: `{verdict}`\n- Phase 2 허용: `{passed}`\n- 다음 단계: `{result['next_step']}`\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
