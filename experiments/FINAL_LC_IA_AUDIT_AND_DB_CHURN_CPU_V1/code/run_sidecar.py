#!/usr/bin/env python3
"""CPU-only sidecar. Reads IA-v6 through a read-only SQLite connection."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import resource
import sqlite3
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "FINAL_LC_IA_AUDIT_AND_DB_CHURN_CPU_V1"
V6 = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1" / "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED"
OLD = ROOT / "experiments" / "FINAL_LC_CPU_ANALYSIS_SIDECAR_V1"
CLEAN = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
RECAL = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_text(value: str):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows):
    rows = list(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(value: str):
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def words(value: str):
    return re.findall(r"[^\W_]+", normalized(value).casefold(), flags=re.UNICODE)


def common_prefix_ratio(left: str, right: str):
    left, right = normalized(left), normalized(right)
    count = 0
    for a, b in zip(left, right):
        if a != b: break
        count += 1
    return count / max(1, min(len(left), len(right)))


def wilson(successes, total, z=1.959963984540054):
    if not total: return [None, None]
    p = successes / total; denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def verify_precommit():
    path = EXP / "configs" / "SIDECAR_PRECOMMIT.json"
    expected = (EXP / "configs" / "SIDECAR_PRECOMMIT.sha256").read_text().split()[0]
    if sha_file(path) != expected: raise RuntimeError("sidecar precommit drift")
    precommit = json.loads(path.read_text())
    for name, digest in precommit["input_sha256"].items():
        if sha_file(Path(name)) != digest: raise RuntimeError(f"input drift: {name}")
    for name, digest in precommit["code_sha256"].items():
        if sha_file(ROOT / name) != digest: raise RuntimeError(f"code drift: {name}")
    return precommit


def checkpoint(stage, **fields):
    payload = {"campaign": EXP.name, "stage": stage, "updated_utc": now(), "gpu_used": False, **fields}
    write_json(EXP / "STATUS.json", payload)
    lines = [f"# {EXP.name}", "", f"- Stage: `{stage}`", f"- Updated UTC: `{payload['updated_utc']}`",
             "- GPU used: `False`"] + [f"- {key}: `{value}`" for key, value in fields.items()]
    temporary = EXP / "STATUS.md.tmp"; temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(EXP / "STATUS.md")


def ia_scope(precommit):
    heartbeat = json.loads((V6 / "HEARTBEAT.json").read_text()) if (V6 / "HEARTBEAT.json").exists() else {}
    ids = list(precommit["cohort"]["ordered_ids"])
    if "PREFLIGHT" in heartbeat.get("stage", "") or not (V6 / "IA_V6_PREFLIGHT_RESULT.json").exists():
        ids = list(precommit["preflight"]["ids"]); phase = "PREFLIGHT"
    else:
        phase = "FULL"
    return phase, ids, heartbeat


def read_ia_tables():
    database = V6 / "runtime" / "ia_v6.sqlite3"
    if not database.exists(): return [], []
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=15)
    questions = list(connection.execute(
        "SELECT target_id,membership,slot,question,generated_token_ids_json,valid,reason FROM question_slot"))
    judgments = list(connection.execute(
        "SELECT target_id,membership,slot,judgment,valid,reason FROM judgment_slot"))
    connection.close()
    return questions, judgments


def audit_ia(v6_precommit):
    phase, scope_ids, heartbeat = ia_scope(v6_precommit)
    scope = set(scope_ids)
    questions, judgments = read_ia_tables()
    questions = [row for row in questions if row[0] in scope]
    judgments = [row for row in judgments if row[0] in scope]
    q_by = defaultdict(list); j_by = defaultdict(list)
    for row in questions: q_by[row[0]].append(row)
    for row in judgments: j_by[row[0]].append(row)
    targets = {row["document_id"]: row["membership"] for row in csv.DictReader(
        Path(v6_precommit["cohort"]["path"]).open(encoding="utf-8"))}

    validity, pair_rows = [], []
    suspicious_sessions = set()
    for target_id in scope_ids:
        qs = sorted(q_by[target_id], key=lambda row: row[2]); js = sorted(j_by[target_id], key=lambda row: row[2])
        q_slots = [row[2] for row in qs]; j_slots = [row[2] for row in js]
        q_text = [row[3] for row in qs]; q_tokens = [tuple(json.loads(row[4])) for row in qs]
        normalized_duplicate = len(q_text) - len(set(map(normalized, q_text)))
        token_duplicate = len(q_tokens) - len(set(q_tokens))
        malformed = sum(not row[5] or not row[3].strip() for row in qs)
        forbidden = sum((not row[4]) or row[3] not in ("Yes", "No", "Unknown") for row in js)
        valid = (q_slots == list(range(1, 16)) and j_slots == list(range(1, 16))
                 and malformed == 0 and forbidden == 0 and normalized_duplicate == 0 and token_duplicate == 0)
        validity.append({"target_id": target_id, "membership": targets[target_id], "questions": len(qs),
                         "judgments": len(js), "exact_q15": q_slots == list(range(1, 16)),
                         "exact_j15": j_slots == list(range(1, 16)), "token_exact_duplicates": token_duplicate,
                         "normalized_exact_duplicates": normalized_duplicate, "malformed": malformed,
                         "forbidden_judgments": forbidden, "complete_valid": valid})
        for left_index in range(len(q_text)):
            for right_index in range(left_index + 1, len(q_text)):
                left, right = q_text[left_index], q_text[right_index]
                lt, rt = set(words(left)), set(words(right)); union = lt | rt
                jaccard = len(lt & rt) / max(1, len(union))
                character = SequenceMatcher(None, normalized(left), normalized(right), autojunk=False).ratio()
                prefix = common_prefix_ratio(left, right)
                unique_ratio = len(lt ^ rt) / max(1, len(union))
                suspicious = (jaccard >= 0.85 and character >= 0.90) or prefix >= 0.90
                if suspicious: suspicious_sessions.add(target_id)
                pair_rows.append({"target_id": target_id, "membership": targets[target_id],
                                  "left_slot": left_index + 1, "right_slot": right_index + 1,
                                  "token_jaccard": jaccard, "character_similarity": character,
                                  "common_prefix_ratio": prefix, "unique_token_ratio": unique_ratio,
                                  "suspicious": suspicious})

    slot_rows = []
    for slot in range(1, 16):
        values = [row[3] for row in questions if row[2] == slot]
        labels = [row[3] for row in judgments if row[2] == slot and row[4]]
        tokens = [token for value in values for token in words(value)]
        starts = Counter(" ".join(words(value)[:3]) for value in values if words(value))
        slot_rows.append({"slot": slot, "generated_questions": len(values),
                          "mean_question_words": statistics.fmean(len(words(value)) for value in values) if values else None,
                          "lexical_type_token_ratio": len(set(tokens)) / len(tokens) if tokens else None,
                          "unique_question_rate": len(set(map(normalized, values))) / len(values) if values else None,
                          "dominant_starting_phrase_rate": max(starts.values()) / len(values) if values else None,
                          "judgments": len(labels), "yes": labels.count("Yes"), "no": labels.count("No"),
                          "unknown": labels.count("Unknown")})

    complete = [row for row in validity if row["questions"] == 15 and row["judgments"] == 15]
    valid_complete = [row for row in complete if row["complete_valid"]]
    valid_member = [row for row in valid_complete if row["membership"] == "member"]
    valid_nonmember = [row for row in valid_complete if row["membership"] == "nonmember"]
    complete_member = [row for row in complete if row["membership"] == "member"]
    complete_nonmember = [row for row in complete if row["membership"] == "nonmember"]
    member_rate = len(valid_member) / len(complete_member) if complete_member else None
    nonmember_rate = len(valid_nonmember) / len(complete_nonmember) if complete_nonmember else None
    invalid_known = sum(row["malformed"] or row["forbidden_judgments"] or row["token_exact_duplicates"] or row["normalized_exact_duplicates"] for row in validity)
    max_invalid = 2 if phase == "PREFLIGHT" else 100
    status = {
        "phase": phase, "scope_sessions": len(scope_ids), "question_rows": len(questions),
        "judgment_rows": len(judgments), "complete_sessions": len(complete),
        "valid_complete_sessions": len(valid_complete),
        "valid_percent_among_complete": len(valid_complete) / len(complete) if complete else None,
        "member_valid_percent_among_complete": member_rate,
        "nonmember_valid_percent_among_complete": nonmember_rate,
        "validity_gap": abs(member_rate - nonmember_rate) if member_rate is not None and nonmember_rate is not None else None,
        "known_invalid_sessions": invalid_known,
        "mathematically_possible": invalid_known <= max_invalid,
        "mathematical_invalid_limit": max_invalid,
        "suspicious_near_duplicate_sessions": len(suspicious_sessions),
        "sessions_with_at_least_two_questions": sum(len(q_by[target]) >= 2 for target in scope_ids),
        "runner_heartbeat": heartbeat, "performance_metrics_computed": False,
    }
    write_csv(EXP / "tables" / "IA_STREAMING_VALIDITY.csv", validity)
    write_csv(EXP / "tables" / "IA_PAIR_NEAR_DUPLICATE_DIAGNOSTIC.csv", pair_rows)
    write_csv(EXP / "tables" / "IA_SLOT_DIVERSITY.csv", slot_rows)
    write_json(EXP / "audits" / "IA_STREAMING_STATUS.json", status)
    return status


def audit_churn():
    base_rows = read_jsonl(CLEAN / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")
    metadata = {row["document_id"]: row for row in base_rows}
    with gzip.open(CLEAN / "inputs" / "COMMON_ELIGIBLE_POOL.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle): metadata.setdefault(row["document_id"], row)
    embedding = np.load(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy", mmap_mode="r")
    embedded_ids = {row["document_id"] for row in base_rows}
    audit_rows = []
    affected_missing = False
    for version in ("V0", "V10", "V25", "V50"):
        path = OLD / "manifests" / f"DB_CHURN_{version}.json"; manifest = json.loads(path.read_text())
        ids = list(manifest["ordered_document_ids"])
        missing_metadata = [item for item in ids if item not in metadata]
        hashes = [metadata[item]["normalized_text_hash"] for item in ids if item in metadata]
        missing_embeddings = [item for item in ids if item not in embedded_ids]
        affected_missing |= bool(missing_embeddings)
        audit_rows.append({"version": version, "documents": len(ids), "removed": manifest["removed_count"],
                           "added": manifest["added_count"], "member_targets_retained": manifest["membership_audit"]["all_member_targets_present"],
                           "nonmember_targets_present": len(manifest["membership_audit"]["nonmember_targets_present"]),
                           "normalized_duplicate_leakage": len(hashes) - len(set(hashes)),
                           "missing_metadata": len(missing_metadata), "available_frozen_embeddings": len(ids) - len(missing_embeddings),
                           "missing_frozen_embeddings": len(missing_embeddings),
                           "ordered_id_hash_match": sha_text("\n".join(ids)) == manifest["ordered_document_id_sha256"],
                           "ordered_text_hash_match": sha_text("\n".join(hashes)) == manifest["ordered_normalized_text_hash_sha256"],
                           "manifest_sha256": sha_file(path)})
    index_status = {"frozen_embedding_file": str(LC / "cache" / "CORPUS_EMBEDDINGS.float16.npy"),
                    "embedding_shape": list(embedding.shape), "faiss_available": False,
                    "verdict": "DB_CHURN_GPU_EMBEDDING_REQUIRED" if affected_missing else "ALL_EMBEDDINGS_AVAILABLE",
                    "gpu_embedding_run": False, "indices_built": 0,
                    "reason": "Churn-added document embeddings are absent from the frozen 3,000-document embedding cache; affected indices and score-only metrics are not constructed."}
    try:
        import faiss  # noqa: F401
        index_status["faiss_available"] = True
    except ImportError:
        pass
    write_csv(EXP / "tables" / "DB_CHURN_MANIFEST_AUDIT.csv", audit_rows)
    write_json(EXP / "audits" / "DB_CHURN_INDEX_STATUS.json", index_status)

    frozen_detection = list(csv.DictReader((OLD / "tables" / "CORE5_DETECTION_STATISTICS.csv").open()))
    frozen_fpr = list(csv.DictReader((OLD / "tables" / "BENIGN_DOMAIN_FPR_WILSON.csv").open()))
    lc_all = next(row for row in frozen_fpr if row["method"] == "Final LC" and row["domain"] == "ALL")
    churn_rows = []
    for version in ("V0", "V10", "V25", "V50"):
        for method, calibration in (("Original MIRABEL", "canonical"), ("Final LC", "strict-transfer"), ("Final LC", "benign-refresh")):
            base = version == "V0" and method == "Final LC"
            churn_rows.append({"db": version, "method": method, "calibration": calibration,
                               "benign_fpr": float(lc_all["fpr"]) if base else None,
                               "worst_domain_fpr": float(next(row for row in frozen_fpr if row["method"] == "Final LC" and row["domain"] == "WORST_DOMAIN")["fpr"]) if base else None,
                               **{attack: float(next(row for row in frozen_detection if row["attack"] == attack)["final_lc_tpr"]) if base else None
                                  for attack in ("MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2")},
                               "status": "FROZEN_V0_BASELINE_REUSED" if base else ("DB_CHURN_GPU_EMBEDDING_REQUIRED" if version != "V0" else "NOT_RECOMPUTED_NO_NEW_INDEX")})
    write_csv(EXP / "tables" / "DB_CHURN_SCORE_ONLY.csv", churn_rows)
    return audit_rows, index_status


def calibration_sensitivity():
    calibration = read_jsonl(RECAL / "inputs" / "TOPIOCQA_BENIGN_RECALIBRATION_500.jsonl")
    ordered = sorted(calibration, key=lambda row: (hashlib.sha256(row["query_id"].encode()).hexdigest(), row["query_id"]))
    query_hashes = {}
    for size in (100, 200, 300, 500):
        query_hashes[size] = sha_text("\n".join(row["query_id"] for row in ordered[:size]))
    result = json.loads((RECAL / "GOLD_RECALIBRATION_RESULT.json").read_text())
    thresholds = list(csv.DictReader((RECAL / "tables" / "GOLD_REFRESH_THRESHOLDS.csv").open()))
    row500 = next(row for row in thresholds if row["method"] == "FINAL_LC_REFRESH" and float(row["budget"]) == 0.025)
    rows = []
    for size in (100, 200, 300, 500):
        available = size == 500
        rows.append({"calibration_n": size, "subset_query_id_sha256": query_hashes[size],
                     "threshold": float(row500["threshold"]) if available else None,
                     "score_quantile": "existing frozen 2.5%-budget operating point" if available else None,
                     "locked_test_fpr": result["locked_intervention"] if available else None,
                     "wilson_low": wilson(round(result["locked_intervention"] * 1000), 1000)[0] if available else None,
                     "wilson_high": wilson(round(result["locked_intervention"] * 1000), 1000)[1] if available else None,
                     "core5_mean_tpr": None, "menta_tpr": None, "rag_mia_tpr": None,
                     "status": "FROZEN_N500_RESULT_REUSED" if available else "CALIBRATION_RAW_MARGIN_CACHE_MISSING"})
    status = {"verdict": "CALIBRATION_SIZE_SENSITIVITY_INPUT_INSUFFICIENT",
              "available": [500], "unavailable": [100, 200, 300],
              "missing": "per-query raw MIRABEL margins for the 500 calibration queries (and frozen Gold corpus document embeddings)",
              "new_bge_forward_run": False, "gpu_used": False,
              "note": "N=500 reports the frozen result only; Core5 scores were calibrated against a different benign reference and are not mixed."}
    write_csv(EXP / "tables" / "CALIBRATION_SIZE_SENSITIVITY.csv", rows)
    write_json(EXP / "audits" / "CALIBRATION_SIZE_STATUS.json", status)
    return rows, status


def report(ia, churn, index_status, calibration_status, started, terminal):
    elapsed = time.monotonic() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cost = [{"metric": "sidecar_elapsed", "value": elapsed, "unit": "seconds"},
            {"metric": "peak_rss", "value": peak, "unit": "KiB"},
            {"metric": "workers", "value": 1, "unit": "process"},
            {"metric": "gpu_used", "value": False, "unit": "boolean"},
            {"metric": "answer_generations", "value": 0, "unit": "count"},
            {"metric": "trainable_updates", "value": 0, "unit": "count"}]
    write_csv(EXP / "tables" / "COST_CPU_TIME.csv", cost)
    lines = ["# Final LC IA Audit and DB Churn CPU Sidecar", "",
             f"- Updated UTC: `{now()}`", f"- IA terminal: `{terminal}`", "- GPU used: `False`",
             "- Answer generation: `0`", "- Final LC changes: `0`", "",
             "## IA STREAMING VALIDITY", "",
             f"- Phase: {ia['phase']}", f"- Questions/judgments read: {ia['question_rows']}/{ia['judgment_rows']}",
             f"- Complete valid sessions: {ia['valid_complete_sessions']}/{ia['complete_sessions']}",
             f"- Known invalid sessions: {ia['known_invalid_sessions']}",
             f"- Gate still mathematically possible: {ia['mathematically_possible']}", "",
             "## IA NEAR-DUPLICATE DIAGNOSTIC", "",
             f"- Suspicious sessions: {ia['suspicious_near_duplicate_sessions']}",
             "- Diagnostic only; it does not reject sessions or alter generation.", "",
             "## DB CHURN", "",
             f"- Index verdict: `{index_status['verdict']}`", "- V10/V25/V50 scoring was not fabricated.",
             "- Missing frozen embeddings by version: " + ", ".join(f"{row['version']}={row['missing_frozen_embeddings']}" for row in churn), "",
             "## CALIBRATION SIZE SENSITIVITY", "",
             f"- Verdict: `{calibration_status['verdict']}`", "- N=500 frozen result retained; N=100/200/300 were not approximated.", "",
             "## COST / CPU", "", f"- Elapsed: {elapsed:.1f}s", f"- Peak RSS: {peak/1024:.1f} MiB"]
    (EXP / "reports" / "FINAL_REPORT_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"campaign": EXP.name, "verdict": "COMPLETE" if terminal else "STREAMING",
              "updated_utc": now(), "ia": ia, "db_churn": index_status,
              "calibration_sensitivity": calibration_status, "gpu_used": False,
              "answer_generations": 0, "final_lc_modified": False, "runtime_seconds": elapsed}
    write_json(EXP / ("FINAL_RESULT.json" if terminal else "INTERIM_RESULT.json"), result)
    checkpoint("COMPLETE" if terminal else "IA_STREAMING", ia_phase=ia["phase"],
               ia_complete=ia["complete_sessions"], ia_valid=ia["valid_complete_sessions"],
               churn=index_status["verdict"], calibration=calibration_status["verdict"])


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
    precommit = verify_precommit(); started = time.monotonic()
    v6_precommit = json.loads((V6 / "configs" / "IA_STD_Q15_V6_PRECOMMIT.json").read_text())
    checkpoint("STATIC_AUDITS")
    churn, index_status = audit_churn()
    _, calibration_status = calibration_sensitivity()
    while True:
        ia = audit_ia(v6_precommit)
        terminal = (V6 / "FINAL_RESULT.json").exists()
        report(ia, churn, index_status, calibration_status, started, terminal)
        if terminal: break
        time.sleep(int(precommit["resource_policy"]["stream_interval_seconds"]))


if __name__ == "__main__":
    main()
