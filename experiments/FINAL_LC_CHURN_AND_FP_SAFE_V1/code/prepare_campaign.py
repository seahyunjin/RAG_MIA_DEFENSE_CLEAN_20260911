#!/usr/bin/env python3
"""Freeze DB-churn V2 inputs, duplicate exception, and evaluation lineage."""
from __future__ import annotations

import csv
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from common import (ATTACKS, BGE, CLEAN, CORE6, DOMAINS, EXP, FINAL8, K_LOCAL,
                    LC, OLD_SIDECAR, OUTER_BUDGET, QWEN, RAW, RECAL,
                    ROOT, STRICT_THRESHOLD, V6, VERSIONS, atomic_json, atomic_text,
                    checkpoint, now, read_jsonl, sha_file, sha_text, write_csv,
                    write_jsonl)


def normalized(value: str) -> str:
    # Match CLEAN_CORE3_DEV_V1/prepare_substrate.py byte-for-byte.  In
    # particular, use lower() rather than casefold() because SciDocs contains
    # OCR-like Unicode for which those operations are not interchangeable.
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def raw_documents(required: set[str]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    by_domain: dict[str, set[str]] = defaultdict(set)
    for document_id in required:
        prefix, local = document_id.split("::", 1)
        by_domain[prefix.removeprefix("BeIR_")].add(local)
    for domain, local_ids in by_domain.items():
        source = RAW / domain / "corpus.jsonl"
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                local = str(row.get("_id") or row.get("id"))
                if local not in local_ids:
                    continue
                title = str(row.get("title") or "").strip()
                text = str(row.get("text") or "").strip()
                source_text = "\n".join(item for item in (title, text) if item)
                output[f"BeIR_{domain}::{local}"] = {
                    "document_id": f"BeIR_{domain}::{local}", "local_document_id": local,
                    "domain": domain, "title": title, "text": text,
                    "source_text": source_text,
                    "normalized_text_hash": sha_text(normalized(source_text)),
                    "source_file": str(source),
                }
    missing = required - set(output)
    if missing:
        raise RuntimeError(f"raw texts missing for {len(missing)} churn IDs: {sorted(missing)[:5]}")
    return output


def prepare_churn() -> tuple[dict, list[dict]]:
    base = read_jsonl(CLEAN / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")
    base_by_id = {row["document_id"]: row for row in base}
    manifests = {
        version: json.loads((OLD_SIDECAR / f"manifests/DB_CHURN_{version}.json").read_text())
        for version in VERSIONS
    }
    union_ids = set().union(*(set(manifests[v]["ordered_document_ids"]) for v in VERSIONS))
    added_ids = union_ids - set(base_by_id)
    added = raw_documents(added_ids)
    all_rows = {**base_by_id, **added}

    version_audit = []
    duplicate_map: dict[str, list[str]] = defaultdict(list)
    for row in base:
        duplicate_map[row["normalized_text_hash"]].append(row["document_id"])
    inherited = {key: sorted(values) for key, values in duplicate_map.items() if len(values) > 1}
    inherited_pairs = sum(len(values) - 1 for values in inherited.values())
    if inherited_pairs != 2:
        raise RuntimeError(f"expected two inherited duplicate excess rows, got {inherited_pairs}")

    for version, manifest in manifests.items():
        ids = manifest["ordered_document_ids"]
        rows = [all_rows[item] for item in ids]
        hashes = [row["normalized_text_hash"] for row in rows]
        groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            groups[row["normalized_text_hash"]].append(row["document_id"])
        duplicates = {key: sorted(values) for key, values in groups.items() if len(values) > 1}
        new_duplicate_groups = {key: values for key, values in duplicates.items() if key not in inherited}
        if sha_text("\n".join(ids)) != manifest["ordered_document_id_sha256"]:
            raise RuntimeError(f"ordered ID hash mismatch: {version}")
        if sha_text("\n".join(hashes)) != manifest["ordered_normalized_text_hash_sha256"]:
            raise RuntimeError(f"ordered text hash mismatch: {version}")
        if new_duplicate_groups:
            raise RuntimeError(f"churn-added duplicate groups: {version}: {len(new_duplicate_groups)}")
        if not manifest["membership_audit"]["all_member_targets_present"]:
            raise RuntimeError(f"member lost: {version}")
        if manifest["membership_audit"]["nonmember_targets_present"]:
            raise RuntimeError(f"nonmember influx: {version}")
        version_audit.append({
            "version": version, "documents": len(rows),
            "removed": manifest["removed_count"], "added": manifest["added_count"],
            "member_targets_retained": True, "nonmember_targets_present": 0,
            "inherited_duplicate_excess_rows": sum(len(v) - 1 for v in duplicates.values()),
            "new_duplicate_groups": len(new_duplicate_groups),
            "ordered_document_id_sha256": manifest["ordered_document_id_sha256"],
            "ordered_text_hash_sha256": manifest["ordered_normalized_text_hash_sha256"],
            "source_manifest_sha256": sha_file(OLD_SIDECAR / f"manifests/DB_CHURN_{version}.json"),
        })

    union_rows = [added[item] for item in sorted(added)]
    write_jsonl(EXP / "inputs/MISSING_EMBEDDING_UNION.jsonl", union_rows)
    atomic_json(EXP / "audits/INHERITED_BASELINE_DUPLICATES.json", {
        "classification": "INHERITED_BASELINE_DUPLICATES",
        "v0_duplicate_excess_rows": inherited_pairs,
        "groups": inherited,
        "rule": "V0 is not deduplicated; only newly introduced duplicate groups fail the campaign.",
    })
    write_csv(EXP / "audits/DB_CHURN_V2_INPUT_AUDIT.csv", version_audit)
    return manifests, union_rows


def prepare_queries() -> dict:
    large_rows = read_jsonl(LC / "cache/LARGE_DETECTION_SCORES.jsonl")
    if len(large_rows) != 16000:
        raise RuntimeError("large detection cache row drift")
    large_index = {row["query_id"]: index for index, row in enumerate(large_rows)}
    if len(large_index) != len(large_rows):
        raise RuntimeError("large query IDs not unique")

    output = []
    for index, row in enumerate(large_rows[:1000]):
        output.append({
            "query_id": row["query_id"], "query": row["query"], "domain": row["domain"],
            "split": "REFERENCE", "attack": None, "membership": None,
            "session_id": row["session_id"], "query_index": row["query_index"],
            "embedding_source": "LC_LARGE_FROZEN", "embedding_index": index,
        })

    final_rows = read_jsonl(FINAL8 / "cache/FINAL_RETRIEVAL_AND_DETECTION.jsonl")
    if len(final_rows) != 17000:
        raise RuntimeError("Final8 cache row drift")
    for row in final_rows:
        attack = row["attack"]
        source = "LC_LARGE_FROZEN" if row["query_id"] in large_index else "NEW_BGE_S2"
        if source == "NEW_BGE_S2" and attack != "S²-MIA":
            raise RuntimeError(f"unexpected query absent from frozen embedding map: {row['query_id']}")
        output.append({
            "query_id": row["query_id"], "query": row["query"], "domain": row["domain"],
            "split": "HOLDOUT" if attack == "BENIGN" else "ATTACK",
            "attack": None if attack == "BENIGN" else attack,
            "membership": row.get("membership"), "target_id": row.get("target_id"),
            "session_id": row["session_id"], "query_index": row["query_index"],
            "embedding_source": source,
            "embedding_index": large_index.get(row["query_id"]),
        })

    dcmi_rows = read_jsonl(CORE6 / "cache/CORE5_DCMI_RETRIEVAL_AND_DETECTION.jsonl")
    if len(dcmi_rows) != 4000:
        raise RuntimeError("DCMI cache row drift")
    for index, row in enumerate(dcmi_rows):
        output.append({
            "query_id": row["query_id"], "query": row["query"], "domain": row["domain"],
            "split": "ATTACK", "attack": "DCMI-Std-Q2",
            "membership": row["membership"], "target_id": row.get("target_id"),
            "session_id": row["session_id"], "query_index": row["query_index"],
            "embedding_source": "DCMI_FROZEN", "embedding_index": index,
        })
    ids = [row["query_id"] for row in output]
    if len(ids) != len(set(ids)):
        counts = Counter(ids)
        raise RuntimeError(f"query ID collision: {[x for x,n in counts.items() if n>1][:5]}")
    if Counter(row["split"] for row in output) != Counter({"REFERENCE": 1000, "HOLDOUT": 1000, "ATTACK": 20000}):
        raise RuntimeError("query split count drift")
    attack_counts = Counter(row["attack"] for row in output if row["split"] == "ATTACK")
    if set(attack_counts) != set(ATTACKS):
        raise RuntimeError(f"Core5 attack set drift: {attack_counts}")
    write_jsonl(EXP / "inputs/CHURN_QUERY_MANIFEST.jsonl", output)
    audit = {
        "rows": len(output), "reference": 1000, "holdout": 1000,
        "attacks": dict(attack_counts),
        "new_query_embeddings": sum(row["embedding_source"] == "NEW_BGE_S2" for row in output),
        "frozen_lc_embeddings": sum(row["embedding_source"] == "LC_LARGE_FROZEN" for row in output),
        "frozen_dcmi_embeddings": sum(row["embedding_source"] == "DCMI_FROZEN" for row in output),
        "ordered_query_id_sha256": sha_text("\n".join(ids)),
        "manifest_sha256": sha_file(EXP / "inputs/CHURN_QUERY_MANIFEST.jsonl"),
    }
    atomic_json(EXP / "audits/CHURN_QUERY_LINEAGE_AUDIT.json", audit)
    return audit


def main() -> None:
    for folder in ("inputs", "audits", "cache", "tables", "reports", "configs", "logs", "checkpoints", "runtime"):
        (EXP / folder).mkdir(parents=True, exist_ok=True)
    ia = json.loads((V6 / "FINAL_RESULT.json").read_text())
    if ia.get("verdict") != "IA_STD_Q15_V6_FAILED_FINAL":
        raise RuntimeError("IA-v6 is not in the required terminal state")
    manifests, union = prepare_churn()
    query_audit = prepare_queries()

    code_files = sorted((EXP / "code").glob("*"))
    code_hashes = {str(path.relative_to(ROOT)): sha_file(path) for path in code_files if path.is_file()}
    source_files = [
        V6 / "FINAL_RESULT.json",
        CLEAN / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl",
        LC / "cache/CORPUS_EMBEDDINGS.float16.npy",
        LC / "cache/QUERY_EMBEDDINGS.float16.npy",
        LC / "cache/LARGE_DETECTION_SCORES.jsonl",
        FINAL8 / "cache/FINAL_RETRIEVAL_AND_DETECTION.jsonl",
        CORE6 / "cache/CORE5_DCMI_RETRIEVAL_AND_DETECTION.jsonl",
        CORE6 / "cache/STANDARDIZED_QUERY_EMBEDDINGS.float16.npy",
        FINAL8 / "configs/FINAL_DEFENSE_MANIFEST.json",
        RECAL / "GOLD_RECALIBRATION_RESULT.json",
        RECAL / "tables/GOLD_REFRESH_FP_SUBSET_AUDIT.csv",
    ] + [OLD_SIDECAR / f"manifests/DB_CHURN_{version}.json" for version in VERSIONS]
    precommit = {
        "campaign": EXP.name, "created_utc": now(),
        "ia_final_status": ["IA_STD_Q15_V6_FAILED_FINAL", "STANDARDIZED_ATTACK_UNAVAILABLE"],
        "frozen_detector": {
            "retriever": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
            "mirabel": "canonical full-corpus Gumbel margin, confidence=0.95",
            "local_calibration": "200-NN benign empirical upper tail",
            "risk": "R_LC=-log(p_local)", "k": K_LOCAL,
            "outer_budget": OUTER_BUDGET, "strict_operator": ">",
            "strict_threshold": STRICT_THRESHOLD,
            "locator": "MIRABEL retrieval top-1", "action": "Simple Hide",
            "generator": "Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "source_token_budget": 2048,
        },
        "churn": {
            "versions": {version: {
                "manifest": str(OLD_SIDECAR / f"manifests/DB_CHURN_{version}.json"),
                "documents": manifests[version]["documents"],
                "added": manifests[version]["added_count"],
            } for version in VERSIONS},
            "missing_embedding_union": len(union),
            "duplicate_exception": "exactly the two V0 inherited duplicate excess rows; no new groups",
            "mode_a": "V0 reference margins and frozen outer threshold",
            "mode_b": "version-specific benign reference margins and benign-only outer refresh",
            "success": {"refresh_fpr_max": .05, "preferred_fpr": [.02, .04],
                        "per_attack_tpr_drop_max": .05, "preferred_mean_drop_max": .03},
        },
        "evaluation": query_audit,
        "fp_safe": {
            "conditional_open": "only if current Simple Hide discards freed evidence budget",
            "candidate": "SAFE_CONTEXT_REDISTRIBUTION_V1",
            "fp_ids": 27, "candidate_generation_allowed_before_packing_audit": False,
        },
        "prohibitions": ["detector modification", "training", "attack-specific calibration",
                         "k search", "LC formula changes", "IA v7", "automatic full Core5 candidate generation"],
        "model_paths": {"bge": str(BGE), "qwen": str(QWEN)},
        "input_sha256": {str(path): sha_file(path) for path in source_files},
        "derived_input_sha256": {
            str(EXP / "inputs/MISSING_EMBEDDING_UNION.jsonl"): sha_file(EXP / "inputs/MISSING_EMBEDDING_UNION.jsonl"),
            str(EXP / "inputs/CHURN_QUERY_MANIFEST.jsonl"): sha_file(EXP / "inputs/CHURN_QUERY_MANIFEST.jsonl"),
            str(EXP / "audits/INHERITED_BASELINE_DUPLICATES.json"): sha_file(EXP / "audits/INHERITED_BASELINE_DUPLICATES.json"),
        },
        "code_sha256": code_hashes,
    }
    path = EXP / "configs/DB_CHURN_V2_PRECOMMIT.json"
    atomic_json(path, precommit)
    atomic_text(path.with_suffix(".sha256"), f"{sha_file(path)}  {path.name}\n")
    atomic_json(EXP / "checkpoints/DB_CHURN_V2_PRECOMMITTED.json", {
        "stage": "DB_CHURN_V2_PRECOMMITTED", "precommit_sha256": sha_file(path),
        "missing_embedding_union": len(union), "updated_utc": now(),
    })
    checkpoint("DB_CHURN_V2_PRECOMMITTED", missing_embedding_union=len(union),
               query_rows=query_audit["rows"], precommit_sha256=sha_file(path))
    print(json.dumps({"precommit_sha256": sha_file(path), "missing_union": len(union),
                      "query_audit": query_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
