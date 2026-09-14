#!/usr/bin/env python3
"""Build the fresh target/benign substrate and freeze the V2 precommit.

No retrieval score, detector output, generated attack query, or answer is read
when selecting the cohort.  Selection depends only on immutable IDs, normalized
text hashes, the frozen protected DB membership contract, and qrels provenance.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata
from datetime import datetime, timezone


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
V1 = ROOT / "experiments" / "BC_DUALTAIL_DETECTOR_V1_STRICT"
EXP = ROOT / "experiments" / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2"
RAW = Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir/_raw")
DOMAINS = ("nfcorpus", "scidocs", "trec-covid")
TARGET_QUOTAS = {"nfcorpus": 34, "scidocs": 33, "trec-covid": 33}
BENIGN_QUOTAS = {"nfcorpus": 500, "scidocs": 500, "trec-covid": 0}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def global_id(domain: str, local_id: str) -> str:
    return f"BeIR_{domain}::{local_id}"


def load_qrels(domain: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for path in sorted((RAW / domain / "qrels").glob("*.tsv")):
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row or len(row) < 3 or row[0].lower() in {"query-id", "query_id"}:
                    continue
                try:
                    positive = float(row[2]) > 0
                except ValueError:
                    continue
                if positive:
                    result.setdefault(row[0], set()).add(global_id(domain, row[1]))
    return result


def main() -> None:
    for name in ("inputs", "audits", "manifests", "configs", "logs", "checkpoints", "tables", "reports", "runtime", "cache"):
        (EXP / name).mkdir(parents=True, exist_ok=True)

    v1_result = json.loads((V1 / "PHASE1_RESULT.json").read_text(encoding="utf-8"))
    if v1_result["verdict"] != "BC_DUALTAIL_V1_NO_TPR_GAIN":
        raise RuntimeError("frozen V1 verdict changed")
    v1_precommit_digest = (V1 / "configs" / "BC_DUALTAIL_V1_STRICT_PRECOMMIT.sha256").read_text(encoding="utf-8").split()[0]
    if sha_file(V1 / "configs" / "BC_DUALTAIL_V1_STRICT_PRECOMMIT.json") != v1_precommit_digest:
        raise RuntimeError("V1 precommit checksum mismatch")

    db_rows = read_jsonl(PARENT / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl")
    db_by_id = {r["document_id"]: r for r in db_rows}
    db_ids = set(db_by_id)
    db_hashes = {r["normalized_text_hash"] for r in db_rows}
    old_targets = list(csv.DictReader((PARENT / "inputs" / "SHARED_TARGETS.csv").open(encoding="utf-8")))
    old_ids = {r["document_id"] for r in old_targets}
    old_hashes = {r["normalized_text_hash"] for r in old_targets}

    with gzip.open(PARENT / "inputs" / "COMMON_ELIGIBLE_POOL.csv.gz", "rt", encoding="utf-8", newline="") as f:
        pool = list(csv.DictReader(f))

    raw_docs: dict[str, dict] = {}
    for domain in DOMAINS:
        for row in read_jsonl(RAW / domain / "corpus.jsonl"):
            local_id = str(row.get("_id") or row.get("id"))
            gid = global_id(domain, local_id)
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or "").strip()
            raw_docs[gid] = {
                "document_id": gid,
                "local_document_id": local_id,
                "domain": domain,
                "title": title,
                "text": text,
                "source_text": "\n".join(x for x in (title, text) if x),
            }

    targets: list[dict] = []
    availability = {}
    for domain in DOMAINS:
        member_candidates = sorted(
            [
                r for r in pool
                if r["domain"] == domain
                and r["document_id"] in db_ids
                and r["document_id"] not in old_ids
                and r["normalized_text_hash"] not in old_hashes
            ],
            key=lambda r: (r["selection_key"], r["document_id"]),
        )
        nonmember_candidates = sorted(
            [
                r for r in pool
                if r["domain"] == domain
                and r["document_id"] not in db_ids
                and r["normalized_text_hash"] not in db_hashes
                and r["document_id"] not in old_ids
                and r["normalized_text_hash"] not in old_hashes
            ],
            key=lambda r: (r["selection_key"], r["document_id"]),
        )
        quota = TARGET_QUOTAS[domain]
        availability[domain] = {"member": len(member_candidates), "nonmember": len(nonmember_candidates), "quota_each": quota}
        if len(member_candidates) < quota or len(nonmember_candidates) < quota:
            raise RuntimeError(f"insufficient fresh target candidates for {domain}")
        for membership, candidates in (("member", member_candidates), ("nonmember", nonmember_candidates)):
            for selection_index, candidate in enumerate(candidates[:quota]):
                raw = raw_docs[candidate["document_id"]]
                targets.append({
                    **raw,
                    "membership": membership,
                    "normalized_text_hash": candidate["normalized_text_hash"],
                    "selection_key": candidate["selection_key"],
                    "domain_selection_index": selection_index,
                })

    targets.sort(key=lambda r: (r["domain"], r["membership"], r["selection_key"], r["document_id"]))
    target_ids = {r["document_id"] for r in targets}
    target_hashes = {r["normalized_text_hash"] for r in targets}
    member = [r for r in targets if r["membership"] == "member"]
    nonmember = [r for r in targets if r["membership"] == "nonmember"]
    membership_audit = {
        "verdict": "FRESH_MEMBERSHIP_AUDIT_PASS",
        "member_targets": len(member),
        "nonmember_targets": len(nonmember),
        "member_actual_db_inclusion": sum(r["document_id"] in db_ids for r in member),
        "nonmember_actual_db_exclusion": sum(r["document_id"] not in db_ids for r in nonmember),
        "nonmember_normalized_text_exclusion": sum(r["normalized_text_hash"] not in db_hashes for r in nonmember),
        "fresh_old_target_id_overlap": len(target_ids & old_ids),
        "fresh_old_target_text_hash_overlap": len(target_hashes & old_hashes),
        "member_nonmember_id_overlap": len({r["document_id"] for r in member} & {r["document_id"] for r in nonmember}),
        "member_nonmember_text_hash_overlap": len({r["normalized_text_hash"] for r in member} & {r["normalized_text_hash"] for r in nonmember}),
        "domain_counts": {
            domain: {
                "member": sum(r["domain"] == domain and r["membership"] == "member" for r in targets),
                "nonmember": sum(r["domain"] == domain and r["membership"] == "nonmember" for r in targets),
            }
            for domain in DOMAINS
        },
        "availability_before_selection": availability,
    }
    checks = [
        len(member) == 100,
        len(nonmember) == 100,
        membership_audit["member_actual_db_inclusion"] == 100,
        membership_audit["nonmember_actual_db_exclusion"] == 100,
        membership_audit["nonmember_normalized_text_exclusion"] == 100,
        membership_audit["fresh_old_target_id_overlap"] == 0,
        membership_audit["fresh_old_target_text_hash_overlap"] == 0,
        membership_audit["member_nonmember_id_overlap"] == 0,
        membership_audit["member_nonmember_text_hash_overlap"] == 0,
    ]
    if not all(checks):
        membership_audit["verdict"] = "FRESH_MEMBERSHIP_AUDIT_FAILED"
        write_json(EXP / "audits" / "FRESH_MEMBERSHIP_AUDIT.json", membership_audit)
        raise RuntimeError("fresh membership audit failed")

    targets_path = EXP / "inputs" / "FRESH_SHARED_TARGETS.csv"
    target_fields = [
        "document_id", "local_document_id", "domain", "membership", "normalized_text_hash",
        "selection_key", "domain_selection_index", "title", "source_text",
    ]
    with targets_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in target_fields} for r in targets)
    write_json(EXP / "audits" / "FRESH_MEMBERSHIP_AUDIT.json", membership_audit)

    old_benign = read_jsonl(PARENT / "inputs" / "BENIGN_CALIBRATION.jsonl") + read_jsonl(PARENT / "inputs" / "BENIGN_HOLDOUT.jsonl")
    old_benign_ids = {r["query_id"] for r in old_benign}
    old_benign_hashes = {r["query_hash"] for r in old_benign}
    benign: list[dict] = []
    benign_availability = {}
    for domain in DOMAINS:
        qrels = load_qrels(domain)
        candidates = []
        for row in read_jsonl(RAW / domain / "queries.jsonl"):
            local_id = str(row.get("_id") or row.get("id"))
            query = str(row.get("text") or "").strip()
            query_id = f"BENIGN::{domain}::{local_id}"
            query_hash = sha_text(normalize_text(query))
            gold_ids = sorted(qrels.get(local_id, set()))
            if not query or not gold_ids:
                continue
            if query_id in old_benign_ids or query_hash in old_benign_hashes:
                continue
            if target_ids.intersection(gold_ids):
                continue
            candidates.append({
                "query_id": query_id,
                "local_query_id": local_id,
                "domain": domain,
                "query": query,
                "query_hash": query_hash,
                "gold_document_ids": gold_ids,
                "primary_gold_document_id": min(gold_ids, key=sha_text),
                "selection_key": sha_text("V2_FRESH_BENIGN||" + query_id + "||" + query_hash),
            })
        candidates.sort(key=lambda r: (r["selection_key"], r["query_id"]))
        quota = BENIGN_QUOTAS[domain]
        benign_availability[domain] = {"available": len(candidates), "selected": quota}
        if len(candidates) < quota:
            raise RuntimeError(f"insufficient fresh benign queries for {domain}: {len(candidates)} < {quota}")
        benign.extend(candidates[:quota])
    benign.sort(key=lambda r: (r["domain"], r["selection_key"], r["query_id"]))
    if len(benign) != 1000 or len({r["query_id"] for r in benign}) != 1000 or len({r["query_hash"] for r in benign}) != 1000:
        raise RuntimeError("fresh benign cohort is not unique 1,000")
    benign_path = EXP / "inputs" / "FRESH_BENIGN_1000.jsonl"
    write_jsonl(benign_path, benign)
    benign_audit = {
        "verdict": "FRESH_BENIGN_AUDIT_PASS",
        "queries": len(benign),
        "domain_counts": {domain: sum(r["domain"] == domain for r in benign) for domain in DOMAINS},
        "availability": benign_availability,
        "old_query_id_overlap": len({r["query_id"] for r in benign} & old_benign_ids),
        "old_query_hash_overlap": len({r["query_hash"] for r in benign} & old_benign_hashes),
        "target_gold_document_overlap": sum(bool(target_ids.intersection(r["gold_document_ids"])) for r in benign),
        "trec_covid_status": "NO_UNUSED_QREL_QUERY_AVAILABLE_AFTER_PRIOR_500_500_COHORT",
    }
    if any(benign_audit[k] for k in ("old_query_id_overlap", "old_query_hash_overlap", "target_gold_document_overlap")):
        benign_audit["verdict"] = "FRESH_BENIGN_AUDIT_FAILED"
        write_json(EXP / "audits" / "FRESH_BENIGN_AUDIT.json", benign_audit)
        raise RuntimeError("fresh benign audit failed")
    write_json(EXP / "audits" / "FRESH_BENIGN_AUDIT.json", benign_audit)

    v1_thresholds = json.loads((V1 / "configs" / "FROZEN_DEPLOYMENT_THRESHOLDS.json").read_text(encoding="utf-8"))
    protocol_root = ROOT / "experiments" / "CORE6_PROTOCOL_RECOVERY_V1" / "protocols"
    code_hashes = {path.name: sha_file(path) for path in sorted((EXP / "code").glob("*.py"))}
    precommit = {
        "campaign": "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2",
        "created_utc": utcnow(),
        "old_v1_verdict_immutable": "BC_DUALTAIL_V1_NO_TPR_GAIN",
        "old_v1_result_sha256": sha_file(V1 / "PHASE1_RESULT.json"),
        "detector_freeze": {
            "v1_precommit_sha256": v1_precommit_digest,
            "v1_implementation_sha256": sha_file(V1 / "code" / "run_phase1.py"),
            "formula_M": "frozen canonical MIRABEL full-corpus margin",
            "formula_C": "s1 - mean(s2,s3,s4)",
            "tail_p": "(1 + count_reference[X_b >= X_q]) / 251",
            "formula_S": "max(-ln(p_M), -ln(p_C))",
            "v1_tail_reference_ids": json.loads((V1 / "configs" / "BC_DUALTAIL_V1_STRICT_PRECOMMIT.json").read_text(encoding="utf-8"))["inputs"]["tail_reference_ids"],
            "v1_frozen_deployment_thresholds": v1_thresholds["thresholds"],
        },
        "protected_db": {
            "path": str(PARENT / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl"),
            "sha256": sha_file(PARENT / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl"),
            "documents": len(db_rows),
        },
        "fresh_targets": {
            "path": str(targets_path), "sha256": sha_file(targets_path),
            "member": 100, "nonmember": 100,
            "domain_quota_each_membership": TARGET_QUOTAS,
            "selection": "per domain and membership candidate set, ascending (frozen common-pool selection_key, document_id); excludes old target ID/text hash",
            "ids_and_labels": [{"document_id": r["document_id"], "normalized_text_hash": r["normalized_text_hash"], "domain": r["domain"], "membership": r["membership"]} for r in targets],
        },
        "fresh_benign": {
            "path": str(benign_path), "sha256": sha_file(benign_path), "n": 1000,
            "domain_quotas": BENIGN_QUOTAS,
            "selection": "unused qrel-backed queries excluding prior 1,000 query IDs/text hashes and all fresh target gold documents; ascending SHA256(V2_FRESH_BENIGN||query_id||query_hash)",
            "trec_covid_limitation": "all 50 available qrel queries were used previously; no fresh TREC-COVID normal query reused",
            "ids": [r["query_id"] for r in benign],
        },
        "fresh_attacks": {
            "MEntA": {"targets": 200, "queries_per_target": 5, "expected_queries": 1000, "model": "gpt-4.1-nano", "summary_temperature": 0.3, "question_temperature": 0.7, "one_generation_per_target": True},
            "MBA": {"targets": 200, "queries_per_target": 1, "mask_count": 5, "generator": "openai-community/gpt2-xl@15ea56dee5df4983c59b2538573817e1667135e2"},
            "RAG-MIA": {"targets": 200, "queries_per_target": 1, "prompt": "recovered frozen paper-faithful query generator"},
            "old_attack_query_reuse": False,
            "protocol_hashes": {
                attack: {
                    "query_generator": sha_file(protocol_root / folder / "query_generator.py"),
                    "scorer": sha_file(protocol_root / folder / "scorer.py"),
                }
                for attack, folder in (("MEntA", "menta"), ("MBA", "mba"), ("RAG-MIA", "rag_mia"))
            },
        },
        "primary_classes": {"positive": "fresh member-target attack query", "negative": "fresh benign query", "nonmember_attack": "separate alarm-rate diagnostic only"},
        "matched_fpr": {"unit": "query", "negative_n": 1000, "primary": 0.03, "secondary": [0.01, 0.05], "method": "empirical ROC linear interpolation"},
        "bootstrap": {"iterations": 2000, "seed": 20260912, "MEntA_unit": "target/session preserving Q1-Q5", "MBA_RAG_MIA_unit": "target", "benign": "fixed"},
        "phase1_gate": {
            "MEntA_delta_tpr3": ">=0.05", "MEntA_bootstrap_delta_ci_low": ">0",
            "MBA_delta_tpr3": ">=-0.05", "RAG-MIA_delta_tpr3": ">=-0.05", "Core3_macro_member_tpr3": "DualTail>MIRABEL",
        },
        "recovery_warning": "TARGET_CONCENTRATED_GAIN if one target contributes >=50% of recovered query gain",
        "phase1_failure": "DUALTAIL_MEMBER_EXPOSURE_NOT_REPLICATED; no generation; close dense retrieval handcrafted geometry family",
        "phase2_if_pass": {
            "conditions": ["NO_DEFENSE", "BC_MIRABEL_SIMPLE_HIDE", "DUALTAIL_SIMPLE_HIDE"],
            "locator": "frozen MIRABEL/similarity Top1", "action": "hide Top1 without replacement", "generator": "frozen Qwen2.5-3B one generation",
            "MEntA": "DualTail native AUC < BC-MIRABEL; preferred <=0.55",
            "MBA": "native AUC <=0.55 or degradation vs BC-MIRABEL <=0.03",
            "RAG-MIA": "native AUC <=0.55 or degradation vs BC-MIRABEL <=0.03",
            "E-AUC": "secondary only",
        },
        "code_sha256_at_precommit": code_hashes,
        "written_before": ["fresh attack query generation", "fresh retrieval", "detector scoring", "matched-FPR result", "answer generation", "privacy scoring"],
    }
    precommit_path = EXP / "configs" / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2_PRECOMMIT.json"
    write_json(precommit_path, precommit)
    digest = sha_file(precommit_path)
    (EXP / "configs" / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2_PRECOMMIT.sha256").write_text(f"{digest}  {precommit_path.name}\n", encoding="utf-8")
    write_json(EXP / "checkpoints" / "PRECOMMIT_WRITTEN.json", {"stage": "PRECOMMIT_WRITTEN", "created_utc": utcnow(), "sha256": digest})
    (EXP / "STATUS.md").write_text(
        "# DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2\n\n"
        "- 상태: `PRECOMMIT_WRITTEN`\n"
        f"- PRECOMMIT SHA-256: `{digest}`\n"
        "- 기존 V1 판정: `BC_DUALTAIL_V1_NO_TPR_GAIN` (불변)\n"
        "- 다음 단계: fresh attack query generation\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": "V2_SUBSTRATE_AND_PRECOMMIT_PASS", "member": 100, "nonmember": 100, "fresh_benign": 1000, "benign_domains": benign_audit["domain_counts"], "precommit_sha256": digest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
