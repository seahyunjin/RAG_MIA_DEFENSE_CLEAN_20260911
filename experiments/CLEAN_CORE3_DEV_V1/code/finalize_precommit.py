#!/usr/bin/env python3
"""Freeze the Core3 design and hard preflight before retrieval/outcomes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
CAMPAIGN = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
RECOVERY = ROOT / "experiments" / "CORE6_PROTOCOL_RECOVERY_V1"
INPUTS = CAMPAIGN / "inputs"
AUDITS = CAMPAIGN / "audits"
MANIFESTS = CAMPAIGN / "manifests"
CONFIGS = CAMPAIGN / "configs"
HF = Path("/home/traffic_3/workspace/.cache/huggingface/hub")


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


def main() -> None:
    targets = list(csv.DictReader((INPUTS / "SHARED_TARGETS.csv").open(encoding="utf-8")))
    benign_cal = read_jsonl(INPUTS / "BENIGN_CALIBRATION.jsonl")
    benign_hold = read_jsonl(INPUTS / "BENIGN_HOLDOUT.jsonl")
    menta = read_jsonl(INPUTS / "MENTA_ATTACK_QUERIES.jsonl")
    mba = read_jsonl(INPUTS / "MBA_ATTACK_QUERIES.jsonl")
    rag = read_jsonl(INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl")
    db_manifest = json.loads((MANIFESTS / "CLEAN_CORE3_DB_MANIFEST.json").read_text(encoding="utf-8"))
    membership = json.loads((AUDITS / "CORE3_MEMBERSHIP_AUDIT.json").read_text(encoding="utf-8"))
    core_manifest = json.loads((RECOVERY / "CORE6_PROTOCOL_MANIFEST.json").read_text(encoding="utf-8"))

    tests = subprocess.run(
        ["python", "-m", "unittest", "discover", "-s", str(RECOVERY / "protocols"), "-p", "test_*.py"],
        cwd=RECOVERY, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    checks = []
    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    add("MEntA recovered implementation hash correct", sha_file(RECOVERY / "protocols/menta/MANIFEST.json") == core_manifest["protocol_manifests"]["menta"], core_manifest["protocol_manifests"]["menta"])
    add("MBA recovered implementation hash correct", sha_file(RECOVERY / "protocols/mba/MANIFEST.json") == core_manifest["protocol_manifests"]["mba"], core_manifest["protocol_manifests"]["mba"])
    add("RAG-MIA recovered implementation hash correct", sha_file(RECOVERY / "protocols/rag_mia/MANIFEST.json") == core_manifest["protocol_manifests"]["rag_mia"], core_manifest["protocol_manifests"]["rag_mia"])
    add("unit tests still pass", tests.returncode == 0 and "Ran 20 tests" in tests.stdout and "OK" in tests.stdout, f"returncode={tests.returncode}")
    add("shared targets fixed", len(targets) == 40 and len({r["document_id"] for r in targets}) == 40, f"N={len(targets)}")
    add("member 20/20 actually in DB", membership["member_targets"] == 20 and membership["member_inclusion"] == 20, json.dumps(membership))
    add("nonmember 20/20 actually absent", membership["nonmember_targets"] == 20 and membership["nonmember_id_in_db"] == 0, json.dumps(membership))
    add("normalized text duplicate leakage zero", membership["nonmember_normalized_text_duplicate_in_db"] == 0, str(membership["nonmember_normalized_text_duplicate_in_db"]))
    add("benign calibration/holdout overlap zero", not ({r["query_id"] for r in benign_cal} & {r["query_id"] for r in benign_hold}), "500/500 IDs")
    benign_hashes = {sha_text(r["query"].strip().lower()) for r in benign_cal + benign_hold}
    attack_hashes = {sha_text(r["query"].strip().lower()) for r in menta + mba + rag}
    add("attack/benign exact-query overlap zero", not (benign_hashes & attack_hashes), f"overlap={len(benign_hashes & attack_hashes)}")
    add("DB hash frozen", sha_file(Path(db_manifest["corpus_file"])) == db_manifest["corpus_sha256"], db_manifest["corpus_sha256"])
    retriever_path = HF / "models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
    generator_path = HF / "models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    add("retriever frozen", retriever_path.is_dir(), str(retriever_path))
    add("generator frozen", generator_path.is_dir(), str(generator_path))
    add("scorers frozen", all((RECOVERY / "protocols" / p / "scorer.py").is_file() for p in ("menta", "mba", "rag_mia")), "three recovered scorer.py files")
    menta_sessions = {r["session_id"] for r in menta}
    add("MEntA valid ordered Q1-Q5", len(menta) == 200 and len(menta_sessions) == 40 and all(sorted(x["query_index"] for x in menta if x["session_id"] == sid) == [1,2,3,4,5] for sid in menta_sessions), f"queries={len(menta)} sessions={len(menta_sessions)}")
    add("MBA valid shared targets", len(mba) > 0 and {r["target_id"] for r in mba}.issubset({r["document_id"] for r in targets}), f"valid={len(mba)}")
    add("RAG-MIA valid shared targets", len(rag) == 40 and {r["target_id"] for r in rag} == {r["document_id"] for r in targets}, f"valid={len(rag)}")

    failures = [r for r in checks if r["status"] != "PASS"]
    with (AUDITS / "HARD_PREFLIGHT_CHECKS.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "evidence"])
        writer.writeheader(); writer.writerows(checks)
    if failures:
        write_json(CAMPAIGN / "PREFLIGHT_RESULT.json", {"verdict": "CLEAN_CORE3_HARD_PREFLIGHT_FAILED", "failures": failures})
        raise RuntimeError(f"hard preflight failed: {[x['check'] for x in failures]}")

    protocol_manifests = {}
    scorer_hashes = {}
    query_generator_hashes = {}
    for p in ("menta", "mba", "rag_mia"):
        protocol_manifests[p] = sha_file(RECOVERY / "protocols" / p / "MANIFEST.json")
        scorer_hashes[p] = sha_file(RECOVERY / "protocols" / p / "scorer.py")
        query_generator_hashes[p] = sha_file(RECOVERY / "protocols" / p / "query_generator.py")

    precommit = {
        "campaign": "CLEAN_CORE3_DEV_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_attacks": ["MEntA", "MBA", "RAG-MIA"],
        "excluded_attacks": {"S²-MIA": "PROTOCOL_REVISE", "DCMI": "PROTOCOL_REVISE", "IA": "PROTOCOL_UNAVAILABLE"},
        "historical_results_policy": "HISTORICAL_DEVELOPMENT_RESULT_ONLY; no prior query/retrieval/answer/score in main table",
        "protocol_recovery_manifest": {"path": str(RECOVERY / "CORE6_PROTOCOL_MANIFEST.json"), "sha256": sha_file(RECOVERY / "CORE6_PROTOCOL_MANIFEST.json")},
        "protocol_manifest_sha256": protocol_manifests,
        "query_generator_sha256": query_generator_hashes,
        "scorer_sha256": scorer_hashes,
        "common_eligibility_rule": json.loads((MANIFESTS / "COMMON_ELIGIBLE_POOL_MANIFEST.json").read_text(encoding="utf-8")),
        "target_selection": {
            "rule": "domain quotas 14/14/12; ascending SHA256(global document ID || normalized text hash); alternating member/nonmember per domain",
            "target_ids_and_membership": [{"document_id": r["document_id"], "domain": r["domain"], "membership": r["membership"], "normalized_text_hash": r["normalized_text_hash"]} for r in targets],
            "targets_file_sha256": sha_file(INPUTS / "SHARED_TARGETS.csv"),
        },
        "database_manifest": {"path": str(MANIFESTS / "CLEAN_CORE3_DB_MANIFEST.json"), "sha256": sha_file(MANIFESTS / "CLEAN_CORE3_DB_MANIFEST.json"), "corpus_sha256": db_manifest["corpus_sha256"]},
        "benign_split": {
            "calibration_n": 500, "holdout_n": 500,
            "calibration_ids": [r["query_id"] for r in benign_cal],
            "holdout_ids": [r["query_id"] for r in benign_hold],
            "calibration_sha256": sha_file(INPUTS / "BENIGN_CALIBRATION.jsonl"),
            "holdout_sha256": sha_file(INPUTS / "BENIGN_HOLDOUT.jsonl"),
        },
        "attack_query_artifacts": {
            "MEntA": {"path": str(INPUTS / "MENTA_ATTACK_QUERIES.jsonl"), "sha256": sha_file(INPUTS / "MENTA_ATTACK_QUERIES.jsonl"), "queries": len(menta), "sessions": len(menta_sessions)},
            "MBA": {"path": str(INPUTS / "MBA_ATTACK_QUERIES.jsonl"), "sha256": sha_file(INPUTS / "MBA_ATTACK_QUERIES.jsonl"), "queries": len(mba), "mask_count": 5, "mask_count_status": "fixed smallest paper-evaluated M before attack outcomes; no ROC-AUC search in this campaign"},
            "RAG-MIA": {"path": str(INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl"), "sha256": sha_file(INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl"), "queries": len(rag)},
        },
        "retriever": {"checkpoint": str(retriever_path), "revision": "5617a9f61b028005a4858fdac845db406aefb181", "snapshot_manifest_sha256": "77d0b2540f35b75b31d7c1390257ac8fb58264f0b39921c2ccbac0db3350e88f", "normalized_embeddings": True, "similarity": "inner product", "source_text": "title\\ntext", "top_k": 4, "stable_ties": "frozen corpus order"},
        "mirabel": {"formula": "canonical_official_v2 full-corpus M(q)=s_max-tau_q; top1 excluded from background moments; full corpus N retained", "confidence": 0.95, "rho": 0.05, "original_alarm": "strict M(q)>0", "locator": "highest-similarity Top1 source"},
        "bc_mirabel": {"score": "same frozen M(q)", "primary": "q97 benign calibration", "nominal_fpr": 0.03, "sensitivity": {"q99": 0.01, "q95": 0.05}, "empirical_cutoff": "for alpha and N, tau is ceil(alpha*N)-th largest calibration margin; alarm strict M(q)>tau"},
        "generator": {"checkpoint": str(generator_path), "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1", "snapshot_manifest_sha256": "ba3fa03be5e0bde35615328f34e49cdea40d9654f6d4e46561bd2247a3cdf18d", "dtype": "bfloat16", "attention": "sdpa", "do_sample": False, "num_beams": 1, "seed": 20260911, "max_prompt_tokens": 3072, "source_token_budget": 2048, "max_new_tokens": 160, "packing": "equal waterfill across available frozen Top4; hide removes selected Top1 without rank5 replacement", "system_prompt": "Answer the user's question using only the retrieved context. Follow any required output format exactly. If the context is insufficient, answer exactly: I don't know."},
        "conditions": ["NO_DEFENSE", "ORIGINAL_MIRABEL_TOP1_HIDE", "BC_MIRABEL_Q97_TOP1_HIDE"],
        "protection_action": "when alarmed, remove the same similarity Top1 source; no replacement, special prompt, clipping, postprocessing, or fixed refusal",
        "metric_provenance": {
            "MEntA": {"primary": "mean over five I_entail-I_idk and native ROC-AUC", "polarity": "higher_is_more_member", "scorer_sha256": scorer_hashes["menta"]},
            "MBA": {"primary": "mask reconstruction accuracy and native ROC-AUC", "polarity": "higher_is_more_member", "scorer_sha256": scorer_hashes["mba"]},
            "RAG-MIA": {"primary": "paper prompt #2 Yes/No TPR/FPR/accuracy; binary ROC-AUC only", "polarity": "Yes_is_member", "scorer_sha256": scorer_hashes["rag_mia"]},
            "E-AUC": "INTERNAL_DIAGNOSTIC_ONLY",
        },
        "bootstrap": {"iterations": 10000, "seed": 20260911, "unit": "target/session"},
        "bottleneck_rules": {
            "ALREADY_CONTROLLED": "No-Defense native ROC-AUC <=0.55",
            "RETRIEVAL_LIMITED": "member Target Retrieval@4 <0.50",
            "DETECTION_LIMITED": "retrieval not limited and alarm-on-retrieved-member <0.50",
            "LOCATOR_LIMITED": "retrieval/detection not limited and locator hit among alarm+retrieved-member <0.50",
            "PROTECTION_LIMITED": "EPO among member queries >=0.50 but defended native ROC-AUC >0.65",
            "MIXED": "none of the preceding hierarchical rules",
        },
        "written_before": ["retrieval", "MIRABEL margin", "BC threshold", "RAG answer generation", "attack scoring", "bottleneck labeling"],
    }
    CONFIGS.mkdir(parents=True, exist_ok=True)
    path = CONFIGS / "CLEAN_CORE3_DEV_V1_PRECOMMIT.json"
    write_json(path, precommit)
    digest = sha_file(path)
    (CONFIGS / "CLEAN_CORE3_DEV_V1_PRECOMMIT.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    write_json(CAMPAIGN / "PREFLIGHT_RESULT.json", {"verdict": "CLEAN_CORE3_HARD_PREFLIGHT_PASS", "precommit_sha256": digest, "checks": len(checks), "failures": 0})
    print(json.dumps({"verdict": "CLEAN_CORE3_HARD_PREFLIGHT_PASS", "precommit_sha256": digest, "checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
