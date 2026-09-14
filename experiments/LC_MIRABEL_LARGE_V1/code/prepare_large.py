#!/usr/bin/env python3
"""Build the fresh 1000/1000 target cohort, benign bank/holdout, and precommit."""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from common import (ATTACKS, BGE, DOMAINS, EXP, K_LOCAL, ORTH, PARENT, RAW,
                    RECOVERY, REQUEST, ROOT, V2, atomic_json, atomic_text,
                    normalize, now, read_jsonl, sha_file, sha_text, write_csv,
                    write_jsonl, checkpoint)


TARGET_QUOTAS = {"nfcorpus": 334, "scidocs": 333, "trec-covid": 333}
BENIGN_SPLIT_QUOTAS = {"nfcorpus": 500, "scidocs": 475, "trec-covid": 25}
MBA_INELIGIBLE = EXP / "configs/MBA_PROTOCOL_INELIGIBLE_TARGETS.json"


def load_qrels(domain: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for path in sorted((RAW / domain / "qrels").glob("*.tsv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) < 3 or row[0].lower() in {"query-id", "query_id"}:
                    continue
                try:
                    positive = float(row[2]) > 0
                except ValueError:
                    continue
                if positive:
                    result[row[0]].add(f"BeIR_{domain}::{row[1]}")
    return result


def collect_prior_targets(pool_by_id: dict[str, dict]) -> tuple[set[str], set[str], list[dict]]:
    ids: set[str] = set()
    provenance: list[dict] = []
    for path in sorted((ROOT / "experiments").glob("**/*TARGETS*.csv")):
        if EXP in path.parents:
            continue
        try:
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
        except Exception:
            continue
        found = {str(row.get("document_id") or row.get("target_id") or "") for row in rows}
        found.discard("")
        if found:
            ids.update(found)
            provenance.append({"path": str(path), "sha256": sha_file(path), "target_ids": len(found)})
    for path in sorted((ROOT / "experiments").glob("**/*ATTACK_QUERIES*.jsonl")):
        if EXP in path.parents:
            continue
        found: set[str] = set()
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        target = str(json.loads(line).get("target_id") or "")
                        if target:
                            found.add(target)
        except Exception:
            continue
        if found:
            ids.update(found)
            provenance.append({"path": str(path), "sha256": sha_file(path), "target_ids": len(found)})
    hashes = {pool_by_id[target]["normalized_text_hash"] for target in ids if target in pool_by_id}
    return ids, hashes, provenance


def main() -> None:
    for name in ("configs", "inputs", "audits", "cache", "tables", "reports", "runtime",
                 "logs", "checkpoints", "manifests", "tests"):
        (EXP / name).mkdir(parents=True, exist_ok=True)
    checkpoint("PREPARING_LARGE_SUBSTRATE")

    if json.loads((V2 / "PHASE1_RESULT.json").read_text())["verdict"] != "DUALTAIL_MEMBER_EXPOSURE_NOT_REPLICATED":
        raise RuntimeError("frozen V2 verdict drift")
    if json.loads((ORTH / "RESULT.json").read_text())["final_verdict"] != "TRAINING_FREE_HANDCRAFTED_DETECTOR_SEARCH_CLOSED":
        raise RuntimeError("frozen Sparse closure drift")

    with gzip.open(PARENT / "inputs/COMMON_ELIGIBLE_POOL.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        pool = list(csv.DictReader(handle))
    pool_by_id = {row["document_id"]: row for row in pool}
    eligibility = json.loads(MBA_INELIGIBLE.read_text())
    if eligibility.get("verdict") != "MBA_PROTOCOL_INPUT_EXCLUSIONS_PRECOMMITTED":
        raise RuntimeError("MBA protocol eligibility exclusion manifest is invalid")
    mba_ineligible_ids = set(eligibility["document_ids"])
    if len(mba_ineligible_ids) != 6 or not mba_ineligible_ids <= set(pool_by_id):
        raise RuntimeError("MBA protocol eligibility exclusion set drift")
    mba_ineligible_hashes = {pool_by_id[target]["normalized_text_hash"] for target in mba_ineligible_ids}
    db_rows = read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")
    db_by_id = {row["document_id"]: row for row in db_rows}
    db_ids = set(db_by_id)
    db_hashes = {row["normalized_text_hash"] for row in db_rows}
    old_ids, old_hashes, old_provenance = collect_prior_targets(pool_by_id)

    selected: list[dict] = []
    availability: dict[str, dict] = {}
    for domain in DOMAINS:
        domain_rows = [row for row in pool if row["domain"] == domain]
        member = sorted((row for row in domain_rows if row["document_id"] in db_ids
                         and row["document_id"] not in old_ids
                         and row["normalized_text_hash"] not in old_hashes
                         and row["document_id"] not in mba_ineligible_ids
                         and row["normalized_text_hash"] not in mba_ineligible_hashes),
                        key=lambda row: (row["selection_key"], row["document_id"]))
        nonmember = sorted((row for row in domain_rows if row["document_id"] not in db_ids
                            and row["normalized_text_hash"] not in db_hashes
                            and row["document_id"] not in old_ids
                            and row["normalized_text_hash"] not in old_hashes
                            and row["document_id"] not in mba_ineligible_ids
                            and row["normalized_text_hash"] not in mba_ineligible_hashes),
                           key=lambda row: (row["selection_key"], row["document_id"]))
        quota = TARGET_QUOTAS[domain]
        availability[domain] = {"member": len(member), "nonmember": len(nonmember), "selected_each": quota}
        if min(len(member), len(nonmember)) < quota:
            raise RuntimeError(f"insufficient fresh targets for {domain}")
        for membership, rows in (("member", member[:quota]), ("nonmember", nonmember[:quota])):
            for index, row in enumerate(rows):
                selected.append({**row, "membership": membership, "domain_selection_index": index})

    selected_ids = {row["document_id"] for row in selected}
    raw_selected: dict[str, dict] = {}
    for domain in DOMAINS:
        needed = {target.split("::", 1)[1] for target in selected_ids if target.startswith(f"BeIR_{domain}::")}
        with (RAW / domain / "corpus.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                local = str(row.get("_id") or row.get("id"))
                if local in needed:
                    title = str(row.get("title") or "").strip()
                    text = str(row.get("text") or "").strip()
                    raw_selected[f"BeIR_{domain}::{local}"] = {
                        "title": title, "source_text": "\n".join(value for value in (title, text) if value)
                    }
    if set(raw_selected) != selected_ids:
        raise RuntimeError("selected source text recovery incomplete")
    targets = []
    for row in sorted(selected, key=lambda value: (value["membership"], value["domain"], value["selection_key"], value["document_id"])):
        targets.append({
            "document_id": row["document_id"], "local_document_id": row["local_document_id"],
            "domain": row["domain"], "membership": row["membership"],
            "normalized_text_hash": row["normalized_text_hash"], "selection_key": row["selection_key"],
            "domain_selection_index": row["domain_selection_index"],
            "title": raw_selected[row["document_id"]]["title"],
            "source_text": raw_selected[row["document_id"]]["source_text"],
        })
    target_path = EXP / "inputs/LARGE_SHARED_TARGETS.csv"
    write_csv(target_path, targets)

    member = [row for row in targets if row["membership"] == "member"]
    nonmember = [row for row in targets if row["membership"] == "nonmember"]
    membership_audit = {
        "verdict": "LARGE_MEMBERSHIP_AUDIT_PASS", "member": len(member), "nonmember": len(nonmember),
        "member_actual_db_inclusion": sum(row["document_id"] in db_ids for row in member),
        "nonmember_actual_db_exclusion": sum(row["document_id"] not in db_ids for row in nonmember),
        "nonmember_text_hash_exclusion": sum(row["normalized_text_hash"] not in db_hashes for row in nonmember),
        "member_nonmember_id_overlap": len({r["document_id"] for r in member} & {r["document_id"] for r in nonmember}),
        "member_nonmember_text_overlap": len({r["normalized_text_hash"] for r in member} & {r["normalized_text_hash"] for r in nonmember}),
        "prior_target_id_overlap": len(selected_ids & old_ids),
        "prior_target_text_overlap": len({r["normalized_text_hash"] for r in targets} & old_hashes),
        "domain_counts": {domain: {label: sum(r["domain"] == domain and r["membership"] == label for r in targets)
                                    for label in ("member", "nonmember")} for domain in DOMAINS},
        "availability": availability, "prior_target_sources": old_provenance,
    }
    required = [len(member) == 1000, len(nonmember) == 1000,
                membership_audit["member_actual_db_inclusion"] == 1000,
                membership_audit["nonmember_actual_db_exclusion"] == 1000,
                membership_audit["nonmember_text_hash_exclusion"] == 1000,
                membership_audit["member_nonmember_id_overlap"] == 0,
                membership_audit["member_nonmember_text_overlap"] == 0,
                membership_audit["prior_target_id_overlap"] == 0,
                membership_audit["prior_target_text_overlap"] == 0]
    if not all(required):
        membership_audit["verdict"] = "LARGE_MEMBERSHIP_AUDIT_FAILED"
        atomic_json(EXP / "audits/LARGE_MEMBERSHIP_AUDIT.json", membership_audit)
        raise RuntimeError(membership_audit["verdict"])
    atomic_json(EXP / "audits/LARGE_MEMBERSHIP_AUDIT.json", membership_audit)

    benign_candidates: dict[str, list[dict]] = {}
    for domain in DOMAINS:
        qrels = load_qrels(domain)
        rows = []
        for source in read_jsonl(RAW / domain / "queries.jsonl"):
            local = str(source.get("_id") or source.get("id"))
            query = str(source.get("text") or "").strip()
            if not query or local not in qrels:
                continue
            query_hash = sha_text(normalize(query))
            rows.append({"query_id": f"BENIGN::{domain}::{local}", "local_query_id": local,
                         "domain": domain, "query": query, "query_hash": query_hash,
                         "gold_document_ids": sorted(qrels[local]),
                         "selection_key": sha_text("LC_MIRABEL_BENIGN||" + domain + "||" + local + "||" + query_hash)})
        benign_candidates[domain] = sorted(rows, key=lambda row: (row["selection_key"], row["query_id"]))
    reference: list[dict] = []
    holdout: list[dict] = []
    globally_used_hashes: set[str] = set()
    for domain in DOMAINS:
        quota = BENIGN_SPLIT_QUOTAS[domain]
        rows = []
        for row in benign_candidates[domain]:
            if row["query_hash"] in globally_used_hashes:
                continue
            globally_used_hashes.add(row["query_hash"])
            rows.append(row)
        if len(rows) < 2 * quota:
            raise RuntimeError(f"insufficient official benign queries for {domain}")
        reference.extend({**row, "split": "BENIGN_REFERENCE"} for row in rows[:quota])
        holdout.extend({**row, "split": "BENIGN_DEPLOYMENT_HOLDOUT"} for row in rows[quota:2*quota])
    reference.sort(key=lambda row: (row["domain"], row["selection_key"], row["query_id"]))
    holdout.sort(key=lambda row: (row["domain"], row["selection_key"], row["query_id"]))
    ref_ids, hold_ids = {r["query_id"] for r in reference}, {r["query_id"] for r in holdout}
    ref_hashes, hold_hashes = {r["query_hash"] for r in reference}, {r["query_hash"] for r in holdout}
    target_text_hashes = {sha_text(normalize(r["source_text"])) for r in targets}
    benign_audit = {
        "verdict": "BENIGN_REFERENCE_AUDIT_PASS", "reference": len(reference), "holdout": len(holdout),
        "reference_holdout_id_overlap": len(ref_ids & hold_ids),
        "reference_holdout_text_overlap": len(ref_hashes & hold_hashes),
        "target_exact_text_overlap": len((ref_hashes | hold_hashes) & target_text_hashes),
        "attack_queries_present": 0,
        "domain_counts": {domain: {"reference": sum(r["domain"] == domain for r in reference),
                                    "holdout": sum(r["domain"] == domain for r in holdout)} for domain in DOMAINS},
        "note": "Qrel relevance to a target is not treated as identity overlap; exact ID/text leakage is audited separately.",
    }
    if len(reference) < 1000 or len(holdout) < 1000 or any(benign_audit[key] for key in
            ("reference_holdout_id_overlap", "reference_holdout_text_overlap", "target_exact_text_overlap")):
        benign_audit["verdict"] = "BENIGN_REFERENCE_AUDIT_FAILED"
        atomic_json(EXP / "audits/BENIGN_REFERENCE_AUDIT.json", benign_audit)
        raise RuntimeError(benign_audit["verdict"])
    write_jsonl(EXP / "inputs/BENIGN_REFERENCE.jsonl", reference)
    write_jsonl(EXP / "inputs/BENIGN_DEPLOYMENT_HOLDOUT.jsonl", holdout)
    atomic_json(EXP / "audits/BENIGN_REFERENCE_AUDIT.json", benign_audit)

    code_hashes = {path.name: sha_file(path) for path in sorted((EXP / "code").glob("*")) if path.is_file()}
    protocol_hashes = {attack: {
        "query_generator": sha_file(RECOVERY / "protocols" / folder / "query_generator.py"),
        "scorer": sha_file(RECOVERY / "protocols" / folder / "scorer.py")}
        for attack, folder in (("MEntA", "menta"), ("MBA", "mba"), ("RAG-MIA", "rag_mia"))}
    precommit = {
        "campaign": "LC_MIRABEL_LARGE_V1", "development_name": "LC-MIRABEL",
        "created_utc": now(), "request": {"path": str(REQUEST), "sha256": sha_file(REQUEST)},
        "scientific_rationale": "Re-rank frozen MIRABEL margin only by its conditional rarity among semantically local benign queries.",
        "status": "DEVELOPMENT_PRECOMMITTED", "k": K_LOCAL,
        "detector": {
            "M": "frozen canonical MIRABEL full-corpus margin",
            "embedding": "BAAI/bge-m3 frozen query embedding",
            "p_local": "(1 + count_{b in 200-NN benign reference}[M(b)>=M(q)]) / 201",
            "R_LC": "-log(p_local)",
            "p_global": "(1 + count_{b in all benign reference}[M(b)>=M(q)]) / (N_ref+1)",
            "R_GLOBAL": "-log(p_global)", "comparison_operator": "strict > for ROC thresholds; p<=0.03 for deployment",
            "forbidden": ["fusion", "multiple-k search", "domain routing", "QLL", "LOO", "DualTail", "Sparse", "learned model"]},
        "substrate": {
            "protected_db": {"path": str(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl"),
                             "sha256": sha_file(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl"), "documents": 3000},
            "targets": {"path": str(target_path), "sha256": sha_file(target_path), "member": 1000,
                        "nonmember": 1000, "domain_quota_each_membership": TARGET_QUOTAS,
                        "selection": "within fixed domain/membership strata, ascending frozen common-pool selection_key after exact prior-target exclusion"},
            "benign_reference": {"path": str(EXP / "inputs/BENIGN_REFERENCE.jsonl"),
                                 "sha256": sha_file(EXP / "inputs/BENIGN_REFERENCE.jsonl"), "n": 1000},
            "benign_holdout": {"path": str(EXP / "inputs/BENIGN_DEPLOYMENT_HOLDOUT.jsonl"),
                               "sha256": sha_file(EXP / "inputs/BENIGN_DEPLOYMENT_HOLDOUT.jsonl"), "n": 1000}},
        "attack_generation": {
            "MEntA": {"targets": 2000, "queries_each": 5, "queries": 10000, "model": "gpt-4.1-nano",
                      "one_api_call_per_target": True, "joint_output": "summary + QUERY_1..QUERY_5",
                      "parser_policy": (
                          "Accept either LABEL: value on one line or a label-only line followed immediately "
                          "by exactly one value line; then require exactly SUMMARY and QUERY_1..QUERY_5 in order"
                      )},
            "MBA": {"targets": 2000, "queries_each": 1, "queries": 2000, "mask_count": 5,
                    "proxy": "openai-community/gpt2-xl frozen"},
            "RAG-MIA": {"targets": 2000, "queries_each": 1, "queries": 2000, "paper_prompt": 2},
            "protocol_hashes": protocol_hashes},
        "input_eligibility": {
            "MBA": {
                "manifest": str(MBA_INELIGIBLE),
                "manifest_sha256": sha_file(MBA_INELIGIBLE),
                "excluded_document_count": len(mba_ineligible_ids),
                "rule": "Exclude only documents deterministically rejected by the frozen MBA generator before any detector score was computed",
            }
        },
        "evaluation": {"positive": "member-target attack query", "negative": "benign deployment holdout query",
                       "nonmember_attack": "diagnostic only", "matched_fpr": [0.01, 0.03, 0.05],
                       "primary": "MEntA query-level TPR@3%", "bootstrap_iterations": 2000,
                       "bootstrap_units": {"MEntA": "target/session", "MBA": "target", "RAG-MIA": "target"}},
        "phase1_gate": {"MEntA_delta_tpr3": ">=0.05", "MEntA_bootstrap_ci_low": ">0",
                        "MBA_delta_tpr3": ">=-0.02", "RAG-MIA_delta_tpr3": ">=-0.02",
                        "Core3_mean_delta_tpr3": ">=0.02"},
        "deployment": {"original": "M>0 (rho=0.05 canonical rule)", "global": "p_global<=0.03",
                       "local": "p_local<=0.03", "holdout_retuning": False},
        "protection_if_pass": "alarm -> MIRABEL top1 locator -> Simple Hide -> Qwen2.5-3B one generation",
        "e2e_conditions": ["NO_DEFENSE", "ORIGINAL_MIRABEL_SIMPLE_HIDE", "GLOBAL_BC_SIMPLE_HIDE", "LC_MIRABEL_SIMPLE_HIDE"],
        "e2e_gate": {"MEntA": "LC native AUC < Global BC native AUC",
                     "MBA": "LC native AUC <= Global BC native AUC", "RAG-MIA": "LC native AUC <= Global BC native AUC"},
        "churn": {"versions": {"V0": 0.0, "V10": 0.10, "V25": 0.25, "V50": 0.50},
                  "operation": "replace deterministic SHA256-selected background docs; preserve all evaluation membership labels",
                  "allowed_update": "benign MIRABEL score cache only", "training_steps": 0,
                  "gate": "LC FPR<=0.05 and each attack TPR degradation from V0<=0.05"},
        "stop_rules": {"detection_fail": "LC_MIRABEL_DETECTION_NOT_SUPPORTED; no generation",
                       "no_posthoc": ["change k", "fusion", "new local feature", "domain routing", "learned calibration"]},
        "protocol_repair": {
            "reason": "One successful MEntA API response used label/value line breaks, yielding 12 physical lines for six complete fields",
            "scope": "deterministic formatting canonicalization only; no regeneration, content editing, exclusion, or label-dependent handling",
            "performance_results_observed_before_reprecommit": False,
            "preserved_failure_history": str(EXP / "history/V1_INPUT_INCOMPATIBLE_20260912T085954Z"),
        },
        "mba_input_repair": {
            "reason": "Six targets were deterministically rejected because frozen MBA selection produced adjacent masks",
            "scope": "exclude those six protocol-ineligible documents and refill the same membership/domain strata by the frozen ordering",
            "performance_results_observed_before_reprecommit": False,
            "preserved_failure_history": str(EXP / "history/V1_LOCAL_INCOMPATIBLE_20260912T113134Z"),
        },
        "model_hashes": {"bge_snapshot": BGE.name, "bge_modules": sha_file(BGE / "modules.json")},
        "lineage": {"frozen_v2_result": sha_file(V2 / "PHASE1_RESULT.json"),
                    "frozen_sparse_result": sha_file(ORTH / "RESULT.json"), "code_sha256": code_hashes},
    }
    precommit_path = EXP / "configs/LC_MIRABEL_LARGE_V1_PRECOMMIT.json"
    atomic_json(precommit_path, precommit)
    atomic_text(precommit_path.with_suffix(".sha256"), sha_file(precommit_path) + "\n")
    atomic_json(EXP / "manifests/SUBSTRATE_MANIFEST.json", {
        "targets_sha256": sha_file(target_path), "member": 1000, "nonmember": 1000,
        "benign_reference_sha256": sha_file(EXP / "inputs/BENIGN_REFERENCE.jsonl"),
        "benign_holdout_sha256": sha_file(EXP / "inputs/BENIGN_DEPLOYMENT_HOLDOUT.jsonl"),
        "precommit_sha256": sha_file(precommit_path)})
    checkpoint("PRECOMMIT_COMPLETE", member=1000, nonmember=1000, benign_reference=1000,
               benign_holdout=1000, precommit_sha256=sha_file(precommit_path))
    print(json.dumps({"verdict": "LARGE_SUBSTRATE_AND_PRECOMMIT_PASS",
                      "precommit_sha256": sha_file(precommit_path),
                      "membership": membership_audit, "benign": benign_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
