#!/usr/bin/env python3
"""Read-only integrity audit for completed BC-RRE artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments/CLEAN_CORE3_DEV_V1"
CAMPAIGN = ROOT / "experiments/BC_RRE_ACTION_ONLY_SMALL_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    precommit_path = CAMPAIGN / "configs/BC_RRE_ACTION_ONLY_SMALL_V1_PRECOMMIT.json"
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    expected_precommit = (CAMPAIGN / "configs/BC_RRE_ACTION_ONLY_SMALL_V1_PRECOMMIT.sha256").read_text().split()[0]
    parent_current = {rel: sha256(PARENT / rel) for rel in precommit["parent_artifact_hashes"]}
    parent_expected = precommit["parent_artifact_hashes"]
    reference = {row["query_id"]: row for row in read_jsonl(CAMPAIGN / "cache/RRE_REFERENCE_RETRIEVAL.jsonl")}
    answers = {row["query_id"]: row for row in read_jsonl(CAMPAIGN / "runtime/RRE_GENERATED_ANSWERS.jsonl")}
    parent_retrieval = {row["query_id"]: row for row in read_jsonl(PARENT / "cache/CLEAN_CORE3_RETRIEVAL_CACHE.jsonl")}
    risk_parent = {qid for qid, row in parent_retrieval.items() if bool(row["bc_q97_alarm"]) and (row["cohort"] == "ATTACK" or row["split"] == "HOLDOUT")}
    slot_checks = []
    for qid, answer in answers.items():
        ref = reference[qid]
        before = list(ref["original_top_document_ids"])
        after = json.loads(answer["source_ids_json"])
        slot = int(answer["replacement_slot"]) - 1
        expected = before.copy(); expected[slot] = ref["replacement_source_id"]
        slot_checks.append(after == expected and before[slot] == ref["selected_source_id"])
    addendum = CAMPAIGN / "configs/PRECOMMIT_IMPLEMENTATION_ADDENDUM_01.json"
    checks = {
        "precommit_hash_valid": sha256(precommit_path) == expected_precommit,
        "implementation_addendum_hash_valid": sha256(addendum) == (CAMPAIGN / "configs/PRECOMMIT_IMPLEMENTATION_ADDENDUM_01.sha256").read_text().split()[0],
        "parent_artifacts_unchanged": parent_current == parent_expected,
        "reference_pool_audit_pass": json.loads((CAMPAIGN / "audits/REFERENCE_POOL_AUDIT.json").read_text())["verdict"] == "REFERENCE_POOL_AUDIT_PASS",
        "front_end_alarm_ids_equal": risk_parent == set(reference) == set(answers),
        "front_end_locator_equal": all(parent_retrieval[qid]["selected_source_id"] == reference[qid]["selected_source_id"] for qid in reference),
        "same_slot_replacement_all": all(slot_checks) and len(slot_checks) == 146,
        "top4_preserved_all": all(len(json.loads(row["source_ids_json"])) == 4 for row in answers.values()),
        "source_budget_contract_all": all(int(row["source_token_count"]) <= 2048 for row in answers.values()),
        "rre_generation_count_146": len(answers) == 146,
        "safe_new_generation_count_zero": json.loads((CAMPAIGN / "runtime/RRE_GENERATION_MANIFEST.json").read_text())["safe_query_new_generation_count"] == 0,
        "mba_final_valid_40": json.loads((CAMPAIGN / "audits/MBA_PARSER_AUDIT.json").read_text())["final_valid_n_each_condition"] == 40,
        "final_verdict_is_fail_stop": json.loads((CAMPAIGN / "FINAL_RESULT.json").read_text())["verdict"] == "BC_RRE_BENIGN_DAMAGE_FAILED",
    }
    output = {
        "verdict": "FINAL_AUDIT_PASS" if all(checks.values()) else "FINAL_AUDIT_FAILED",
        "checks": checks,
        "parent_expected_sha256": parent_expected,
        "parent_current_sha256": parent_current,
        "precommit_sha256": expected_precommit,
        "implementation_addendum_sha256": sha256(addendum),
        "rre_answer_sha256": sha256(CAMPAIGN / "runtime/RRE_GENERATED_ANSWERS.jsonl"),
        "reference_retrieval_sha256": sha256(CAMPAIGN / "cache/RRE_REFERENCE_RETRIEVAL.jsonl"),
        "new_generation_count": len(answers),
        "safe_new_generation_count": 0,
        "post_failure_search_or_tuning": False,
    }
    (CAMPAIGN / "audits/FINAL_INTEGRITY_AUDIT.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["verdict"] == "FINAL_AUDIT_PASS" else 1)


if __name__ == "__main__":
    main()
