#!/usr/bin/env python3
"""Precommit an exact-ID native-scorer metadata join for Phase B."""
from __future__ import annotations

import json
from pathlib import Path

from common import EXP, ROOT, checkpoint, freeze_json, now, read_jsonl, sha_file


RETRIEVAL = EXP / "cache/PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl"
SOURCES = {
    "MBA": (ROOT / "experiments/LC_MIRABEL_LARGE_V1/inputs/LARGE_MBA_ATTACK_QUERIES.jsonl", "mask_answers", 400),
    "S²-MIA": (ROOT / "experiments/FINAL_8ATTACK_E2E_FRAMEWORK_V1/inputs/S2_ATTACK_QUERIES.jsonl", "s2_full_target", 802),
    "DCMI-Std-Q2": (ROOT / "experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1/inputs/DCMI_STD_Q2_QUERIES.jsonl", "variant", 800),
}


def main() -> None:
    if (EXP / "PHASE_B_RESULT.json").exists():
        raise RuntimeError("Phase-B result already exists; repair precommit refused")
    rows = read_jsonl(RETRIEVAL)
    audits = {}
    for attack, (path, field, expected_ids) in SOURCES.items():
        source = {row["query_id"]: row for row in read_jsonl(path)}
        selected = [row for row in rows if row.get("attack") == attack]
        ids = {row["query_id"] for row in selected}
        missing, conflicts = [], []
        for row in selected:
            original = source.get(row["query_id"])
            if original is None:
                missing.append(row["query_id"]); continue
            for key in ("query", "membership", "target_id", "session_id"):
                if original.get(key) != row.get(key): conflicts.append(f"{row['query_id']}:{key}")
            if field not in original or original[field] in (None, "", {}): conflicts.append(f"{row['query_id']}:{field}")
        if len(ids) != expected_ids or len(selected) != expected_ids * 4 or missing or conflicts:
            raise RuntimeError(f"{attack} Phase-B repair mismatch")
        audits[attack] = {"unique_query_ids": len(ids), "retrieval_rows": len(selected),
                          "missing": len(missing), "conflicts": len(conflicts), "joined_field": field}
    original = EXP / "code/run_phase_b_scoring.py"
    runner = EXP / "code/run_phase_b_scoring_repair.py"
    this_file = Path(__file__).resolve()
    answers = EXP / "runtime/PHASE_B_CLOSED_BOOK_ANSWERS.jsonl"
    manifest = EXP / "runtime/PHASE_B_GENERATION_MANIFEST.json"
    precommit = {
        "campaign": EXP.name, "repair": "PHASE_B_NATIVE_SCORER_METADATA_JOIN_V1",
        "created_utc": now(), "status": "PRECOMMITTED_BEFORE_REPAIRED_PHASE_B_SCORING",
        "cause": "the frozen Phase-A2 retrieval substrate omitted native-scorer-only MBA/S2/DCMI fields",
        "allowed_change": "exact-query-id read-time join only",
        "forbidden_changes": ["generation rerun", "answer/retrieval change", "fuzzy matching", "imputation",
                              "native metric change", "detector/model/threshold change"],
        "audits": audits,
        "inputs_sha256": {str(RETRIEVAL): sha_file(RETRIEVAL), str(answers): sha_file(answers),
                          str(manifest): sha_file(manifest),
                          **{str(path): sha_file(path) for path, _, _ in SOURCES.values()}},
        "code_sha256": {str(original): sha_file(original), str(runner): sha_file(runner), str(this_file): sha_file(this_file)},
    }
    path = EXP / "configs/PHASE_B_SCORING_REPAIR_PRECOMMIT.json"
    digest = freeze_json(path, precommit)
    checkpoint("PHASE_B_SCORING_REPAIR_PRECOMMITTED", precommit_sha256=digest,
               mba_exact=400, s2_exact=802, dcmi_exact=800, conflicts=0)
    print(json.dumps(precommit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
