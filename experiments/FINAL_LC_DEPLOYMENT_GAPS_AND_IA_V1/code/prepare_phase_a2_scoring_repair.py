#!/usr/bin/env python3
"""Precommit a metadata-only repair for the Phase-A2 MBA scorer input."""
from __future__ import annotations

import json
from pathlib import Path

from common import EXP, ROOT, checkpoint, freeze_json, now, read_jsonl, sha_file


MBA_SOURCE = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1" / "inputs" / "LARGE_MBA_ATTACK_QUERIES.jsonl"
SELECTED = EXP / "inputs" / "PHASE_A2_SELECTED_QUERIES.jsonl"
RETRIEVAL = EXP / "cache" / "PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl"
ORIGINAL_SCORER = EXP / "code" / "run_phase_a2_scoring.py"
REPAIRED_SCORER = EXP / "code" / "run_phase_a2_scoring_repair.py"


def validate_alignment(selected_rows: list[dict], source_rows: list[dict]) -> dict:
    selected = [row for row in selected_rows if row.get("attack") == "MBA"]
    source = {row["query_id"]: row for row in source_rows if row.get("attack") == "MBA"}
    if len(selected) != 400 or len({row["query_id"] for row in selected}) != 400:
        raise RuntimeError(f"MBA selected cohort drift: {len(selected)}")
    missing = [row["query_id"] for row in selected if row["query_id"] not in source]
    conflicts = []
    for row in selected:
        original = source.get(row["query_id"])
        if original is None:
            continue
        for key in ("query", "membership", "target_id", "session_id"):
            if original.get(key) != row.get(key):
                conflicts.append({"query_id": row["query_id"], "field": key})
        if not isinstance(original.get("mask_answers"), dict) or not original["mask_answers"]:
            conflicts.append({"query_id": row["query_id"], "field": "mask_answers"})
    if missing or conflicts:
        raise RuntimeError(f"MBA immutable-source mismatch: missing={len(missing)}, conflicts={len(conflicts)}")
    return {
        "selected_mba_queries": len(selected),
        "matched_by_exact_query_id": len(selected) - len(missing),
        "missing": len(missing),
        "query_membership_target_session_conflicts": len(conflicts),
    }


def main() -> None:
    if (EXP / "PHASE_A2_RESULT.json").exists():
        raise RuntimeError("Phase-A2 result already exists; repair precommit refused")
    selected_rows = read_jsonl(SELECTED)
    source_rows = read_jsonl(MBA_SOURCE)
    alignment = validate_alignment(selected_rows, source_rows)
    retrieval_rows = read_jsonl(RETRIEVAL)
    mba_retrieval = [row for row in retrieval_rows if row.get("attack") == "MBA"]
    if len(mba_retrieval) != 1600:
        raise RuntimeError(f"MBA retrieval row drift: {len(mba_retrieval)}")
    if any("mask_answers" in row for row in mba_retrieval):
        raise RuntimeError("unexpected pre-existing mask_answers in frozen retrieval artifact")
    precommit = {
        "campaign": EXP.name,
        "repair": "PHASE_A2_MBA_SCORING_METADATA_JOIN_V1",
        "created_utc": now(),
        "status": "PRECOMMITTED_BEFORE_REPAIRED_SCORING",
        "cause": "PHASE_A2_SELECTED_QUERIES omitted paper-faithful MBA mask_answers required only by the frozen scorer",
        "allowed_change": "join immutable mask_answers by exact query_id at scoring read time",
        "forbidden_changes": [
            "generation rerun", "answer modification", "retrieval modification",
            "membership inference or imputation", "fuzzy ID matching",
            "native scorer semantic change", "detector/model/threshold change",
        ],
        "alignment": alignment,
        "inputs_sha256": {
            str(SELECTED): sha_file(SELECTED),
            str(RETRIEVAL): sha_file(RETRIEVAL),
            str(MBA_SOURCE): sha_file(MBA_SOURCE),
            str(EXP / "runtime" / "PHASE_A2_GENERATED_ANSWERS.jsonl"): sha_file(EXP / "runtime" / "PHASE_A2_GENERATED_ANSWERS.jsonl"),
            str(EXP / "runtime" / "PHASE_A2_GENERATION_MANIFEST.json"): sha_file(EXP / "runtime" / "PHASE_A2_GENERATION_MANIFEST.json"),
        },
        "code_sha256": {
            str(ORIGINAL_SCORER): sha_file(ORIGINAL_SCORER),
            str(REPAIRED_SCORER): sha_file(REPAIRED_SCORER),
            str(Path(__file__).resolve()): sha_file(Path(__file__).resolve()),
        },
        "expected_join": {"unique_query_ids": 400, "retrieval_rows": 1600, "versions_per_query": 4},
    }
    path = EXP / "configs" / "PHASE_A2_SCORING_REPAIR_PRECOMMIT.json"
    digest = freeze_json(path, precommit)
    checkpoint("PHASE_A2_SCORING_REPAIR_PRECOMMITTED", precommit_sha256=digest, **alignment)
    print(json.dumps(precommit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
