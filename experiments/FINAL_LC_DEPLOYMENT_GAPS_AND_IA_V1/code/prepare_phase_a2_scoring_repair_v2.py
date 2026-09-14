#!/usr/bin/env python3
"""Precommit the complete metadata-only Phase-A2 native-scorer repair."""
from __future__ import annotations

import json
from pathlib import Path

from common import EXP, ROOT, checkpoint, freeze_json, now, read_jsonl, sha_file


SELECTED = EXP / "inputs" / "PHASE_A2_SELECTED_QUERIES.jsonl"
RETRIEVAL = EXP / "cache" / "PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl"
SOURCES = {
    "MBA": (ROOT / "experiments/LC_MIRABEL_LARGE_V1/inputs/LARGE_MBA_ATTACK_QUERIES.jsonl", "mask_answers", 400),
    "S²-MIA": (ROOT / "experiments/FINAL_8ATTACK_E2E_FRAMEWORK_V1/inputs/S2_ATTACK_QUERIES.jsonl", "s2_full_target", 802),
    "DCMI-Std-Q2": (ROOT / "experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1/inputs/DCMI_STD_Q2_QUERIES.jsonl", "variant", 800),
}


def audit_source(selected: list[dict], attack: str, path: Path, field: str, expected: int) -> dict:
    chosen = [row for row in selected if row.get("attack") == attack]
    source = {row["query_id"]: row for row in read_jsonl(path)}
    if len(chosen) != expected or len({row["query_id"] for row in chosen}) != expected:
        raise RuntimeError(f"{attack} selected cohort drift: {len(chosen)}")
    missing, conflicts, field_missing = [], [], []
    for row in chosen:
        original = source.get(row["query_id"])
        if original is None:
            missing.append(row["query_id"])
            continue
        for key in ("query", "membership", "target_id", "session_id"):
            if original.get(key) != row.get(key):
                conflicts.append(f"{row['query_id']}:{key}")
        if field not in original or original[field] in (None, "", {}):
            field_missing.append(row["query_id"])
    if missing or conflicts or field_missing:
        raise RuntimeError(f"{attack} source mismatch: missing={len(missing)} conflicts={len(conflicts)} field_missing={len(field_missing)}")
    return {"selected": len(chosen), "exact_id_matches": len(chosen), "missing": 0,
            "provenance_conflicts": 0, "field": field, "field_missing": 0}


def main() -> None:
    if (EXP / "PHASE_A2_RESULT.json").exists():
        raise RuntimeError("Phase-A2 result already exists; repair precommit refused")
    selected = read_jsonl(SELECTED)
    audits = {attack: audit_source(selected, attack, *spec) for attack, spec in SOURCES.items()}
    retrieval = read_jsonl(RETRIEVAL)
    expected_rows = {attack: expected * 4 for attack, (_, _, expected) in SOURCES.items()}
    actual_rows = {attack: sum(row.get("attack") == attack for row in retrieval) for attack in SOURCES}
    if actual_rows != expected_rows:
        raise RuntimeError(f"retrieval attack-row drift: {actual_rows} != {expected_rows}")
    this_file = Path(__file__).resolve()
    runner = EXP / "code" / "run_phase_a2_scoring_repair_v2.py"
    original = EXP / "code" / "run_phase_a2_scoring.py"
    precommit = {
        "campaign": EXP.name, "repair": "PHASE_A2_NATIVE_SCORER_METADATA_JOIN_V2",
        "created_utc": now(), "status": "PRECOMMITTED_BEFORE_V2_REPAIRED_SCORING",
        "supersedes_without_deleting": "PHASE_A2_MBA_SCORING_METADATA_JOIN_V1",
        "cause": "A2 substrate retained common fields but omitted MBA, S2-MIA, and DCMI native-scorer-only metadata",
        "allowed_change": "exact-query-id join of mask_answers, s2_full_target, and variant from immutable original attack artifacts at scoring read time",
        "forbidden_changes": ["generation rerun", "answer/retrieval modification", "fuzzy matching",
                              "field inference or imputation", "scorer semantic change", "detector/model/threshold change"],
        "source_audits": audits, "expected_retrieval_rows": expected_rows,
        "inputs_sha256": {
            str(SELECTED): sha_file(SELECTED), str(RETRIEVAL): sha_file(RETRIEVAL),
            str(EXP / "runtime/PHASE_A2_GENERATED_ANSWERS.jsonl"): sha_file(EXP / "runtime/PHASE_A2_GENERATED_ANSWERS.jsonl"),
            str(EXP / "runtime/PHASE_A2_GENERATION_MANIFEST.json"): sha_file(EXP / "runtime/PHASE_A2_GENERATION_MANIFEST.json"),
            **{str(path): sha_file(path) for path, _, _ in SOURCES.values()},
        },
        "code_sha256": {str(original): sha_file(original), str(runner): sha_file(runner), str(this_file): sha_file(this_file)},
    }
    path = EXP / "configs/PHASE_A2_SCORING_REPAIR_V2_PRECOMMIT.json"
    digest = freeze_json(path, precommit)
    checkpoint("PHASE_A2_SCORING_REPAIR_V2_PRECOMMITTED", precommit_sha256=digest,
               mba_exact=400, s2_exact=802, dcmi_exact=800, provenance_conflicts=0)
    print(json.dumps(precommit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
