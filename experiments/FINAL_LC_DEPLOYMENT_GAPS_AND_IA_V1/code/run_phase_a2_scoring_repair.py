#!/usr/bin/env python3
"""Run the frozen Phase-A2 scorer with an exact-ID MBA metadata sidecar join."""
from __future__ import annotations

import json
from pathlib import Path

import run_phase_a2_scoring as frozen_scorer
from common import EXP, ROOT, atomic_json, checkpoint, now, read_jsonl, sha_file, verify_hashed_json


MBA_SOURCE = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1" / "inputs" / "LARGE_MBA_ATTACK_QUERIES.jsonl"
RETRIEVAL = (EXP / "cache" / "PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl").resolve()


def enrich_retrieval(rows: list[dict], source_rows: list[dict]) -> tuple[list[dict], dict]:
    source = {row["query_id"]: row for row in source_rows if row.get("attack") == "MBA"}
    output = []
    joined_ids = set()
    joined_rows = 0
    for row in rows:
        if row.get("attack") != "MBA":
            output.append(row)
            continue
        original = source.get(row["query_id"])
        if original is None:
            raise RuntimeError(f"MBA source missing exact query_id: {row['query_id']}")
        for key in ("query", "membership", "target_id", "session_id"):
            if original.get(key) != row.get(key):
                raise RuntimeError(f"MBA source conflict: {row['query_id']}/{key}")
        mask_answers = original.get("mask_answers")
        if not isinstance(mask_answers, dict) or not mask_answers:
            raise RuntimeError(f"MBA mask_answers absent: {row['query_id']}")
        output.append({**row, "mask_answers": mask_answers})
        joined_ids.add(row["query_id"])
        joined_rows += 1
    audit = {"unique_query_ids": len(joined_ids), "retrieval_rows_joined": joined_rows,
             "fuzzy_matches": 0, "inferred_or_imputed_fields": 0, "conflicts": 0}
    expected = {"unique_query_ids": 400, "retrieval_rows_joined": 1600,
                "fuzzy_matches": 0, "inferred_or_imputed_fields": 0, "conflicts": 0}
    if audit != expected:
        raise RuntimeError(f"MBA repair join count drift: {audit}")
    return output, audit


def main() -> None:
    precommit_path = EXP / "configs" / "PHASE_A2_SCORING_REPAIR_PRECOMMIT.json"
    precommit = verify_hashed_json(precommit_path)
    for path, expected in {**precommit["inputs_sha256"], **precommit["code_sha256"]}.items():
        actual = sha_file(Path(path))
        if actual != expected:
            raise RuntimeError(f"Phase-A2 scoring-repair drift: {path}: {actual} != {expected}")
    original_reader = frozen_scorer.read_jsonl
    source_rows = read_jsonl(MBA_SOURCE)
    audit_holder: dict = {}

    def repaired_reader(path: Path) -> list[dict]:
        rows = original_reader(path)
        if Path(path).resolve() == RETRIEVAL:
            enriched, audit = enrich_retrieval(rows, source_rows)
            audit_holder.update(audit)
            return enriched
        return rows

    frozen_scorer.read_jsonl = repaired_reader
    checkpoint("PHASE_A2_SCORING_REPAIR_STARTED", repair_precommit_sha256=sha_file(precommit_path))
    frozen_scorer.main()
    audit = {
        "verdict": "PHASE_A2_MBA_SCORING_METADATA_REPAIR_APPLIED",
        "completed_utc": now(), **audit_holder,
        "source_path": str(MBA_SOURCE), "source_sha256": sha_file(MBA_SOURCE),
        "generated_answers_unchanged": True, "retrieval_artifact_unchanged": True,
        "frozen_scorer_unchanged": True, "detector_model_threshold_unchanged": True,
        "repair_precommit_sha256": sha_file(precommit_path),
    }
    atomic_json(EXP / "audits" / "PHASE_A2_MBA_SCORING_METADATA_REPAIR.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
