#!/usr/bin/env python3
"""Execute frozen Phase-A2 scoring with complete exact-ID metadata joins."""
from __future__ import annotations

import json
from pathlib import Path

import run_phase_a2_scoring as frozen_scorer
from common import EXP, ROOT, atomic_json, checkpoint, now, read_jsonl, sha_file, verify_hashed_json


RETRIEVAL = (EXP / "cache/PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl").resolve()
SOURCES = {
    "MBA": (ROOT / "experiments/LC_MIRABEL_LARGE_V1/inputs/LARGE_MBA_ATTACK_QUERIES.jsonl", ("mask_answers",)),
    "S²-MIA": (ROOT / "experiments/FINAL_8ATTACK_E2E_FRAMEWORK_V1/inputs/S2_ATTACK_QUERIES.jsonl", ("s2_full_target",)),
    "DCMI-Std-Q2": (ROOT / "experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1/inputs/DCMI_STD_Q2_QUERIES.jsonl", ("variant",)),
}


def enrich_retrieval(rows: list[dict], sources: dict[str, tuple[list[dict], tuple[str, ...]]]) -> tuple[list[dict], dict]:
    indices = {attack: {row["query_id"]: row for row in source_rows}
               for attack, (source_rows, _) in sources.items()}
    output, joined_rows, joined_ids = [], {attack: 0 for attack in sources}, {attack: set() for attack in sources}
    for row in rows:
        attack = row.get("attack")
        if attack not in sources:
            output.append(row)
            continue
        original = indices[attack].get(row["query_id"])
        if original is None:
            raise RuntimeError(f"{attack} source missing exact query_id: {row['query_id']}")
        for key in ("query", "membership", "target_id", "session_id"):
            if original.get(key) != row.get(key):
                raise RuntimeError(f"{attack} source conflict: {row['query_id']}/{key}")
        additions = {}
        for field in sources[attack][1]:
            if field not in original or original[field] in (None, "", {}):
                raise RuntimeError(f"{attack} source field absent: {row['query_id']}/{field}")
            additions[field] = original[field]
        output.append({**row, **additions})
        joined_rows[attack] += 1
        joined_ids[attack].add(row["query_id"])
    audit = {attack: {"unique_query_ids": len(joined_ids[attack]), "retrieval_rows_joined": joined_rows[attack]}
             for attack in sources}
    expected = {"MBA": {"unique_query_ids": 400, "retrieval_rows_joined": 1600},
                "S²-MIA": {"unique_query_ids": 802, "retrieval_rows_joined": 3208},
                "DCMI-Std-Q2": {"unique_query_ids": 800, "retrieval_rows_joined": 3200}}
    if audit != expected:
        raise RuntimeError(f"metadata join count drift: {audit} != {expected}")
    return output, audit


def main() -> None:
    precommit_path = EXP / "configs/PHASE_A2_SCORING_REPAIR_V2_PRECOMMIT.json"
    precommit = verify_hashed_json(precommit_path)
    for path, expected in {**precommit["inputs_sha256"], **precommit["code_sha256"]}.items():
        actual = sha_file(Path(path))
        if actual != expected:
            raise RuntimeError(f"Phase-A2 scoring-repair V2 drift: {path}: {actual} != {expected}")
    sources = {attack: (read_jsonl(path), fields) for attack, (path, fields) in SOURCES.items()}
    original_reader = frozen_scorer.read_jsonl
    audit_holder = {}

    def repaired_reader(path: Path) -> list[dict]:
        rows = original_reader(path)
        if Path(path).resolve() == RETRIEVAL:
            enriched, audit = enrich_retrieval(rows, sources)
            audit_holder.update(audit)
            return enriched
        return rows

    frozen_scorer.read_jsonl = repaired_reader
    checkpoint("PHASE_A2_SCORING_REPAIR_V2_STARTED", repair_precommit_sha256=sha_file(precommit_path))
    frozen_scorer.main()
    audit = {
        "verdict": "PHASE_A2_NATIVE_SCORER_METADATA_REPAIR_V2_APPLIED", "completed_utc": now(),
        "joins": audit_holder, "exact_id_only": True, "fuzzy_matches": 0, "inferred_or_imputed_fields": 0,
        "generated_answers_unchanged": True, "retrieval_artifact_unchanged": True,
        "frozen_scorer_unchanged": True, "detector_model_threshold_unchanged": True,
        "repair_precommit_sha256": sha_file(precommit_path),
    }
    atomic_json(EXP / "audits/PHASE_A2_NATIVE_SCORER_METADATA_REPAIR_V2.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
