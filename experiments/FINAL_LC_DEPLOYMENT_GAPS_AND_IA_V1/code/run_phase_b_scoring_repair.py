#!/usr/bin/env python3
"""Run frozen Phase-B scoring with exact-ID native-scorer metadata joins."""
from __future__ import annotations

import json
from pathlib import Path

import run_phase_b_scoring as frozen_scorer
from common import EXP, ROOT, atomic_json, checkpoint, now, read_jsonl, sha_file, verify_hashed_json


RETRIEVAL = (EXP / "cache/PHASE_A2_RETRIEVAL_AND_DETECTION.jsonl").resolve()
SOURCES = {
    "MBA": (ROOT / "experiments/LC_MIRABEL_LARGE_V1/inputs/LARGE_MBA_ATTACK_QUERIES.jsonl", ("mask_answers",), 400),
    "S²-MIA": (ROOT / "experiments/FINAL_8ATTACK_E2E_FRAMEWORK_V1/inputs/S2_ATTACK_QUERIES.jsonl", ("s2_full_target",), 802),
    "DCMI-Std-Q2": (ROOT / "experiments/FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1/inputs/DCMI_STD_Q2_QUERIES.jsonl", ("variant",), 800),
}


def enrich(rows: list[dict]) -> tuple[list[dict], dict]:
    sources = {attack: {row["query_id"]: row for row in read_jsonl(path)}
               for attack, (path, _, _) in SOURCES.items()}
    output, audits = [], {attack: {"ids": set(), "rows": 0} for attack in SOURCES}
    for row in rows:
        attack = row.get("attack")
        if attack not in SOURCES:
            output.append(row); continue
        original = sources[attack].get(row["query_id"])
        if original is None: raise RuntimeError(f"{attack} exact query_id missing: {row['query_id']}")
        for key in ("query", "membership", "target_id", "session_id"):
            if original.get(key) != row.get(key): raise RuntimeError(f"{attack} provenance conflict: {row['query_id']}/{key}")
        additions = {field: original[field] for field in SOURCES[attack][1]}
        output.append({**row, **additions})
        audits[attack]["ids"].add(row["query_id"]); audits[attack]["rows"] += 1
    serial = {attack: {"unique_query_ids": len(value["ids"]), "retrieval_rows_joined": value["rows"]}
              for attack, value in audits.items()}
    expected = {attack: {"unique_query_ids": count, "retrieval_rows_joined": count * 4}
                for attack, (_, _, count) in SOURCES.items()}
    if serial != expected: raise RuntimeError(f"Phase-B join count drift: {serial} != {expected}")
    return output, serial


def main() -> None:
    precommit_path = EXP / "configs/PHASE_B_SCORING_REPAIR_PRECOMMIT.json"
    precommit = verify_hashed_json(precommit_path)
    for path, expected in {**precommit["inputs_sha256"], **precommit["code_sha256"]}.items():
        actual = sha_file(Path(path))
        if actual != expected: raise RuntimeError(f"Phase-B repair drift: {path}")
    original_reader = frozen_scorer.read_jsonl
    audit_holder = {}
    def repaired_reader(path: Path) -> list[dict]:
        rows = original_reader(path)
        if Path(path).resolve() == RETRIEVAL:
            joined, audit = enrich(rows); audit_holder.update(audit); return joined
        return rows
    frozen_scorer.read_jsonl = repaired_reader
    checkpoint("PHASE_B_SCORING_REPAIR_STARTED", precommit_sha256=sha_file(precommit_path))
    frozen_scorer.main()
    audit = {"verdict": "PHASE_B_NATIVE_SCORER_METADATA_REPAIR_APPLIED", "completed_utc": now(),
             "joins": audit_holder, "exact_id_only": True, "fuzzy_matches": 0, "imputed_fields": 0,
             "generated_answers_unchanged": True, "retrieval_artifact_unchanged": True,
             "frozen_scorer_unchanged": True, "detector_model_threshold_unchanged": True,
             "precommit_sha256": sha_file(precommit_path)}
    atomic_json(EXP / "audits/PHASE_B_NATIVE_SCORER_METADATA_REPAIR.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
