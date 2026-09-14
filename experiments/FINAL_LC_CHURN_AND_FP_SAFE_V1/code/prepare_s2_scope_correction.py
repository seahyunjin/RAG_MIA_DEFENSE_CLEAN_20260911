#!/usr/bin/env python3
"""Precommit a protocol-only repair excluding frozen S2_REFERENCE rows."""
from __future__ import annotations

import json

from common import EXP, FINAL8, atomic_json, atomic_text, now, read_jsonl, sha_file, sha_text


def main() -> None:
    final8 = read_jsonl(FINAL8 / "cache/FINAL_RETRIEVAL_AND_DETECTION.jsonl")
    s2 = [row for row in final8 if row["attack"] == "S²-MIA"]
    evaluation = [row for row in s2 if row.get("evaluation_split") == "S2_EVALUATION"]
    reference = [row for row in s2 if row.get("evaluation_split") == "S2_REFERENCE"]
    if len(s2) != 2000 or len(evaluation) != 1598 or len(reference) != 402:
        raise RuntimeError(f"unexpected S2 scope: total/eval/ref={len(s2)}/{len(evaluation)}/{len(reference)}")
    member = [row for row in evaluation if row["membership"] == "member"]
    nonmember = [row for row in evaluation if row["membership"] == "nonmember"]
    if len(member) != 799 or len(nonmember) != 799:
        raise RuntimeError("S2 evaluation membership symmetry drift")
    original_files = [EXP / "configs/DB_CHURN_V2_PRECOMMIT.json", EXP / "DB_CHURN_V2_RESULT.json",
                      EXP / "FINAL_RESULT.json", EXP / "tables/DB_CHURN_V2_SUMMARY.csv",
                      EXP / "tables/DB_CHURN_CORE5_CLUSTER_BOOTSTRAP.csv"]
    precommit = {
        "campaign": "DB_CHURN_V2_CORRECTED_CORE5_SCOPE",
        "created_utc": now(),
        "reason": "Initial aggregation included 402 S2_REFERENCE rows. Frozen Core5 evaluation contains only 1,598 S2_EVALUATION rows.",
        "classification_of_initial_result": "INVALID_S2_SCOPE_INCLUSION",
        "allowed_change": "metric aggregation IDs only; exclude S2_REFERENCE",
        "forbidden_changes": ["scores", "embeddings", "retrieval", "detector", "threshold", "formula", "other attack IDs"],
        "s2": {"all": 2000, "evaluation": 1598, "excluded_reference": 402,
               "evaluation_member": 799, "evaluation_nonmember": 799,
               "evaluation_id_sha256": sha_text("\n".join(row["query_id"] for row in evaluation)),
               "excluded_id_sha256": sha_text("\n".join(row["query_id"] for row in reference))},
        "original_artifact_sha256": {str(path): sha_file(path) for path in original_files},
        "score_cache_sha256": {str(path): sha_file(path) for path in sorted((EXP / "cache").glob("V*_QUERY_SCORES.npz"))},
        "code_sha256": {
            "prepare": sha_file(EXP / "code/prepare_s2_scope_correction.py"),
            "run": sha_file(EXP / "code/run_s2_scope_correction.py"),
        },
    }
    path = EXP / "configs/DB_CHURN_V2_S2_SCOPE_CORRECTION_PRECOMMIT.json"
    atomic_json(path, precommit)
    atomic_text(path.with_suffix(".sha256"), f"{sha_file(path)}  {path.name}\n")
    print(json.dumps({"precommit_sha256": sha_file(path), "s2": precommit["s2"]}, indent=2))


if __name__ == "__main__":
    main()
