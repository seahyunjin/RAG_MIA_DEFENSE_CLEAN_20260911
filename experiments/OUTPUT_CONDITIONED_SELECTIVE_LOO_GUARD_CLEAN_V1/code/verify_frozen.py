#!/usr/bin/env python3
"""Fail-closed verification of every frozen CLEAN_V1 Phase-1 input."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from common import ROOT, DOMAINS, sha256_file, sha256_text, atomic_json


def main() -> None:
    precommit_path = ROOT / "configs/PRECOMMIT.json"
    expected = (ROOT / "configs/PRECOMMIT.sha256").read_text().split()[0]
    actual = sha256_file(precommit_path)
    if actual != expected:
        raise RuntimeError(f"PRECOMMIT drift: {actual} != {expected}")
    pre = json.loads(precommit_path.read_text())
    if not pre.get("independent_from_legacy_scores") or pre.get("legacy_numeric_inputs_used") != []:
        raise RuntimeError("legacy numeric isolation contract failed")
    attack = pd.read_csv(ROOT / "private/PHASE1_ATTACK_SELECTION.csv.gz", keep_default_na=False, dtype=str)
    benign = pd.read_csv(ROOT / "private/PHASE1_BENIGN_SELECTION.csv.gz", keep_default_na=False, dtype=str)
    if len(attack) != 280 or attack.case_id.nunique() != 280 or attack.session_id.nunique() != 120:
        raise RuntimeError("attack cohort drift")
    if len(benign) != 500 or benign.case_id.nunique() != 500 or benign.query_sha256.nunique() != 500:
        raise RuntimeError("benign cohort drift")
    if benign.groupby("split").size().to_dict() != {"CALIBRATION": 250, "HOLDOUT": 250}:
        raise RuntimeError("benign split drift")
    if set(attack.query_sha256) & set(benign.query_sha256):
        raise RuntimeError("attack/benign query overlap")
    if set(attack.target_id) & set(benign.target_id):
        raise RuntimeError("attack target/benign gold overlap")
    corpus_checks = {}
    for domain in DOMAINS:
        path = ROOT / "private" / f"CORPUS_{domain}.csv.gz"
        expected_file = pre["corpus"]["audit"][domain]["corpus_file_sha256"]
        if sha256_file(path) != expected_file:
            raise RuntimeError(f"corpus file drift: {domain}")
        corpus = pd.read_csv(path, keep_default_na=False, dtype={"document_id": str})
        if len(corpus) != 1000 or corpus.document_id.nunique() != 1000:
            raise RuntimeError(f"corpus size/ID drift: {domain}")
        ordered = sha256_text("\n".join(corpus.sort_values("corpus_order").document_id.astype(str)))
        expected_order = pre["corpus"]["audit"][domain]["ordered_document_id_sha256"]
        if ordered != expected_order:
            raise RuntimeError(f"corpus order drift: {domain}")
        corpus_checks[domain] = {"rows": len(corpus), "file_sha256": expected_file,
                                 "ordered_document_id_sha256": ordered}
    addendum = ROOT / "configs/PRECOMMIT_ADDENDUM_BEFORE_RETRIEVAL.json"
    addendum_hash = ROOT / "configs/PRECOMMIT_ADDENDUM_BEFORE_RETRIEVAL.sha256"
    if not addendum.exists() or not addendum_hash.exists():
        raise RuntimeError("pre-retrieval addendum missing")
    if sha256_file(addendum) != addendum_hash.read_text().split()[0]:
        raise RuntimeError("pre-retrieval addendum drift")
    add = json.loads(addendum.read_text())
    if add["parent_precommit_sha256"] != actual or not add["written_before_retrieval"]:
        raise RuntimeError("addendum chain failure")
    for relative, digest in add["code_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"code drift: {relative}")
    result = {"verdict": "FROZEN_INPUTS_VERIFIED", "precommit_sha256": actual,
              "addendum_sha256": sha256_file(addendum), "attack_queries": len(attack),
              "attack_sessions": attack.session_id.nunique(), "benign_queries": len(benign),
              "corpora": corpus_checks, "legacy_numeric_inputs_used": []}
    atomic_json(ROOT / "audits/FROZEN_INPUT_VERIFICATION.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
