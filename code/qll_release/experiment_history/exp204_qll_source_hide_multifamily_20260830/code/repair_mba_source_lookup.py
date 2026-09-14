#!/usr/bin/env python3
"""Repair Exp204 MBA source lookup using immutable title+text records.

The frozen Exp87 lookup kept only the `text` field.  Twenty-three TREC-COVID
records have an empty text field but a non-empty title, causing mba_truth to
raise before a response could be scored.  This script does not impute scores or
change MBA semantics.  It restores the canonical source content as title+text,
then reruns the same frozen mba_truth -> normalize_mba -> mba_accuracy route.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp204_qll_source_hide_multifamily_20260830"
AD3 = PROJECT.parent / "AD-test-LLM3"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
EXP152 = PROJECT / "exp152_exp150_clean_3k_confirmation_20260825"
SCORES = ROOT / "private/EXP204_NATIVE_SCORES.private.csv.gz"
RESPONSES = ROOT / "private/EXP204_RESPONSES.private.csv.gz"
REPAIRED = ROOT / "private/EXP204_NATIVE_SCORES_MBA_COMPLETE.private.csv.gz"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_records(targets):
    sources = [
        AD3 / "data/beir/BeIR_trec-covid/corpus_member.jsonl",
        AD3 / "data/beir/BeIR_trec-covid/corpus_nonmember.jsonl",
        Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir/_raw/trec-covid/corpus.jsonl"),
    ]
    records = {}
    provenance = []
    for source in sources:
        if not source.exists():
            continue
        provenance.append({"path": str(source), "sha256": sha256_file(source), "access": "READ_ONLY"})
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                document_id = str(row.get("_id", row.get("id", "")))
                if document_id not in targets:
                    continue
                title = str(row.get("title", "")).strip()
                text = str(row.get("text", "")).strip()
                content = " ".join(value for value in (title, text) if value)
                previous = records.get(document_id)
                if previous is None or len(content) > len(previous["content"]):
                    records[document_id] = {"title": title, "text": text, "content": content,
                                            "source": str(source)}
    pd.DataFrame(provenance).to_csv(ROOT / "provenance/MBA_CANONICAL_SOURCE_FILES.csv", index=False)
    missing = sorted(set(targets)-set(records))
    if missing:
        raise RuntimeError(f"MBA immutable records missing: {missing}")
    return records


def main():
    exp87 = load(EXP87 / "code/exp87_scoring.py", "exp204_mba_exp87")
    sys.path.insert(0, str(AD3))
    from src.exp44_native import mba_accuracy

    scores = pd.read_csv(SCORES, low_memory=False)
    responses = pd.read_csv(RESPONSES, keep_default_na=False, low_memory=False, dtype={"row_id": str})
    manifest = pd.read_csv(EXP152 / "private/EXP152_CLEAN_3K_MANIFEST.private.csv.gz",
                           keep_default_na=False, low_memory=False, dtype={"row_id": str})
    mba_manifest = manifest[manifest.family.eq("MBA")].drop_duplicates("session_id").set_index("session_id")
    missing_mask = scores.attack_family.eq("MBA") & scores.attack_score.isna()
    missing = scores[missing_mask].copy()
    targets = {str(value) for value in missing.source_document_id}
    documents = canonical_records(targets)
    response_map = responses[responses.family.eq("MBA")].set_index(["session_id", "condition"]).response.to_dict()
    audit = []
    for index, row in missing.iterrows():
        session_id = str(row.session_id)
        condition = str(row.condition)
        meta = mba_manifest.loc[session_id]
        document_id = str(row.source_document_id)
        record = documents[document_id]
        if record["text"]:
            raise RuntimeError(f"expected title-only frozen lookup failure, found text: {document_id}")
        if not record["title"]:
            raise RuntimeError(f"immutable title unavailable: {document_id}")
        truth = exp87.mba_truth(record["content"])
        response = str(response_map[(session_id, condition)])
        score = float(mba_accuracy(truth, exp87.normalize_mba(response)))
        scores.loc[index, "attack_score"] = score
        scores.loc[index, "original_family_scorer_rerun"] = True
        audit.append({"condition": condition, "session_id": session_id, "member": int(row.member),
            "target_document_id": document_id, "failure_class": "EMPTY_TEXT_WITH_NONEMPTY_TITLE",
            "repair": "IMMUTABLE_TITLE_PLUS_TEXT_SOURCE_RECONSTRUCTION",
            "canonical_source": record["source"], "title_sha256": hashlib.sha256(record["title"].encode()).hexdigest(),
            "truth_items": len(truth), "response_was_refusal": response.strip().casefold() in {"i don't know.", "i don't know"},
            "repaired_score": score, "membership_used_for_recovery": False,
            "response_used_for_source_recovery": False, "synthetic_score": False})
    if scores.attack_family.eq("MBA").sum() != 720 or scores[scores.attack_family.eq("MBA")].attack_score.isna().any():
        raise RuntimeError("MBA repair did not produce complete 360-session x 2-condition scores")
    scores.to_csv(REPAIRED, index=False, compression="gzip")
    detail = pd.DataFrame(audit)
    detail.to_csv(ROOT / "audits/MBA_SOURCE_LOOKUP_REPAIR_DETAIL.csv", index=False)
    detail.groupby(["condition", "member", "failure_class", "repair"], as_index=False).agg(
        sessions=("session_id", "size"), mean_repaired_score=("repaired_score", "mean"),
        refusals=("response_was_refusal", "sum")).to_csv(ROOT / "tables/TABLE_204_08_MBA_REPAIR_AUDIT.csv", index=False)

    evaluator = load(EXP188 / "code/run_exp188.py", "exp204_mba_metrics")
    auc, low, _ = evaluator.metric_tables(scores, "EXP204_MBA_COMPLETE")
    auc["effective_auc"] = auc.roc_auc.map(lambda value: max(float(value), 1-float(value)))
    auc.to_csv(ROOT / "tables/TABLE_204_09_NATIVE_EAUC_COMPLETE.csv", index=False)
    low.to_csv(ROOT / "tables/TABLE_204_10_NATIVE_LOW_FPR_COMPLETE.csv", index=False)

    gates = pd.read_csv(ROOT / "tables/TABLE_204_07_PRIMARY_GATES.csv")
    primary_mba = float(auc[(auc.condition.eq("QLL_SOURCE_HIDE_SESSION_STICKY")) &
                            auc.attack_family.eq("MBA")].effective_auc.iloc[0])
    gates.loc[gates.gate.eq("native_MBA"), "value"] = primary_mba
    gates.loc[gates.gate.eq("native_MBA"), "passed"] = primary_mba <= .65
    gates.to_csv(ROOT / "tables/TABLE_204_11_PRIMARY_GATES_MBA_COMPLETE.csv", index=False)

    original_result = ROOT / "FINAL_RESULT.json"
    history = ROOT / "provenance/FINAL_RESULT_PRE_MBA_REPAIR.json"
    if not history.exists():
        shutil.copy2(original_result, history)
    result = json.loads(original_result.read_text())
    result.update({"mba_repair_status": "MBA_SOURCE_LOOKUP_REPAIRED_COMPLETE",
        "mba_missing_before": int(len(missing)), "mba_missing_after": 0,
        "mba_sessions_complete_per_condition": 360, "mba_effective_auc_complete": primary_mba,
        "mba_synthetic_scores": 0, "mba_membership_used_for_recovery": False,
        "passed": bool(gates.passed.astype(bool).all()),
        "completed_utc_after_mba_repair": datetime.now(timezone.utc).isoformat()})
    original_result.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+"\n")
    (ROOT / "checkpoints/MBA_REPAIR_COMPLETE.json").write_text(json.dumps({
        "stage": "MBA_REPAIR_COMPLETE", "repaired_score_rows": len(missing),
        "unique_sessions": missing.session_id.nunique(), "missing_after": 0,
        "synthetic_scores": 0, "primary_mba_effective_auc": primary_mba,
        "updated_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
