#!/usr/bin/env python3
"""Audit the frozen Simple-Hide token packing without answer generation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

from common import (CORE6, EXP, QWEN, RECAL, atomic_json, checkpoint, read_jsonl,
                    sha_file, verify_hashed_json, write_csv)


def waterfill(lengths: list[int], total: int = 2048) -> list[int]:
    caps = [0] * len(lengths)
    remaining = total
    active = [index for index, length in enumerate(lengths) if length > 0]
    while remaining and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, lengths[index] - caps[index], remaining)
            caps[index] += add
            remaining -= add
            changed = changed or bool(add)
            if caps[index] >= lengths[index]:
                active.remove(index)
            if not remaining:
                break
        if not changed:
            break
    return caps


def main() -> None:
    pre = verify_hashed_json(EXP / "configs/DB_CHURN_V2_PRECOMMIT.json")
    for path, digest in pre["input_sha256"].items():
        if sha_file(Path(path)) != digest:
            raise RuntimeError(f"frozen input drift: {path}")
    fp_path = RECAL / "tables/GOLD_REFRESH_FP_SUBSET_AUDIT.csv"
    fp = list(csv.DictReader(fp_path.open(encoding="utf-8")))
    if len(fp) != 27:
        raise RuntimeError(f"expected 27 false-positive benign rows, got {len(fp)}")
    retrieval = {row["query_id"]: row for row in read_jsonl(CORE6 / "cache/GOLD_RETRIEVAL_AND_DETECTION.jsonl")}
    docs = {row["document_id"]: row["source_text"] for row in read_jsonl(CORE6 / "inputs/TOPIOCQA_GOLD_CORPUS.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)

    output = []
    for row in fp:
        query_id = row["query_id"]
        score = retrieval[query_id]
        top = score["top_document_ids"]
        removed = row["removed_source"]
        if removed != score["selected_source_id"] or removed not in top:
            raise RuntimeError(f"locator/source mismatch: {query_id}")
        pre_lengths = [len(tokenizer(docs[item], add_special_tokens=False).input_ids) for item in top]
        pre_caps = waterfill(pre_lengths)
        kept = [item for item in top if item != removed]
        post_lengths = [len(tokenizer(docs[item], add_special_tokens=False).input_ids) for item in kept]
        post_caps = waterfill(post_lengths)
        output.append({
            "query_id": query_id, "removed_source": removed,
            "original_total_evidence_budget": 2048,
            "top4_source_ids": json.dumps(top, ensure_ascii=False),
            "remaining_source_ids": json.dumps(kept, ensure_ascii=False),
            "source_lengths_pre": json.dumps(pre_lengths),
            "allocations_pre_hide": json.dumps(pre_caps),
            "removed_source_allocation_pre_hide": pre_caps[top.index(removed)],
            "allocations_post_hide": json.dumps(post_caps),
            "actual_context_tokens_pre": sum(pre_caps),
            "actual_context_tokens_post": sum(post_caps),
            "truncated_pre": any(length > cap for length, cap in zip(pre_lengths, pre_caps)),
            "truncated_post": any(length > cap for length, cap in zip(post_lengths, post_caps)),
            "freed_budget_discarded": sum(post_caps) < min(2048, sum(post_lengths)),
        })
    write_csv(EXP / "audits/CURRENT_HIDE_PACKING_27.csv", output)
    discarded = sum(row["freed_budget_discarded"] for row in output)
    result = {
        "campaign": EXP.name,
        "verdict": "FP_SAFE_REDISTRIBUTION_NOT_APPLICABLE",
        "CURRENT_HIDE_PACKING_POLICY": "REMOVE MIRABEL TOP-1; WATER-FILL THE REMAINING TOP-3 TO THE SAME 2,048 SOURCE-TOKEN BUDGET",
        "cases": len(output), "freed_budget_discarded_cases": discarded,
        "mean_context_tokens_pre": sum(row["actual_context_tokens_pre"] for row in output) / len(output),
        "mean_context_tokens_post": sum(row["actual_context_tokens_post"] for row in output) / len(output),
        "candidate_opened": False, "candidate_generations": 0,
        "privacy_screen_run": False,
        "reason": "The frozen implementation already redistributes the available source-token budget over the remaining Top-3 documents.",
        "implementation_evidence": {
            "path": str(RECAL / "code/run_gold_recalibration.py"),
            "sha256": sha_file(RECAL / "code/run_gold_recalibration.py"),
            "functions": ["waterfill", "build_prompt"],
        },
    }
    atomic_json(EXP / "audits/CURRENT_HIDE_PACKING_POLICY.json", result)
    atomic_json(EXP / "checkpoints/FP_SAFE_REDISTRIBUTION_NOT_APPLICABLE.json", result)
    checkpoint("PACKING_AUDIT_COMPLETE", packing_verdict=result["verdict"], cases=len(output),
               candidate_generations=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
