#!/usr/bin/env python3
"""Frozen-NLI screen for the six Exp196 benign answers changed by G1.

This is an automatic diagnostic, not a human hallucination label.  It uses the
exact G1 source allocation and the already frozen DeBERTa NLI classifier.
"""
from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp196_selective_rank_agnostic_globalcap64_20260829"
EXP170 = PROJECT / "exp170_grounding_dcmi_confirmation_20260828"
EXP188 = PROJECT / "exp188_globalcap_selective_dcmcel_20260829"
EXP195 = PROJECT / "exp195_minimal_qll_exposure_guard_20260829"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
NLI = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def main():
    core = module("exp196_nli_core", EXP188 / "code/exp188_core.py")
    grounding = module("exp196_nli_grounding", EXP170 / "code/run_grounding_confirmation.py")
    cases = pd.read_pickle(EXP195 / "private/EXP195_NORMAL_CASES.private.pkl.gz", compression="gzip")
    answers = pd.read_csv(ROOT / "private/EXP196_G1_RESPONSES.private.csv.gz",
                          keep_default_na=False, dtype={"case_id": str})
    changed = answers[(answers.panel.eq("NORMAL_GOLD")) &
                      answers.response.astype(str).ne(answers.vanilla_response.astype(str))]
    changed = changed.merge(cases[["case_id", "source_ids", "source_texts"]],
                            on="case_id", validate="one_to_one")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    units, pairs = [], []
    for row in changed.itertuples(index=False):
        ids, texts = list(map(str, row.source_ids)), list(map(str, row.source_texts))
        selected = str(row.selected_source_id)
        others = [source for source in ids if source != selected]
        caps = {selected: 64}
        for index, source in enumerate(others):
            caps[source] = 662 if index < 2 else 661
        visible, hidden_parts = [], []
        for source, text in zip(ids, texts):
            token_ids = qwen_tokenizer(text, add_special_tokens=False).input_ids
            visible.append(qwen_tokenizer.decode(token_ids[:caps[source]], skip_special_tokens=True).strip())
            hidden_parts.append(qwen_tokenizer.decode(token_ids[caps[source]:], skip_special_tokens=True).strip())
        visible = visible + [""] * 6
        hidden = "\n\n".join(hidden_parts)
        for claim_index, (_, _, claim) in enumerate(core.claim_spans(str(row.response)), 1):
            key = f"{row.case_id}:{claim_index}"
            units.append({"claim_key": key, "case_id": row.case_id, "row_id": row.row_id,
                          "claim_index": claim_index, "claim": claim, "answer": row.response,
                          "visible": visible, "hidden": hidden})
            for premise_index, premise in enumerate([*visible, hidden]):
                pairs.append({"claim_key": key, "premise_index": premise_index,
                              "premise": premise or "[EMPTY]", "hypothesis": claim})
    pair_frame = pd.DataFrame(pairs)
    tokenizer = AutoTokenizer.from_pretrained(NLI, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI, local_files_only=True, dtype=dtype).to(device).eval()
    entailment_id = model.config.label2id["entailment"]
    contradiction_id = model.config.label2id["contradiction"]
    entailment, contradiction = [], []
    batch = 96 if device == "cuda" else 8
    for offset in range(0, len(pair_frame), batch):
        cell = pair_frame.iloc[offset:offset + batch]
        encoded = tokenizer(cell.premise.astype(str).tolist(), cell.hypothesis.astype(str).tolist(),
                            padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(**encoded).logits.float(), -1).cpu().numpy()
        entailment.extend(probs[:, entailment_id].tolist())
        contradiction.extend(probs[:, contradiction_id].tolist())
    pair_frame["entailment"] = entailment
    pair_frame["contradiction"] = contradiction
    by_claim = {key: cell.sort_values("premise_index") for key, cell in pair_frame.groupby("claim_key")}
    labels = []
    for unit in units:
        cell = by_claim[unit["claim_key"]]
        label, name, rationale, support = grounding.classify(
            unit["claim"], unit["answer"], unit["visible"], unit["hidden"],
            cell.entailment.to_numpy(float), cell.contradiction.to_numpy(float))
        labels.append({key: value for key, value in unit.items() if key not in {"visible", "hidden", "answer"}} |
                      {"grounding_label": label, "grounding_name": name,
                       "evidence_rationale": rationale, "support_location": support,
                       "unsupported_risk": label in {"G4", "G5", "G6"}})
    claims = pd.DataFrame(labels)
    rows = claims.groupby(["case_id", "row_id"], as_index=False).agg(
        claims=("claim_key", "size"), unsupported_risk=("unsupported_risk", "max"))
    summary_path = ROOT / "tables/TABLE_196_07_HALLUCINATION_SCREEN.csv"
    summary = pd.read_csv(summary_path, keep_default_na=False)
    summary = summary.drop(columns=["unsupported_claim_rate", "unsupported_claim_status"], errors="ignore")
    summary["unsupported_claim_rate"] = float(rows.unsupported_risk.mean()) if len(rows) else 0.0
    summary["unsupported_claim_status"] = "FROZEN_NLI_CHANGED_SUBSET_AUTOMATIC_NOT_HUMAN"
    claims.to_csv(ROOT / "private/EXP196_CHANGED_NORMAL_NLI_CLAIMS.private.csv.gz",
                  index=False, compression="gzip")
    rows.to_csv(ROOT / "tables/TABLE_196_09_CHANGED_NORMAL_NLI_ROWS.csv", index=False)
    summary.to_csv(summary_path, index=False)
    result_path = ROOT / "FINAL_RESULT.json"
    result = json.loads(result_path.read_text())
    result["hallucination_screen"] = summary.iloc[0].to_dict()
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"changed_rows": len(rows), "claims": len(claims),
                      "automatic_unsupported_row_rate": float(rows.unsupported_risk.mean()) if len(rows) else 0.0,
                      "device": device, "human_verified": False}, indent=2))
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
