#!/usr/bin/env python3
"""Freeze the final format-only IA-v6 recovery before GPU generation."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
V5 = PARENT / "IA_STD_Q15_V5_SLOTTED_CONSTRAINED"
CHILD = PARENT / "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
CORE = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    v5_result_path = V5 / "FINAL_RESULT.json"
    v5_result = json.loads(v5_result_path.read_text(encoding="utf-8"))
    if v5_result["verdict"] != "IA_STD_Q15_V5_FORMAT_PREFLIGHT_FAILED":
        raise RuntimeError("frozen v5 failure not found")
    if v5_result["preflight"]["valid"] != 1 or v5_result["preflight"]["duplicate_questions"] != 98:
        raise RuntimeError("v5 failure lineage drift")

    targets_path = LC / "inputs" / "LARGE_SHARED_TARGETS.csv"
    targets = list(csv.DictReader(targets_path.open(encoding="utf-8")))
    if len(targets) != 2000:
        raise RuntimeError("target cohort drift")
    member = sum(row["membership"] == "member" for row in targets)
    nonmember = sum(row["membership"] == "nonmember" for row in targets)
    if (member, nonmember) != (1000, 1000):
        raise RuntimeError("membership cohort drift")

    v5_pre_path = V5 / "configs" / "IA_STD_Q15_V5_PRECOMMIT.json"
    v5_pre = json.loads(v5_pre_path.read_text(encoding="utf-8"))
    preflight_ids = list(v5_pre["preflight"]["session_ids"])
    if len(preflight_ids) != 100:
        raise RuntimeError("preflight cohort drift")

    question_prompt = """Based on the target corpus below, generate exactly ONE specific, indirect, natural yes/no question for IA-Std-Q15.

Target corpus:
{target_text}

This is frozen slot {slot} of 15.
Previously frozen questions for this target:
{previous_questions}

Preserve the existing IA-Std intent: probe a concrete fact from the target indirectly, preserve abbreviations, and do not mention 'the study' or 'the passage'. The question must be answerable from the target as Yes, No, or Unknown. It must differ exactly from every previous question.

The first word must be one of: Is, Are, Was, Were, Does, Do, Did, Can, Could, Has, Have, Had, Will, Would, Should.
Output only the question text, with no JSON, quote, label, prefix, bullet, or explanation. End with exactly one question mark."""
    judgment_prompt = """Based solely on the target context, classify the frozen yes/no question.

Target context:
{target_text}

Frozen question:
{question}

Choose exactly one label: Yes, No, or Unknown. Yes means the context supports the proposition; No means it contradicts the proposition; Unknown means it is not determined. Output only the label."""

    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    label_trie = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("Yes", "No", "Unknown")}
    if any(not ids for ids in label_trie.values()):
        raise RuntimeError("empty judgment token sequence")

    code_paths = [CHILD / "code" / "ia_v6_constraints.py", CHILD / "code" / "prepare_ia_v6.py",
                  CHILD / "code" / "run_ia_v6.py", CHILD / "tests" / "test_ia_v6_constraints.py"]
    model_files = [QWEN / name for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")]
    precommit = {
        "campaign": "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED",
        "name": "IA-Std-Q15-v6-DuplicateConstrained",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "final format-only standardized IA recovery; not Original IA and not paper-exact IA",
        "v5_frozen_failure": {"path": str(v5_result_path), "sha256": sha_file(v5_result_path),
                              "verdict": v5_result["verdict"], "valid": 1,
                              "duplicate_sessions": 98, "performance_observed": False},
        "cohort": {"path": str(targets_path), "sha256": sha_file(targets_path),
                   "sessions": 2000, "member": member, "nonmember": nonmember,
                   "ordered_ids": [row["document_id"] for row in targets],
                   "ordered_id_sha256": hashlib.sha256("\n".join(row["document_id"] for row in targets).encode()).hexdigest()},
        "generation_model": {"model": "Qwen/Qwen2.5-3B-Instruct", "snapshot": QWEN.name,
                             "files_sha256": {name.name: sha_file(name) for name in model_files}},
        "question_stage": {"slots": 15, "prompt": question_prompt, "output": "QUESTION TEXT ONLY",
                           "host_serialization": {"id": "slot 1..15", "question": "exact accepted decoded text"},
                           "duplicate_constraint": "block only a next token that exactly completes a previously accepted generated token sequence for that session",
                           "normalization_audit": "Unicode NFKC; trim surrounding whitespace; collapse repeated whitespace",
                           "semantic_duplicate_filter": False, "retry": False, "max_new_tokens": 96},
        "judgment_stage": {"slots": 15, "prompt": judgment_prompt, "labels": ["Yes", "No", "Unknown"],
                           "allowed_token_trie": label_trie, "constraint": "exact label trie followed by EOS",
                           "fuzzy_mapping": False, "inference": False, "interpolation": False},
        "decoding": {"do_sample": False, "num_beams": 1, "temperature": None, "top_p": None,
                     "seed": 20260912, "input_truncation": 4096, "question_batch": 10,
                     "judgment_batch": 24, "attempts_per_slot": 1},
        "frozen_attack_semantics": {"Q": 15, "indirect_natural": True,
                                    "judgment_definition": "Yes/No/Unknown supported solely by target context",
                                    "unknown_penalty_lambda": 5,
                                    "session_score": "(sum exact Yes/No correctness - 5*sum response-UNK)/15; session ROC-AUC"},
        "allowed_change": "output validity constraints only",
        "forbidden_changes": ["target cohort", "membership labels", "Q15", "attack strengthening",
                              "model/checkpoint/tokenizer", "prompt semantics", "sampling policy",
                              "GT definition", "Final LC", "MIRABEL", "benign calibration",
                              "locator", "Simple Hide", "Qwen target RAG generator", "semantic duplicate filter",
                              "retry", "beam candidate selection", "temperature/top-p search"],
        "preflight": {"ids": preflight_ids, "n": 100, "member": 50, "nonmember": 50,
                      "id_sha256": hashlib.sha256("\n".join(preflight_ids).encode()).hexdigest(),
                      "performance_metrics_forbidden": True, "included_in_full": True,
                      "gate": {"valid_min": 98, "validity_gap_max": 0.02,
                               "exact_q15_among_valid": True, "exact_15_judgments_among_valid": True,
                               "duplicates_among_valid": 0, "forbidden_judgments": 0,
                               "inferred_fields": 0, "fuzzy_recovery": 0}},
        "full_gate": {"sessions": 2000, "valid_min": 1900, "coverage_min": 0.95,
                      "validity_gap_max": 0.02, "automatic_after_preflight_pass": True,
                      "additional_recovery": False},
        "stop_rule": {"preflight_fail": "IA_STD_Q15_V6_FAILED_FINAL and no v7",
                      "full_pass": "IA_STD_Q15_V6_READY then frozen detection/E2E only",
                      "full_fail": "IA_STD_Q15_V6_FAILED_FINAL"},
        "post_ready_evaluation": {"detection": "MIRABEL vs Final LC at same benign FPR 2.5%; query TPR, alarms/15, any/all alarm, session bootstrap 2000",
                                  "e2e": "No Defense vs matched MIRABEL vs matched Final LC using standardized v6 scorer"},
        "preserved_results": {
            "core5_detection": {"path": str(CORE / "PHASE_A_RESULT.json"), "sha256": sha_file(CORE / "PHASE_A_RESULT.json")},
            "core5_e2e": {"path": str(CORE / "PHASE_B_RESULT.json"), "sha256": sha_file(CORE / "PHASE_B_RESULT.json")},
            "gold_recalibration": {"path": str(PARENT / "GOLD_RECALIBRATION_RESULT.json"), "sha256": sha_file(PARENT / "GOLD_RECALIBRATION_RESULT.json")}},
        "training_steps": 0, "performance_results_opened": False,
        "code_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in code_paths},
    }
    output = CHILD / "configs" / "IA_STD_Q15_V6_PRECOMMIT.json"
    output.write_text(json.dumps(precommit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha_file(output)
    (CHILD / "configs" / "IA_STD_Q15_V6_PRECOMMIT.sha256").write_text(
        f"{digest}  IA_STD_Q15_V6_PRECOMMIT.json\n", encoding="utf-8")
    (CHILD / "checkpoints" / "PRECOMMITTED.json").write_text(
        json.dumps({"precommit_sha256": digest, "created_utc": precommit["created_utc"]}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"precommit_sha256": digest, "unit_test_required": "PASS 8/8",
                      "sessions": 2000, "preflight": 100}, indent=2))


if __name__ == "__main__":
    main()
