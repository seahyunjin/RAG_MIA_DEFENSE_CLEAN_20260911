#!/usr/bin/env python3
"""Create and freeze the IA_STEALTH_CONFIRMATION_V1 precommit."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ia_st1_protocol import (
    GT_MODEL,
    GT_PROMPT,
    GT_SCHEMA,
    GT_SEED,
    QUESTION_MODEL,
    QUESTION_PROMPT,
    QUESTION_SCHEMA,
    QUESTION_SEED,
)


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1"
TARGETS = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1" / "inputs" / "LARGE_SHARED_TARGETS.csv"
FINAL_LC = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1" / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"
IA_PARENT = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
PAPER = ROOT / "papers" / "IA.pdf"
OFFICIAL_QUERY = ROOT / "code" / "menta_official" / "IA-MIA" / "generate_queries.py"
OFFICIAL_GT = ROOT / "code" / "menta_official" / "IA-MIA" / "generate_ground_truth.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    for name in ["audits", "checkpoints", "configs", "inputs", "logs", "outputs", "reports", "runtime", "tables", "tests"]:
        (EXP / name).mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(TARGETS.open(encoding="utf-8")))
    if len(rows) != 2000:
        raise SystemExit(f"expected 2000 targets, got {len(rows)}")
    membership = {m: sum(r["membership"] == m for r in rows) for m in ["member", "nonmember"]}
    if membership != {"member": 1000, "nonmember": 1000}:
        raise SystemExit(f"bad membership counts: {membership}")

    # Preserve the deterministic 50/50 target selection used by IA v4-v6.
    selected = (
        sorted((r for r in rows if r["membership"] == "member"), key=lambda r: r["document_id"])[:50]
        + sorted((r for r in rows if r["membership"] == "nonmember"), key=lambda r: r["document_id"])[:50]
    )
    ids = [r["document_id"] for r in selected]
    id_sha = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    v6_precommit_path = IA_PARENT / "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED" / "configs" / "IA_STD_Q15_V6_PRECOMMIT.json"
    v6_precommit = json.loads(v6_precommit_path.read_text(encoding="utf-8"))
    v6_ids = v6_precommit["preflight"]["ids"]
    if ids != v6_ids:
        raise SystemExit("deterministic ST1 preflight IDs do not match preserved IA v4-v6 preflight IDs")

    prior_results = []
    candidates = [
        IA_PARENT / "IA_V3_EARLY_STOP_RESULT.json",
        IA_PARENT / "IA_STD_Q15_V4_TWO_STAGE" / "FINAL_RESULT.json",
        IA_PARENT / "IA_STD_Q15_V5_SLOTTED_CONSTRAINED" / "FINAL_RESULT.json",
        IA_PARENT / "IA_STD_Q15_V6_DUPLICATE_CONSTRAINED" / "FINAL_RESULT.json",
        ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1" / "IA_PARSER_V2_RESULT.json",
    ]
    for path in candidates:
        if path.exists():
            prior_results.append({"path": str(path), "sha256": sha(path)})

    initial_invalid = EXP / "history" / "INITIAL_OVERRESTRICTIVE_VALIDATOR" / "IA_ST1_PREFLIGHT_RESULT.json"

    code_paths = sorted((EXP / "code").glob("*")) + sorted((EXP / "tests").glob("*.py"))
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_paths if p.is_file() and p.name != "__pycache__"}

    precommit = {
        "campaign": "IA_STEALTH_CONFIRMATION_V1",
        "attack_name": "IA-Std-Q15-ST1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "standardized IA-principle-preserving stealth stress test; not Original IA and not paper-exact IA",
        "final_lc_freeze": {
            "path": str(FINAL_LC),
            "sha256": sha(FINAL_LC),
            "operating_budget": 0.025,
            "mutation_forbidden": True,
        },
        "prior_ia_v1_v6_failures_preserved": prior_results,
        "specification_correction": {
            "reason": "initial host validator incorrectly promoted meta wording and domain terms such as 'the study', medical 'index', and bibliographic 'database' to the direct-membership hard gate",
            "question_api_calls_reused": True,
            "question_regeneration": 0,
            "performance_metrics_seen": False,
            "initial_invalid_result": {"path": str(initial_invalid), "sha256": sha(initial_invalid)},
            "initial_invalid_label": "INVALID_OVERRESTRICTIVE_VALIDATOR",
        },
        "paper_provenance": {
            "paper": {"path": str(PAPER), "sha256": sha(PAPER)},
            "official_query_code": {"path": str(OFFICIAL_QUERY), "sha256": sha(OFFICIAL_QUERY)},
            "official_gt_code": {"path": str(OFFICIAL_GT), "sha256": sha(OFFICIAL_GT)},
            "paper_models": {"query_family": "GPT-4o", "shadow_family": "GPT-4o-mini"},
        },
        "cohort": {
            "path": str(TARGETS),
            "sha256": sha(TARGETS),
            "sessions": 2000,
            "member": 1000,
            "nonmember": 1000,
            "full_ordered_id_sha256": hashlib.sha256(("\n".join(r["document_id"] for r in rows) + "\n").encode()).hexdigest(),
            "preflight_ids": ids,
            "preflight_id_sha256": id_sha,
            "preflight_id_hash_method": "SHA256 over UTF-8 ordered IDs joined by LF with terminal LF",
            "preserved_v6_declared_preflight_id_sha256": v6_precommit["preflight"]["id_sha256"],
            "matches_preserved_v4_v6_preflight_ids": True,
            "preflight_member": 50,
            "preflight_nonmember": 50,
            "preflight_included_in_full": True,
        },
        "attack_semantics": {
            "indirect_natural_probing": True,
            "direct_target_string_reconstruction": False,
            "queries_per_session": 15,
            "member_nonmember_same_substrate": True,
            "session_aggregation": True,
            "unknown_penalty_lambda": 5,
            "runtime_membership_label_available_to_attacker": False,
        },
        "question_generation": {
            "api_endpoint": "/v1/chat/completions",
            "execution": "synchronous checkpointed; exactly one accepted structured generation per target",
            "model": QUESTION_MODEL,
            "seed": QUESTION_SEED,
            "temperature": 0.2,
            "max_tokens": 1200,
            "prompt": QUESTION_PROMPT,
            "schema": QUESTION_SCHEMA,
            "ids_assigned_by_host": True,
        },
        "gt_generation": {
            "api_endpoint": "/v1/chat/completions",
            "model": GT_MODEL,
            "seed": GT_SEED,
            "temperature": 0.0,
            "max_tokens": 200,
            "prompt": GT_PROMPT,
            "schema": GT_SCHEMA,
            "labels": ["Yes", "No", "Unknown"],
            "ids_assigned_by_host": True,
        },
        "validity": {
            "exact_q15": True,
            "exact_gt15": True,
            "nonempty": True,
            "normalized_exact_duplicate": 0,
            "direct_membership_question": 0,
            "meta_reference_wording": "diagnostic only; not a hard gate",
            "formatting_inference": 0,
            "fuzzy_parsing": 0,
            "trivial_copy_audit": "full normalized question substring and 12-token verbatim overlap; diagnostic only",
            "near_duplicate_audit": "token Jaccard>=0.85 or SequenceMatcher>=0.90; diagnostic only",
        },
        "transport_policy": {
            "api_key_path": "/home/traffic_3/workspace/workspace/SH/.secrets/openai_api_key",
            "key_never_copied_to_artifact": True,
            "retry": "identical request only after HTTP 429; no content/schema retries",
            "orphaned_inflight": "invalid; never automatically regenerated",
        },
        "preflight_gate": {
            "sessions": 100,
            "valid_min": 98,
            "member_nonmember_validity_gap_max": 0.02,
            "q15_exact_among_valid": True,
            "gt15_exact_among_valid": True,
            "exact_duplicate": 0,
            "inferred_fields": 0,
            "performance_metrics_forbidden": True,
            "failure_verdict": "IA_STD_Q15_ST1_UNAVAILABLE",
            "pass_verdict": "IA_STD_Q15_ST1_PREFLIGHT_PASS",
        },
        "full_gate": {
            "sessions": 2000,
            "valid_min": 1900,
            "member_nonmember_validity_gap_max": 0.02,
            "automatic_after_preflight_pass": True,
            "additional_prompt_parser_or_model_search_if_fail": False,
            "pass_verdict": "IA_STD_Q15_ST1_READY",
        },
        "post_ready_detection": {
            "primary_fpr": 0.025,
            "secondary_fpr": [0.01, 0.03, 0.05],
            "comparison": "Final LC vs MIRABEL on identical frozen benign evaluation set",
            "primary_metric": "query-level TPR",
            "cluster_bootstrap": 2000,
            "session_metrics": ["alarms/15", "protected/15", "any-alarm", "all-15-alarm", "first-alarm-index"],
            "gate": "Final LC TPR@2.5% >= MIRABEL TPR@2.5% - 0.03 and no catastrophic stealth miss",
        },
        "post_ready_e2e": {
            "conditions": ["No Defense", "MIRABEL@2.5%", "Final LC@2.5%"],
            "same_benign_intervention_budget": True,
            "action": "on alarm hide MIRABEL top-1 source; otherwise use A0; reuse global caches",
            "scorer": "IA-Std-Q15-ST1, Q15 session aggregation, lambda=5",
            "metrics": ["native standardized metric", "member/nonmember distribution", "Raw AUC", "E-AUC", "bootstrap CI"],
        },
        "q30": "not automatically executed in this campaign",
        "next": "CROSS_DOMAIN_PRIVACY_GENERALIZATION regardless of IA PASS/FAIL; Final LC remains frozen",
        "performance_results_opened_before_freeze": False,
        "training_steps": 0,
        "code_sha256": code_hashes,
    }

    target_out = EXP / "inputs" / "PREFLIGHT_TARGETS.jsonl"
    target_out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in selected), encoding="utf-8")
    precommit["cohort"]["preflight_target_artifact"] = {"path": str(target_out), "sha256": sha(target_out)}

    out = EXP / "configs" / "IA_STEALTH_CONFIRMATION_PRECOMMIT.json"
    write_json(out, precommit)
    digest = sha(out)
    (EXP / "configs" / "IA_STEALTH_CONFIRMATION_PRECOMMIT.sha256").write_text(f"{digest}  {out}\n", encoding="utf-8")
    write_json(EXP / "checkpoints" / "PRECOMMIT_FROZEN.json", {"precommit": str(out), "sha256": digest})
    (EXP / "STATUS.md").write_text(
        "# IA_STEALTH_CONFIRMATION_V1\n\n"
        "- state: PRECOMMIT_FROZEN\n"
        f"- precommit_sha256: `{digest}`\n"
        "- performance metrics: forbidden during 100-session format preflight\n",
        encoding="utf-8",
    )
    print(json.dumps({"precommit": str(out), "sha256": digest, "preflight_ids": len(ids)}, indent=2))


if __name__ == "__main__":
    main()
