#!/usr/bin/env python3
"""Fail-close protocol audit conducted before any new attack result is opened."""
from __future__ import annotations

import json
from pathlib import Path

from common import EXP, RECOVERY, ROOT, checkpoint, now, sha_file, write_csv, atomic_json, atomic_text


OFFICIAL = ROOT / "code" / "menta_official"


def code_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha_file(path) for path in paths if path.is_file()}


def main() -> None:
    checkpoint("PHASE_A_PROTOCOL_AUDIT_STARTED")
    recovery_table = RECOVERY / "audits" / "FINAL_PROTOCOL_TABLE.csv"
    rows = [
        {
            "attack": "MEntA",
            "role": "DEVELOPMENT",
            "status": "PAPER_PROTOCOL_READY",
            "query_protocol": "GPT-4.1-nano summary plus five diverse questions; immutable existing Q5 packet",
            "member_nonmember": "target document included/excluded from protected DB",
            "required_gt": "target source plus generated answer",
            "scorer": "mean_q(I_entail(answer,target)-I_idk(answer))",
            "polarity": "larger means member",
            "aggregation": "mean over five queries per target",
            "native_metric": "ROC-AUC, accuracy, low-FPR TPR",
            "source": "MEntA paper and preserved reproduction package",
            "execution": "READY_EXISTING_IMMUTABLE_QUERIES",
        },
        {
            "attack": "MBA",
            "role": "DEVELOPMENT",
            "status": "PAPER_PROTOCOL_READY",
            "query_protocol": "GPT2-XL difficulty-ranked five masks with frozen validity constraints",
            "member_nonmember": "target document included/excluded from protected DB",
            "required_gt": "exact masked tokens",
            "scorer": "correctly reconstructed indexed masks / 5",
            "polarity": "larger means member",
            "aggregation": "one masked query per target",
            "native_metric": "ROC-AUC, accuracy, low-FPR TPR",
            "source": "MBA paper and preserved reproduction package",
            "execution": "READY_EXISTING_IMMUTABLE_QUERIES",
        },
        {
            "attack": "RAG-MIA",
            "role": "DEVELOPMENT",
            "status": "PAPER_FAITHFUL_REIMPLEMENTATION",
            "query_protocol": "paper black-box prompt #2, exact Yes/No output",
            "member_nonmember": "target document included/excluded from protected DB",
            "required_gt": "membership manifest",
            "scorer": "Yes=1; No or no explicit Yes=0; Yes+No ambiguity fails closed",
            "polarity": "larger means member",
            "aggregation": "one query per target",
            "native_metric": "TPR/FPR and ROC-AUC",
            "source": "RAG-MIA paper; no frozen author scorer was recovered",
            "execution": "READY_EXISTING_IMMUTABLE_QUERIES",
        },
        {
            "attack": "S²-MIA",
            "role": "CONFIRMATION_STRESS",
            "status": "PAPER_FAITHFUL_REIMPLEMENTATION",
            "query_protocol": "split target at deterministic character midpoint/next whitespace; first half is query",
            "member_nonmember": "target document included/excluded from protected DB",
            "required_gt": "full target, generated-token log probabilities, frozen balanced reference split",
            "scorer": "S²-MIA-T: BLEU(full target,output) high AND generation perplexity low",
            "polarity": "high BLEU and low perplexity mean member",
            "aggregation": "greedy two-threshold fit on precommitted 20% reference; evaluate remaining 80%",
            "native_metric": "balanced accuracy/TPR/FPR; ROC-AUC secondary",
            "source": "S²-MIA arXiv 2406.19234 v2, Eq.4 and Sec.3.2; local repo query template",
            "execution": "READY_TO_BUILD_ON_SHARED_TARGETS",
        },
        {
            "attack": "DCMI",
            "role": "CONFIRMATION_STRESS",
            "status": "PROTOCOL_UNAVAILABLE",
            "query_protocol": "original and 6% antonym-perturbed Yes/No pair",
            "member_nonmember": "target document included/excluded from protected DB",
            "required_gt": "immutable 6% perturbations generated with fully identified model/config",
            "scorer": "f_rag(q)-f_rag(q') with Yes=1, No=0",
            "polarity": "1 means member",
            "aggregation": "paired differential per target",
            "native_metric": "Accuracy and ROC-AUC",
            "source": "DCMI arXiv 2509.06026 and official repository",
            "execution": "BLOCKED_NO_IMMUTABLE_PERTURBATION_BUNDLE_AND_NO_API_CREDENTIAL",
        },
        {
            "attack": "IA",
            "role": "CONFIRMATION_STRESS",
            "status": "PROTOCOL_UNAVAILABLE",
            "query_protocol": "30 GPT-generated Yes/No questions per target",
            "member_nonmember": "target document included/excluded from protected DB",
            "required_gt": "immutable original 30-query and GPT-4o-mini GT bundle",
            "scorer": "mean(I(response=GT)-5*I(response=UNK))",
            "polarity": "larger means member",
            "aggregation": "mean over 30 queries",
            "native_metric": "ROC-AUC, accuracy, low-FPR TPR",
            "source": "IA arXiv 2502.00306 and official repository commit audited in CORE6 recovery",
            "execution": "BLOCKED_ORIGINAL_GT_BUNDLE_NOT_RECOVERED",
        },
        {
            "attack": "RAGLeak",
            "role": "CONFIRMATION_STRESS",
            "status": "PROTOCOL_UNAVAILABLE",
            "query_protocol": "crop target; former part input, latter part reference",
            "member_nonmember": "target document included/excluded from protected DB",
            "required_gt": "author-faithful crop rule, prompt, similarity implementation",
            "scorer": "black-box output/reference similarity; gray-box output perplexity",
            "polarity": "larger similarity or lower perplexity means member",
            "aggregation": "one cropped target query",
            "native_metric": "paper reports accuracy; threshold protocol required",
            "source": "RAGLeak ACISP 2025 accepted manuscript metadata; official code not recovered",
            "execution": "BLOCKED_EXACT_CROP_PROMPT_AND_SCORER_NOT_RECOVERED",
        },
        {
            "attack": "BudgetLeak",
            "role": "CONFIRMATION_STRESS",
            "status": "PROTOCOL_UNAVAILABLE",
            "query_protocol": "same QA query once at output budgets 10..270 step 20",
            "member_nonmember": "1,000 member and 1,000 mutually exclusive nonmember QA samples",
            "required_gt": "question/reference-answer pairs and exact BudgetLeak-Z/P scorer",
            "scorer": "multi-metric sequences; P=attention LSTM, Z=two-cluster semantic+lexical vector",
            "polarity": "higher predicted member probability means member",
            "aggregation": "14 output-budget generations per target",
            "native_metric": "Accuracy, ROC-AUC, TPR at 0.1% FPR",
            "source": "BudgetLeak arXiv 2511.12043 v1",
            "execution": "BLOCKED_SHARED_BEIR_TARGETS_HAVE_NO_PAIRED_QA_REFERENCES_AND_AUTHOR_SCORER_NOT_RECOVERED",
        },
    ]
    write_csv(EXP / "audits" / "ATTACK_PROTOCOL_STATUS.csv", rows)
    source_paths = [
        recovery_table,
        OFFICIAL / "S2-MIA" / "create_queries.py",
        OFFICIAL / "S2-MIA" / "evaluate.py",
        OFFICIAL / "DCMI" / "perturb_docs.py",
        OFFICIAL / "DCMI" / "evaluate.py",
        OFFICIAL / "IA-MIA" / "generate_queries.py",
        OFFICIAL / "IA-MIA" / "generate_ground_truth.py",
        OFFICIAL / "IA-MIA" / "evaluate.py",
    ]
    evidence = {
        "campaign": EXP.name,
        "audited_utc": now(),
        "performance_results_opened": False,
        "ready_or_reimplementation": [row["attack"] for row in rows if row["status"] != "PROTOCOL_UNAVAILABLE"],
        "unavailable": [row["attack"] for row in rows if row["status"] == "PROTOCOL_UNAVAILABLE"],
        "code_and_prior_audit_hashes": code_hashes(source_paths),
        "claim_rule": "Unavailable protocols remain blank in the main table; no proxy score is relabeled as original.",
        "sources": {
            "S2-MIA": "https://arxiv.org/abs/2406.19234",
            "DCMI": "https://arxiv.org/abs/2509.06026",
            "IA": "https://arxiv.org/abs/2502.00306",
            "RAGLeak": "https://doi.org/10.1007/978-981-96-9101-2_8",
            "BudgetLeak": "https://arxiv.org/abs/2511.12043",
        },
    }
    atomic_json(EXP / "audits" / "PHASE_A_PROTOCOL_AUDIT.json", evidence)
    report = [
        "# Phase A — 8개 공격 프로토콜 감사", "",
        "성능 숫자를 보기 전에 원 쿼리·원 점수식을 복구할 수 있는지를 먼저 판정했다.", "",
        "| 공격 | 판정 | 실제 생성 가능 여부 |", "|---|---|---|",
    ]
    report.extend(f"| {r['attack']} | {r['status']} | {r['execution']} |" for r in rows)
    report += ["", "DCMI·IA·RAGLeak·BudgetLeak에는 임의 대체 점수를 넣지 않는다. 나머지 공격의 실행은 계속한다."]
    atomic_text(EXP / "reports" / "PHASE_A_PROTOCOL_AUDIT_KO.md", "\n".join(report) + "\n")
    checkpoint("PHASE_A_PROTOCOL_AUDIT_COMPLETE", ready=4, unavailable=4, verdict="PROTOCOL_PARTIAL_4_OF_8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

