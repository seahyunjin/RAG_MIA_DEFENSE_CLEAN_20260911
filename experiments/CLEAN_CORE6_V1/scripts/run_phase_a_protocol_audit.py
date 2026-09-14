#!/usr/bin/env python3
"""Generate the fail-closed Phase-A protocol audit for CLEAN_CORE6_V1.

This script intentionally performs no retrieval, generation, scoring, or model
loading.  It only fingerprints the preserved local assets and emits the frozen
protocol audit.  Phase B is forbidden while the verdict is not PHASE_A_PASS.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
CAMPAIGN = PROJECT / "experiments" / "CLEAN_CORE6_V1"
CODE = PROJECT / "code"
PDF = PROJECT.parent / "LLM_pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


assets = [
    ("MEntA paper", PDF / "MentA.pdf"),
    ("IA paper", PDF / "IA.pdf"),
    ("DCMI paper", PDF / "DCMI.pdf"),
    ("MBA paper", PDF / "MBA.pdf"),
    ("MEntA scorer", CODE / "menta_official" / "MEntA" / "evaluate.py"),
    ("IA reproduction scorer", CODE / "menta_official" / "IA-MIA" / "evaluate.py"),
    ("DCMI reproduction scorer", CODE / "menta_official" / "DCMI" / "evaluate.py"),
    ("S2 reproduction scorer", CODE / "menta_official" / "S2-MIA" / "evaluate.py"),
    ("MBA reproduction scorer", CODE / "menta_official" / "MBA" / "evaluate.py"),
    ("legacy native helpers", CODE / "ad_mirabel_core" / "exp44_native.py"),
]


rows = [
    {
        "attack": "MEntA",
        "source_paper": "Five Queries Are Enough (USENIX Security 2026; arXiv:2605.24312)",
        "official_repository": "MEntA reproduction package; preserved locally (package tree hash recorded in report)",
        "target_document_requirements": "descriptive document from which five distinct information-seeking questions and a short summary can be produced",
        "member_definition": "target document is present in the protected retrieval corpus",
        "nonmember_definition": "target document and normalized duplicate are absent from the protected retrieval corpus",
        "native_query_count": "5",
        "query_generation": "GPT-4.1-nano; five specific natural questions; target summary prepended at attack time",
        "ordering_session": "one document-level session; fixed query indices 0..4; mean aggregation",
        "immutable_gt": "target document text; generated query manifest; NLI model/checkpoint",
        "original_scorer": "mean over five queries of entailment-hit minus abstention indicator; DeBERTa long-NLI over atomic claims",
        "original_metrics": "ROC-AUC, accuracy, TPR at low FPR",
        "local_asset": "code/menta_official/MEntA/* and LLM_pdf/MentA.pdf",
        "missing_asset": "hosted GPT model revision is not content-addressable; generation output must be frozen after creation",
        "status": "PROTOCOL_AND_SCORER_AVAILABLE",
    },
    {
        "attack": "IA",
        "source_paper": "Riddle Me This! Stealthy Membership Inference for RAG",
        "official_repository": "https://github.com/ali7naseh/RAG_MIA @ a53e8fda4f1492f204d968d7afb53ce8287c193c",
        "target_document_requirements": "document supports diverse yes/no questions; TF-IDF near-duplicate exclusion at 0.95",
        "member_definition": "target document is included in the RAG database",
        "nonmember_definition": "target document and near duplicates are excluded from the RAG database",
        "native_query_count": "30 in paper; official repository bundles 15 per document and config N=15",
        "query_generation": "paper: GPT-4o produces 30 yes/no questions and document description; repository consumes a pre-generated target_data_with_questions.json",
        "ordering_session": "document-level session; question selection/ranking then aggregation over selected questions",
        "immutable_gt": "GPT-4o-mini yes/no ground-truth answer for every query; exact query pool and selection lineage",
        "original_scorer": "(1/n) sum[1(response=GT) - lambda*1(response=UNK)], lambda=5",
        "original_metrics": "ROC-AUC, accuracy, low-FPR TPR",
        "local_asset": "paper and MEntA reproduction only; official repo audited separately",
        "missing_asset": "old immutable GT; exact 30-query artifact for new shared targets; version-pinned GPT-4o/GPT-4o-mini outputs; official repo paper-consistent lambda=5 scorer",
        "status": "IA_PROTOCOL_UNAVAILABLE",
    },
    {
        "attack": "DCMI",
        "source_paper": "DCMI: A Differential Calibration Membership Inference Attack against RAG (arXiv:2509.06026)",
        "official_repository": "https://github.com/Xinyu140203/RAG_MIA @ ef961a1f23d09c62c5fbf69d798e325fb0362e55",
        "target_document_requirements": "target sample can be embedded verbatim in a yes/no query and minimally perturbed",
        "member_definition": "original target document is present in the RAG corpus",
        "nonmember_definition": "original target document is absent from the RAG corpus",
        "native_query_count": "2 (original plus perturbed)",
        "query_generation": "ask whether original/perturbed target sample is correct/present; perturb 6% of selected words for paper black-box setting",
        "ordering_session": "paired original then perturbed query; pairwise difference",
        "immutable_gt": "exact original-to-perturbed mapping and perturbation provenance",
        "original_scorer": "black-box binary calibrated score f_rag(q)-f_rag(q') in {-1,0,1}",
        "original_metrics": "ROC-AUC, accuracy and thresholded classification metrics",
        "local_asset": "paper, official repo, and MEntA reproduction",
        "missing_asset": "official repo is not an executable exact E2E scorer: MIA.py contains placeholder score arrays; perturb.py uses 3%, while the paper black-box protocol fixes 6%",
        "status": "SPECIFICATION_AVAILABLE_IMPLEMENTATION_INCOMPLETE",
    },
    {
        "attack": "S2-MIA",
        "source_paper": "Generating Is Believing: Membership Inference Attacks against RAG (arXiv:2406.19234)",
        "official_repository": "no authoritative repository located in Phase A",
        "target_document_requirements": "target sample can be split into a query half and held-out continuation half",
        "member_definition": "target sample is present in the RAG database",
        "nonmember_definition": "target sample is absent from the RAG database",
        "native_query_count": "1",
        "query_generation": "first half of target becomes query; generator output is compared with target/remaining content",
        "ordering_session": "single query",
        "immutable_gt": "exact text split and output-token log probabilities when perplexity is used",
        "original_scorer": "paper S2-MIA-T/M use BLEU plus generation perplexity; paper also reports BLEU-only ablation",
        "original_metrics": "ROC-AUC and PR-AUC",
        "local_asset": "MEntA baseline reproduction only",
        "missing_asset": "authoritative code and frozen choice of T vs M variant; local evaluator uses sentence-embedding cosine as primary score instead of the paper's BLEU/perplexity definition",
        "status": "SPECIFICATION_AVAILABLE_SCORER_MISMATCH",
    },
    {
        "attack": "MBA",
        "source_paper": "Mask-Based Membership Inference Attacks for RAG (arXiv:2410.20142)",
        "official_repository": "no authoritative attack repository located in Phase A",
        "target_document_requirements": "enough valid non-stopword tokens/spans to construct mask reconstruction queries",
        "member_definition": "target document is included in the retrieval database",
        "nonmember_definition": "target document is excluded from the retrieval database",
        "native_query_count": "1 masked query per target; M selected from {5,10,15,20} in paper experiments",
        "query_generation": "proxy LM ranks difficult-to-predict words; masked fragments request reconstruction",
        "ordering_session": "single reconstruction query with indexed masks",
        "immutable_gt": "mask positions, accepted target spans/variants, proxy model and tokenizer revision",
        "original_scorer": "fraction of masked targets reconstructed correctly",
        "original_metrics": "ROC-AUC and thresholded accuracy/F1-style metrics",
        "local_asset": "paper and MEntA baseline reproduction",
        "missing_asset": "authoritative code/frozen M-selection rule; local pipeline defaults to 5 masks and 'important' masking rather than the paper's difficult-word proxy protocol",
        "status": "SPECIFICATION_AVAILABLE_QUERY_GENERATOR_MISMATCH",
    },
    {
        "attack": "RAG-MIA",
        "source_paper": "Membership Inference Attacks against RAG (arXiv:2405.20446)",
        "official_repository": "no authoritative repository located in Phase A",
        "target_document_requirements": "target sample can be inserted verbatim into the attack prompt",
        "member_definition": "target sample is present in the RAG database",
        "nonmember_definition": "target sample is absent from the RAG database",
        "native_query_count": "1 per chosen prompt; paper studies 5 prompt templates",
        "query_generation": "best black-box prompt #2 asks whether the target sample appears in context and requests Yes/No",
        "ordering_session": "single query per prompt condition",
        "immutable_gt": "chosen prompt ID, exact target sample serialization, deterministic output parsing policy",
        "original_scorer": "black-box discrete Yes/No membership decision; gray-box uses token probabilities/ensemble",
        "original_metrics": "black-box TPR/FPR and ROC-AUC; gray-box ROC-AUC and TPR at low FPR",
        "local_asset": "only a legacy helper parser in exp44_native.py; no implementation in preserved MEntA package",
        "missing_asset": "official code; frozen black-box output parser and handling of responses without clear Yes/No",
        "status": "PAPER_PROTOCOL_AVAILABLE_OFFICIAL_SCORER_MISSING",
    },
]


def main() -> None:
    (CAMPAIGN / "audits").mkdir(parents=True, exist_ok=True)
    inventory = []
    for label, path in assets:
        inventory.append(
            {
                "asset": label,
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else "MISSING",
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
        )

    with (CAMPAIGN / "audits" / "LOCAL_ASSET_INVENTORY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    with (CAMPAIGN / "audits" / "CORE6_PROTOCOL_AUDIT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "campaign": "CLEAN_CORE6_V1",
        "phase": "A_CORE6_PROTOCOL_AUDIT",
        "verdict": "IA_PROTOCOL_UNAVAILABLE",
        "campaign_state": "CORE6_PHASE_A_FAIL_CLOSED",
        "phase_b_started": False,
        "retrieval_run": False,
        "generation_run": False,
        "model_loaded": False,
        "old_scores_or_answers_reused": False,
        "reason": (
            "The immutable IA ground-truth artifact cannot be recovered, and the paper's exact 30-query "
            "GPT-4o/GPT-4o-mini lineage cannot be reproduced from the preserved repository without changing "
            "the protocol or fabricating provenance. Additional scorer mismatches also remain for S2-MIA and MBA."
        ),
        "required_to_resume": [
            "IA: immutable per-query GT for the frozen shared targets, or an author-provided/version-pinned exact reproduction bundle",
            "IA: exact 30-query pool generation/selection lineage and lambda=5 UNK policy",
            "S2-MIA: freeze the original BLEU/perplexity T or M scorer; do not substitute cosine",
            "MBA: freeze the paper's difficult-word mask construction and M-selection protocol",
            "RAG-MIA: freeze exact black-box prompt, parser, and missing-output policy",
            "DCMI: freeze an executable 6%-perturbation paired scorer implementation",
        ],
        "official_repository_commits_audited": {
            "IA": "a53e8fda4f1492f204d968d7afb53ce8287c193c",
            "DCMI": "ef961a1f23d09c62c5fbf69d798e325fb0362e55",
        },
    }
    with (CAMPAIGN / "PHASE_A_RESULT.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
