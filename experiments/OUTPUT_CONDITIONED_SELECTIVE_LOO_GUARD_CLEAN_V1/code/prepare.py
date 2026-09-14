#!/usr/bin/env python3
"""Create and freeze the independent CLEAN_V1 Phase-1 substrate.

Only original attack query text/target labels are read from the preserved source
packet. No legacy retrieval, detector, QLL, LOO, answer or threshold value is
read or joined.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ATTACK_QUERY_SOURCE, BGE, CAMPAIGN, CORPUS_SIZE, DOMAINS, FAMILIES, FOLDERS,
    MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, NATIVE_BUDGET, QWEN, RAW_ROOT, REQUEST,
    ROOT, SEED, SOURCE_TOKEN_BUDGET, SYSTEM_PROMPT, TOP_K, atomic_csv, atomic_json,
    atomic_text, checkpoint, normalize_family, now, sha256_file, sha256_text,
)

NORMAL_QUOTAS = {"BeIR_nfcorpus": 225, "BeIR_scidocs": 225, "BeIR_trec-covid": 50}
CALIBRATION_QUOTAS = {"BeIR_nfcorpus": 112, "BeIR_scidocs": 113, "BeIR_trec-covid": 25}
OFFICIAL = ROOT.parents[1] / "code" / "menta_official"
NLI = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def domain_from_session(value: str) -> str:
    hits = [part for part in str(value).split("|") if part in DOMAINS]
    if len(hits) != 1:
        raise ValueError(f"cannot recover unique domain from session_id: {value}")
    return hits[0]


def source_only_attack_queries() -> pd.DataFrame:
    allowed = ["case_id", "session_id", "family", "member", "turn", "query", "target_id"]
    frame = pd.read_csv(ATTACK_QUERY_SOURCE, usecols=allowed, keep_default_na=False,
                        dtype={column: str for column in allowed})
    frame["family"] = frame.family.map(normalize_family)
    frame = frame[frame.family.isin(FAMILIES)].copy()
    frame["member"] = frame.member.astype(int)
    frame["turn"] = frame.turn.astype(int)
    frame["domain"] = frame.session_id.map(domain_from_session)
    frame["query_sha256"] = frame["query"].map(sha256_text)
    frame["kind"] = "ATTACK"
    if frame.case_id.duplicated().any() or frame["query"].eq("").any() or frame.target_id.eq("").any():
        raise RuntimeError("attack source packet has duplicate IDs or missing query/target")
    expected = {(family, member): 40 for family in FAMILIES for member in (0, 1)}
    got = frame.groupby(["family", "member"]).session_id.nunique().to_dict()
    if got != expected:
        raise RuntimeError(f"source attack session contract failed: {got}")
    sessions = []
    for (family, member, session_id), cell in frame.groupby(
            ["family", "member", "session_id"], sort=False):
        ordered = cell.sort_values(["turn", "case_id"])
        budget = NATIVE_BUDGET[family]
        if len(ordered) != budget or ordered.turn.tolist() != list(range(1, budget + 1)):
            raise RuntimeError(f"native protocol/order mismatch: {session_id}")
        key = min(sha256_text(case_id) for case_id in ordered.case_id.astype(str))
        sessions.append({"family": family, "member": member, "session_id": session_id,
                         "session_selection_sha256": key})
    session_frame = pd.DataFrame(sessions)
    selected_sessions = pd.concat(
        [cell.sort_values(["session_selection_sha256", "session_id"]).head(20)
         for _, cell in session_frame.groupby(["family", "member"], sort=True)],
        ignore_index=True,
    )
    selected = frame.merge(selected_sessions, on=["family", "member", "session_id"],
                           validate="many_to_one")
    selected = selected.sort_values(
        ["family", "member", "session_selection_sha256", "turn", "case_id"]
    ).reset_index(drop=True)
    expected_rows = 40 * sum(NATIVE_BUDGET.values())
    if len(selected) != expected_rows:
        raise RuntimeError(f"attack small-cohort size mismatch: {len(selected)} != {expected_rows}")
    return selected


def load_qrels(domain: str) -> pd.DataFrame:
    paths = sorted((RAW_ROOT / FOLDERS[domain] / "qrels").glob("*.tsv"))
    if not paths:
        raise RuntimeError(f"missing qrels: {domain}")
    frames = []
    for path in paths:
        part = pd.read_csv(path, sep="\t", dtype=str)
        part.columns = [str(column).strip().lower().replace("_", "-") for column in part.columns]
        query_col = next(column for column in part if column in {"query-id", "queryid", "query"})
        corpus_col = next(column for column in part if column in {"corpus-id", "doc-id", "document-id"})
        score_col = next((column for column in part if column in {"score", "relevance"}), None)
        frames.append(pd.DataFrame({
            "query_id": part[query_col].astype(str),
            "gold_source_id": part[corpus_col].astype(str),
            "relevance": (pd.to_numeric(part[score_col], errors="coerce").fillna(1.0)
                          if score_col else 1.0),
        }))
    return pd.concat(frames, ignore_index=True)


def select_benign(attacks: pd.DataFrame) -> pd.DataFrame:
    attack_hashes = set(attacks.query_sha256.astype(str))
    selected = []
    used_benign_hashes = set()
    for domain in DOMAINS:
        forbidden_targets = set(attacks[attacks.domain.eq(domain)].target_id.astype(str))
        query_rows = read_jsonl(RAW_ROOT / FOLDERS[domain] / "queries.jsonl")
        query_map = {str(row.get("_id", row.get("id"))): str(row.get("text", ""))
                     for row in query_rows}
        qrels = load_qrels(domain)
        qrels = qrels[~qrels.gold_source_id.astype(str).isin(forbidden_targets)]
        qrels = qrels.sort_values(
            ["query_id", "relevance", "gold_source_id"], ascending=[True, False, True]
        ).drop_duplicates("query_id", keep="first")
        rows = []
        for item in qrels.itertuples(index=False):
            query = query_map.get(str(item.query_id), "").strip()
            if not query:
                continue
            query_hash = sha256_text(query)
            if query_hash in attack_hashes:
                continue
            case_id = f"BENIGN|{domain}|{item.query_id}"
            rows.append({
                "case_id": case_id, "kind": "BENIGN", "family": "BENIGN_GOLD",
                "member": -1, "session_id": case_id, "turn": 1, "domain": domain,
                "query": query, "query_sha256": query_hash,
                "target_id": str(item.gold_source_id), "selection_sha256": sha256_text(case_id),
            })
        pool = pd.DataFrame(rows).sort_values(["selection_sha256", "case_id"])
        pool = pool.drop_duplicates("query_sha256", keep="first")
        pool = pool[~pool.query_sha256.isin(used_benign_hashes)]
        quota = NORMAL_QUOTAS[domain]
        if len(pool) < quota:
            raise RuntimeError(f"insufficient benign gold pool: {domain}/{len(pool)}/{quota}")
        chosen = pool.head(quota).copy()
        calibration_quota = CALIBRATION_QUOTAS[domain]
        chosen["split"] = np.where(np.arange(len(chosen)) < calibration_quota, "CALIBRATION", "HOLDOUT")
        used_benign_hashes.update(chosen.query_sha256.astype(str))
        selected.append(chosen)
    frame = pd.concat(selected, ignore_index=True).sort_values(
        ["selection_sha256", "case_id"]
    ).reset_index(drop=True)
    if len(frame) != 500 or frame.case_id.duplicated().any() or frame.query_sha256.duplicated().any():
        raise RuntimeError("benign identity/duplicate contract failed")
    if frame.groupby("split").size().to_dict() != {"CALIBRATION": 250, "HOLDOUT": 250}: raise RuntimeError("stratified benign split failed")
    return frame


def raw_documents(domain: str) -> dict[str, str]:
    rows = read_jsonl(RAW_ROOT / FOLDERS[domain] / "corpus.jsonl")
    output = {}
    for row in rows:
        source_id = str(row.get("_id", row.get("id")))
        output[source_id] = "\n".join(
            value for value in (str(row.get("title", "")).strip(), str(row.get("text", "")).strip())
            if value
        )
    return output


def build_corpora(attacks: pd.DataFrame, benign: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    audit = {}
    membership_rows = []
    for domain in DOMAINS:
        documents = raw_documents(domain)
        member_targets = set(attacks[(attacks.domain.eq(domain)) & attacks.member.eq(1)].target_id)
        nonmember_targets = set(attacks[(attacks.domain.eq(domain)) & attacks.member.eq(0)].target_id)
        gold_targets = set(benign[benign.domain.eq(domain)].target_id)
        if member_targets & nonmember_targets:
            raise RuntimeError(f"member/nonmember target conflict: {domain}")
        required = member_targets | gold_targets
        if required & nonmember_targets:
            raise RuntimeError(f"required document conflicts with nonmember exclusion: {domain}")
        if (required | nonmember_targets) - set(documents):
            raise RuntimeError(f"source documents missing from raw corpus: {domain}")
        fillers = sorted(set(documents) - required - nonmember_targets,
                         key=lambda value: (sha256_text(value), value))
        if len(required) > CORPUS_SIZE or len(fillers) < CORPUS_SIZE - len(required):
            raise RuntimeError(f"cannot construct 1000-document corpus: {domain}/{len(required)}")
        selected_ids = sorted(required | set(fillers[:CORPUS_SIZE - len(required)]),
                              key=lambda value: (sha256_text(value), value))
        corpus = pd.DataFrame([
            {"corpus_order": index, "document_id": source_id, "text": documents[source_id],
             "text_sha256": sha256_text(documents[source_id]),
             "selected_member_target": source_id in member_targets,
             "selected_benign_gold": source_id in gold_targets,
             "deterministic_filler": source_id not in required}
            for index, source_id in enumerate(selected_ids)
        ])
        corpus_path = ROOT / "private" / f"CORPUS_{domain}.csv.gz"
        atomic_csv(corpus, corpus_path, "gzip")
        audit[domain] = {
            "documents": len(corpus), "member_targets_included": len(member_targets),
            "nonmember_targets_excluded": len(nonmember_targets),
            "benign_gold_included": len(gold_targets),
            "ordered_document_id_sha256": sha256_text("\n".join(selected_ids)),
            "corpus_file_sha256": sha256_file(corpus_path),
            "raw_corpus_sha256": sha256_file(RAW_ROOT / FOLDERS[domain] / "corpus.jsonl"),
        }
        for source_id in sorted(member_targets):
            membership_rows.append({"domain": domain, "document_id": source_id, "label": "MEMBER",
                                    "included_in_corpus": True})
        for source_id in sorted(nonmember_targets):
            membership_rows.append({"domain": domain, "document_id": source_id, "label": "NONMEMBER",
                                    "included_in_corpus": False})
    membership = pd.DataFrame(membership_rows)
    atomic_csv(membership, ROOT / "private/MEMBER_NONMEMBER_SPLIT.csv.gz", "gzip")
    atomic_json(ROOT / "audits/CORPUS_CONSTRUCTION_AUDIT.json", audit)
    return audit, membership


def file_manifest(root: Path) -> tuple[list[dict], str]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                     "sha256": sha256_file(path)})
    combined = sha256_text("\n".join(f"{row['sha256']}  {row['relative_path']}" for row in rows))
    return rows, combined


def freeze_precommit(attacks: pd.DataFrame, benign: pd.DataFrame, corpus_audit: dict,
                     membership: pd.DataFrame) -> None:
    attack_path = ROOT / "private/PHASE1_ATTACK_SELECTION.csv.gz"
    benign_path = ROOT / "private/PHASE1_BENIGN_SELECTION.csv.gz"
    atomic_csv(attacks, attack_path, "gzip")
    atomic_csv(benign, benign_path, "gzip")
    qwen_files, qwen_hash = file_manifest(QWEN)
    bge_files, bge_hash = file_manifest(BGE)
    nli_files, nli_hash = file_manifest(NLI)
    atomic_json(ROOT / "configs/MODEL_FILE_MANIFESTS.json", {
        "qwen": qwen_files, "bge": bge_files, "nli": nli_files,
        "combined_sha256": {"qwen": qwen_hash, "bge": bge_hash, "nli": nli_hash},
    })
    scorer_paths = {
        "MEntA": OFFICIAL / "MEntA/evaluate.py",
        "S2-MIA": OFFICIAL / "S2-MIA/evaluate.py",
        "MBA": OFFICIAL / "MBA/evaluate.py",
    }
    raw_input_hashes = {}
    for domain in DOMAINS:
        folder = RAW_ROOT / FOLDERS[domain]
        raw_input_hashes[domain] = {
            "corpus": sha256_file(folder / "corpus.jsonl"),
            "queries": sha256_file(folder / "queries.jsonl"),
            "qrels": {str(path.relative_to(folder)): sha256_file(path)
                       for path in sorted((folder / "qrels").glob("*.tsv"))},
        }
    payload = {
        "campaign": CAMPAIGN, "phase": 1, "created_utc": now(),
        "written_before_retrieval_qll_generation_and_loo": True,
        "independent_from_legacy_scores": True, "legacy_numeric_inputs_used": [],
        "source_only_input": {
            "path": str(ATTACK_QUERY_SOURCE), "sha256": sha256_file(ATTACK_QUERY_SOURCE),
            "allowed_columns": ["case_id", "session_id", "family", "member", "turn", "query", "target_id"],
            "explicitly_ignored_columns": ["target_rank", "alarm", "stage1", "stage2", "stage3",
                                           "selected_rank", "selected_source_id", "no_defense_answer",
                                           "b2cf_answer", "changed", "empty", "generation_seconds"],
        },
        "selection": {
            "attacks": "per family/member: session key=min SHA256(case_id); first 20; retain all native turns",
            "benign": "BEIR qrels gold query; SHA256(case_id); fixed domain quotas; first 500",
            "benign_split": "domain-stratified SHA256 order with fixed quotas: 250 calibration, 250 locked holdout",
            "seed": SEED,
        },
        "cohort": {
            "attack_queries": len(attacks), "attack_sessions": attacks.session_id.nunique(),
            "benign_calibration": int(benign.split.eq("CALIBRATION").sum()),
            "benign_holdout": int(benign.split.eq("HOLDOUT").sum()),
            "attack_case_id_sha256": sha256_text("\n".join(attacks.case_id.astype(str))),
            "benign_case_id_sha256": sha256_text("\n".join(benign.case_id.astype(str))),
            "attack_query_sha256": sha256_text("\n".join(attacks.query_sha256.astype(str))),
            "benign_query_sha256": sha256_text("\n".join(benign.query_sha256.astype(str))),
        },
        "corpus": {
            "documents_per_domain": CORPUS_SIZE,
            "construction": "include selected member targets and benign gold; exclude selected nonmember targets; SHA256(document_id) fillers/order",
            "audit": corpus_audit, "membership_rows": len(membership), "raw_input_hashes": raw_input_hashes,
        },
        "retriever": {
            "checkpoint": str(BGE), "snapshot_sha256": bge_hash,
            "normalized_embeddings": True, "similarity": "inner product", "max_seq_length": 512,
            "top_k": TOP_K, "source_text": "title\\ntext", "stable_ties": "corpus order",
        },
        "generator": {
            "checkpoint": str(QWEN), "snapshot_sha256": qwen_hash,
            "tokenizer": str(QWEN), "dtype": "bfloat16", "attention": "sdpa",
            "do_sample": False, "num_beams": 1, "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS, "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "source_token_budget": SOURCE_TOKEN_BUDGET, "packing": "equal waterfill across Top-4",
            "system_prompt": SYSTEM_PROMPT,
        },
        "locators": {
            "MIRABEL": "argmax normalized query-document similarity on clean frozen corpus; official MIRABEL target-document locator",
            "QLL": "argmax mean query-token log P(query | source) under frozen Qwen",
            "UNION2": "unique(MIRABEL, QLL), at most two sources; max LOO influence; no weights",
            "ORACLE": "target source only when retrieved in Top-4; analysis upper bound",
        },
        "loo": "I_i = token-average logP(A0|q,D) - token-average logP(A0|q,D without d_i)",
        "thresholds": {"calibration": "benign calibration only", "target_fpr": [0.01, 0.03, 0.05],
                       "comparison": "strict >", "attack_samples_used": 0},
        "analysis": {
            "session_score": "max query risk within original ordered session",
            "bootstrap_unit": "session", "bootstrap_iterations": 10000, "bootstrap_seed": SEED,
            "oracle_gate": "MEntA AUC>=0.60 with bootstrap AUC lower>0.50 and member-minus-nonmember mean CI lower>0; same for at least one of S2-MIA/MBA",
            "locator_gate": "UNION2 Hit@2>=0.90 on retrieved member targets; MEntA TPR@3% > MIRABEL; UNION2/Oracle MEntA TPR retention>=0.80; S2/MBA UNION2 not below MIRABEL by >0.05",
        },
        "support_nli_evaluator": {
            "checkpoint": str(NLI), "snapshot_sha256": nli_hash,
            "status": "FROZEN_FOR_PHASE2_NOT_USED_IN_PHASE1",
        },
        "attack_scorers": {family: {"path": str(path), "sha256": sha256_file(path),
                                     "status": "FROZEN_FOR_PHASE2_NOT_USED_IN_PHASE1"}
                           for family, path in scorer_paths.items()},
        "request_sha256": sha256_file(REQUEST),
        "forbidden": ["legacy numeric result join", "attack threshold calibration", "learned classifier",
                      "attack-family routing", "weight search", "Phase2 before Phase1 PASS"],
    }
    precommit = ROOT / "configs/PRECOMMIT.json"
    if precommit.exists():
        raise RuntimeError("PRECOMMIT already exists; clean campaign must not be silently rewritten")
    atomic_json(precommit, payload)
    atomic_text(ROOT / "configs/PRECOMMIT.sha256", f"{sha256_file(precommit)}  PRECOMMIT.json\n")
    code_rows, code_hash = file_manifest(ROOT / "code")
    atomic_json(ROOT / "configs/CODE_MANIFEST.json", {"files": code_rows, "combined_sha256": code_hash})


def main() -> None:
    checkpoint("CLEAN_PREPARE_STARTED")
    attacks = source_only_attack_queries()
    benign = select_benign(attacks)
    corpus_audit, membership = build_corpora(attacks, benign)
    freeze_precommit(attacks, benign, corpus_audit, membership)
    checkpoint("CLEAN_PROVENANCE_FROZEN", attack_queries=len(attacks),
               attack_sessions=attacks.session_id.nunique(), benign_queries=len(benign),
               precommit_sha256=sha256_file(ROOT / "configs/PRECOMMIT.json"),
               next_stage="UNIT_TESTS_THEN_PHASE1")


if __name__ == "__main__":
    main()
