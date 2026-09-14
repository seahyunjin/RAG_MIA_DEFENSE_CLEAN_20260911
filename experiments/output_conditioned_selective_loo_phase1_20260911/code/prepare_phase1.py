#!/usr/bin/env python3
"""Freeze the small Phase-1 cohort and a deterministic 1K-document RAG corpus."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_common import (
    ATTACK_QUERIES, ATTACK_SCORES, BGE, CORPUS_SIZE, DOMAINS, FAMILIES, FOLDERS,
    MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, NATIVE_BUDGET, QWEN, RAW_ROOT, REQUEST,
    ROOT, SEED, SOURCE_TOKEN_BUDGET, SYSTEM_PROMPT, TOP_K, atomic_csv, atomic_json,
    atomic_text, checkpoint, normalize_family, sha256_file, sha256_text,
)


NORMAL_QUOTAS = {"BeIR_nfcorpus": 225, "BeIR_scidocs": 225, "BeIR_trec-covid": 50}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_attack() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(
        ATTACK_SCORES, keep_default_na=False, low_memory=False,
        dtype={"case_id": str, "session_id": str, "target_id": str},
    )
    queries = pd.read_csv(
        ATTACK_QUERIES, keep_default_na=False, low_memory=False,
        dtype={"case_id": str, "session_id": str, "target_id": str},
        usecols=["case_id", "session_id", "family", "member", "turn", "query", "target_id", "target_rank"],
    )
    scores["family"] = scores.family.map(normalize_family)
    queries["family"] = queries.family.map(normalize_family)
    keep = ["case_id", "domain", "query_hash", "family", "member", "session_id", "turn", "target_id"]
    frame = queries.merge(scores[keep], on=["case_id", "family", "member", "session_id", "turn", "target_id"],
                          validate="one_to_one", suffixes=("", "_frozen"))
    frame["query_sha256"] = frame.query.map(sha256_text)
    if not frame.query_sha256.eq(frame.query_hash).all():
        raise RuntimeError("attack query hash mismatch")
    frame = frame[frame.family.isin(FAMILIES)].copy()
    expected = {(family, member): 40 for family in FAMILIES for member in (0, 1)}
    got = frame.groupby(["family", "member"]).session_id.nunique().to_dict()
    if got != expected:
        raise RuntimeError(f"attack session balance drift: {got}")
    sessions = []
    for (family, member, session_id), cell in frame.groupby(["family", "member", "session_id"], sort=False):
        ordered = cell.sort_values(["turn", "case_id"])
        if len(ordered) != NATIVE_BUDGET[family] or ordered.turn.nunique() != NATIVE_BUDGET[family]:
            raise RuntimeError(f"native query budget drift: {session_id}")
        session_key = min(sha256_text(case_id) for case_id in ordered.case_id.astype(str))
        sessions.append({"family": family, "member": int(member), "session_id": str(session_id),
                         "session_selection_sha256": session_key})
    session_frame = pd.DataFrame(sessions)
    selected_sessions = []
    for _, cell in session_frame.groupby(["family", "member"], sort=True):
        selected_sessions.append(cell.sort_values(["session_selection_sha256", "session_id"]).head(20))
    selected_sessions = pd.concat(selected_sessions, ignore_index=True)
    selected = frame.merge(selected_sessions, on=["family", "member", "session_id"], validate="many_to_one")
    selected = selected.sort_values(["family", "member", "session_selection_sha256", "turn", "case_id"]).reset_index(drop=True)
    expected_rows = 40 * (NATIVE_BUDGET["MEntA"] + NATIVE_BUDGET["S²-MIA"] + NATIVE_BUDGET["MBA"])
    if len(selected) != expected_rows or selected.case_id.duplicated().any():
        raise RuntimeError("small attack selection size/identity failure")
    return selected, scores


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
        out = pd.DataFrame({"query_id": part[query_col].astype(str), "gold_source_id": part[corpus_col].astype(str),
                            "relevance": pd.to_numeric(part[score_col], errors="coerce").fillna(1.0)
                            if score_col else 1.0})
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def select_benign(all_attack: pd.DataFrame) -> pd.DataFrame:
    attack_hashes = set(all_attack.query_hash.astype(str))
    nonmember = {
        domain: set(all_attack[(all_attack.domain.eq(domain)) & all_attack.member.eq(0)].target_id.astype(str))
        for domain in DOMAINS
    }
    selected = []
    for domain in DOMAINS:
        queries = read_jsonl(RAW_ROOT / FOLDERS[domain] / "queries.jsonl")
        query_map = {str(row.get("_id", row.get("id"))): str(row.get("text", "")) for row in queries}
        qrels = load_qrels(domain)
        qrels = qrels[~qrels.gold_source_id.isin(nonmember[domain])].copy()
        qrels = qrels.sort_values(["query_id", "relevance", "gold_source_id"], ascending=[True, False, True])
        qrels = qrels.drop_duplicates("query_id", keep="first")
        rows = []
        for item in qrels.itertuples(index=False):
            query = query_map.get(str(item.query_id), "").strip()
            if not query:
                continue
            query_hash = sha256_text(query)
            if query_hash in attack_hashes:
                continue
            case_id = f"BENIGN|{domain}|{item.query_id}"
            rows.append({"case_id": case_id, "kind": "BENIGN", "family": "BENIGN_GOLD", "member": -1,
                         "session_id": case_id, "turn": 1, "domain": domain, "query": query,
                         "query_sha256": query_hash, "target_id": str(item.gold_source_id),
                         "selection_sha256": sha256_text(case_id)})
        pool = pd.DataFrame(rows).sort_values(["selection_sha256", "case_id"])
        quota = NORMAL_QUOTAS[domain]
        if len(pool) < quota:
            raise RuntimeError(f"insufficient benign gold pool: {domain}/{len(pool)}/{quota}")
        selected.append(pool.head(quota))
    frame = pd.concat(selected, ignore_index=True).sort_values(["selection_sha256", "case_id"]).reset_index(drop=True)
    if len(frame) != 500 or frame.case_id.duplicated().any() or frame.query_sha256.duplicated().any():
        raise RuntimeError("benign 500 identity/duplicate failure")
    frame["split"] = np.where(np.arange(len(frame)) < 250, "CALIBRATION", "HOLDOUT")
    if frame.groupby("split").size().to_dict() != {"CALIBRATION": 250, "HOLDOUT": 250}:
        raise RuntimeError("benign 250/250 split failure")
    return frame


def build_corpora(all_attack: pd.DataFrame, benign: pd.DataFrame) -> dict[str, dict]:
    audit = {}
    for domain in DOMAINS:
        raw = read_jsonl(RAW_ROOT / FOLDERS[domain] / "corpus.jsonl")
        docs = {}
        for row in raw:
            document_id = str(row.get("_id", row.get("id")))
            text = "\n".join(value for value in (str(row.get("title", "")), str(row.get("text", ""))) if value)
            docs[document_id] = text
        members = set(all_attack[(all_attack.domain.eq(domain)) & all_attack.member.eq(1)].target_id.astype(str))
        nonmembers = set(all_attack[(all_attack.domain.eq(domain)) & all_attack.member.eq(0)].target_id.astype(str))
        gold = set(benign[benign.domain.eq(domain)].target_id.astype(str))
        if members & nonmembers:
            raise RuntimeError(f"member/nonmember target conflict: {domain}")
        required = members | gold
        if required & nonmembers:
            raise RuntimeError(f"required/nonmember target conflict: {domain}")
        missing = required - set(docs)
        if missing:
            raise RuntimeError(f"missing required raw documents: {domain}/{len(missing)}")
        eligible = set(docs) - nonmembers
        fillers = sorted(eligible - required, key=lambda value: (sha256_text(value), value))
        if len(required) > CORPUS_SIZE or len(fillers) < CORPUS_SIZE - len(required):
            raise RuntimeError(f"cannot form deterministic 1K corpus: {domain}")
        chosen = sorted(required | set(fillers[:CORPUS_SIZE-len(required)]), key=lambda value: (sha256_text(value), value))
        frame = pd.DataFrame([{"document_id": document_id, "text": docs[document_id],
                               "text_sha256": sha256_text(docs[document_id]),
                               "required_member_target": document_id in members,
                               "required_benign_gold": document_id in gold}
                              for document_id in chosen])
        path = ROOT / "private" / f"CORPUS_{domain}.csv.gz"
        atomic_csv(frame, path, "gzip")
        audit[domain] = {
            "documents": len(frame), "required_member_targets": len(members),
            "excluded_nonmember_targets": len(nonmembers), "required_benign_gold": len(gold),
            "ordered_id_sha256": sha256_text("\n".join(chosen)), "file_sha256": sha256_file(path),
        }
    atomic_json(ROOT / "audits/CORPUS_CONSTRUCTION_AUDIT.json", audit)
    return audit


def write_precommit(attack: pd.DataFrame, benign: pd.DataFrame, corpus_audit: dict) -> None:
    attack_path = ROOT / "private/PHASE1_ATTACK_SELECTION.csv.gz"
    benign_path = ROOT / "private/PHASE1_BENIGN_SELECTION.csv.gz"
    atomic_csv(attack, attack_path, "gzip")
    atomic_csv(benign, benign_path, "gzip")
    payload = {
        "experiment": "OUTPUT_CONDITIONED_SELECTIVE_LOO_GUARD_PHASE1",
        "written_before_new_retrieval_qll_generation_or_loo_scores": True,
        "candidate": "MIRABEL_RANK1_PLUS_QLL_TOP1_UNION2_OUTPUT_CONDITIONED_LOO",
        "phase": 1,
        "rebase_reason": ("The approved cleanup removed the original AD-test-LLM3 1K corpus, frozen BGE cache, "
                          "and per-source QLL locator IDs. Saved L_full/L_minus_i values remain exact, but cannot "
                          "be joined to QLL-selected sources without fabrication. Phase 1 is therefore recomputed "
                          "on this deterministic, precommitted 1K-per-domain substrate; legacy and rebase scores "
                          "must never be mixed."),
        "attack_selection": "per family/member, sort sessions by minimum SHA256(case_id), first 20; retain every native turn",
        "benign_selection": {"total": 500, "calibration": 250, "holdout": 250,
                             "domain_quotas": NORMAL_QUOTAS,
                             "rule": "BEIR qrels gold queries; SHA256(case_id) order; no attack-query duplicate"},
        "corpus": {"documents_per_domain": CORPUS_SIZE,
                   "rule": "include all frozen member targets and selected benign gold sources; exclude all frozen nonmember targets; SHA256(document_id) fillers",
                   "audit": corpus_audit},
        "retrieval": {"model": str(BGE), "top_k": TOP_K, "normalized_dense_inner_product": True,
                      "max_seq_length": 512, "source_text": "title\\ntext"},
        "generator": {"model": str(QWEN), "dtype": "bfloat16", "attention": "sdpa",
                      "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                      "system_prompt": SYSTEM_PROMPT, "max_prompt_tokens": MAX_PROMPT_TOKENS,
                      "source_token_budget": SOURCE_TOKEN_BUDGET, "packing": "equal waterfill Top-4"},
        "qll_locator": "argmax mean query-token log P(query|Context: source) using frozen Qwen",
        "mirabel_locator": "retrieval rank-1 source (canonical MIRABEL source locator)",
        "oracle_locator": "target source when target is in Top-4; zero influence otherwise",
        "loo": "I_i=mean_t logP(A0_t|q,D,A0_<t)-mean_t logP(A0_t|q,D\\{d_i},A0_<t)",
        "union2": "unique(MIRABEL rank1, QLL top1), max influence; no weights",
        "thresholds": {"source": "benign calibration 250 only", "target_fprs": [0.01, 0.03, 0.05],
                       "comparison": "strict >", "ties": "unsplit"},
        "score_unit": "query thresholds; native session score=max query risk; detected if any query exceeds threshold",
        "bootstrap": {"unit": "session", "iterations": 10000, "seed": SEED},
        "forbidden": ["attack-specific routing", "attack calibration", "learned classifier", "weight search",
                      "new fusion", "Phase-2 generation before Phase-1 GO"],
        "hashes": {
            "request": sha256_file(REQUEST), "attack_parent": sha256_file(ATTACK_SCORES),
            "query_parent": sha256_file(ATTACK_QUERIES), "attack_selection": sha256_file(attack_path),
            "benign_selection": sha256_file(benign_path), "bge_config": sha256_file(BGE / "config.json"),
            "qwen_config": sha256_file(QWEN / "config.json"), "code": sha256_file(Path(__file__)),
        },
    }
    path = ROOT / "configs/PRECOMMIT.json"
    if path.exists():
        old = json.loads(path.read_text())
        old.pop("created_utc", None)
        compare = dict(payload)
        if old != compare:
            raise RuntimeError("existing PRECOMMIT differs from rebuilt payload")
    else:
        atomic_json(path, {**payload, "created_utc": __import__("phase1_common").now()})
        atomic_text(ROOT / "configs/PRECOMMIT.sha256", sha256_file(path) + "  PRECOMMIT.json\n")


def main() -> None:
    checkpoint("PREPARE_STARTED")
    attack, all_attack = select_attack()
    benign = select_benign(all_attack)
    corpus_audit = build_corpora(all_attack, benign)
    write_precommit(attack, benign, corpus_audit)
    selection = attack.groupby(["family", "member"], as_index=False).agg(
        sessions=("session_id", "nunique"), queries=("case_id", "size"))
    atomic_csv(selection, ROOT / "audits/ATTACK_SELECTION_COUNTS.csv")
    atomic_csv(benign.groupby(["split", "domain"], as_index=False).agg(queries=("case_id", "size")),
               ROOT / "audits/BENIGN_SELECTION_COUNTS.csv")
    atomic_text(ROOT / "reports/INPUT_REBASE_NOTE_KO.md",
                "# Phase 1 입력 재구성\n\n"
                "기존 `L_full/L_-i`는 수식상 정확하지만 승인된 정리 과정에서 QLL-selected source ID와 "
                "AD-test-LLM3의 1,000문서 BGE retrieval cache가 삭제되었다. 서로 다른 retrieval lineage의 "
                "source ID를 붙이면 결과가 조작되므로, 원문 질의와 원본 BEIR 문서로 도메인별 1,000문서 "
                "substrate를 결과 확인 전에 고정했다. 이 Phase 1 결과는 과거 frozen score와 섞지 않는다.\n")
    checkpoint("PRECOMMIT_FROZEN", attack_queries=len(attack), attack_sessions=attack.session_id.nunique(),
               benign_calibration=250, benign_holdout=250, corpus_documents_per_domain=CORPUS_SIZE,
               precommit_sha256=sha256_file(ROOT / "configs/PRECOMMIT.json"))


if __name__ == "__main__":
    main()
