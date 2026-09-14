#!/usr/bin/env python3
"""Run frozen BGE retrieval, QLL location, A0 generation, and exact answer LOO scoring."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from common import (
    BGE, DOMAINS, MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, QWEN, ROOT, SOURCE_TOKEN_BUDGET,
    SYSTEM_PROMPT, TOP_K, atomic_csv, checkpoint, sha256_file, sha256_text, waterfill,
)


RETRIEVAL = ROOT / "private/PHASE1_RETRIEVAL.csv.gz"
QLL_DB = ROOT / "private/QLL_SCORES.sqlite3"
GEN_DB = ROOT / "private/A0_GENERATIONS.sqlite3"
LOO_DB = ROOT / "private/LOO_SCORES.sqlite3"
PACKING = ROOT / "private/PHASE1_PACKING.csv.gz"
QUERY_SCORES = ROOT / "private/PHASE1_QUERY_SCORES.csv.gz"


def preflight() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    precommit = ROOT / "configs/PRECOMMIT.json"
    digest = ROOT / "configs/PRECOMMIT.sha256"
    if not precommit.exists() or not digest.exists():
        raise RuntimeError("PRECOMMIT missing; run prepare.py first")
    expected = digest.read_text().split()[0]
    if sha256_file(precommit) != expected:
        raise RuntimeError("PRECOMMIT hash drift")
    attack = pd.read_csv(ROOT / "private/PHASE1_ATTACK_SELECTION.csv.gz", keep_default_na=False,
                         dtype={"case_id": str, "session_id": str, "target_id": str})
    benign = pd.read_csv(ROOT / "private/PHASE1_BENIGN_SELECTION.csv.gz", keep_default_na=False,
                         dtype={"case_id": str, "session_id": str, "target_id": str})
    attack["kind"] = "ATTACK"
    frame = pd.concat([attack, benign], ignore_index=True, sort=False)
    if len(frame) != 780 or frame.case_id.duplicated().any():
        raise RuntimeError(f"Phase-1 cohort drift: {frame.shape}")
    corpora = {}
    for domain in DOMAINS:
        part = pd.read_csv(ROOT / "private" / f"CORPUS_{domain}.csv.gz", keep_default_na=False,
                           dtype={"document_id": str})
        if len(part) != 1000 or part.document_id.duplicated().any():
            raise RuntimeError(f"corpus drift: {domain}/{len(part)}")
        corpora[domain] = part
    checkpoint("GPU_PREFLIGHT_COMPLETE", queries=len(frame), attacks=len(attack), benign=len(benign),
               precommit_sha256=expected, cuda_required=True)
    return frame, corpora


def retrieve(frame: pd.DataFrame, corpora: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if RETRIEVAL.exists():
        output = pd.read_csv(RETRIEVAL, keep_default_na=False, dtype={"case_id": str, "target_id": str})
        if len(output) == len(frame) and set(output.case_id) == set(frame.case_id):
            checkpoint("RETRIEVAL_REUSED", queries=len(output), file_sha256=sha256_file(RETRIEVAL))
            return output
    import torch
    from sentence_transformers import SentenceTransformer
    checkpoint("RETRIEVAL_STARTED", model=str(BGE), domains=len(DOMAINS), documents_per_domain=1000)
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    rows = []
    for domain in DOMAINS:
        corpus = corpora[domain]
        subset = frame[frame.domain.eq(domain)].sort_values("case_id").reset_index(drop=True)
        checkpoint("RETRIEVAL_DOMAIN_STARTED", domain=domain, documents=len(corpus), queries=len(subset))
        doc_embeddings = model.encode(corpus.text.astype(str).tolist(), batch_size=32, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        query_embeddings = model.encode(subset["query"].astype(str).tolist(), batch_size=32, convert_to_numpy=True,
                                        normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        matrix = query_embeddings @ doc_embeddings.T
        order = np.argsort(-matrix, axis=1, kind="stable")[:, :TOP_K]
        ids = corpus.document_id.astype(str).tolist()
        for number, item in enumerate(subset.itertuples(index=False)):
            indexes = order[number]
            source_ids = [ids[index] for index in indexes]
            scores = [float(matrix[number, index]) for index in indexes]
            target = str(item.target_id)
            rows.append({"case_id": str(item.case_id), "source_ids": json.dumps(source_ids),
                         "retrieval_scores": json.dumps(scores),
                         "target_rank": source_ids.index(target)+1 if target in source_ids else 0,
                         "retrieval_top1_source": source_ids[0]})
        checkpoint("RETRIEVAL_DOMAIN_COMPLETE", domain=domain, queries=len(subset))
        del doc_embeddings, query_embeddings, matrix, order
        gc.collect(); torch.cuda.empty_cache()
    del model; gc.collect(); torch.cuda.empty_cache()
    result = frame.merge(pd.DataFrame(rows), on="case_id", validate="one_to_one", suffixes=("_source", ""))
    if len(result) != len(frame) or result.source_ids.eq("").any():
        raise RuntimeError("retrieval output incomplete")
    atomic_csv(result, RETRIEVAL, "gzip")
    summary = result.groupby(["kind", "family", "member"], as_index=False).agg(
        queries=("case_id", "size"), target_retrieval_at4=("target_rank", lambda x: float(np.mean(np.asarray(x)>0))))
    atomic_csv(summary, ROOT / "audits/TARGET_RETRIEVAL_AT4.csv")
    checkpoint("RETRIEVAL_COMPLETE", queries=len(result), file_sha256=sha256_file(RETRIEVAL))
    return result


def document_lookup(corpora: dict[str, pd.DataFrame]) -> dict[tuple[str, str], str]:
    return {(domain, str(row.document_id)): str(row.text)
            for domain, corpus in corpora.items() for row in corpus.itertuples(index=False)}


def init_dbs() -> tuple[sqlite3.Connection, sqlite3.Connection, sqlite3.Connection]:
    qll = sqlite3.connect(QLL_DB); qll.execute("PRAGMA journal_mode=WAL"); qll.execute("PRAGMA synchronous=FULL")
    qll.execute("""CREATE TABLE IF NOT EXISTS score (
      task_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, source_rank INTEGER NOT NULL,
      source_id TEXT NOT NULL, mean_logp REAL NOT NULL, query_tokens INTEGER NOT NULL,
      prefix_tokens INTEGER NOT NULL, wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL)""")
    gen = sqlite3.connect(GEN_DB); gen.execute("PRAGMA journal_mode=WAL"); gen.execute("PRAGMA synchronous=FULL")
    gen.execute("""CREATE TABLE IF NOT EXISTS answer (
      case_id TEXT PRIMARY KEY, answer TEXT NOT NULL, prompt_sha256 TEXT NOT NULL,
      prompt_tokens INTEGER NOT NULL, answer_tokens INTEGER NOT NULL,
      wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL)""")
    loo = sqlite3.connect(LOO_DB); loo.execute("PRAGMA journal_mode=WAL"); loo.execute("PRAGMA synchronous=FULL")
    loo.execute("""CREATE TABLE IF NOT EXISTS score (
      task_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, condition TEXT NOT NULL,
      source_rank INTEGER NOT NULL, mean_logp REAL NOT NULL, prompt_tokens INTEGER NOT NULL,
      answer_tokens INTEGER NOT NULL, wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL)""")
    return qll, gen, loo


def qll_score(model, tokenizer, source: str, query: str) -> tuple[float, int, int, float]:
    import torch
    before = tokenizer("Context:\n", add_special_tokens=False).input_ids
    source_ids = tokenizer(str(source), add_special_tokens=False).input_ids
    bridge = tokenizer("\n\nUser query:\n", add_special_tokens=False).input_ids
    query_ids = tokenizer(str(query), add_special_tokens=False).input_ids
    if not query_ids:
        query_ids = [tokenizer.eos_token_id]
    maximum = int(model.config.max_position_embeddings)
    source_budget = max(0, maximum-len(before)-len(bridge)-len(query_ids))
    source_ids = source_ids[:source_budget]
    sequence = before + source_ids + bridge + query_ids
    ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(ids, use_cache=False, logits_to_keep=len(query_ids)+1).logits.float()[0, :-1]
        targets = ids[0, -len(query_ids):]
        value = torch.log_softmax(logits[-len(query_ids):], dim=-1).gather(1, targets[:, None]).mean()
    elapsed = time.perf_counter()-started
    result = float(value.cpu())
    del ids, logits, targets, value
    return result, len(query_ids), len(sequence)-len(query_ids), elapsed


def run_qll(model, tokenizer, retrieval: pd.DataFrame, documents: dict[tuple[str, str], str], db: sqlite3.Connection) -> None:
    complete = {row[0] for row in db.execute("SELECT task_id FROM score")}
    tasks = []
    for row in retrieval.sort_values("case_id").itertuples(index=False):
        for rank, source_id in enumerate(json.loads(row.source_ids), 1):
            task_id = sha256_text(f"QLL\0{row.case_id}\0{rank}\0{source_id}")
            tasks.append((task_id, row, rank, str(source_id)))
    pending = [task for task in tasks if task[0] not in complete]
    checkpoint("QLL_STARTED", total=len(tasks), cached=len(complete), pending=len(pending), microbatch=1)
    started = time.monotonic(); initial = len(complete)
    for index, (task_id, row, rank, source_id) in enumerate(pending, 1):
        value, query_tokens, prefix_tokens, wall = qll_score(
            model, tokenizer, documents[(str(row.domain), source_id)], str(row.query))
        db.execute("INSERT OR REPLACE INTO score VALUES (?,?,?,?,?,?,?,?,?)",
                   (task_id, str(row.case_id), rank, source_id, value, query_tokens, prefix_tokens, wall,
                    __import__("common").now()))
        done = initial + index
        if index % 20 == 0 or index == len(pending):
            db.commit(); rate = index / max(time.monotonic()-started, 1e-9)
            checkpoint("QLL_PROGRESS", total=len(tasks), completed=done, pending=len(tasks)-done,
                       percent=round(100*done/len(tasks), 2), tasks_per_second=round(rate, 3),
                       eta_seconds=round((len(pending)-index)/max(rate, 1e-9)))
    db.commit(); checkpoint("QLL_COMPLETE", tasks=len(tasks))


def pack_sources(tokenizer, source_ids: list[str], source_texts: list[str]) -> tuple[list[str], list[int]]:
    encoded = [tokenizer(str(text), add_special_tokens=False).input_ids for text in source_texts]
    caps = waterfill([len(value) for value in encoded], SOURCE_TOKEN_BUDGET)
    visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip() for value, cap in zip(encoded, caps)]
    return visible, caps


def render_prompt(tokenizer, query: str, source_ids: list[str], visible: list[str], removed_rank: int = 0) -> tuple[str, list[int]]:
    blocks = [f"[Document {rank}]\n{text}" for rank, text in enumerate(visible, 1)
              if rank != int(removed_rank)]
    joined_blocks = "\n\n".join(blocks)
    user = f"Retrieved context:\n{joined_blocks}\n\nUser query:\n{query}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids[-MAX_PROMPT_TOKENS:]
    return rendered, ids


def prepare_packing(tokenizer, retrieval: pd.DataFrame, documents: dict[tuple[str, str], str]) -> pd.DataFrame:
    if PACKING.exists():
        output = pd.read_csv(PACKING, keep_default_na=False, dtype={"case_id": str})
        if len(output) == len(retrieval):
            return output
    rows = []
    for row in retrieval.sort_values("case_id").itertuples(index=False):
        source_ids = list(map(str, json.loads(row.source_ids)))
        texts = [documents[(str(row.domain), source)] for source in source_ids]
        visible, caps = pack_sources(tokenizer, source_ids, texts)
        _, prompt_ids = render_prompt(tokenizer, str(row.query), source_ids, visible, 0)
        rows.append({"case_id": str(row.case_id), "source_ids": json.dumps(source_ids),
                     "packed_texts": json.dumps(visible), "source_caps": json.dumps(caps),
                     "source_tokens": int(sum(caps)), "full_prompt_tokens": len(prompt_ids),
                     "left_truncated_to": MAX_PROMPT_TOKENS})
    output = pd.DataFrame(rows)
    atomic_csv(output, PACKING, "gzip")
    checkpoint("PACKING_COMPLETE", queries=len(output), max_source_tokens=int(output.source_tokens.max()),
               max_prompt_tokens=int(output.full_prompt_tokens.max()))
    return output


def run_generation(model, tokenizer, retrieval: pd.DataFrame, packing: pd.DataFrame, db: sqlite3.Connection) -> None:
    import torch
    complete = {row[0] for row in db.execute("SELECT case_id FROM answer")}
    packing_map = packing.set_index("case_id")
    tasks = []
    for row in retrieval.sort_values("case_id").itertuples(index=False):
        if str(row.case_id) in complete:
            continue
        packed = packing_map.loc[str(row.case_id)]
        source_ids = list(map(str, json.loads(packed.source_ids)))
        visible = list(map(str, json.loads(packed.packed_texts)))
        rendered, prompt_ids = render_prompt(tokenizer, str(row.query), source_ids, visible, 0)
        tasks.append((str(row.case_id), rendered, prompt_ids))
    checkpoint("A0_GENERATION_STARTED", total=len(retrieval), cached=len(complete), pending=len(tasks), batch_size=8)
    tokenizer.padding_side = "left"
    started = time.monotonic(); done0 = len(complete)
    for offset in range(0, len(tasks), 8):
        batch = tasks[offset:offset+8]
        texts = [item[1] for item in batch]
        encoded = tokenizer(texts, add_special_tokens=False, padding=True, truncation=True,
                            max_length=MAX_PROMPT_TOKENS, return_tensors="pt").to(model.device)
        before = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                        use_cache=True, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter()-before
        generated = output_ids[:, encoded.input_ids.shape[1]:]
        answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for (case_id, rendered, prompt_ids), answer, ids in zip(batch, answers, generated):
            clean = str(answer).strip()
            db.execute("INSERT OR REPLACE INTO answer VALUES (?,?,?,?,?,?,?)",
                       (case_id, clean, sha256_text(json.dumps(prompt_ids)), len(prompt_ids), int(ids.ne(tokenizer.pad_token_id).sum()),
                        elapsed/len(batch), __import__("common").now()))
        db.commit(); done = done0 + min(offset+len(batch), len(tasks))
        rate = max(1, done-done0)/max(time.monotonic()-started, 1e-9)
        checkpoint("A0_GENERATION_PROGRESS", total=len(retrieval), completed=done, pending=len(retrieval)-done,
                   percent=round(100*done/len(retrieval), 2), queries_per_second=round(rate, 3),
                   eta_seconds=round((len(tasks)-min(offset+len(batch),len(tasks)))/max(rate,1e-9)))
        del encoded, output_ids, generated
    db.commit(); checkpoint("A0_GENERATION_COMPLETE", answers=db.execute("SELECT COUNT(*) FROM answer").fetchone()[0])


def teacher_force(model, tokenizer, prompt_ids: list[int], answer: str) -> tuple[float, int, float]:
    import torch
    answer_ids = tokenizer(str(answer), add_special_tokens=False).input_ids
    if not answer_ids:
        answer_ids = [tokenizer.eos_token_id]
    sequence = prompt_ids + answer_ids
    ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(ids, use_cache=False, logits_to_keep=len(answer_ids)+1).logits.float()[0, :-1]
        targets = ids[0, -len(answer_ids):]
        value = torch.log_softmax(logits[-len(answer_ids):], dim=-1).gather(1, targets[:, None]).mean()
    elapsed = time.perf_counter()-started
    result = float(value.cpu())
    del ids, logits, targets, value
    return result, len(answer_ids), elapsed


def run_loo(model, tokenizer, retrieval: pd.DataFrame, packing: pd.DataFrame,
            gen_db: sqlite3.Connection, loo_db: sqlite3.Connection) -> None:
    answers = {str(case): str(answer) for case, answer in gen_db.execute("SELECT case_id,answer FROM answer")}
    complete = {row[0] for row in loo_db.execute("SELECT task_id FROM score")}
    pmap = packing.set_index("case_id")
    tasks = []
    for row in retrieval.sort_values("case_id").itertuples(index=False):
        packed = pmap.loc[str(row.case_id)]
        source_ids = list(map(str, json.loads(packed.source_ids)))
        visible = list(map(str, json.loads(packed.packed_texts)))
        for condition, rank in [("FULL", 0), *[(f"REMOVE_{value}", value) for value in range(1, 5)]]:
            task_id = sha256_text(f"LOO\0{row.case_id}\0{condition}\0{sha256_text(answers[str(row.case_id)])}")
            if task_id in complete:
                continue
            _, prompt_ids = render_prompt(tokenizer, str(row.query), source_ids, visible, rank)
            tasks.append((task_id, str(row.case_id), condition, rank, prompt_ids, answers[str(row.case_id)]))
    total = 5*len(retrieval); checkpoint("LOO_SCORING_STARTED", total=total, cached=len(complete), pending=len(tasks), microbatch=1)
    started = time.monotonic(); done0 = len(complete)
    for index, (task_id, case_id, condition, rank, prompt_ids, answer) in enumerate(tasks, 1):
        value, answer_tokens, wall = teacher_force(model, tokenizer, prompt_ids, answer)
        loo_db.execute("INSERT OR REPLACE INTO score VALUES (?,?,?,?,?,?,?,?,?)",
                       (task_id, case_id, condition, rank, value, len(prompt_ids), answer_tokens, wall,
                        __import__("common").now()))
        done = done0 + index
        if index % 20 == 0 or index == len(tasks):
            loo_db.commit(); rate = index/max(time.monotonic()-started, 1e-9)
            checkpoint("LOO_SCORING_PROGRESS", total=total, completed=done, pending=total-done,
                       percent=round(100*done/total, 2), tasks_per_second=round(rate, 3),
                       eta_seconds=round((len(tasks)-index)/max(rate,1e-9)))
    loo_db.commit(); checkpoint("LOO_SCORING_COMPLETE", tasks=loo_db.execute("SELECT COUNT(*) FROM score").fetchone()[0])


def aggregate(retrieval: pd.DataFrame, qll_db: sqlite3.Connection, gen_db: sqlite3.Connection,
              loo_db: sqlite3.Connection) -> pd.DataFrame:
    qll = pd.read_sql_query("SELECT case_id,source_rank,source_id,mean_logp FROM score", qll_db)
    loo = pd.read_sql_query("SELECT case_id,condition,source_rank,mean_logp,prompt_tokens,answer_tokens,wall_seconds FROM score", loo_db)
    answers = pd.read_sql_query("SELECT * FROM answer", gen_db)
    if len(qll) != 4*len(retrieval) or len(loo) != 5*len(retrieval) or len(answers) != len(retrieval):
        raise RuntimeError(f"incomplete aggregate inputs qll={len(qll)} loo={len(loo)} answers={len(answers)}")
    rows = []
    for row in retrieval.itertuples(index=False):
        qcell = qll[qll.case_id.eq(str(row.case_id))].sort_values(["mean_logp", "source_rank", "source_id"],
                                                                      ascending=[False, True, True])
        qtop = qcell.iloc[0]
        values = qcell.mean_logp.to_numpy(float)
        probabilities = np.exp(values-values.max()); probabilities /= probabilities.sum()
        lcell = loo[loo.case_id.eq(str(row.case_id))]
        full = float(lcell[lcell.condition.eq("FULL")].iloc[0].mean_logp)
        influence = {rank: full-float(lcell[lcell.source_rank.eq(rank)].iloc[0].mean_logp) for rank in range(1,5)}
        target_rank = int(row.target_rank)
        qrank = int(qtop.source_rank)
        candidates = sorted(set([1, qrank]))
        union_rank = max(candidates, key=lambda rank: (influence[rank], -rank))
        rows.append({"case_id": str(row.case_id), "kind": str(row.kind), "family": str(row.family),
                     "member": int(row.member), "session_id": str(row.session_id), "turn": int(row.turn),
                     "domain": str(row.domain), "query_sha256": str(row.query_sha256),
                     "target_id": str(row.target_id), "target_rank": target_rank,
                     "target_in_top4": target_rank > 0, "source_ids": str(row.source_ids),
                     "mirabel_source_rank": 1, "mirabel_source_id": json.loads(row.source_ids)[0],
                     "qll_source_rank": qrank, "qll_source_id": str(qtop.source_id),
                     "qll_dominance": float(probabilities[0]), "union_candidate_ranks": json.dumps(candidates),
                     "union_selected_rank": int(union_rank), "L_full": full,
                     **{f"L_minus_{rank}": full-influence[rank] for rank in range(1,5)},
                     **{f"I_{rank}": influence[rank] for rank in range(1,5)},
                     "risk_oracle": influence[target_rank] if target_rank > 0 else 0.0,
                     "risk_mirabel": influence[1], "risk_qll": influence[qrank],
                     "risk_union2": influence[union_rank],
                     "oracle_selected_rank": target_rank, "answer_sha256": sha256_text(
                         answers[answers.case_id.eq(str(row.case_id))].iloc[0].answer),
                     "answer_tokens": int(answers[answers.case_id.eq(str(row.case_id))].iloc[0].answer_tokens)})
    output = pd.DataFrame(rows)
    atomic_csv(output, QUERY_SCORES, "gzip")
    answer_review = retrieval[["case_id", "kind", "family", "member", "session_id", "turn", "domain", "query"]].merge(
        answers[["case_id", "answer", "prompt_tokens", "answer_tokens", "wall_seconds"]], on="case_id", validate="one_to_one")
    atomic_csv(answer_review, ROOT / "tables/PHASE1_QUERY_A0_REVIEW.csv.gz", "gzip")
    checkpoint("GPU_PHASE1_COMPLETE", query_scores=len(output), file_sha256=sha256_file(QUERY_SCORES),
               next_stage="CPU_ANALYSIS")
    return output


def main() -> None:
    frame, corpora = preflight()
    retrieval = retrieve(frame, corpora)
    documents = document_lookup(corpora)
    qll_db, gen_db, loo_db = init_dbs()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    checkpoint("QWEN_LOADING", model=str(QWEN), cuda_name=torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    torch.manual_seed(20260911)
    torch.cuda.manual_seed_all(20260911)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    checkpoint("QWEN_LOADED", peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
    run_qll(model, tokenizer, retrieval, documents, qll_db)
    packing = prepare_packing(tokenizer, retrieval, documents)
    run_generation(model, tokenizer, retrieval, packing, gen_db)
    run_loo(model, tokenizer, retrieval, packing, gen_db, loo_db)
    aggregate(retrieval, qll_db, gen_db, loo_db)
    qll_db.close(); gen_db.close(); loo_db.close()
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
