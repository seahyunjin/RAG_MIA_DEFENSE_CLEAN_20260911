#!/usr/bin/env python3
"""Phase A: existing-signal complementarity audit on the frozen V2 substrate."""
from __future__ import annotations

import csv
import json
import math
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import (ALPHAS, ATTACKS, EXP, PARENT, PRIMARY_ALPHA, QWEN, V2, atomic_json,
                    checkpoint, matched_binary_threshold, now, read_jsonl, sha_file,
                    spearman, tpr_at_fpr, write_csv)


PRE = EXP / "configs/COMPLEMENTARITY_AUDIT_PRECOMMIT.json"
PRE_SHA = PRE.with_suffix(".sha256")
SCORES = V2 / "cache/FRESH_REPLICATION_RETRIEVAL_SCORES.jsonl"
QLL_DB = EXP / "private/PHASE_A_QLL.sqlite3"


def preflight() -> tuple[dict, list[dict], dict[str, str]]:
    expected = PRE_SHA.read_text().strip()
    if sha_file(PRE) != expected: raise RuntimeError("Phase A precommit hash drift")
    pre = json.loads(PRE.read_text())
    if sha_file(Path(__file__)) != pre["code_sha256"][Path(__file__).name]:
        raise RuntimeError("Phase A implementation changed after precommit")
    if sha_file(V2 / "PHASE1_RESULT.json") != pre["frozen_v2_result"]["sha256"]:
        raise RuntimeError("frozen V2 result drift")
    if json.loads((V2 / "PHASE1_RESULT.json").read_text())["verdict"] != "DUALTAIL_MEMBER_EXPOSURE_NOT_REPLICATED":
        raise RuntimeError("frozen verdict changed")
    if sha_file(SCORES) != pre["fresh_retrieval"]["sha256"]:
        raise RuntimeError("fresh retrieval drift")
    rows = read_jsonl(SCORES)
    if len(rows) != 2400 or len({r["query_id"] for r in rows}) != 2400:
        raise RuntimeError("fresh substrate identity/count mismatch")
    docs = {r["document_id"]: r["source_text"] for r in read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")}
    if len(docs) != 3000 or any(any(x not in docs for x in r["top_document_ids"]) for r in rows):
        raise RuntimeError("protected document lookup mismatch")
    checkpoint("PHASE_A_PREFLIGHT_PASS", rows=len(rows), benign=sum(r["cohort"] == "BENIGN" for r in rows),
               precommit_sha256=expected, development_notice="THIS_COHORT_IS_NOW_DEVELOPMENT_DATA")
    return pre, rows, docs


def initialize_qll_db() -> sqlite3.Connection:
    db = sqlite3.connect(QLL_DB); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
    db.execute("""CREATE TABLE IF NOT EXISTS score (
      task_id TEXT PRIMARY KEY, query_id TEXT NOT NULL, source_rank INTEGER NOT NULL,
      source_id TEXT NOT NULL, mean_logp REAL NOT NULL, query_tokens INTEGER NOT NULL,
      prefix_tokens INTEGER NOT NULL, wall_seconds REAL NOT NULL, completed_utc TEXT NOT NULL)""")
    db.commit(); return db


def qll_scores(rows: list[dict], docs: dict[str, str]) -> dict[str, float]:
    """Exact canonical QLL definition, batched only for throughput and resume safety."""
    db = initialize_qll_db()
    complete = {r[0] for r in db.execute("SELECT task_id FROM score")}
    tasks = []
    for row in sorted(rows, key=lambda r: r["query_id"]):
        for rank, source_id in enumerate(row["top_document_ids"], 1):
            task_id = __import__("hashlib").sha256(f"QLL\0{row['query_id']}\0{rank}\0{source_id}".encode()).hexdigest()
            if task_id not in complete:
                tasks.append((task_id, row["query_id"], rank, source_id, docs[source_id], row["query"]))
    checkpoint("PHASE_A_QLL_STARTED", total_pairs=4*len(rows), cached_pairs=len(complete), pending_pairs=len(tasks),
               eligibility="ANALYSIS_ONLY_NON_BLACKBOX")
    if tasks:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable for exact QLL recalculation")
        tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            QWEN, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        before = tokenizer("Context:\n", add_special_tokens=False).input_ids
        bridge = tokenizer("\n\nUser query:\n", add_special_tokens=False).input_ids
        maximum = int(model.config.max_position_embeddings)
        encoded = []
        for task in tasks:
            source_ids = tokenizer(str(task[4]), add_special_tokens=False).input_ids
            query_ids = tokenizer(str(task[5]), add_special_tokens=False).input_ids or [tokenizer.eos_token_id]
            source_ids = source_ids[:max(0, maximum-len(before)-len(bridge)-len(query_ids))]
            encoded.append((task, before+source_ids+bridge+query_ids, len(query_ids)))
        encoded.sort(key=lambda x: x[0][0])
        started = time.monotonic(); done0 = len(complete); offset = 0
        while offset < len(encoded):
            # Deliberately preserve the canonical microbatch=1 calculation.
            # Left-padded multi-example batches showed non-negligible numerical
            # drift in the frozen Qwen implementation during pre-result audit.
            batch = encoded[offset:offset+1]
            max_len = max(len(x[1]) for x in batch); max_q = max(x[2] for x in batch)
            input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
            attention = torch.zeros((len(batch), max_len), dtype=torch.long, device="cuda")
            for i, (_, ids, _) in enumerate(batch):
                input_ids[i, -len(ids):] = torch.tensor(ids, dtype=torch.long, device="cuda")
                attention[i, -len(ids):] = 1
            t0 = time.perf_counter()
            with torch.inference_mode():
                output = model(input_ids=input_ids, attention_mask=attention, use_cache=False,
                               logits_to_keep=max_q+1).logits
            wall = time.perf_counter()-t0
            for i, (task, ids, qlen) in enumerate(batch):
                logits = output[i, -qlen-1:-1].float()
                targets = input_ids[i, -qlen:]
                value = -torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
                db.execute("INSERT OR REPLACE INTO score VALUES (?,?,?,?,?,?,?,?,?)",
                           (task[0], task[1], task[2], task[3], float(value.cpu()), qlen,
                            len(ids)-qlen, wall/len(batch), now()))
            db.commit(); offset += len(batch)
            if offset % 80 < len(batch) or offset == len(encoded):
                elapsed = max(time.monotonic()-started, 1e-9); rate = offset/elapsed
                checkpoint("PHASE_A_QLL_PROGRESS", total_pairs=4*len(rows), completed_pairs=done0+offset,
                           pending_pairs=len(encoded)-offset, percent=round(100*(done0+offset)/(4*len(rows)), 2),
                           pairs_per_second=round(rate, 3), eta_seconds=round((len(encoded)-offset)/rate))
            del input_ids, attention, output
        del model; torch.cuda.empty_cache()
    cells: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for query_id, rank, value in db.execute("SELECT query_id,source_rank,mean_logp FROM score"):
        cells[str(query_id)].append((int(rank), float(value)))
    db.close()
    if len(cells) != len(rows) or any(len(v) != 4 for v in cells.values()):
        raise RuntimeError("QLL cache incomplete")
    dominance = {}
    for qid, pairs in cells.items():
        values = [x[1] for x in sorted(pairs)]; top = max(values)
        weights = [math.exp(x-top) for x in values]
        dominance[qid] = max(weights)/sum(weights)
    checkpoint("PHASE_A_QLL_COMPLETE", queries=len(dominance), pairs=4*len(dominance),
               sqlite_sha256=sha_file(QLL_DB), eligibility="ANALYSIS_ONLY_NON_BLACKBOX")
    return dominance


def binary_cells(rows: list[dict], signal: str, m_t: float, x_t: float) -> tuple[Counter, list[dict]]:
    counts = Counter(); cases = []
    for r in rows:
        mh, xh = r["M"] > m_t, r[signal] > x_t
        cell = "BOTH_HIT" if mh and xh else "MIRABEL_ONLY" if mh else "X_ONLY" if xh else "BOTH_MISS"
        counts[cell] += 1; cases.append({"query_id": r["query_id"], "session_id": r["session_id"], "cell": cell})
    return counts, cases


def main() -> None:
    started = time.monotonic(); pre, rows, docs = preflight()
    unavailable = [{"signal": "Answer-LOO", "status": "SIGNAL_UNAVAILABLE",
                    "reason": "No exact fresh A0 answer plus four leave-one-source-out likelihood records exist on this 2,400-query substrate.",
                    "runtime_eligibility": "ANALYSIS_ONLY_POST_GENERATION"}]
    write_csv(EXP / "audits/PHASE_A_SIGNAL_AVAILABILITY.csv", [
        {"signal": "MIRABEL", "status": "AVAILABLE", "runtime_eligibility": "STRICT_BLACKBOX", "reason": "frozen V2 score"},
        {"signal": "DualTail", "status": "AVAILABLE", "runtime_eligibility": "STRICT_BLACKBOX", "reason": "frozen V2 score"},
        {"signal": "QLL", "status": "RECALCULATED", "runtime_eligibility": "ANALYSIS_ONLY_NON_BLACKBOX", "reason": "exact frozen model/formula/source lineage"},
        unavailable[0]])
    qll = qll_scores(rows, docs)
    for r in rows: r["QLL"] = qll[r["query_id"]]
    benign = [r for r in rows if r["cohort"] == "BENIGN"]
    signals = [("S", "DualTail", "STRICT_BLACKBOX"), ("QLL", "QLL", "ANALYSIS_ONLY_NON_BLACKBOX")]
    matched_rows = []; overlap_rows = []; oracle_rows = []; corr_rows = []; menta_session_rows = []
    conclusions = {}
    for key, name, eligibility in signals:
        neg_m, neg_x = [r["M"] for r in benign], [r[key] for r in benign]
        for alpha in ALPHAS:
            for attack in ATTACKS:
                member = [r for r in rows if r.get("attack") == attack and r.get("membership") == "member"]
                mt = tpr_at_fpr([r["M"] for r in member], neg_m, alpha)
                xt = tpr_at_fpr([r[key] for r in member], neg_x, alpha)
                matched_rows.append({"signal": name, "eligibility": eligibility, "attack": attack,
                                     "fpr": alpha, "member_queries": len(member), "mirabel_tpr": mt,
                                     "signal_tpr": xt, "delta": xt-mt})
        mt, mn = matched_binary_threshold(neg_m, PRIMARY_ALPHA)
        xt, xn = matched_binary_threshold(neg_x, PRIMARY_ALPHA)
        bc, _ = binary_cells(benign, key, mt, xt)
        benign_jaccard = bc["BOTH_HIT"] / max(1, bc["BOTH_HIT"]+bc["MIRABEL_ONLY"]+bc["X_ONLY"])
        for attack in ATTACKS:
            member = [r for r in rows if r.get("attack") == attack and r.get("membership") == "member"]
            cells, cases = binary_cells(member, key, mt, xt)
            unique, lost = cells["X_ONLY"], cells["MIRABEL_ONLY"]
            added_fp = bc["X_ONLY"]
            overlap_rows.append({"signal": name, "eligibility": eligibility, "attack": attack,
                "member_n": len(member), "both_hit": cells["BOTH_HIT"], "mirabel_only": lost,
                "signal_only_unique_recovery": unique, "both_miss": cells["BOTH_MISS"],
                "unique_recovery_rate": unique/len(member), "lost_coverage_rate": lost/len(member),
                "net_recovery": unique-lost, "net_recovery_rate": (unique-lost)/len(member),
                "benign_both_fp": bc["BOTH_HIT"], "benign_mirabel_only_fp": bc["MIRABEL_ONLY"],
                "benign_signal_only_fp": added_fp, "benign_neither": bc["BOTH_MISS"],
                "benign_alarm_jaccard": benign_jaccard,
                "useful_recovery_per_added_fp": None if added_fp == 0 else unique/added_fp})
            union_member = sum((r["M"] > mt) or (r[key] > xt) for r in member)
            union_benign = sum((r["M"] > mt) or (r[key] > xt) for r in benign)
            oracle_rows.append({"signal": name, "attack": attack, "member_n": len(member),
                "union_member_coverage": union_member/len(member), "benign_n": len(benign),
                "union_benign_fpr": union_benign/len(benign), "unique_member_recovery": unique,
                "unique_benign_fp": added_fp, "diagnostic_only": True})
            if attack == "MEntA":
                by_session: dict[str, list[dict]] = defaultdict(list)
                for r, case in zip(member, cases): by_session[r["session_id"]].append(case)
                for sid, cc in sorted(by_session.items()):
                    menta_session_rows.append({"signal": name, "session_id": sid,
                        "unique_recovered_queries": sum(x["cell"] == "X_ONLY" for x in cc),
                        "lost_queries": sum(x["cell"] == "MIRABEL_ONLY" for x in cc),
                        "unique_recovered_session": any(x["cell"] == "X_ONLY" for x in cc),
                        "lost_session": any(x["cell"] == "MIRABEL_ONLY" for x in cc)})
        for group, subset in [("benign", benign)] + [(a, [r for r in rows if r.get("attack") == a and r.get("membership") == "member"]) for a in ATTACKS]:
            corr_rows.append({"signal": name, "eligibility": eligibility, "group": group, "n": len(subset),
                              "spearman_rho_M_X": spearman([r["M"] for r in subset], [r[key] for r in subset])})
        menta = next(r for r in overlap_rows if r["signal"] == name and r["attack"] == "MEntA")
        conclusions[name] = {"eligibility": eligibility, "menta_unique_recovery_rate": menta["unique_recovery_rate"],
                             "menta_net_recovery_rate": menta["net_recovery_rate"],
                             "meaningful_5pp_net_recovery": menta["net_recovery_rate"] >= .05-1e-12}
    blackbox = any(v["meaningful_5pp_net_recovery"] and v["eligibility"] == "STRICT_BLACKBOX" for v in conclusions.values())
    generator = any(v["meaningful_5pp_net_recovery"] and v["eligibility"] != "STRICT_BLACKBOX" for v in conclusions.values())
    verdict = "A_EXISTING_BLACKBOX_COMPLEMENT_FOUND" if blackbox else "A_ONLY_GENERATOR_SIDE_COMPLEMENT_FOUND" if generator else "A_NO_MEANINGFUL_COMPLEMENT"
    write_csv(EXP / "tables/PHASE_A_MATCHED_FPR_TPR.csv", matched_rows)
    write_csv(EXP / "tables/PHASE_A_COMPLEMENTARITY.csv", overlap_rows)
    write_csv(EXP / "tables/PHASE_A_ORACLE_HEADROOM.csv", oracle_rows)
    write_csv(EXP / "tables/PHASE_A_CORRELATION.csv", corr_rows)
    write_csv(EXP / "tables/PHASE_A_MENTA_SESSION_RECOVERY.csv", menta_session_rows)
    result = {"verdict": verdict, "completed_utc": now(), "runtime_seconds": time.monotonic()-started,
        "precommit_sha256": sha_file(PRE), "frozen_v2_verdict_preserved": "DUALTAIL_MEMBER_EXPOSURE_NOT_REPLICATED",
        "development_notice": "THIS_COHORT_IS_NOW_DEVELOPMENT_DATA", "signal_availability": {
            "MIRABEL": "STRICT_BLACKBOX", "DualTail": "STRICT_BLACKBOX",
            "QLL": "ANALYSIS_ONLY_NON_BLACKBOX", "Answer-LOO": "SIGNAL_UNAVAILABLE"},
        "conclusions": conclusions,
        "answers": {"Q1_existing_5pp_complement": blackbox or generator,
                    "Q2_strict_blackbox_runtime_eligible": blackbox,
                    "Q3_generator_side_evidence_required_for_headroom": generator and not blackbox},
        "phase_b_required": True}
    atomic_json(EXP / "PHASE_A_RESULT.json", result)
    report = ["# Phase A — Existing-signal Complementarity Audit", "", f"- 판정: `{verdict}`",
              "- QLL: exact recalculation, mechanism audit only (non-black-box)",
              "- Answer-LOO: `SIGNAL_UNAVAILABLE` (fresh A0/ablation provenance 없음)", "",
              "| Signal | MEntA unique recovery | MEntA net recovery | Runtime |", "|---|---:|---:|---|"]
    for name, value in conclusions.items():
        report.append(f"| {name} | {value['menta_unique_recovery_rate']:.3f} | {value['menta_net_recovery_rate']:+.3f} | {value['eligibility']} |")
    report += ["", "Phase A에서는 fusion/model selection을 하지 않았다. 명세대로 Phase B로 이동한다."]
    (EXP / "reports/PHASE_A_REPORT_KO.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    checkpoint("PHASE_A_COMPLETE", verdict=verdict, phase_b_required=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
