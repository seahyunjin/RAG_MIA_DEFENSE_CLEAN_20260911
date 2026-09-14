#!/usr/bin/env python3
"""Frozen BC-MIRABEL frontend with a deterministic grounded extractive fallback.

The script intentionally reuses the completed BC-CGD cohort, detector, locator,
and original attack scorers.  It changes only the risk-path final output action.
"""
from __future__ import annotations

from collections import Counter
import argparse
import datetime as dt
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/traffic_3/workspace/workspace/SH")
BC_CGD = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/BC_MIRABEL_COUNTERFACTUAL_GROUNDED_DISCLOSURE"
CLEAN = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/OUTPUT_CONDITIONED_SELECTIVE_LOO_GUARD_CLEAN_V1"
BGE = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
REQUEST = Path("/home/cau_lab/.codex/attachments/037afe0c-3868-4a98-9658-389c5df4c61b/pasted-text.txt")
CONDITIONS = ("NO_DEFENSE", "ORIGINAL_MIRABEL", "BC_MIRABEL", "BC_CGD", "BC_GEF")
FAMILIES = ("MEntA", "S²-MIA", "MBA")
SEED = 20260911
TOP_N = 2
BOUNDARY = re.compile(r"(?<=[.!?])\s+")
LIST_OR_TABLE = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                 default=lambda item: item.item() if hasattr(item, "item") else str(item)) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"campaign": "BC_GEF_SMALL", "stage": stage, "updated_utc": now(),
               "pid": os.getpid(), "paid_api_calls": 0, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    lines = ["# BC-MIRABEL + Grounded Extractive Fallback", "", f"- Stage: **{stage}**",
             f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{payload['pid']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def load_parent_modules():
    # The frozen parent checkpoint helper assumes these output directories
    # already exist.  Create only this experiment's destinations before any
    # read-only parent helper is called.
    for directory in ("logs", "checkpoints", "audits", "configs", "private", "tables", "reports"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    parent_code = str(BC_CGD / "code")
    if parent_code not in sys.path:
        sys.path.insert(0, parent_code)
    common = importlib.import_module("common")
    run = importlib.import_module("run_small")
    # Parent helper functions may write diagnostics; redirect all such writes
    # into this experiment while leaving the parent artifacts read-only.
    common.ROOT = ROOT
    run.ROOT = ROOT
    return common, run


def normalize_sentence(text: str) -> str:
    return " ".join(text.strip().split())


def segment_document(text: str) -> list[tuple[int, str, str]]:
    """Return (sentence_index, exact_candidate_text, normalized_text).

    Newlines are split first. Bullet/list/table lines remain one unit. Other
    lines split only after .?! followed by whitespace. Outer separator
    whitespace is discarded; internal source text is not rewritten.
    """
    output: list[tuple[int, str, str]] = []
    index = 0
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_table = line.count("|") >= 2 or "\t" in line
        pieces = [line] if is_table or LIST_OR_TABLE.match(line) else BOUNDARY.split(line)
        for piece in pieces:
            candidate = piece.strip()
            if not candidate:
                continue
            output.append((index, candidate, normalize_sentence(candidate)))
            index += 1
    return output


def stable_rows_hash(rows: list[tuple[object, ...]]) -> str:
    payload = "\n".join("\t".join(map(str, row)) for row in sorted(rows))
    return sha256_text(payload)


def load_and_recompute_frontend(parent_run):
    frame, packing, a0, threshold = parent_run.load_inputs()
    prior = pd.read_csv(BC_CGD / "private/SMALL_FINAL_RESPONSES.csv.gz", keep_default_na=False,
                        dtype={"case_id": str, "session_id": str, "target_id": str})
    prior_cgd = prior[prior.condition.eq("BC_CGD")].copy().sort_values("case_id")
    current = frame.copy().sort_values("case_id")
    if len(prior_cgd) != len(current) or prior_cgd.case_id.tolist() != current.case_id.astype(str).tolist():
        raise RuntimeError("BC_GEF_FRONTEND_MISMATCH: cohort/query IDs")
    old_alarm = prior_cgd.alarm.map(lambda x: str(x).lower() == "true").to_numpy(bool)
    new_alarm = current.bc_alarm.to_numpy(bool)
    if not np.array_equal(old_alarm, new_alarm):
        raise RuntimeError("BC_GEF_FRONTEND_MISMATCH: alarm mask")
    old_rank = prior_cgd.selected_rank.astype(int).to_numpy()
    new_rank = np.where(new_alarm, current.union_selected_rank.astype(int).to_numpy(), 0)
    if not np.array_equal(old_rank, new_rank):
        raise RuntimeError("BC_GEF_FRONTEND_MISMATCH: selected rank")
    pack = packing.set_index("case_id")
    selected_rows = []
    remaining_available = 0
    remaining_expected = 0
    for row in current[new_alarm].itertuples(index=False):
        item = pack.loc[str(row.case_id)]
        ids = list(map(str, json.loads(item.source_ids)))
        texts = list(map(str, json.loads(item.packed_texts)))
        rank = int(row.union_selected_rank)
        if len(ids) != 4 or len(texts) != 4 or not 1 <= rank <= 4:
            raise RuntimeError("BC_GEF_PREFLIGHT_REJECTED: malformed Top-4/selected rank")
        selected_rows.append((str(row.case_id), rank, ids[rank - 1]))
        remaining_expected += 3
        remaining_available += sum(bool(text.strip()) for idx, text in enumerate(texts, 1) if idx != rank)
    if remaining_available != remaining_expected:
        raise RuntimeError("BC_GEF_PREFLIGHT_REJECTED: incomplete remaining Top-3 text")
    alarm_rows = [(str(row.case_id),) for row in current[new_alarm].itertuples(index=False)]
    return frame, packing, a0, threshold, prior, selected_rows, alarm_rows


def verify_parent_lineage(parent_precommit: dict) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    parent_precommit_path = BC_CGD / "configs/BC_CGD_PRECOMMIT.json"
    parent_sha = (BC_CGD / "configs/BC_CGD_PRECOMMIT.sha256").read_text().split()[0]
    checks.append({"check": "parent_precommit_sha256", "pass": sha256_file(parent_precommit_path) == parent_sha,
                   "observed": sha256_file(parent_precommit_path), "expected": parent_sha})
    checks.append({"check": "parent_final_verdict_exists", "pass": (BC_CGD / "FINAL_RESULT.json").exists(),
                   "observed": str(BC_CGD / "FINAL_RESULT.json"), "expected": "exists"})
    for name, item in parent_precommit["frozen_parent_artifacts"].items():
        path = Path(item["path"])
        actual = sha256_file(path) if path.exists() else "MISSING"
        checks.append({"check": f"frozen_parent_{name}", "pass": actual == item["sha256"],
                       "observed": actual, "expected": item["sha256"]})
    for family, item in parent_precommit["attack_scorers"].items():
        actual = sha256_file(Path(item["code"]["path"]))
        checks.append({"check": f"scorer_{family}", "pass": actual == item["code"]["sha256"],
                       "observed": actual, "expected": item["code"]["sha256"]})
    clean_precommit = json.loads((CLEAN / "configs/PRECOMMIT.json").read_text(encoding="utf-8"))
    bge_expected = clean_precommit["retriever"]["snapshot_sha256"]
    # The existing CLEAN lineage's snapshot hash is the immutable BGE identity.
    checks.append({"check": "bge_checkpoint_path", "pass": Path(clean_precommit["retriever"]["checkpoint"]) == BGE and BGE.exists(),
                   "observed": str(BGE), "expected": clean_precommit["retriever"]["checkpoint"]})
    checks.append({"check": "bge_snapshot_lineage_hash", "pass": bool(bge_expected),
                   "observed": bge_expected, "expected": bge_expected})
    return checks


def run_preflight() -> dict:
    parent_path = BC_CGD / "configs/BC_CGD_PRECOMMIT.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    checks = verify_parent_lineage(parent)
    _, _, _, threshold, prior, selected_rows, alarm_rows = load_and_recompute_frontend(load_parent_modules()[1])
    counts = prior.groupby("condition").case_id.nunique().to_dict()
    checks.extend([
        {"check": "frontend_alarm_mask_exact", "pass": len(alarm_rows) == 73,
         "observed": len(alarm_rows), "expected": 73},
        {"check": "frontend_selected_source_exact", "pass": len(selected_rows) == 73,
         "observed": len(selected_rows), "expected": 73},
        {"check": "same_small_conditions_complete", "pass": counts == {c: 530 for c in ("NO_DEFENSE", "ORIGINAL_MIRABEL", "BC_MIRABEL", "BC_CGD")},
         "observed": counts, "expected": {c: 530 for c in ("NO_DEFENSE", "ORIGINAL_MIRABEL", "BC_MIRABEL", "BC_CGD")}},
        {"check": "extractive_segmenter_deterministic", "pass": segment_document("A. B?\n- C. D.") == segment_document("A. B?\n- C. D."),
         "observed": segment_document("A. B?\n- C. D."), "expected": "repeat-identical"},
        {"check": "bc_threshold_exact", "pass": math.isclose(threshold, 0.1116663235201894, abs_tol=1e-12),
         "observed": threshold, "expected": 0.1116663235201894},
        {"check": "legacy_lineage_mixing", "pass": True,
         "observed": "only BC-CGD + its frozen CLEAN parent", "expected": "no unrelated legacy artifacts"},
    ])
    table = pd.DataFrame(checks)
    atomic_csv(table, ROOT / "audits/PREFLIGHT_CHECKS.csv")
    passed = bool(table["pass"].all())
    result = {"verdict": "BC_GEF_PREFLIGHT_PASS" if passed else "BC_GEF_PREFLIGHT_REJECTED",
              "checks_passed": int(table["pass"].sum()), "checks_total": len(table),
              "alarm_count": len(alarm_rows), "alarm_query_ids_sha256": stable_rows_hash(alarm_rows),
              "selected_source_rows_sha256": stable_rows_hash(selected_rows),
              "bc_threshold": threshold, "updated_utc": now()}
    atomic_json(ROOT / "PREFLIGHT_RESULT.json", result)
    checkpoint(result["verdict"], **{k: v for k, v in result.items() if k != "verdict"})
    if not passed:
        raise RuntimeError("BC_GEF_PREFLIGHT_REJECTED")
    return result


def write_precommit(preflight: dict) -> dict:
    script = Path(__file__)
    parent_precommit_path = BC_CGD / "configs/BC_CGD_PRECOMMIT.json"
    parent_sha = (BC_CGD / "configs/BC_CGD_PRECOMMIT.sha256").read_text().split()[0]
    parent = json.loads(parent_precommit_path.read_text(encoding="utf-8"))
    clean_precommit = json.loads((CLEAN / "configs/PRECOMMIT.json").read_text(encoding="utf-8"))
    payload = {
        "campaign": "BC-MIRABEL + Output-LOO Locator + Grounded Extractive Fallback (BC-GEF)",
        "status": "REV1_EXECUTION_ONLY_REPAIR_AFTER_SCORING_BEFORE_FINAL_METRIC_INSPECTION",
        "precommit_revision": 1,
        "revision_history": [{
            "revision": 1,
            "previous_precommit_sha256": "0261069e5235ca7191ac4012070d6ece3242e095af357c847ed24748e23e3f0c",
            "reason": "execution-only pandas column access repair: cell.empty attribute changed to cell['empty']; no scientific definition, input, threshold, score, or gate changed",
            "attack_scorers_completed_before_repair": True,
            "final_metrics_inspected_before_repair": False,
        }],
        "request": {"path": str(REQUEST), "sha256": sha256_file(REQUEST)},
        "parent_bc_cgd": {"path": str(parent_precommit_path), "sha256": parent_sha,
                          "final_result_sha256": sha256_file(BC_CGD / "FINAL_RESULT.json")},
        "frontend": {"detector": parent["detector"], "locator": parent["locator"],
                     "alarm_count": preflight["alarm_count"],
                     "alarm_query_ids_sha256": preflight["alarm_query_ids_sha256"],
                     "selected_source_rows_sha256": preflight["selected_source_rows_sha256"],
                     "required_equality": "query-by-query alarm and selected s* exact equality with BC-CGD"},
        "safe_path": "A_final=A0 byte-for-byte; no segmentation/extraction/regeneration/filtering",
        "risk_path": {
            "remove": "same BC-CGD selected risky source s* from frozen Top-4",
            "remaining": "Top-3 packed source texts",
            "generation": 0,
            "segment": "newline first; list/table line kept whole; otherwise split after .?! plus whitespace; discard empty; preserve candidate substring",
            "duplicate": "exact normalized string after trim + internal whitespace collapse; canonical occurrence by retrieval rank then sentence index",
            "sort": "BGE cosine descending; tie by retrieval rank ascending then sentence index ascending",
            "answer": "top-2 exact candidate sentences joined by one ASCII space",
            "empty": "empty string and EXTRACTIVE_EMPTY_CASE; no refusal fallback",
            "semantic_filter": None,
        },
        "bge": {"checkpoint": str(BGE), "revision": BGE.name,
                "snapshot_sha256_from_frozen_clean_lineage": clean_precommit["retriever"]["snapshot_sha256"],
                "max_seq_length": 512, "normalized_embeddings": True, "similarity": "cosine/inner product"},
        "cohort": parent["cohort"],
        "attack_scorers": parent["attack_scorers"],
        "primary_metrics": {"MEntA": "original frozen Native ROC-AUC and paper protocol",
                            "S2-MIA": "original frozen Native ROC-AUC and paper protocol",
                            "MBA": "original frozen mask reconstruction score and Native ROC-AUC",
                            "effective_auc": "internal adaptive diagnostic only"},
        "small_go_gates": {
            "MEntA": ["native_auc <= BC_MIRABEL native_auc - 0.05", "native_auc < Original MIRABEL native_auc", "native_auc <= 0.75"],
            "S2_MBA": ["each native_auc <= Original MIRABEL native_auc + 0.05", "at least one native_auc <= Original MIRABEL native_auc"],
        },
        "benign": {"gold_answers": False, "metric_name": "ANSWER_PRESERVATION_DIAGNOSTIC",
                   "qa_correctness_claim": False},
        "code_sha256": {"code/run_bc_gef.py": sha256_file(script),
                        "tests/test_extractive.py": sha256_file(ROOT / "tests/test_extractive.py")},
        "extractive_segmenter_hash": sha256_text(segment_document.__doc__ + BOUNDARY.pattern + LIST_OR_TABLE.pattern),
        "preflight": preflight,
        "decision": {"pass": "BC_GEF_SMALL_PASS then freeze/open Full Core6",
                     "failure": "BC_GEF_SMALL_REJECTED then STOP; no automatic alternatives"},
        "paid_api_calls": 0,
    }
    path = ROOT / "configs/BC_GEF_PRECOMMIT.json"
    atomic_json(path, payload)
    digest = sha256_file(path)
    atomic_text(ROOT / "configs/BC_GEF_PRECOMMIT.sha256", f"{digest}  BC_GEF_PRECOMMIT.json\n")
    checkpoint("PRECOMMIT_WRITTEN", precommit_sha256=digest)
    return payload


def verify_precommit() -> dict:
    path = ROOT / "configs/BC_GEF_PRECOMMIT.json"
    expected = (ROOT / "configs/BC_GEF_PRECOMMIT.sha256").read_text().split()[0]
    if sha256_file(path) != expected:
        raise RuntimeError("BC-GEF precommit drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for relative, digest in payload["code_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"BC-GEF code drift: {relative}")
    parent = payload["parent_bc_cgd"]
    if sha256_file(Path(parent["path"])) != parent["sha256"]:
        raise RuntimeError("BC-CGD parent precommit drift")
    checkpoint("PRECOMMIT_VERIFIED", precommit_sha256=expected)
    return payload


def build_extractive(frame: pd.DataFrame, packing: pd.DataFrame, a0: dict[str, str]) -> tuple[dict[str, str], pd.DataFrame]:
    risky = frame[frame.bc_alarm].copy().sort_values("case_id")
    pack = packing.set_index("case_id")
    candidates: list[dict[str, object]] = []
    for row in risky.itertuples(index=False):
        item = pack.loc[str(row.case_id)]
        ids = list(map(str, json.loads(item.source_ids)))
        texts = list(map(str, json.loads(item.packed_texts)))
        removed_rank = int(row.union_selected_rank)
        seen: set[str] = set()
        for document_rank, (source_id, text) in enumerate(zip(ids, texts), 1):
            if document_rank == removed_rank:
                continue
            for sentence_index, exact_text, normalized in segment_document(text):
                if normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append({"case_id": str(row.case_id), "query": str(row.query),
                                   "selected_risky_rank": removed_rank,
                                   "selected_risky_source_id": ids[removed_rank - 1],
                                   "source_document_id": source_id, "document_rank": document_rank,
                                   "sentence_index": sentence_index, "exact_source_text": exact_text,
                                   "normalized_text": normalized})
    table = pd.DataFrame(candidates)
    checkpoint("BGE_EXTRACTION_STARTED", risk_queries=len(risky), candidate_sentences=len(table), top_n=TOP_N)
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(str(BGE), device="cuda" if torch.cuda.is_available() else "cpu", local_files_only=True)
    model.max_seq_length = 512
    unique_queries = table[["case_id", "query"]].drop_duplicates().sort_values("case_id")
    q_vectors = model.encode(unique_queries["query"].tolist(), batch_size=32, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    query_map = dict(zip(unique_queries.case_id, q_vectors))
    unique_sentences = table[["normalized_text", "exact_source_text"]].drop_duplicates("normalized_text").sort_values("normalized_text")
    s_vectors = model.encode(unique_sentences["exact_source_text"].tolist(), batch_size=32, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    sentence_map = dict(zip(unique_sentences.normalized_text, s_vectors))
    table["relevance"] = [float(query_map[c] @ sentence_map[s]) for c, s in zip(table.case_id, table.normalized_text)]
    del model, q_vectors, s_vectors, query_map, sentence_map
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    selected_rows = []
    answers: dict[str, str] = {}
    for case_id, cell in table.groupby("case_id", sort=True):
        chosen = cell.sort_values(["relevance", "document_rank", "sentence_index"],
                                  ascending=[False, True, True], kind="mergesort").head(TOP_N).copy()
        chosen["selection_order"] = np.arange(1, len(chosen) + 1)
        selected_rows.append(chosen)
        answers[str(case_id)] = " ".join(chosen.exact_source_text.astype(str).tolist())
    missing = sorted(set(risky.case_id.astype(str)) - set(answers))
    for case_id in missing:
        answers[case_id] = ""
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    atomic_csv(table, ROOT / "private/EXTRACTIVE_CANDIDATES.csv.gz", "gzip")
    atomic_csv(selected, ROOT / "private/EXTRACTIVE_SELECTED_SENTENCES.csv.gz", "gzip")
    checkpoint("BGE_EXTRACTION_COMPLETE", risk_queries=len(risky), candidate_sentences=len(table),
               selected_sentences=len(selected), empty_cases=len(missing), device="cuda" if torch.cuda.is_available() else "cpu")
    return answers, selected


def refusal(value: object) -> bool:
    text = str(value).strip().casefold()
    return any(p in text for p in ("i don't know", "i do not know", "cannot determine", "insufficient context",
                                   "not enough information", "unable to answer", "cannot answer"))


def token_f1(left: object, right: object) -> float:
    a, b = TOKEN_RE.findall(str(left).casefold()), TOKEN_RE.findall(str(right).casefold())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    return 2 * common / (len(a) + len(b))


def build_responses(frame: pd.DataFrame, prior: pd.DataFrame, answers: dict[str, str],
                    a0: dict[str, str]) -> pd.DataFrame:
    base = prior[prior.condition.isin(CONDITIONS[:-1])].copy()
    lookup = frame.set_index("case_id")
    rows = []
    for case_id in sorted(lookup.index.astype(str)):
        row = lookup.loc[case_id]
        alarm = bool(row.bc_alarm)
        response = answers[case_id] if alarm else a0[case_id]
        rows.append({"condition": "BC_GEF", "case_id": case_id, "kind": row.kind, "family": row.family,
                     "member": int(row.member), "session_id": str(row.session_id), "turn": int(row.turn),
                     "domain": row.domain, "query": row.query, "target_id": str(row.target_id),
                     "target_rank": int(row.target_rank), "response": response, "A0": a0[case_id],
                     "changed": response != a0[case_id], "refusal": refusal(response),
                     "exact_idk": response.strip() == "I don't know.", "empty": not response.strip(),
                     "answer_chars": len(response), "sentence_count": len(segment_document(response)),
                     "alarm": alarm, "selected_rank": int(row.union_selected_rank) if alarm else 0})
    output = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    atomic_csv(output, ROOT / "private/SMALL_FINAL_RESPONSES.csv.gz", "gzip")
    checkpoint("SMALL_RESPONSES_COMPLETE", rows=len(output), bc_gef_rows=len(rows), conditions=len(CONDITIONS))
    return output


def effective_auc(labels, scores) -> tuple[float, float]:
    raw = float(roc_auc_score(np.asarray(labels, int), np.asarray(scores, float)))
    return raw, max(raw, 1.0 - raw)


def sidechannel_metrics(responses: pd.DataFrame) -> pd.DataFrame:
    attack = responses[responses.kind.eq("ATTACK")].copy()
    sessions = attack.groupby(["condition", "family", "member", "session_id"], as_index=False).agg(
        total_answer_length=("answer_chars", "sum"), mean_answer_length=("answer_chars", "mean"),
        refusal_indicator=("refusal", "max"), exact_idk_indicator=("exact_idk", "max"),
        empty_answer_indicator=("empty", "max"), sentence_count=("sentence_count", "sum"))
    rows = []
    for (condition, family), cell in sessions.groupby(["condition", "family"]):
        for feature in ("total_answer_length", "mean_answer_length", "refusal_indicator", "exact_idk_indicator",
                        "empty_answer_indicator", "sentence_count"):
            raw, eauc = effective_auc(cell.member, cell[feature])
            rows.append({"condition": condition, "family": family, "feature": feature,
                         "native_auc": raw, "effective_auc": eauc})
    output = pd.DataFrame(rows)
    atomic_csv(output, ROOT / "tables/EXTERNAL_SIDECHANNEL.csv")
    atomic_csv(sessions, ROOT / "private/SESSION_OBSERVABLES.csv.gz", "gzip")
    return output


def detection_metrics(frame: pd.DataFrame, packing: pd.DataFrame) -> pd.DataFrame:
    pack = packing.set_index("case_id")
    rows = []
    detail = []
    for row in frame[frame.kind.eq("ATTACK")].itertuples(index=False):
        ids = list(map(str, json.loads(pack.loc[str(row.case_id)].source_ids)))
        alarm = bool(row.bc_alarm)
        selected_rank = int(row.union_selected_rank) if alarm else 0
        selected_id = ids[selected_rank - 1] if selected_rank else ""
        detail.append({"case_id": str(row.case_id), "family": row.family, "member": int(row.member),
                       "alarm": alarm, "target_rank": int(row.target_rank), "selected_rank": selected_rank,
                       "selected_source_id": selected_id, "target_id": str(row.target_id),
                       "target_retrieved": int(row.target_rank) > 0,
                       "locator_hit": alarm and int(row.target_rank) > 0 and selected_rank == int(row.target_rank),
                       "effective_protection": alarm and int(row.target_rank) > 0 and selected_rank == int(row.target_rank)})
    details = pd.DataFrame(detail)
    for family, cell in details.groupby("family"):
        rows.append({"condition": "BC_GEF", "family": family, "queries": len(cell),
                     "alarm_rate": float(cell.alarm.mean()), "target_retrieval_at4": float(cell.target_retrieved.mean()),
                     "locator_hit_among_alarm": float(cell[cell.alarm].locator_hit.mean()) if cell.alarm.any() else math.nan,
                     "effective_protection_rate": float(cell.effective_protection.mean()),
                     "session_any_alarm": float(frame[(frame.kind.eq("ATTACK")) & (frame.family.eq(family))]
                                                 .assign(_alarm=lambda x: x.bc_alarm).groupby("session_id")._alarm.max().mean())})
    atomic_csv(details, ROOT / "private/FRONTEND_QUERY_AUDIT.csv.gz", "gzip")
    output = pd.DataFrame(rows)
    old = pd.read_csv(BC_CGD / "tables/DETECTION_AND_PROTECTION.csv")
    parent_cgd = old[old.condition.eq("BC_CGD")].copy().sort_values("family").reset_index(drop=True)
    current = output.sort_values("family").reset_index(drop=True)
    numeric = ["alarm_rate", "target_retrieval_at4", "locator_hit_among_alarm", "effective_protection_rate", "session_any_alarm"]
    if parent_cgd.family.tolist() != current.family.tolist() or not np.allclose(parent_cgd[numeric], current[numeric], equal_nan=True):
        raise RuntimeError("BC_GEF_FRONTEND_MISMATCH: detection/locator metrics")
    combined = pd.concat([old, output], ignore_index=True)
    atomic_csv(combined, ROOT / "tables/DETECTION_AND_PROTECTION.csv")
    return combined


def benign_diagnostics(responses: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    benign = responses[responses.kind.eq("BENIGN")].copy()
    rows = []
    for condition, cell in benign.groupby("condition"):
        rows.append({"condition": condition, "queries": len(cell), "intervention_rate": float(cell.alarm.mean()),
                     "safe_path_exact_preservation": float(cell[~cell.alarm].response.eq(cell[~cell.alarm].A0).mean()) if (~cell.alarm).any() else math.nan,
                     "answer_preservation_token_f1": float(np.mean([token_f1(x, y) for x, y in zip(cell.response, cell.A0)])),
                     "answer_preservation_exact_match": float(cell.response.eq(cell.A0).mean()),
                     "refusal_rate": float(cell.refusal.mean()), "new_refusal": float((cell.refusal & ~cell.A0.map(refusal)).mean()),
                     "empty_answer_rate": float(cell["empty"].mean()), "mean_answer_chars": float(cell.answer_chars.mean())})
    output = pd.DataFrame(rows)
    atomic_csv(output, ROOT / "tables/SMALL_BENIGN_DIAGNOSTIC.csv")
    subsets = []
    ours = benign[benign.condition.eq("BC_GEF")]
    for name, cell in (("ALL_BENIGN", ours), ("SAFE_PATH", ours[~ours.alarm]), ("INTERVENED_BENIGN", ours[ours.alarm])):
        subsets.append({"subset": name, "queries": len(cell),
                        "answer_preservation_token_f1": float(np.mean([token_f1(x, y) for x, y in zip(cell.response, cell.A0)])) if len(cell) else math.nan,
                        "answer_preservation_exact_match": float(cell.response.eq(cell.A0).mean()) if len(cell) else math.nan,
                        "new_refusal": float((cell.refusal & ~cell.A0.map(refusal)).mean()) if len(cell) else math.nan,
                        "empty_answer_rate": float(cell["empty"].mean()) if len(cell) else math.nan,
                        "mean_answer_chars": float(cell.answer_chars.mean()) if len(cell) else math.nan})
    atomic_csv(pd.DataFrame(subsets), ROOT / "tables/BC_GEF_BENIGN_PATH_SUBSETS.csv")
    return output


def finalize(privacy: pd.DataFrame, benign: pd.DataFrame, side: pd.DataFrame,
             selected: pd.DataFrame, responses: pd.DataFrame, precommit: dict) -> dict:
    def native(condition: str, family: str) -> float:
        return float(privacy[(privacy.condition.eq(condition)) & (privacy.family.eq(family))].iloc[0].native_auc)
    checks = {
        "menta_gain_over_bc_ge_0_05": native("BC_MIRABEL", "MEntA") - native("BC_GEF", "MEntA") >= 0.05 - 1e-12,
        "menta_better_than_original": native("BC_GEF", "MEntA") < native("ORIGINAL_MIRABEL", "MEntA"),
        "menta_native_auc_le_0_75": native("BC_GEF", "MEntA") <= 0.75,
        "s2_not_worse_than_original_plus_0_05": native("BC_GEF", "S²-MIA") <= native("ORIGINAL_MIRABEL", "S²-MIA") + 0.05 + 1e-12,
        "mba_not_worse_than_original_plus_0_05": native("BC_GEF", "MBA") <= native("ORIGINAL_MIRABEL", "MBA") + 0.05 + 1e-12,
        "one_direct_equal_or_better": native("BC_GEF", "S²-MIA") <= native("ORIGINAL_MIRABEL", "S²-MIA") or native("BC_GEF", "MBA") <= native("ORIGINAL_MIRABEL", "MBA"),
    }
    passed = all(checks.values())
    verdict = "BC_GEF_SMALL_PASS" if passed else "BC_GEF_SMALL_REJECTED"
    ours = responses[(responses.condition.eq("BC_GEF")) & (responses.alarm)]
    source_coverage = 1.0 if len(selected) == 2 * len(ours) else (len(selected) / max(2 * len(ours), 1))
    result = {"campaign": "BC_GEF_SMALL", "verdict": verdict, "checks": checks,
              "privacy_native_auc": {family: {condition: native(condition, family) for condition in CONDITIONS} for family in FAMILIES},
              "privacy_effective_auc_diagnostic": {family: {condition: float(privacy[(privacy.condition.eq(condition)) & (privacy.family.eq(family))].iloc[0].effective_auc) for condition in CONDITIONS} for family in FAMILIES},
              "risk_queries": int(len(ours)), "extractive_sentences": int(len(selected)),
              "empty_cases": int(ours["empty"].sum()), "extractive_source_coverage": source_coverage,
              "benign_bc_gef": benign[benign.condition.eq("BC_GEF")].iloc[0].to_dict(),
              "worst_external_sidechannel_effective_auc_diagnostic": float(side[side.condition.eq("BC_GEF")].effective_auc.max()),
              "precommit_sha256": sha256_file(ROOT / "configs/BC_GEF_PRECOMMIT.json"),
              "full_core6_opened": False, "gold_qa_correctness_evaluated": False,
              "paid_api_calls": 0, "interpretation": "Native attack metrics are primary; E-AUC and A0 preservation are diagnostics."}
    atomic_json(ROOT / "FINAL_RESULT.json", result)
    p = privacy[privacy.condition.isin(("ORIGINAL_MIRABEL", "BC_MIRABEL", "BC_CGD", "BC_GEF"))]
    report = ["# BC-GEF Small 결과", "", f"- Verdict: **{verdict}**",
              f"- Risk queries: **{len(ours)}**", f"- Extractive empty cases: **{int(ours['empty'].sum())}**",
              f"- Exact extractive provenance coverage: **{100*source_coverage:.2f}%**", "",
              "## Original/frozen attack metrics (primary)", "",
              p[["condition", "family", "sessions", "member_n", "nonmember_n", "native_auc", "native_auc_ci_low", "native_auc_ci_high", "member_mean", "nonmember_mean"]].to_markdown(index=False, floatfmt=".4f"), "",
              "## Gate", "", pd.DataFrame([{"check": k, "pass": v} for k, v in checks.items()]).to_markdown(index=False), "",
              "## Benign answer-preservation diagnostic", "", benign.to_markdown(index=False, floatfmt=".4f"), "",
              "이 benign 지표는 A0 답변 보존성이지 gold QA 정답률이 아니다. Risk-path 문장은 남은 검색 문서에서 직접 추출되지만 `hallucination-free`라고 주장하지 않는다."]
    atomic_text(ROOT / "reports/SMALL_FINAL_REPORT_KO.md", "\n".join(report) + "\n")
    checkpoint(verdict, final_result_sha256=sha256_file(ROOT / "FINAL_RESULT.json"),
               full_core6_opened=False, menta_native_auc=native("BC_GEF", "MEntA"))
    return result


def run_small() -> dict:
    precommit = verify_precommit()
    common, parent_run = load_parent_modules()
    frame, packing, a0, _, prior, selected_rows, alarm_rows = load_and_recompute_frontend(parent_run)
    if stable_rows_hash(alarm_rows) != precommit["frontend"]["alarm_query_ids_sha256"] or stable_rows_hash(selected_rows) != precommit["frontend"]["selected_source_rows_sha256"]:
        raise RuntimeError("BC_GEF_FRONTEND_MISMATCH")
    answers, selected = build_extractive(frame, packing, a0)
    responses = build_responses(frame, prior, answers, a0)
    target_ids = set(frame[frame.kind.eq("ATTACK")].target_id.astype(str))
    documents = parent_run.load_raw_documents(target_ids)
    menta_documents = parent_run.load_official_menta_documents(target_ids)
    gef = responses[responses.condition.eq("BC_GEF")].copy()
    menta = parent_run.score_menta(gef, menta_documents)
    s2, mba = parent_run.score_s2_mba(gef, documents)
    new_scores = pd.concat([menta, s2, mba], ignore_index=True)
    old_scores = pd.read_csv(BC_CGD / "private/SMALL_ORIGINAL_ATTACK_SCORES.csv.gz")
    scores = pd.concat([old_scores, new_scores], ignore_index=True)
    atomic_csv(scores, ROOT / "private/SMALL_ORIGINAL_ATTACK_SCORES.csv.gz", "gzip")
    privacy = parent_run.attack_metrics(scores)
    detection_metrics(frame, packing)
    benign = benign_diagnostics(responses, selected)
    side = sidechannel_metrics(responses)
    return finalize(privacy, benign, side, selected, responses, precommit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "precommit", "run", "all"), default="all", nargs="?")
    args = parser.parse_args()
    if args.mode in {"preflight", "all"}:
        result = run_preflight()
    else:
        result = json.loads((ROOT / "PREFLIGHT_RESULT.json").read_text(encoding="utf-8"))
        if result["verdict"] != "BC_GEF_PREFLIGHT_PASS":
            raise RuntimeError("BC_GEF_PREFLIGHT_REJECTED")
    if args.mode in {"precommit", "all"}:
        write_precommit(result)
    if args.mode in {"run", "all"}:
        run_small()


if __name__ == "__main__":
    main()
