#!/usr/bin/env python3
"""Stage 3: frozen Stateless QLL Source Hide transfer from Qwen to Llama.

The MPNet retrieval universe, Top-4 policy, strict threshold, Qwen-tokenized
water-fill views, hide count, and all scorers are frozen.  Only the model used
for source-conditioned query likelihood and final answer generation changes to
the local Llama-3.2-3B-Instruct snapshot.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import numpy as np
import pandas as pd


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
CAMPAIGN = PROJECT / "final_validation_stateless_qll_source_hide_20260901"
ROOT = CAMPAIGN / "stage_03_llama_transfer"
EXP212 = PROJECT / "exp212_retriever_transfer_stateless_qll_20260831"
EXP212_CODE = EXP212 / "code/run_exp212.py"
EXP179_CODE = PROJECT / "exp179_cross_family_qwen_source_influence_20260828/code/run_exp179.py"
EXP210 = PROJECT / "exp210_stateless_qll_source_hide_recovery_20260831"
LLAMA = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/006f5dcd1393c3add266de40994ba96225e9689d")
QWEN_TOKENIZER = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
PRECOMMIT = ROOT / "configs/PRECOMMIT.json"
STRICT_THRESHOLD = 0.5300846414247485
PRIORITY_RELEASE = ROOT / "checkpoints/PRIORITY_ANALYSES_DONE.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module("final_validation_exp212_base", EXP212_CODE)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    suffix = ".csv.gz" if str(path).endswith(".gz") else ".csv"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"campaign": "Final Validation Campaign — Stateless QLL Source Hide",
               "stage": stage, "updated_utc": now(), "pid": os.getpid(), **details}
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    lines = ["# Stage 3 — Llama transfer", "", f"- Stage: **{stage}**",
             f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{os.getpid()}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class LlamaGeneratorAdapter:
    GENERATION_CONFIG = {"batch_size": 16}

    @staticmethod
    def make_task(*, task_type: str, row_id: str, prompt: str, system_prompt: str, max_new_tokens: int):
        payload = {"task_type": str(task_type), "row_id": str(row_id), "prompt": str(prompt),
                   "system_prompt": str(system_prompt), "max_new_tokens": int(max_new_tokens),
                   "generator_revision": LLAMA.name, "max_input_tokens": 3072,
                   "temperature": 0.0, "seed": 42}
        stable = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        payload["task_key"] = hashlib.sha256(stable.encode()).hexdigest()
        return payload

    @staticmethod
    def run_generation(tasks, store_path, _unused, progress_stage):
        store_path = Path(store_path).with_name("MPNET_LLAMA_RESPONSES.sqlite3")
        store_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(store_path, timeout=60)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE IF NOT EXISTS responses (task_key TEXT PRIMARY KEY, row_id TEXT NOT NULL, response TEXT NOT NULL, created_utc TEXT NOT NULL)")
        connection.commit()
        existing = {row[0]: (row[1], row[2]) for row in connection.execute("SELECT task_key,row_id,response FROM responses")}
        pending = [task for task in tasks if task["task_key"] not in existing]
        cached = len(tasks) - len(pending)
        checkpoint(progress_stage + "_STARTED", total=len(tasks), exact_cache_reused=cached,
                   pending=len(pending), generator="Llama-3.2-3B-Instruct")
        if pending:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.manual_seed(42)
            tokenizer = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                LLAMA, local_files_only=True, dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map="cuda").eval()
            groups: dict[int, list[dict]] = defaultdict(list)
            for task in pending:
                groups[int(task["max_new_tokens"])].append(task)
            generated = 0
            started = time.perf_counter()
            for maximum, group in sorted(groups.items()):
                for offset in range(0, len(group), int(LlamaGeneratorAdapter.GENERATION_CONFIG["batch_size"])):
                    batch = group[offset:offset + int(LlamaGeneratorAdapter.GENERATION_CONFIG["batch_size"])]
                    messages = [[{"role": "system", "content": item["system_prompt"]},
                                 {"role": "user", "content": item["prompt"]}] for item in batch]
                    rendered = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
                                for item in messages]
                    encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True,
                                        max_length=3072).to(model.device)
                    with torch.inference_mode():
                        output = model.generate(**encoded, do_sample=False, num_beams=1, temperature=None,
                                                max_new_tokens=maximum, pad_token_id=tokenizer.pad_token_id)
                    decoded = tokenizer.batch_decode(output[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
                    rows = [(task["task_key"], task["row_id"], answer.strip() or "I don't know.", now())
                            for task, answer in zip(batch, decoded)]
                    connection.executemany("INSERT OR REPLACE INTO responses VALUES (?,?,?,?)", rows)
                    connection.commit()
                    existing.update({key: (row_id, answer) for key, row_id, answer, _ in rows})
                    generated += len(batch)
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    rate = generated / elapsed
                    checkpoint(progress_stage, total=len(tasks), exact_cache_reused=cached,
                               newly_generated=generated, pending=len(pending) - generated,
                               rows_per_second=rate, eta_seconds=(len(pending)-generated)/rate,
                               max_new_tokens=maximum, device="cuda")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        result = {}
        for task in tasks:
            row_id, response = existing[task["task_key"]]
            if row_id != task["row_id"]:
                raise RuntimeError("Llama response-cache row identity drift")
            result[row_id] = response
        connection.close()
        return result


def qll_scores_llama(retriever: str, retrieval: pd.DataFrame, corpora) -> pd.DataFrame:
    output_path = ROOT / f"private/{retriever}_LLAMA_QLL_CASES.private.csv.gz"
    llama_threshold_path = ROOT / f"audits/{retriever}_LLAMA_THRESHOLD_FREEZE.json"
    base_threshold_path = ROOT / f"audits/{retriever}_THRESHOLD_FREEZE.json"
    if output_path.exists():
        # The frozen Exp212 evaluator reads the generic threshold artifact name.
        # Preserve the explicit Llama artifact and materialize an identical
        # compatibility copy; this does not recompute or change the threshold.
        if not base_threshold_path.exists():
            if not llama_threshold_path.exists():
                raise FileNotFoundError(llama_threshold_path)
            atomic_json(base_threshold_path, json.loads(llama_threshold_path.read_text()))
        return pd.read_csv(output_path, keep_default_na=False, low_memory=False,
                           dtype={"row_id": str, "case_id": str, "session_id": str, "target_document_id": str})
    qll = load_module("final_validation_llama_qll", EXP179_CODE)
    cell = ROOT / retriever.lower()
    cell.mkdir(exist_ok=True)
    qll.ROOT = cell
    qll.DB = ROOT / f"private/{retriever}_LLAMA_QLL.sqlite3"
    qll.PRIOR_DB = ROOT / "private/NO_PRIOR_LLAMA_QLL.sqlite3"
    qll.QWEN = LLAMA
    qll.checkpoint = lambda stage, **details: checkpoint(f"{retriever}_LLAMA_QLL_{stage}", **details)
    documents = {}
    for dataset, (ids, texts) in corpora.items():
        documents.update({(dataset, str(document_id)): text for document_id, text in zip(ids, texts)})
    rows = []
    for item in retrieval.itertuples(index=False):
        query_hash = base.sha256_text(item.query)
        source_ids = list(map(str, json.loads(item.retrieved_document_ids)))
        for rank, source_id in enumerate(source_ids, 1):
            rows.append({"task_id": base.sha256_text(f"LLAMA\0{retriever}\0{item.case_id}\0{source_id}\0{query_hash}"),
                         "case_id": str(item.case_id), "attack_family": str(item.attack_family),
                         "cohort": str(item.kind), "dataset": str(item.dataset), "member": int(item.member),
                         "query": str(item.query), "query_sha256": query_hash, "source_id": source_id,
                         "source_rank": rank, "target_document_id": str(item.target_document_id),
                         "target_rank": int(item.target_rank), "is_labeled_source": source_id == str(item.target_document_id)})
    tasks = pd.DataFrame(rows)
    if len(tasks) != 4 * len(retrieval) or tasks.task_id.nunique() != len(tasks):
        raise RuntimeError("Llama QLL task identity failure")
    started = time.perf_counter()
    scored = qll.score_tasks(tasks, documents)
    source_scores, case_scores = qll.case_scores(scored)
    atomic_csv(source_scores, ROOT / f"private/{retriever}_LLAMA_QLL_SOURCE_SCORES.private.csv.gz", "gzip")
    merged = retrieval.merge(case_scores[["case_id", "qll_top1_source", "qll_top2_source", "margin",
                                          "dominance", "entropy", "labeled_source_qll_rank"]],
                             on="case_id", validate="one_to_one")
    normal = merged[merged.kind.eq("BENIGN")].dominance.to_numpy(float)
    protocol = float(np.quantile(normal, .95, method="higher"))
    threshold = {"retriever": retriever, "qll_generator": str(LLAMA),
                 "strict_numeric_threshold": STRICT_THRESHOLD,
                 "benign_only_protocol_threshold": protocol, "quantile": .95, "method": "higher",
                 "comparison": "strict greater than", "benign_examples": len(normal), "attack_examples": 0,
                 "strict_benign_exceed_rate": float(np.mean(normal > STRICT_THRESHOLD)),
                 "protocol_benign_exceed_rate": float(np.mean(normal > protocol)),
                 "frozen_before_attack_response_generation": True}
    atomic_json(llama_threshold_path, threshold)
    atomic_json(base_threshold_path, threshold)
    atomic_csv(merged, output_path, "gzip")
    atomic_json(ROOT / f"efficiency/{retriever}_LLAMA_QLL.json",
                {"pairs": len(tasks), "seconds": time.perf_counter()-started,
                 "pairs_per_second": len(tasks)/max(time.perf_counter()-started, 1e-9)})
    checkpoint(f"{retriever}_LLAMA_QLL_COMPLETE", cases=len(merged), pairs=len(tasks),
               protocol_threshold=protocol, attack_calibration_examples=0)
    return merged


def native_scores_family_safe(retriever, condition, answers, source, sessions, families, cell_root):
    """Run the frozen native scorer with a cache identity unique to the family set.

    Exp212 reused ``private/NATIVE_SCORES.private.csv.gz`` for Stage A and Stage B.
    Consequently Stage B could return the Stage-A DCMI/S² cache instead of
    evaluating MEntA/RAG-MIA/MBA.  This changes only artifact identity; attack
    queries, answers, scorer implementation, thresholds, and defense remain frozen.
    """
    family_tag = "__".join(str(value).replace("²", "2").replace("-", "_") for value in families)
    evaluator = load_module(f"llama_native_{retriever}_{condition}_{family_tag}", base.EXP211_CODE)
    evaluator.ROOT = cell_root
    evaluator.DATASET = "FiQA-2018"
    evaluator.MPNET = MPNET
    evaluator.checkpoint = lambda stage, **details: checkpoint(
        f"{retriever}_{condition}_{family_tag}_{stage}", **details)
    score_path = cell_root / "private" / f"NATIVE_SCORES_{family_tag}.private.csv.gz"
    result = evaluator.native_scores(answers, source, sessions, families, score_path)
    observed = set(result.attack_family.astype(str).unique())
    expected = set(map(str, families))
    if observed != expected:
        raise RuntimeError(f"native family cache identity failure: {observed} != {expected}")
    return result


def generate_stage_a_prioritized(retriever: str, packing: pd.DataFrame) -> pd.DataFrame:
    """Finish regular Llama answers, yield the GPU, then run BudgetLeak.

    The pause is orchestration-only.  It does not alter prompts, decoding,
    cohorts, budgets, thresholds, or response-cache identities.
    """
    final_path = ROOT / f"private/{retriever}_STAGE_A_RESPONSES.private.csv.gz"
    if final_path.exists():
        return pd.read_csv(final_path, keep_default_na=False, low_memory=False,
                           dtype={"row_id": str, "case_id": str, "session_id": str,
                                  "target_document_id": str})
    regular = packing[((packing.kind.eq("BENIGN"))) |
                      ((packing.attack_family.isin(base.STAGE_A_NATIVE + ("RAGLeak",))))].copy()
    budget = packing[packing.attack_family.eq("BudgetLeak-Z") &
                     packing.condition.ne("NO_DEFENSE")].copy()
    regular_path = ROOT / f"private/{retriever}_STAGE_A_REGULAR.private.csv.gz"
    regular_answers = base.generate_rows(retriever, regular, regular_path, "STAGE_A_REGULAR")
    checkpoint(f"{retriever}_PRIORITY_BOUNDARY_WAIT", regular_answers=len(regular_answers),
               budget_generation_started=False,
               waiting_for=str(PRIORITY_RELEASE))
    while not PRIORITY_RELEASE.exists():
        time.sleep(20)
        checkpoint(f"{retriever}_PRIORITY_BOUNDARY_WAIT", regular_answers=len(regular_answers),
                   budget_generation_started=False,
                   waiting_for=str(PRIORITY_RELEASE))
    checkpoint(f"{retriever}_PRIORITY_BOUNDARY_RELEASED", regular_answers=len(regular_answers),
               budget_generation_started=True, release_sha256=sha256_file(PRIORITY_RELEASE))
    budget_answers = base.generate_rows(
        retriever, budget, ROOT / f"private/{retriever}_STAGE_A_BUDGET.private.csv.gz",
        "STAGE_A_BUDGET", True)
    output = pd.concat([regular_answers, budget_answers], ignore_index=True)
    atomic_csv(output, final_path, "gzip")
    return output


def preflight() -> pd.DataFrame:
    required = {"campaign_precommit": PRECOMMIT, "frozen_candidate": EXP210 / "models/STATELESS_QLL_SOURCE_HIDE_FROZEN.json",
                "exp212_code": EXP212_CODE, "qll_code": EXP179_CODE, "llama_config": LLAMA / "config.json",
                "qwen_view_tokenizer": QWEN_TOKENIZER / "tokenizer_config.json", "mpnet_config": MPNET / "config.json"}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing frozen Llama-transfer input: {missing}")
    config = json.loads(PRECOMMIT.read_text())
    if config.get("candidate_changes_allowed") or config.get("attack_examples_for_threshold") != 0:
        raise RuntimeError("Llama-transfer precommit contract failure")
    hash_keys = {"frozen_candidate": "candidate_model_json", "exp212_code": "exp212_base_code",
                 "qll_code": "exp179_qll_code", "llama_config": "llama_config",
                 "qwen_view_tokenizer": "qwen_tokenizer_config", "mpnet_config": "mpnet_config"}
    for input_key, hash_key in hash_keys.items():
        actual = sha256_file(required[input_key])
        if actual != config["frozen_hashes"][hash_key]:
            raise RuntimeError(f"frozen Llama-transfer hash drift: {input_key}")
    rows = pd.DataFrame([{"key": key, "path": str(path), "bytes": path.stat().st_size,
                          "sha256_before": sha256_file(path), "access": "READ_ONLY"}
                         for key, path in required.items()])
    atomic_csv(rows, ROOT / "provenance/FROZEN_INPUTS.csv")
    checkpoint("PREFLIGHT_COMPLETE", frozen_inputs=len(rows), retriever="MPNet", qll="Llama", generator="Llama",
               view_tokenizer="frozen Qwen tokenizer", model_changed=False)
    return rows


def postrun(before: pd.DataFrame) -> None:
    output = before.copy()
    output["sha256_after"] = [sha256_file(Path(path)) for path in output.path]
    output["unchanged"] = output.sha256_before.eq(output.sha256_after)
    atomic_csv(output, ROOT / "provenance/FROZEN_INPUTS_POSTRUN.csv")
    if not output.unchanged.all():
        raise RuntimeError("frozen Llama-transfer input changed")


def report(result: dict) -> None:
    lines = ["# Stage 3 — MPNet + Llama 전이 결과", "", f"- Verdict: **{result['verdict']}**",
             f"- Transfer level: **{result['transfer_level']}**", "- Retriever: **MPNet (frozen)**",
             "- QLL / generator: **Llama-3.2-3B-Instruct**", "- View allocation tokenizer: **frozen Qwen tokenizer**",
             "- Attack examples used for threshold: **0**", ""]
    for condition in ("STRICT", "PROTOCOL"):
        cell = result["conditions"][condition]
        privacy = {row["Attack"]: row for row in cell["full_privacy"]}
        benign = cell["screen"]["benign"]
        budget = cell["screen"]["budgetleak"]
        ranks = budget.get("rank_effective_auc", {})
        lines.extend([f"## {condition}", "", f"- Final pass: **{cell['final_pass']}**",
                      f"- DCMI / S²-MIA / RAGLeak: **{privacy.get('DCMI',{}).get('E-AUC',float('nan')):.4f} / {privacy.get('S²-MIA',{}).get('E-AUC',float('nan')):.4f} / {privacy.get('RAGLeak',{}).get('E-AUC',float('nan')):.4f}**",
                      f"- BudgetLeak overall: **{privacy.get('BudgetLeak-Z',{}).get('E-AUC',float('nan')):.4f}**",
                      f"- BudgetLeak rank 1–4: **{ranks.get('1',float('nan')):.4f} / {ranks.get('2',float('nan')):.4f} / {ranks.get('3',float('nan')):.4f} / {ranks.get('4',float('nan')):.4f}**",
                      f"- MEntA / RAG-MIA / MBA: **{privacy.get('MEntA',{}).get('E-AUC',float('nan')):.4f} / {privacy.get('RAG-MIA',{}).get('E-AUC',float('nan')):.4f} / {privacy.get('MBA',{}).get('E-AUC',float('nan')):.4f}**",
                      f"- Benign hide / Token-F1 / new refusal: **{100*benign['hide_rate']:.2f}% / {benign['relative_token_f1']:.4f} / {100*benign['new_refusal_rate']:.2f}%**",
                      f"- FP-benign Token-F1 / semantic / new refusal: **{benign['hidden_token_f1']:.4f} / {benign['hidden_semantic_similarity']:.4f} / {100*benign['hidden_new_refusal_rate']:.2f}%**", ""])
    atomic_text(ROOT / "reports/STAGE_03_LLAMA_TRANSFER_KO.md", "\n".join(lines) + "\n")


def main() -> None:
    for name in ("code", "configs", "tests", "scripts", "logs", "checkpoints", "provenance", "audits",
                 "reports", "tables", "private", "efficiency", "mpnet"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    try:
        before = preflight()
        base.ROOT = ROOT
        base.PRECOMMIT = PRECOMMIT
        base.RETRIEVERS = {"MPNET": MPNET}
        base.QWEN = QWEN_TOKENIZER
        base.THRESHOLD = STRICT_THRESHOLD
        base.qll_scores = qll_scores_llama
        base.native_scores_cell = native_scores_family_safe
        base.generate_stage_a = generate_stage_a_prioritized
        base.generator_module = lambda: LlamaGeneratorAdapter
        base.checkpoint = checkpoint
        source, sessions, external = base.load_source_sessions_external()
        frame = base.build_query_frame(source, sessions, external)
        corpora = base.load_corpora(source, sessions)
        result = base.run_retriever("MPNET", frame, corpora, source, sessions, before)
        postrun(before)
        level = result["transfer_level"]
        verdict = ("LLAMA_STRICT_TRANSFER_PASS" if level == "STRICT" else
                   "LLAMA_CALIBRATION_PROTOCOL_TRANSFER_PASS" if level == "PROTOCOL" else
                   "LLAMA_TRANSFER_FAILED")
        final = {"verdict": verdict, "transfer_level": level, "passed": level != "FAILED",
                 "retriever": "MPNet", "qll_model": str(LLAMA), "generator": str(LLAMA),
                 "view_tokenizer": str(QWEN_TOKENIZER), "strict_threshold": STRICT_THRESHOLD,
                 "attack_examples_used_for_calibration": 0, "new_trainable_parameters": 0,
                 "session_state": 0, "result": result, "completed_utc": now()}
        atomic_json(ROOT / "FINAL_RESULT.json", final)
        report(result)
        checkpoint("COMPLETE", verdict=verdict, transfer_level=level, passed=final["passed"])
    except Exception as error:
        checkpoint("FAILED_EXCEPTION", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
