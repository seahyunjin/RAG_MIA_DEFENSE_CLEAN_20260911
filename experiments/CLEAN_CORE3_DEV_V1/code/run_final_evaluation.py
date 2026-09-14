#!/usr/bin/env python3
"""Paper-faithful scoring and final bottleneck report for CLEAN_CORE3_DEV_V1."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
CAMPAIGN = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
RECOVERY = ROOT / "experiments" / "CORE6_PROTOCOL_RECOVERY_V1"
INPUTS = CAMPAIGN / "inputs"
CACHE = CAMPAIGN / "cache"
RUNTIME = CAMPAIGN / "runtime"
TABLES = CAMPAIGN / "tables"
AUDITS = CAMPAIGN / "audits"
REPORTS = CAMPAIGN / "reports"
CONFIGS = CAMPAIGN / "configs"
CHECKPOINTS = CAMPAIGN / "checkpoints"
CLAIM_MODEL = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Babelscape--t5-base-summarization-claim-extractor/snapshots/94775fb1c8dc2c3ef1bfec413f9f961e6ba5a1c8")
NLI_MODEL = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--tasksource--deberta-base-long-nli/snapshots/04dcf11f844b07bc57015169fca2b7d6df8299d5")
QWEN_TOKENIZER = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
SEED = 20260911
BOOTSTRAP_ITERATIONS = 10000
CONDITIONS = ["NO_DEFENSE", "ORIGINAL_MIRABEL_TOP1_HIDE", "BC_MIRABEL_Q97_TOP1_HIDE"]
DISPLAY = {"NO_DEFENSE": "No Defense", "ORIGINAL_MIRABEL_TOP1_HIDE": "Original MIRABEL",
           "BC_MIRABEL_Q97_TOP1_HIDE": "BC-MIRABEL q97"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"campaign": "CLEAN_CORE3_DEV_V1", "stage": stage, "updated_utc": now(),
               "pid": os.getpid(), **details}
    write_json(CAMPAIGN / "HEARTBEAT.json", payload)
    write_json(CHECKPOINTS / f"{stage}.json", payload)
    lines = ["# CLEAN_CORE3_DEV_V1 STATUS", "", f"- Current stage: `{stage}`",
             f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{os.getpid()}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    (CAMPAIGN / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def auc_and_ci(labels: list[int], scores: list[float]) -> tuple[float, float, float]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if len(set(labels)) != 2:
        return math.nan, math.nan, math.nan
    observed = float(roc_auc_score(labels_array, scores_array))
    member = np.where(labels_array == 1)[0]
    nonmember = np.where(labels_array == 0)[0]
    rng = np.random.default_rng(SEED)
    values = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for index in range(BOOTSTRAP_ITERATIONS):
        selected = np.concatenate((rng.choice(member, len(member), replace=True),
                                   rng.choice(nonmember, len(nonmember), replace=True)))
        values[index] = roc_auc_score(labels_array[selected], scores_array[selected])
    return observed, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def token_f1(left: str, right: str) -> float:
    a, b = left.lower().split(), right.lower().split()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(b), common / len(a)
    return 2 * precision * recall / (precision + recall)


def is_refusal(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    phrases = ("i don't know", "i do not know", "cannot determine", "can't determine",
               "insufficient information", "not enough information")
    return any(value in normalized for value in phrases)


def load_inputs() -> tuple[list[dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    retrieval = read_jsonl(CACHE / "CLEAN_CORE3_RETRIEVAL_CACHE.jsonl")
    retrieval = [row for row in retrieval if row["cohort"] == "ATTACK" or
                 (row["split"] == "HOLDOUT" and row["bc_q97_alarm"])]
    retrieval_map = {row["query_id"]: row for row in retrieval}
    answers_path = RUNTIME / "CLEAN_CORE3_GENERATED_ANSWERS.jsonl"
    answers = read_jsonl(answers_path)
    # The generation tensor is padded with EOS rather than tokenizer.pad_token;
    # recompute this metadata from the decoded answer.  Text and scores are not
    # changed.  This also normalizes checkpoints produced before this fix.
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER, local_files_only=True)
    changed = False
    for row in answers:
        correct = len(qwen_tokenizer(row["answer"], add_special_tokens=False).input_ids)
        if int(row["answer_tokens"]) != correct:
            row["answer_tokens"] = correct
            changed = True
    if changed:
        temporary = answers_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in answers:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, answers_path)
        manifest_path = RUNTIME / "CLEAN_CORE3_GENERATION_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_sha256"] = sha_file(answers_path)
        manifest["answer_token_metadata_recomputed"] = True
        write_json(manifest_path, manifest)
    answer_map = {(row["query_id"], row["condition"]): row for row in answers}
    expected = {(row["query_id"], condition) for row in retrieval for condition in CONDITIONS}
    if set(answer_map) != expected:
        raise RuntimeError(f"generated-answer key mismatch actual={len(answer_map)} expected={len(expected)}")
    attack_inputs: dict[str, dict] = {}
    for filename in ("MENTA_ATTACK_QUERIES.jsonl", "MBA_ATTACK_QUERIES.jsonl", "RAG_MIA_ATTACK_QUERIES.jsonl"):
        for row in read_jsonl(INPUTS / filename):
            attack_inputs[row["query_id"]] = row
    targets = {row["document_id"]: row for row in csv.DictReader((INPUTS / "SHARED_TARGETS.csv").open(encoding="utf-8"))}
    return retrieval, retrieval_map, answer_map, {"attacks": attack_inputs, "targets": targets}


def score_discrete(answer_map: dict, bundle: dict) -> list[dict]:
    sys.path.insert(0, str(RECOVERY))
    mba = load_module("clean_core3_mba_scorer", RECOVERY / "protocols/mba/scorer.py")
    rag = load_module("clean_core3_ragmia_scorer", RECOVERY / "protocols/rag_mia/scorer.py")
    inputs = bundle["attacks"]
    rows: list[dict] = []
    for query_id, item in sorted(inputs.items()):
        if item["attack"] not in ("MBA", "RAG-MIA"):
            continue
        for condition in CONDITIONS:
            answer = answer_map[(query_id, condition)]["answer"]
            valid, score, error = True, math.nan, ""
            try:
                if item["attack"] == "MBA":
                    score = float(mba.reconstruction_accuracy(answer, item["mask_answers"]))
                else:
                    score = float(rag.black_box_membership_score(answer))
            except Exception as exception:
                valid, error = False, f"{type(exception).__name__}: {exception}"
            rows.append({"attack": item["attack"], "query_id": query_id,
                         "session_id": item["session_id"], "target_id": item["target_id"],
                         "membership": item["membership"], "condition": condition,
                         "answer": answer, "valid": valid, "native_score": score,
                         "parse_error": error})
    write_csv(TABLES / "DISCRETE_ATTACK_QUERY_SCORES.csv", rows)
    return rows


def score_menta(answer_map: dict, bundle: dict) -> tuple[list[dict], list[dict]]:
    compute = load_module("clean_core3_compute_entailment", ROOT / "code/menta_official/MEntA/compute_entailment.py")
    menta_scorer = load_module("clean_core3_menta_scorer", RECOVERY / "protocols/menta/scorer.py")
    attack_inputs = bundle["attacks"]
    target_rows = bundle["targets"]
    cases = []
    for query_id, item in sorted(attack_inputs.items()):
        if item["attack"] != "MEntA":
            continue
        for condition in CONDITIONS:
            cases.append({"key": f"{query_id}::{condition}", "query_id": query_id,
                          "session_id": item["session_id"], "target_id": item["target_id"],
                          "membership": item["membership"], "query_index": item["query_index"],
                          "condition": condition, "answer": answer_map[(query_id, condition)]["answer"]})

    checkpoint("MENTA_CLAIM_EXTRACTION_LOADING", answers=len(cases), model=str(CLAIM_MODEL))
    split_tokenizer = AutoTokenizer.from_pretrained(CLAIM_MODEL, local_files_only=True, use_fast=False)
    split_model = AutoModelForSeq2SeqLM.from_pretrained(CLAIM_MODEL, local_files_only=True).to("cuda").eval()
    claims = compute.split_into_atomic_claims_batch(
        [row["answer"] for row in cases], split_tokenizer, split_model, "cuda",
        batch_size=32, max_length=512, min_claim_length=20, min_words=5, show_progress=True,
    )
    for row, extracted in zip(cases, claims):
        row["claims"] = extracted
    write_json(RUNTIME / "MENTA_EXTRACTED_CLAIMS.json", {row["key"]: row["claims"] for row in cases})
    del split_model, split_tokenizer
    torch.cuda.empty_cache()

    checkpoint("MENTA_NLI_LOADING", answers=len(cases), model=str(NLI_MODEL))
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, local_files_only=True, use_fast=False)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, local_files_only=True).to("cuda").eval()
    units_by_target = {}
    for target_id, row in target_rows.items():
        title = row["title"]
        source_text = row["source_text"]
        prefix = title + "\n"
        body = source_text[len(prefix):] if source_text.startswith(prefix) else source_text
        units_by_target[target_id] = compute.split_text(f"title: {title}; text: {body}", min_length=10)

    metadata, premises, hypotheses = [], [], []
    for case_index, row in enumerate(cases):
        units = units_by_target[row["target_id"]]
        for claim_index, claim in enumerate(row["claims"]):
            pair_index = len(metadata)
            metadata.append((case_index, claim_index, claim, len(units)))
            for unit in units:
                premises.append(unit); hypotheses.append(claim)
    checkpoint("MENTA_DOCUMENT_NLI_STARTED", nli_pairs=len(premises), claims=len(metadata))
    results = compute.check_entailment_batch(
        premises, hypotheses, nli_tokenizer, nli_model, "cuda", batch_size=128,
        max_length=2048, desc="CLEAN CORE3 MEntA document entailment",
    ) if premises else []
    best_by_claim = {}
    cursor = 0
    for pair_index, (case_index, claim_index, claim, unit_count) in enumerate(metadata):
        chunk = results[cursor:cursor + unit_count]
        cursor += unit_count
        best_by_claim[(case_index, claim_index)] = max(chunk, key=lambda value: value[0])

    idk_hypotheses = [
        "I don't have enough information to answer this question",
        "The provided text does not contain the answer to this question",
        "I cannot determine the answer from the given information",
        "This information is not mentioned in the provided text",
        "I am unable to answer based on the available information",
        "There is no information provided about this",
        "The text does not specify this information",
    ]
    idk_meta, idk_premises, idk_hypothesis_text = [], [], []
    for case_index, row in enumerate(cases):
        for claim_index, claim in enumerate(row["claims"]):
            for hypothesis in idk_hypotheses:
                idk_meta.append((case_index, claim_index))
                idk_premises.append(claim); idk_hypothesis_text.append(hypothesis)
    checkpoint("MENTA_IDK_NLI_STARTED", nli_pairs=len(idk_premises))
    idk_results = compute.check_entailment_batch(
        idk_premises, idk_hypothesis_text, nli_tokenizer, nli_model, "cuda",
        batch_size=128, max_length=2048, desc="CLEAN CORE3 MEntA IDK entailment",
    ) if idk_premises else []
    idk_max: dict[tuple[int, int], float] = defaultdict(float)
    for key, result in zip(idk_meta, idk_results):
        idk_max[key] = max(idk_max[key], float(result[0]))
    del nli_model, nli_tokenizer
    torch.cuda.empty_cache()

    query_rows = []
    for case_index, row in enumerate(cases):
        entailed = False
        best_ent = 0.0
        idk = False
        max_idk = 0.0
        details = []
        for claim_index, claim in enumerate(row["claims"]):
            ent, neutral, contradiction = best_by_claim[(case_index, claim_index)]
            hit = ent >= neutral and ent >= contradiction
            entailed = entailed or hit
            best_ent = max(best_ent, float(ent))
            claim_idk = idk_max[(case_index, claim_index)] > 0.5
            idk = idk or claim_idk
            max_idk = max(max_idk, idk_max[(case_index, claim_index)])
            details.append({"claim": claim, "entailment_probability": float(ent),
                            "neutral_probability": float(neutral),
                            "contradiction_probability": float(contradiction),
                            "entailed": bool(hit), "max_idk_entailment_probability": idk_max[(case_index, claim_index)],
                            "idk": bool(claim_idk)})
        # The preserved evaluator gives IDK precedence over entailment.
        effective_entailed = bool(entailed and not idk)
        contribution = -1 if idk else int(effective_entailed)
        query_rows.append({**{key: row[key] for key in ("query_id", "session_id", "target_id", "membership", "query_index", "condition", "answer")},
                           "claims": json.dumps(row["claims"], ensure_ascii=False),
                           "claim_count": len(row["claims"]), "entailed": effective_entailed,
                           "idk": bool(idk), "contribution": contribution,
                           "best_entailment_probability": best_ent,
                           "max_idk_entailment_probability": max_idk,
                           "claim_details": json.dumps(details, ensure_ascii=False)})
    write_csv(TABLES / "MENTA_QUERY_EVIDENCE.csv", query_rows)

    session_rows = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in query_rows:
        groups[(row["session_id"], row["condition"])].append(row)
    for (session_id, condition), group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["query_index"]))
        evidence = [{"entailed": bool(row["entailed"]), "idk": bool(row["idk"])} for row in group]
        score = float(menta_scorer.score_session(evidence))
        session_rows.append({"attack": "MEntA", "session_id": session_id,
                             "target_id": group[0]["target_id"], "membership": group[0]["membership"],
                             "condition": condition, "valid": True, "native_score": score,
                             "entailed_queries": sum(row["entailed"] for row in group),
                             "idk_queries": sum(row["idk"] for row in group)})
    write_csv(TABLES / "MENTA_SESSION_SCORES.csv", session_rows)
    checkpoint("MENTA_SCORING_COMPLETE", query_rows=len(query_rows), sessions=len(session_rows),
               document_nli_pairs=len(premises), idk_nli_pairs=len(idk_premises))
    return query_rows, session_rows


def summarize_privacy(discrete_rows: list[dict], menta_sessions: list[dict]) -> list[dict]:
    all_scores = discrete_rows + menta_sessions
    output = []
    for attack in ("MEntA", "MBA", "RAG-MIA"):
        for condition in CONDITIONS:
            rows = [row for row in all_scores if row["attack"] == attack and row["condition"] == condition]
            valid = [row for row in rows if row["valid"] and math.isfinite(float(row["native_score"]))]
            labels = [int(row["membership"] == "member") for row in valid]
            scores = [float(row["native_score"]) for row in valid]
            auc, low, high = auc_and_ci(labels, scores)
            member_scores = [score for label, score in zip(labels, scores) if label]
            nonmember_scores = [score for label, score in zip(labels, scores) if not label]
            record = {"attack": attack, "condition": condition, "condition_display": DISPLAY[condition],
                      "primary_metric": "native ROC-AUC", "total_n": len(rows), "valid_n": len(valid),
                      "invalid_n": len(rows)-len(valid), "member_n": len(member_scores),
                      "nonmember_n": len(nonmember_scores), "member_score_mean": np.mean(member_scores) if member_scores else math.nan,
                      "nonmember_score_mean": np.mean(nonmember_scores) if nonmember_scores else math.nan,
                      "native_roc_auc": auc, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                      "effective_auc_internal_only": max(auc, 1-auc) if math.isfinite(auc) else math.nan}
            if attack == "RAG-MIA" and valid:
                predictions = [int(value) for value in scores]
                record["accuracy"] = float(np.mean(np.asarray(predictions) == np.asarray(labels)))
                record["tpr"] = float(np.mean([score for label, score in zip(labels, predictions) if label]))
                record["fpr"] = float(np.mean([score for label, score in zip(labels, predictions) if not label]))
            output.append(record)
    write_csv(TABLES / "POST_GENERATION_PRIVACY.csv", output)
    return output


def detection_tables(retrieval: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    attack_rows = [row for row in retrieval if row["cohort"] == "ATTACK"]
    summary = []
    for attack in ("MEntA", "MBA", "RAG-MIA"):
        rows = [row for row in attack_rows if row["attack"] == attack]
        members = [row for row in rows if row["membership"] == "member"]
        for condition, alarm_field in (("ORIGINAL_MIRABEL_TOP1_HIDE", "original_alarm"),
                                       ("BC_MIRABEL_Q97_TOP1_HIDE", "bc_q97_alarm")):
            retrieved = [row for row in members if row["target_retrieved_at4"]]
            alarm_retrieved = [row for row in retrieved if row[alarm_field]]
            located = [row for row in members if row["selected_source_id"] == row["target_id"]]
            located_retrieved = [row for row in retrieved if row["selected_source_id"] == row["target_id"]]
            epo = [row for row in members if row[alarm_field] and row["target_retrieved_at4"] and row["selected_source_id"] == row["target_id"]]
            summary.append({"attack": attack, "condition": condition, "all_query_n": len(rows),
                            "member_query_n": len(members), "member_target_retrieval_at1": np.mean([r["target_retrieved_at1"] for r in members]),
                            "member_target_retrieval_at4": np.mean([r["target_retrieved_at4"] for r in members]),
                            "query_alarm_rate_all": np.mean([r[alarm_field] for r in rows]),
                            "query_alarm_rate_member": np.mean([r[alarm_field] for r in members]),
                            "alarm_on_retrieved_member": np.mean([r[alarm_field] for r in retrieved]) if retrieved else math.nan,
                            "locator_hit_at1_overall_member": len(located)/len(members) if members else math.nan,
                            "locator_hit_at1_given_target_retrieved": len(located_retrieved)/len(retrieved) if retrieved else math.nan,
                            "locator_hit_given_alarm_and_retrieved": np.mean([r["selected_source_id"] == r["target_id"] for r in alarm_retrieved]) if alarm_retrieved else math.nan,
                            "effective_protection_opportunity": len(epo)/len(members) if members else math.nan})
    write_csv(TABLES / "DETECT_LOCATE_PROTECT.csv", summary)

    session_rows = []
    menta = [row for row in attack_rows if row["attack"] == "MEntA"]
    for condition, alarm_field in (("ORIGINAL_MIRABEL_TOP1_HIDE", "original_alarm"),
                                   ("BC_MIRABEL_Q97_TOP1_HIDE", "bc_q97_alarm")):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in menta:
            groups[row["session_id"]].append(row)
        for session_id, rows in sorted(groups.items()):
            rows.sort(key=lambda row: row["query_index"])
            alarmed = [row for row in rows if row[alarm_field]]
            removed = [row for row in rows if row[alarm_field] and row["selected_source_id"] == row["target_id"]]
            session_rows.append({"condition": condition, "session_id": session_id,
                                 "membership": rows[0]["membership"], "alarmed_queries": len(alarmed),
                                 "target_removed_queries": len(removed), "any_alarm": bool(alarmed),
                                 "all_5_alarm": len(alarmed) == 5, "any_target_removed": bool(removed),
                                 "all_5_target_removed": len(removed) == 5,
                                 "first_alarm_query_index": alarmed[0]["query_index"] if alarmed else ""})
    write_csv(TABLES / "MENTA_SESSION_COVERAGE.csv", session_rows)

    aggregate = []
    for condition in ("ORIGINAL_MIRABEL_TOP1_HIDE", "BC_MIRABEL_Q97_TOP1_HIDE"):
        rows = [row for row in session_rows if row["condition"] == condition]
        for membership in ("ALL", "member", "nonmember"):
            subset = rows if membership == "ALL" else [row for row in rows if row["membership"] == membership]
            aggregate.append({"condition": condition, "membership": membership, "sessions": len(subset),
                              "mean_alarmed_queries_per_5": np.mean([r["alarmed_queries"] for r in subset]),
                              "mean_target_removed_queries_per_5": np.mean([r["target_removed_queries"] for r in subset]),
                              "any_alarm_session_rate": np.mean([r["any_alarm"] for r in subset]),
                              "all_5_alarm_session_rate": np.mean([r["all_5_alarm"] for r in subset]),
                              "any_target_removed_session_rate": np.mean([r["any_target_removed"] for r in subset]),
                              "all_5_target_removed_session_rate": np.mean([r["all_5_target_removed"] for r in subset])})
    write_csv(TABLES / "MENTA_SESSION_COVERAGE_SUMMARY.csv", aggregate)
    return summary, session_rows, aggregate


def false_positive_and_cost(retrieval: list[dict], answer_map: dict) -> tuple[list[dict], list[dict]]:
    holdout = [row for row in retrieval if row["cohort"] == "BENIGN"]
    false_positives = []
    for row in holdout:
        if not row["bc_q97_alarm"]:
            continue
        original = answer_map[(row["query_id"], "NO_DEFENSE")]["answer"]
        defended = answer_map[(row["query_id"], "BC_MIRABEL_Q97_TOP1_HIDE")]["answer"]
        false_positives.append({"query_id": row["query_id"], "domain": row["domain"],
                                "query": row["query"], "mirabel_margin": row["mirabel_margin"],
                                "selected_removed_document_id": row["selected_source_id"],
                                "original_answer": original, "defended_answer": defended,
                                "original_refusal": is_refusal(original), "defended_refusal": is_refusal(defended),
                                "new_refusal": is_refusal(defended) and not is_refusal(original),
                                "original_answer_words": len(original.split()), "defended_answer_words": len(defended.split()),
                                "answer_changed": original != defended,
                                "answer_preservation_token_f1_diagnostic": token_f1(original, defended),
                                "correctness_assessed": False})
    write_csv(TABLES / "BC_Q97_FALSE_POSITIVE_AUDIT.csv", false_positives)

    answers = list(answer_map.values())
    costs = []
    for condition in CONDITIONS:
        subset = [row for row in answers if row["condition"] == condition]
        generated = [row for row in subset if row["generated"]]
        latency = np.asarray([float(row["wall_seconds"]) for row in generated], dtype=float)
        costs.append({"condition": condition, "evaluation_query_rows": len(subset),
                      "physical_generation_count_in_campaign": len(generated),
                      "exact_cached_answer_reuse_count": len(subset)-len(generated),
                      "logical_deployment_generation_count": (len(subset) if condition == "NO_DEFENSE" else sum(int(row["detector_alarm"]) for row in subset)),
                      "logical_defense_regeneration_count": sum(int(row["detector_alarm"]) for row in subset) if condition != "NO_DEFENSE" else 0,
                      "mean_generation_seconds": float(np.mean(latency)) if len(latency) else 0.0,
                      "p50_generation_seconds": float(np.quantile(latency, .5)) if len(latency) else 0.0,
                      "p95_generation_seconds": float(np.quantile(latency, .95)) if len(latency) else 0.0,
                      "summed_gpu_generation_seconds": float(np.sum(latency))})
    write_csv(TABLES / "COST.csv", costs)
    return false_positives, costs


def query_answer_export(retrieval: list[dict], answer_map: dict) -> None:
    rows = []
    for item in retrieval:
        if item["cohort"] != "ATTACK":
            continue
        for condition in CONDITIONS:
            answer = answer_map[(item["query_id"], condition)]
            rows.append({"attack": item["attack"], "membership": item["membership"],
                         "target_id": item["target_id"], "session_id": item["session_id"],
                         "query_index": item["query_index"], "query_id": item["query_id"],
                         "query": item["query"], "condition": condition, "detector_alarm": answer["detector_alarm"],
                         "removed_document_id": answer["hidden_source_id"] or "", "generated_answer": answer["answer"]})
    write_csv(TABLES / "ATTACK_QUERIES_AND_GENERATED_ANSWERS.csv", rows)


def final_report(retrieval: list[dict], detection: list[dict], privacy: list[dict], false_positives: list[dict], costs: list[dict]) -> None:
    pmap = {(row["attack"], row["condition"]): row for row in privacy}
    dmap = {(row["attack"], row["condition"]): row for row in detection}
    main_rows, bottlenecks = [], []
    for attack in ("MEntA", "MBA", "RAG-MIA"):
        bc = dmap[(attack, "BC_MIRABEL_Q97_TOP1_HIDE")]
        orig = dmap[(attack, "ORIGINAL_MIRABEL_TOP1_HIDE")]
        main_rows.append({"attack": attack,
                          "member_retrieval_at4": bc["member_target_retrieval_at4"],
                          "original_alarm_rate_all": orig["query_alarm_rate_all"],
                          "bc_q97_alarm_rate_all": bc["query_alarm_rate_all"],
                          "bc_locator_hit_at1_member": bc["locator_hit_at1_overall_member"],
                          "bc_effective_protection_opportunity": bc["effective_protection_opportunity"],
                          "no_defense_native_auc": pmap[(attack, "NO_DEFENSE")]["native_roc_auc"],
                          "original_mirabel_native_auc": pmap[(attack, "ORIGINAL_MIRABEL_TOP1_HIDE")]["native_roc_auc"],
                          "bc_mirabel_native_auc": pmap[(attack, "BC_MIRABEL_Q97_TOP1_HIDE")]["native_roc_auc"],
                          "primary_metric": "native ROC-AUC"})
        no_auc = pmap[(attack, "NO_DEFENSE")]["native_roc_auc"]
        bc_auc = pmap[(attack, "BC_MIRABEL_Q97_TOP1_HIDE")]["native_roc_auc"]
        if math.isfinite(no_auc) and no_auc <= .55:
            label = "ALREADY_CONTROLLED"; reason = f"No-Defense native AUC={no_auc:.3f} <= 0.55"
        elif bc["member_target_retrieval_at4"] < .50:
            label = "RETRIEVAL_LIMITED"; reason = f"member Retrieval@4={bc['member_target_retrieval_at4']:.3f} < 0.50"
        elif bc["alarm_on_retrieved_member"] < .50:
            label = "DETECTION_LIMITED"; reason = f"BC alarm on retrieved members={bc['alarm_on_retrieved_member']:.3f} < 0.50"
        elif bc["locator_hit_given_alarm_and_retrieved"] < .50:
            label = "LOCATOR_LIMITED"; reason = f"locator hit among alarm+retrieved={bc['locator_hit_given_alarm_and_retrieved']:.3f} < 0.50"
        elif bc["effective_protection_opportunity"] >= .50 and math.isfinite(bc_auc) and bc_auc > .65:
            label = "PROTECTION_LIMITED"; reason = f"EPO={bc['effective_protection_opportunity']:.3f} but defended AUC={bc_auc:.3f} > 0.65"
        else:
            label = "MIXED"; reason = f"Retrieval@4={bc['member_target_retrieval_at4']:.3f}, alarm|retrieved={bc['alarm_on_retrieved_member']:.3f}, locator={bc['locator_hit_given_alarm_and_retrieved']:.3f}, AUC={bc_auc:.3f}"
        bottlenecks.append({"attack": attack, "bottleneck": label, "numeric_reason": reason})
    write_csv(TABLES / "CORE3_MAIN_TABLE.csv", main_rows)
    write_csv(TABLES / "BOTTLENECK_MAP.csv", bottlenecks)

    fpr_all = list(csv.DictReader((TABLES / "BENIGN_FPR.csv").open(encoding="utf-8")))
    fpr_all = [row for row in fpr_all if row["domain"] == "ALL"]
    lines = ["# CLEAN CORE3 개발 진단 최종 보고서", "",
             "## 판정", "",
             "`CLEAN_CORE3_DEV_V1_DIAGNOSTIC_COMPLETE`", "",
             "이 결과는 **MEntA·MBA·RAG-MIA 개발 공격 3종**의 병목 지도이다. S²-MIA·DCMI·IA는 포함하지 않았고, general/universal 방어 성공을 뜻하지 않는다.", "",
             "## 정상 질의 오탐", "",
             "| Operating point | Calibration target | Locked holdout FPR |", "|---|---:|---:|"]
    for row in fpr_all:
        lines.append(f"| {row['operating_point']} | {100*float(row['nominal_calibration_fpr']):.1f}% | {100*float(row['measured_holdout_fpr']):.1f}% ({row['false_positives']}/{row['holdout_n']}) |")
    lines += ["", "## End-to-end 개인정보 누출", "",
              "아래는 각 공격 논문의 복구된 원 점수로 계산한 native ROC-AUC이다. 0.5에 가까울수록 member/nonmember 구분이 어렵다.", "",
              "| Attack | No Defense | Original MIRABEL | BC-MIRABEL q97 | BC bootstrap 95% CI |", "|---|---:|---:|---:|---:|"]
    for row in main_rows:
        p = pmap[(row["attack"], "BC_MIRABEL_Q97_TOP1_HIDE")]
        lines.append(f"| {row['attack']} | {row['no_defense_native_auc']:.3f} | {row['original_mirabel_native_auc']:.3f} | {row['bc_mirabel_native_auc']:.3f} | [{p['bootstrap_ci_low']:.3f}, {p['bootstrap_ci_high']:.3f}] |")
    lines += ["", "## Detect → Locate → Protect (BC q97)", "",
              "| Attack | Member Retrieval@4 | Alarm | Locator Hit@1 | EPO |", "|---|---:|---:|---:|---:|"]
    for row in main_rows:
        lines.append(f"| {row['attack']} | {100*row['member_retrieval_at4']:.1f}% | {100*row['bc_q97_alarm_rate_all']:.1f}% | {100*row['bc_locator_hit_at1_member']:.1f}% | {100*row['bc_effective_protection_opportunity']:.1f}% |")
    lines += ["", "## 병목", "", "| Attack | Label | 수치 근거 |", "|---|---|---|"]
    for row in bottlenecks:
        lines.append(f"| {row['attack']} | {row['bottleneck']} | {row['numeric_reason']} |")
    fp_f1 = np.mean([row["answer_preservation_token_f1_diagnostic"] for row in false_positives]) if false_positives else math.nan
    new_refusals = sum(row["new_refusal"] for row in false_positives)
    lines += ["", "## BC q97 정상 오탐 피해", "",
              f"- Locked holdout에서 오탐: **{len(false_positives)}/500 ({100*len(false_positives)/500:.1f}%)**",
              f"- 오탐 subset 답변 보존 Token-F1 진단: **{fp_f1:.3f}**",
              f"- 새 refusal: **{new_refusals}/{len(false_positives)}**",
              "- gold answer가 없으므로 정확도/utility라고 부르지 않았다.", "",
              "## 과학적 결론", ""]
    labels = {row["attack"]: row["bottleneck"] for row in bottlenecks}
    lines.append("- 세 공격의 병목이 완전히 동일한지는 위 표의 실제 label로 판단해야 한다.")
    lines.append("- Original MIRABEL은 이 새 corpus의 정상 holdout에서 높은 오탐을 재현했다. BC 보정은 경보 빈도를 크게 줄였지만, 최종 privacy는 반드시 공격별 native AUC와 함께 해석해야 한다.")
    lines.append("- simple top-1 hide가 충분한 공격은 BC native AUC가 0.65 이하이고 오탐 피해가 허용되는 경우에 한해서만 말할 수 있다.")
    lines += ["", "## 다음 방향", "",
              "**새 구조를 자동 구현하지 않는다.** 이 진단에서 다수 공격에 공통으로 나타난 가장 앞단의 병목 하나를 다음 사용자 결정에서 선택한다.", "",
              "## 재현성", "",
              f"- Precommit SHA-256: `{(CONFIGS / 'CLEAN_CORE3_DEV_V1_PRECOMMIT.sha256').read_text().split()[0]}`",
              f"- Retrieval cache SHA-256: `{sha_file(CACHE / 'CLEAN_CORE3_RETRIEVAL_CACHE.jsonl')}`",
              f"- Generated answers SHA-256: `{sha_file(RUNTIME / 'CLEAN_CORE3_GENERATED_ANSWERS.jsonl')}`",
              "- 세 공격은 development set이며 unseen confirmation으로 주장하지 않는다."]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "CLEAN_CORE3_FINAL_REPORT_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"verdict": "CLEAN_CORE3_DEV_V1_DIAGNOSTIC_COMPLETE", "completed_utc": now(),
              "development_attacks": ["MEntA", "MBA", "RAG-MIA"], "excluded": ["S²-MIA", "DCMI", "IA"],
              "main_table": main_rows, "bottlenecks": bottlenecks, "benign_false_positive_audit_n": len(false_positives),
              "precommit_sha256": (CONFIGS / "CLEAN_CORE3_DEV_V1_PRECOMMIT.sha256").read_text().split()[0]}
    write_json(CAMPAIGN / "FINAL_RESULT.json", result)
    checkpoint("CLEAN_CORE3_DEV_V1_DIAGNOSTIC_COMPLETE", attacks=3, false_positive_audit=len(false_positives),
               final_report=str(REPORTS / "CLEAN_CORE3_FINAL_REPORT_KO.md"))


def main() -> None:
    started = time.monotonic()
    retrieval, retrieval_map, answer_map, bundle = load_inputs()
    checkpoint("FINAL_SCORING_STARTED", retrieval_rows=len(retrieval), condition_rows=len(answer_map))
    detection, _, _ = detection_tables(retrieval)
    discrete = score_discrete(answer_map, bundle)
    _, menta_sessions = score_menta(answer_map, bundle)
    privacy = summarize_privacy(discrete, menta_sessions)
    false_positives, costs = false_positive_and_cost(retrieval, answer_map)
    query_answer_export(retrieval, answer_map)
    final_report(retrieval, detection, privacy, false_positives, costs)
    write_json(RUNTIME / "FINAL_EVALUATION_RUNTIME.json", {"runtime_seconds": time.monotonic()-started, "completed_utc": now()})
    print((CAMPAIGN / "FINAL_RESULT.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
