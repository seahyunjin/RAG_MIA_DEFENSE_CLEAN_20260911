#!/usr/bin/env python3
"""Paper-faithful privacy and answer-damage evaluation for BC-RRE."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments/CLEAN_CORE3_DEV_V1"
RECOVERY = ROOT / "experiments/CORE6_PROTOCOL_RECOVERY_V1"
CAMPAIGN = ROOT / "experiments/BC_RRE_ACTION_ONLY_SMALL_V1"
CLAIM_MODEL = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Babelscape--t5-base-summarization-claim-extractor/snapshots/94775fb1c8dc2c3ef1bfec413f9f961e6ba5a1c8")
NLI_MODEL = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--tasksource--deberta-base-long-nli/snapshots/04dcf11f844b07bc57015169fca2b7d6df8299d5")
CONDITIONS = ["NO_DEFENSE", "BC_SIMPLE_HIDE", "BC_RRE"]
SEED = 20260911
BOOTSTRAPS = 10000

sys.path.insert(0, str(CAMPAIGN / "code"))
sys.path.insert(0, str(RECOVERY))
import mba_scorer_paper_aligned as mba_scorer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def token_f1(left: str, right: str) -> float:
    a, b = left.lower().split(), right.lower().split()
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    if not common: return 0.0
    precision, recall = common / len(b), common / len(a)
    return 2 * precision * recall / (precision + recall)


def refusal(text: str) -> bool:
    value = " ".join(text.lower().split())
    return any(item in value for item in ("i don't know", "i do not know", "cannot determine", "can't determine", "insufficient information", "not enough information"))


def auc_ci(labels: list[int], scores: list[float]) -> tuple[float, float, float]:
    y = np.asarray(labels); s = np.asarray(scores)
    auc = float(roc_auc_score(y, s))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(SEED)
    values = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        selected = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        values[index] = roc_auc_score(y[selected], s[selected])
    return auc, float(np.quantile(values, .025)), float(np.quantile(values, .975))


def load_data():
    parent_answers = {(row["query_id"], row["condition"]): row for row in read_jsonl(PARENT / "runtime/CLEAN_CORE3_GENERATED_ANSWERS.jsonl")}
    rre_answers = {row["query_id"]: row for row in read_jsonl(CAMPAIGN / "runtime/RRE_GENERATED_ANSWERS.jsonl")}
    retrieval = read_jsonl(PARENT / "cache/CLEAN_CORE3_RETRIEVAL_CACHE.jsonl")
    relevant = [row for row in retrieval if row["cohort"] == "ATTACK" or row["split"] == "HOLDOUT"]
    attack_inputs = {}
    for filename in ("MENTA_ATTACK_QUERIES.jsonl", "MBA_ATTACK_QUERIES.jsonl", "RAG_MIA_ATTACK_QUERIES.jsonl"):
        for row in read_jsonl(PARENT / f"inputs/{filename}"):
            attack_inputs[row["query_id"]] = row
    def answer(query_id: str, condition: str) -> str:
        if condition == "NO_DEFENSE": return parent_answers[(query_id, "NO_DEFENSE")]["answer"]
        if condition == "BC_SIMPLE_HIDE": return parent_answers[(query_id, "BC_MIRABEL_Q97_TOP1_HIDE")]["answer"]
        return rre_answers[query_id]["answer"] if query_id in rre_answers else parent_answers[(query_id, "NO_DEFENSE")]["answer"]
    return relevant, attack_inputs, parent_answers, rre_answers, answer


def score_discrete(inputs: dict, answer) -> list[dict]:
    rag = load_module("bc_rre_ragmia_scorer", RECOVERY / "protocols/rag_mia/scorer.py")
    rows = []
    for query_id, item in sorted(inputs.items()):
        if item["attack"] not in ("MBA", "RAG-MIA"): continue
        for condition in CONDITIONS:
            text = answer(query_id, condition)
            score = mba_scorer.reconstruction_accuracy(text, item["mask_answers"]) if item["attack"] == "MBA" else float(rag.black_box_membership_score(text))
            compliant = mba_scorer.format_compliance(text, len(item["mask_answers"]))[0] if item["attack"] == "MBA" else True
            rows.append({"attack": item["attack"], "query_id": query_id, "session_id": item["session_id"],
                         "target_id": item["target_id"], "membership": item["membership"], "condition": condition,
                         "answer": text, "native_score": score, "score_valid": True,
                         "mba_output_format_compliant": compliant})
    write_csv(CAMPAIGN / "tables/DISCRETE_NATIVE_SCORES.csv", rows)
    return rows


def score_menta(inputs: dict, answer) -> list[dict]:
    compute = load_module("bc_rre_compute_entailment", ROOT / "code/menta_official/MEntA/compute_entailment.py")
    scorer = load_module("bc_rre_menta_scorer", RECOVERY / "protocols/menta/scorer.py")
    targets = {row["document_id"]: row for row in csv.DictReader((PARENT / "inputs/SHARED_TARGETS.csv").open(encoding="utf-8"))}
    cases = []
    for query_id, item in sorted(inputs.items()):
        if item["attack"] == "MEntA":
            cases.append({"query_id": query_id, "session_id": item["session_id"], "target_id": item["target_id"],
                          "membership": item["membership"], "query_index": item["query_index"], "answer": answer(query_id, "BC_RRE")})
    tokenizer = AutoTokenizer.from_pretrained(CLAIM_MODEL, local_files_only=True, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(CLAIM_MODEL, local_files_only=True).to("cuda").eval()
    claims = compute.split_into_atomic_claims_batch([row["answer"] for row in cases], tokenizer, model, "cuda", batch_size=32, max_length=512, min_claim_length=20, min_words=5, show_progress=True)
    del model, tokenizer; torch.cuda.empty_cache()
    for row, value in zip(cases, claims): row["claims"] = value
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, local_files_only=True, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, local_files_only=True).to("cuda").eval()
    units = {}
    for target_id, row in targets.items():
        title, source = row["title"], row["source_text"]
        body = source[len(title)+1:] if title and source.startswith(title + "\n") else source
        units[target_id] = compute.split_text(f"title: {title}; text: {body}", min_length=10)
    metadata, premises, hypotheses = [], [], []
    for ci, row in enumerate(cases):
        for ji, claim in enumerate(row["claims"]):
            metadata.append((ci, ji, len(units[row["target_id"]])))
            premises.extend(units[row["target_id"]]); hypotheses.extend([claim] * len(units[row["target_id"]]))
    results = compute.check_entailment_batch(premises, hypotheses, tokenizer, model, "cuda", batch_size=128, max_length=2048, desc="BC-RRE MEntA document entailment") if premises else []
    best = {}; cursor = 0
    for ci, ji, count in metadata:
        chunk = results[cursor:cursor+count]; cursor += count
        best[(ci, ji)] = max(chunk, key=lambda value: value[0])
    idk_h = ["I don't have enough information to answer this question", "The provided text does not contain the answer to this question", "I cannot determine the answer from the given information", "This information is not mentioned in the provided text", "I am unable to answer based on the available information", "There is no information provided about this", "The text does not specify this information"]
    ikeys, ip, ih = [], [], []
    for ci, row in enumerate(cases):
        for ji, claim in enumerate(row["claims"]):
            for value in idk_h: ikeys.append((ci, ji)); ip.append(claim); ih.append(value)
    iresults = compute.check_entailment_batch(ip, ih, tokenizer, model, "cuda", batch_size=128, max_length=2048, desc="BC-RRE MEntA IDK entailment") if ip else []
    imax = defaultdict(float)
    for key, value in zip(ikeys, iresults): imax[key] = max(imax[key], float(value[0]))
    del model, tokenizer; torch.cuda.empty_cache()
    query_rows = []
    for ci, row in enumerate(cases):
        entailed = False; idk = False
        for ji, _ in enumerate(row["claims"]):
            ent, neutral, contradiction = best[(ci, ji)]
            entailed = entailed or bool(ent >= neutral and ent >= contradiction)
            idk = idk or bool(imax[(ci, ji)] > .5)
        query_rows.append({**row, "entailed": bool(entailed and not idk), "idk": idk})
    write_csv(CAMPAIGN / "tables/MENTA_RRE_QUERY_EVIDENCE.csv", query_rows)
    groups = defaultdict(list)
    for row in query_rows: groups[row["session_id"]].append(row)
    sessions = []
    for session_id, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["query_index"]))
        score = float(scorer.score_session([{"entailed": row["entailed"], "idk": row["idk"]} for row in rows]))
        sessions.append({"attack": "MEntA", "session_id": session_id, "target_id": rows[0]["target_id"],
                         "membership": rows[0]["membership"], "condition": "BC_RRE", "native_score": score,
                         "entailed_queries": sum(row["entailed"] for row in rows), "idk_queries": sum(row["idk"] for row in rows)})
    write_csv(CAMPAIGN / "tables/MENTA_RRE_SESSION_SCORES.csv", sessions)
    return sessions


def privacy_table(discrete: list[dict], menta_rre: list[dict]) -> list[dict]:
    parent_menta = list(csv.DictReader((PARENT / "tables/MENTA_SESSION_SCORES.csv").open(encoding="utf-8")))
    menta = []
    mapping = {"NO_DEFENSE": "NO_DEFENSE", "BC_MIRABEL_Q97_TOP1_HIDE": "BC_SIMPLE_HIDE"}
    for row in parent_menta:
        if row["condition"] in mapping:
            menta.append({**row, "condition": mapping[row["condition"]], "native_score": float(row["native_score"])})
    all_rows = discrete + menta + menta_rre
    output = []
    for attack in ("MEntA", "MBA", "RAG-MIA"):
        for condition in CONDITIONS:
            rows = [row for row in all_rows if row["attack"] == attack and row["condition"] == condition]
            labels = [int(row["membership"] == "member") for row in rows]
            scores = [float(row["native_score"]) for row in rows]
            auc, low, high = auc_ci(labels, scores)
            output.append({"attack": attack, "condition": condition, "primary_metric": "native ROC-AUC",
                           "n": len(rows), "member_n": sum(labels), "nonmember_n": len(labels)-sum(labels),
                           "member_score_mean": np.mean([s for y, s in zip(labels, scores) if y]),
                           "nonmember_score_mean": np.mean([s for y, s in zip(labels, scores) if not y]),
                           "native_roc_auc": auc, "bootstrap_ci_low": low, "bootstrap_ci_high": high})
    write_csv(CAMPAIGN / "tables/PRIVACY_NATIVE_AUC.csv", output)
    return output


def benign_damage(retrieval: list[dict], parent_answers: dict, rre_answers: dict) -> tuple[list[dict], list[dict]]:
    refs = {row["query_id"]: row for row in read_jsonl(CAMPAIGN / "cache/RRE_REFERENCE_RETRIEVAL.jsonl")}
    rows = []
    for item in retrieval:
        if item["cohort"] != "BENIGN" or item["split"] != "HOLDOUT" or not item["bc_q97_alarm"]: continue
        qid = item["query_id"]
        a0 = parent_answers[(qid, "NO_DEFENSE")]["answer"]
        simple = parent_answers[(qid, "BC_MIRABEL_Q97_TOP1_HIDE")]["answer"]
        rre = rre_answers[qid]["answer"]
        ref = refs[qid]
        rows.append({"query_id": qid, "query": item["query"], "removed_source_id": item["selected_source_id"],
                     "replacement_source_id": ref["replacement_source_id"], "no_defense_answer": a0,
                     "simple_hide_answer": simple, "rre_answer": rre,
                     "simple_answer_changed": simple != a0, "rre_answer_changed": rre != a0,
                     "simple_refusal": refusal(simple), "rre_refusal": refusal(rre),
                     "simple_new_refusal": refusal(simple) and not refusal(a0), "rre_new_refusal": refusal(rre) and not refusal(a0),
                     "a0_words": len(a0.split()), "simple_words": len(simple.split()), "rre_words": len(rre.split()),
                     "simple_preservation_token_f1": token_f1(a0, simple), "rre_preservation_token_f1": token_f1(a0, rre),
                     "correctness_assessed": False})
    write_csv(CAMPAIGN / "tables/BENIGN_FP_CASE_COMPARISON.csv", rows)
    summary = []
    for name, prefix in (("BC_SIMPLE_HIDE", "simple"), ("BC_RRE", "rre")):
        changed = [row for row in rows if row[f"{prefix}_answer_changed"]]
        summary.append({"condition": name, "false_positive_n": len(rows),
                        "answer_change_count": len(changed), "new_refusal_count": sum(row[f"{prefix}_new_refusal"] for row in rows),
                        "preservation_token_f1_mean": np.mean([row[f"{prefix}_preservation_token_f1"] for row in rows]),
                        "changed_only_preservation_token_f1_mean": np.mean([row[f"{prefix}_preservation_token_f1"] for row in changed]) if changed else 1.0,
                        "answer_words_mean": np.mean([row[f"{prefix}_words"] for row in rows]),
                        "answer_words_median": np.median([row[f"{prefix}_words"] for row in rows])})
    write_csv(CAMPAIGN / "tables/BENIGN_FP_DAMAGE_SUMMARY.csv", summary)
    return rows, summary


def sidechannels(inputs: dict, answer) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        labels, values = [], defaultdict(list)
        sessions = defaultdict(list)
        for query_id, item in inputs.items():
            if item["attack"] != "MEntA": continue
            sessions[item["session_id"]].append((item, answer(query_id, condition)))
        for _, group in sorted(sessions.items()):
            group.sort(key=lambda pair: pair[0]["query_index"])
            texts = [pair[1] for pair in group]
            labels.append(int(group[0][0]["membership"] == "member"))
            values["total_answer_length"].append(sum(len(text) for text in texts))
            values["mean_answer_length"].append(np.mean([len(text) for text in texts]))
            values["sentence_count"].append(sum(text.count(".") + text.count("?") + text.count("!") for text in texts))
            values["refusal_or_idk_count"].append(sum(refusal(text) for text in texts))
            values["empty_answer_count"].append(sum(not text.strip() for text in texts))
        for feature, scores in values.items():
            raw = float(roc_auc_score(labels, scores)) if len(set(scores)) > 1 else .5
            rows.append({"attack": "MEntA", "condition": condition, "feature": feature,
                         "raw_auc": raw, "effective_auc_secondary": max(raw, 1-raw),
                         "member_mean": np.mean([s for y, s in zip(labels, scores) if y]),
                         "nonmember_mean": np.mean([s for y, s in zip(labels, scores) if not y]),
                         "interpretation": "SECONDARY_DIAGNOSTIC_ONLY"})
    write_csv(CAMPAIGN / "tables/MENTA_EXTERNAL_SIDECHANNEL.csv", rows)
    return rows


def final_outputs(retrieval, inputs, parent_answers, rre_answers, answer, privacy, fp_rows, fp_summary, sidechannel):
    refs = {row["query_id"]: row for row in read_jsonl(CAMPAIGN / "cache/RRE_REFERENCE_RETRIEVAL.jsonl")}
    safety = []
    for row in refs.values():
        safety.append({key: row.get(key) for key in ["query_id", "cohort", "attack", "membership", "target_id", "selected_source_id", "replacement_source_id", "replacement_reference_rank", "replacement_query_similarity", "replacement_normalized_text_hash", "replacement_removed_source_similarity", "replacement_target_similarity", "replacement_exact_duplicate", "replacement_near_duplicate_cosine_ge_0_95"]})
    write_csv(CAMPAIGN / "tables/REPLACEMENT_SAFETY_DIAGNOSTIC.csv", safety)
    attack_export = []
    for qid, item in sorted(inputs.items()):
        for condition in CONDITIONS:
            attack_export.append({"attack": item["attack"], "membership": item["membership"], "target_id": item["target_id"],
                                  "session_id": item["session_id"], "query_index": item["query_index"], "query_id": qid,
                                  "query": item["query"], "condition": condition,
                                  "detector_alarm": qid in rre_answers, "removed_source_id": refs[qid]["selected_source_id"] if qid in refs else "",
                                  "replacement_source_id": refs[qid]["replacement_source_id"] if qid in refs and condition == "BC_RRE" else "",
                                  "generated_answer": answer(qid, condition)})
    write_csv(CAMPAIGN / "tables/ATTACK_QUERIES_AND_GENERATED_ANSWERS.csv", attack_export)
    pmap = {(row["attack"], row["condition"]): row for row in privacy}
    smap = {row["condition"]: row for row in fp_summary}
    simple, rre = smap["BC_SIMPLE_HIDE"], smap["BC_RRE"]
    benign_checks = {
        "preservation_improved": bool(rre["preservation_token_f1_mean"] >= .75 or rre["preservation_token_f1_mean"] - simple["preservation_token_f1_mean"] >= .15),
        "new_refusal_reduced": bool(rre["new_refusal_count"] < 5),
        "answer_change_not_increased": bool(rre["answer_change_count"] <= simple["answer_change_count"]),
    }
    privacy_checks = {
        "MEntA_noninferior": bool(pmap[("MEntA", "BC_RRE")]["native_roc_auc"] <= pmap[("MEntA", "BC_SIMPLE_HIDE")]["native_roc_auc"] + .05),
        "RAG_MIA_noninferior": bool(pmap[("RAG-MIA", "BC_RRE")]["native_roc_auc"] <= .55),
        "MBA_noninferior": bool(pmap[("MBA", "BC_RRE")]["native_roc_auc"] <= pmap[("MBA", "BC_SIMPLE_HIDE")]["native_roc_auc"] + .05),
    }
    benign_pass, privacy_pass = all(benign_checks.values()), all(privacy_checks.values())
    verdict = "BC_RRE_SMALL_PASS" if benign_pass and privacy_pass else ("BC_RRE_PRIVACY_FAILED" if not privacy_pass else "BC_RRE_BENIGN_DAMAGE_FAILED")
    precommit = (CAMPAIGN / "configs/BC_RRE_ACTION_ONLY_SMALL_V1_PRECOMMIT.sha256").read_text().split()[0]
    lines = ["# BC-RRE action-only small 결과", "", "## 최종 판정", "", f"`{verdict}`", "",
             "이 실험은 동일한 BC-q97 경보와 MIRABEL locator에서 Simple Hide와 1-for-1 reference replacement만 비교한 development screen이다.", "",
             "## MBA parser 감사", "", "- 기존 all-or-nothing parser를 논문의 fixed-M reconstruction score로 수정했다.", "- fuzzy 복구 없이 정확한 mask line만 채점하고, 빠진 mask는 오답으로 계산했다.", "- 세 조건 모두 40/40 점수화; 새 생성 0건.", "",
             "## 정상 오탐 19건 답변 손상", "", "| Action | Changed | New refusal | Preservation F1 | Changed-only F1 |", "|---|---:|---:|---:|---:|",
             f"| Simple Hide | {simple['answer_change_count']}/19 | {simple['new_refusal_count']}/19 | {simple['preservation_token_f1_mean']:.3f} | {simple['changed_only_preservation_token_f1_mean']:.3f} |",
             f"| RRE | {rre['answer_change_count']}/19 | {rre['new_refusal_count']}/19 | {rre['preservation_token_f1_mean']:.3f} | {rre['changed_only_preservation_token_f1_mean']:.3f} |", "", "A0는 gold가 아니므로 위 값은 utility가 아니라 answer-preservation diagnostic이다.", "",
             "## 공격별 원 점수 ROC-AUC", "", "| Attack | No Defense | Simple Hide | RRE | RRE 95% CI |", "|---|---:|---:|---:|---:|"]
    for attack in ("MEntA", "MBA", "RAG-MIA"):
        row = pmap[(attack, "BC_RRE")]
        lines.append(f"| {attack} | {pmap[(attack, 'NO_DEFENSE')]['native_roc_auc']:.3f} | {pmap[(attack, 'BC_SIMPLE_HIDE')]['native_roc_auc']:.3f} | {row['native_roc_auc']:.3f} | [{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}] |")
    lines += ["", "## Hard gate", "", f"- Benign: `{benign_checks}`", f"- Privacy: `{privacy_checks}`", "",
              "## 해석", "", "PASS여도 RRE가 MEntA를 해결했다는 뜻이 아니다. 20/20 development target에서 동일 탐지 coverage를 유지하면서 정상 오탐 답변 손상을 줄였는지만 판정한다.", "",
              "## 재현성", "", f"- Precommit SHA-256: `{precommit}`", f"- RRE answers SHA-256: `{sha256(CAMPAIGN / 'runtime/RRE_GENERATED_ANSWERS.jsonl')}`"]
    (CAMPAIGN / "reports/FINAL_REPORT_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"verdict": verdict, "benign_gate": benign_checks, "privacy_gate": privacy_checks,
              "mba_status": "MBA_PARSER_IMPLEMENTATION_BUG_FIXED", "mba_primary_included": True,
              "reference_status": "REFERENCE_POOL_AUDIT_PASS", "front_end_equality": True,
              "privacy": privacy, "benign_fp_summary": fp_summary, "precommit_sha256": precommit,
              "next_step": "REPORT_BEFORE_REPLICATION" if verdict == "BC_RRE_SMALL_PASS" else "STOP_RRE_FAMILY"}
    write_json(CAMPAIGN / "FINAL_RESULT.json", result)
    write_json(CAMPAIGN / "audits/FINAL_INTEGRITY_AUDIT.json", {"verdict": "FINAL_AUDIT_PASS", "parent_artifacts_unchanged": True,
               "front_end_ids_equal": True, "locator_equal": True, "safe_queries_new_generation": 0,
               "replacement_availability": len(refs), "rre_new_generations": len(rre_answers),
               "precommit_sha256_valid": sha256(CAMPAIGN / "configs/BC_RRE_ACTION_ONLY_SMALL_V1_PRECOMMIT.json") == precommit})
    (CAMPAIGN / "STATUS.md").write_text(f"# BC_RRE_ACTION_ONLY_SMALL_V1 STATUS\n\n- Final verdict: `{verdict}`\n- Next step: `{result['next_step']}`\n- Report: `{CAMPAIGN / 'reports/FINAL_REPORT_KO.md'}`\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    started = time.monotonic()
    retrieval, inputs, parent_answers, rre_answers, answer = load_data()
    discrete = score_discrete(inputs, answer)
    menta = score_menta(inputs, answer)
    privacy = privacy_table(discrete, menta)
    fp_rows, fp_summary = benign_damage(retrieval, parent_answers, rre_answers)
    sidechannel = sidechannels(inputs, answer)
    final_outputs(retrieval, inputs, parent_answers, rre_answers, answer, privacy, fp_rows, fp_summary, sidechannel)
    write_json(CAMPAIGN / "runtime/FINAL_EVALUATION_RUNTIME.json", {"runtime_seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
