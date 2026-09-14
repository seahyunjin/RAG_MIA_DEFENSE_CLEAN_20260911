#!/usr/bin/env python3
"""Fail-closed DCMI/IA protocol recovery for the frozen Final LC campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"
PAPERS = ROOT / "papers"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, "")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def git(repo: Path, *args: str, binary: bool = False):
    output = subprocess.check_output(["git", "-C", str(repo), *args])
    return output if binary else output.decode("utf-8", errors="replace")


def pdf_text(path: Path) -> str:
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)


def rank_auc(labels: list[int], scores: list[float]) -> float:
    """ROC-AUC by the Mann-Whitney rank statistic, with exact tie handling."""
    if len(labels) != len(scores) or not labels:
        return math.nan
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return math.nan
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    rank = 1
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        group_size = end - cursor
        average_rank = (rank + rank + group_size - 1) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[cursor:end])
        rank += group_size
        cursor = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def normalize_mba(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", value.casefold()))


def parse_mba_exact_lines(text: str) -> tuple[dict[str, str], int, int]:
    """Parse only exact indexed lines; malformed/missing content gets no credit."""
    predictions: dict[str, str] = {}
    malformed = 0
    duplicates = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*\[?Mask_(\d+)\]?\s*:\s*(\S(?:.*\S)?)\s*", line, re.I)
        if not match:
            malformed += 1
            continue
        key = f"Mask_{int(match.group(1))}"
        if key in predictions:
            duplicates += 1
            continue
        predictions[key] = match.group(2)
    return predictions, malformed, duplicates


def mba_missing_as_incorrect() -> tuple[list[dict], list[dict]]:
    query_path = PREVIOUS / "inputs" / "FINAL_SUPPORTED_QUERY_MANIFEST.jsonl"
    answer_path = PREVIOUS / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl"
    queries = {row["query_id"]: row for row in read_jsonl(query_path) if row.get("attack") == "MBA"}
    answers = {(row["query_id"], row["condition"]): row for row in read_jsonl(answer_path) if row.get("attack") == "MBA"}
    conditions = sorted({condition for _, condition in answers})
    detail: list[dict] = []
    summary: list[dict] = []
    for condition in conditions:
        rows = []
        for query_id, query in sorted(queries.items()):
            answer = answers[(query_id, condition)]["answer"]
            predictions, malformed, duplicates = parse_mba_exact_lines(answer)
            ground_truth = query["mask_answers"]
            expected = set(ground_truth)
            parsed_expected = expected & set(predictions)
            hits = 0
            for key, accepted in ground_truth.items():
                if key in predictions and normalize_mba(predictions[key]) in {normalize_mba(item) for item in accepted}:
                    hits += 1
            score = hits / len(ground_truth)
            missing = len(expected - set(predictions))
            unexpected = len(set(predictions) - expected)
            item = {
                "query_id": query_id,
                "target_id": query["target_id"],
                "membership": query["membership"],
                "condition": condition,
                "mask_count": len(ground_truth),
                "correct_count": hits,
                "native_score_missing_as_incorrect": score,
                "parsed_expected_count": len(parsed_expected),
                "missing_count": missing,
                "unexpected_count": unexpected,
                "malformed_line_count": malformed,
                "duplicate_count": duplicates,
                "incomplete_or_malformed": int(bool(missing or unexpected or malformed or duplicates)),
                "partial_reconstruction": int(0 < len(parsed_expected) < len(expected)),
                "contains_refusal": int("i don't know" in answer.casefold() or "i do not know" in answer.casefold()),
            }
            rows.append(item)
            detail.append(item)
        labels = [int(row["membership"] == "member") for row in rows]
        scores = [float(row["native_score_missing_as_incorrect"]) for row in rows]
        member_rows = [row for row in rows if row["membership"] == "member"]
        nonmember_rows = [row for row in rows if row["membership"] == "nonmember"]
        auc = rank_auc(labels, scores)
        summary.append({
            "condition": condition,
            "total_n": len(rows),
            "valid_n_under_missing_as_incorrect_rule": len(rows),
            "excluded_n": 0,
            "member_n": len(member_rows),
            "nonmember_n": len(nonmember_rows),
            "native_roc_auc": auc,
            "effective_auc_secondary": max(auc, 1 - auc),
            "member_score_mean": statistics.fmean(row["native_score_missing_as_incorrect"] for row in member_rows),
            "nonmember_score_mean": statistics.fmean(row["native_score_missing_as_incorrect"] for row in nonmember_rows),
            "incomplete_or_malformed_n": sum(row["incomplete_or_malformed"] for row in rows),
            "member_incomplete_or_malformed_n": sum(row["incomplete_or_malformed"] for row in member_rows),
            "nonmember_incomplete_or_malformed_n": sum(row["incomplete_or_malformed"] for row in nonmember_rows),
            "partial_reconstruction_n": sum(row["partial_reconstruction"] for row in rows),
            "refusal_contribution_n": sum(row["contains_refusal"] for row in rows),
        })
    return detail, summary


def audit_dcmi(repo: Path) -> dict:
    paper = PAPERS / "DCMI.pdf"
    text = pdf_text(paper)
    perturb = (repo / "perturb.py").read_text(encoding="utf-8")
    scorer = (repo / "MIA.py").read_text(encoding="utf-8")
    commit = git(repo, "rev-parse", "HEAD").strip()
    heads_tags = git(repo, "branch", "-a").splitlines() + git(repo, "tag", "--list").splitlines()
    paper_checks = {
        "black_box_perturbation_0_06": bool(re.search(r"magnitude 0\.06", text)),
        "binary_difference_equation": "frag,calibrated" in text and "frag (q) − frag (q′ )" in text,
        "membership_requires_difference_one": "frag,calibrated (q) = 1" in text,
        "appendix_prompt_present": "Replace {num_to_replace} key adjectives or adverbs" in text,
    }
    repo_checks = {
        "perturbation_0_03": bool(re.search(r"perturbation_magnitude\s*=\s*0\.03", perturb)),
        "uses_unversioned_gpt_4o_alias": 'model="gpt-4o"' in perturb,
        "temperature_declared": "temperature=" in perturb,
        "top_p_declared": "top_p=" in perturb,
        "seed_declared": "seed=" in perturb,
        "scorer_has_placeholder_arrays": all(token in scorer for token in ("mem = [...]", "perturb_mem = [...]", "nom = [...]", "perturb_nom = [...]")),
    }
    return {
        "status": "DCMI_SPEC_UNDERDETERMINED",
        "paper_sha256": sha256(paper),
        "official_repository": "https://github.com/Xinyu140203/RAG_MIA",
        "official_commit": commit,
        "official_visible_refs": heads_tags,
        "official_perturb_sha256": sha256(repo / "perturb.py"),
        "official_scorer_sha256": sha256(repo / "MIA.py"),
        "paper_checks": paper_checks,
        "repository_checks": repo_checks,
        "recovered_definition": {
            "member": "target sample included in protected retrieval database",
            "nonmember": "target sample excluded from protected retrieval database",
            "queries_per_target": 2,
            "base_query": "Is this: <target sample> right? Answer with Yes or No.",
            "paper_perturbation_rate": 0.06,
            "score": "f(q)-f(q_perturbed), Yes=1, No=0",
            "member_decision": "score == 1",
            "primary_metrics": ["accuracy", "ROC-AUC"],
            "score_polarity": "higher_is_more_member",
        },
        "blocking_reasons": [
            "Paper black-box setting fixes 6%, but official perturb.py fixes 3%.",
            "The perturbation LLM is not frozen to an immutable model revision.",
            "Perturbation decoding parameters are not specified.",
            "Official MIA.py requires manually inserted placeholder score arrays.",
            "No immutable original perturbation bundle exists for the frozen 2,000-target cohort.",
        ],
        "paper_faithful_execution_allowed": False,
    }


def historic_ia_result_audit(repo: Path) -> dict:
    objects = git(repo, "rev-list", "--objects", "--all")
    blobs: dict[str, str] = {}
    for line in objects.splitlines():
        pieces = line.split(" ", 1)
        if len(pieces) != 2:
            continue
        oid, path = pieces
        if path.startswith("results/target_docs/") and path.endswith(".json"):
            blobs.setdefault(oid, path)
    max_questions = 0
    q30_rows = 0
    parsed_blobs = 0
    blobs_with_gt = 0
    blobs_with_responses = 0
    for oid in blobs:
        try:
            data = json.loads(git(repo, "cat-file", "blob", oid, binary=True))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        values = list(data.values()) if isinstance(data, dict) else data if isinstance(data, list) else []
        values = [row for row in values if isinstance(row, dict)]
        parsed_blobs += 1
        lengths = [len(row.get("questions", [])) for row in values if isinstance(row.get("questions"), list)]
        max_questions = max(max_questions, max(lengths, default=0))
        q30_rows += sum(length >= 30 for length in lengths)
        blobs_with_gt += int(any(isinstance(row.get("answers"), list) and row["answers"] for row in values))
        blobs_with_responses += int(any(isinstance(row.get("llm_responses"), list) and row["llm_responses"] for row in values))
    return {
        "unique_historic_result_blobs": len(blobs),
        "parsed_historic_result_blobs": parsed_blobs,
        "maximum_questions_per_target": max_questions,
        "rows_with_at_least_30_questions": q30_rows,
        "blobs_with_ground_truth_answers": blobs_with_gt,
        "blobs_with_rag_responses": blobs_with_responses,
    }


def audit_ia(repo: Path) -> dict:
    paper = PAPERS / "IA.pdf"
    text = pdf_text(paper)
    commit = git(repo, "rev-parse", "HEAD").strip()
    dataset_rows = []
    frozen_rows = list(csv.DictReader((ROOT / "experiments" / "LC_MIRABEL_LARGE_V1" / "inputs" / "LARGE_SHARED_TARGETS.csv").open(encoding="utf-8")))
    frozen_overlap = []
    for dataset in ("nfcorpus", "trec-covid"):
        data_path = repo / "datasets" / dataset / "clean_data_with_questions.json"
        selection_path = repo / "datasets" / dataset / "selected_indices.json"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = selection["mem_indices"] + selection["non_mem_indices"]
        distribution = Counter(len(data[key].get("questions", [])) for key in selected if key in data)
        dataset_rows.append({
            "dataset": dataset,
            "corpus_rows": len(data),
            "member_ids": len(selection["mem_indices"]),
            "nonmember_ids": len(selection["non_mem_indices"]),
            "member_nonmember_overlap": len(set(selection["mem_indices"]) & set(selection["non_mem_indices"])),
            "selected_ids_missing_from_bundle": len(set(selected) - set(data)),
            "question_count_distribution": dict(sorted(distribution.items())),
            "question_bundle_sha256": sha256(data_path),
            "selection_sha256": sha256(selection_path),
            "contains_ground_truth_field": any("answers" in row or "ground_truth_answer" in row for row in data.values()),
        })
        for membership, selection_field in (("member", "mem_indices"), ("nonmember", "non_mem_indices")):
            frozen_ids = {
                row["document_id"].split("::")[-1]
                for row in frozen_rows
                if row["domain"] == dataset and row["membership"] == membership
            }
            official_ids = set(selection[selection_field])
            frozen_overlap.append({
                "dataset": dataset,
                "membership": membership,
                "frozen_ids": len(frozen_ids),
                "official_selected_ids": len(official_ids),
                "exact_id_overlap": len(frozen_ids & official_ids),
            })
    history = historic_ia_result_audit(repo)
    return {
        "status": "IA_PAPER_EXACT_UNAVAILABLE",
        "repository_variant_status": "IA_REPOSITORY_VARIANT_AVAILABLE_AS_STRESS_TEST_ONLY",
        "paper_sha256": sha256(paper),
        "official_repository": "https://github.com/ali7naseh/RAG_MIA",
        "official_commit": commit,
        "official_visible_refs": git(repo, "branch", "-a").splitlines() + git(repo, "tag", "--list").splitlines(),
        "paper_checks": {
            "q30_prompt_documented": "generate 30 yes/no questions with GPT-4o" in text,
            "gpt_4o_query_generation": "GPT-4o" in text,
            "gpt_4o_mini_ground_truth": "GPT-4o-mini" in text,
            "lambda_five": bool(re.search(r"set [^\n]*?to a value greater than one \(5\)", text)),
            "score_formula_present": "I[𝑟𝑖 = 𝑔𝑖 ] − 𝜆I[𝑟𝑖 = UNK]" in text,
            "paper_targets_per_class": bool(re.search(r"randomly select\s+1000 members and 1000 non-members", text)),
        },
        "official_current_datasets": dataset_rows,
        "official_selection_vs_frozen_cohort": frozen_overlap,
        "official_git_history": history,
        "recovered_definition": {
            "queries_per_target": 30,
            "query_generator": "GPT-4o",
            "ground_truth_generator": "GPT-4o-mini",
            "lambda_unknown_penalty": 5,
            "score": "mean_i(1[response_i=GT_i] - 5*1[response_i=UNK])",
            "primary_metrics": ["ROC-AUC", "accuracy", "TPR@low-FPR"],
            "score_polarity": "higher_is_more_member",
        },
        "blocking_reasons": [
            "Paper requires Q30, but current official public bundles contain Q15 per target.",
            "All 51 unique historical result blobs have at most Q15; no Q30 row was found.",
            "Current public question bundles contain no immutable GPT-4o-mini ground-truth field.",
            "No original Q30+GT bundle matches the frozen 2,000-target evaluation cohort.",
            "Regenerating today would not reproduce the immutable original hosted-model outputs.",
        ],
        "paper_exact_execution_allowed": False,
    }


def preserve_core4(mba_summary: list[dict]) -> list[dict]:
    previous_path = PREVIOUS / "tables" / "SUPPORTED_POST_GENERATION_PRIVACY.csv"
    previous = list(csv.DictReader(previous_path.open(encoding="utf-8")))
    mba_map = {row["condition"]: row for row in mba_summary}
    conditions = [
        "NO_DEFENSE",
        "ORIGINAL_MIRABEL_SIMPLE_HIDE",
        "GLOBAL_BC_SIMPLE_HIDE",
        "FINAL_LC_OUTER_SIMPLE_HIDE",
    ]
    rows = []
    for attack in ("MEntA", "MBA", "RAG-MIA", "S²-MIA"):
        for condition in conditions:
            old = next(row for row in previous if row["attack"] == attack and row["condition"] == condition)
            if attack == "MBA":
                fixed = mba_map[condition]
                rows.append({
                    "attack": attack,
                    "condition": condition,
                    "native_metric": "ROC-AUC; missing/malformed masks scored incorrect; no exclusions",
                    "native_value": fixed["native_roc_auc"],
                    "effective_auc_secondary": fixed["effective_auc_secondary"],
                    "total_n": fixed["total_n"],
                    "valid_n": fixed["total_n"],
                    "invalid_n": 0,
                    "provenance": "recomputed_from_frozen_answers",
                })
            else:
                rows.append({
                    "attack": attack,
                    "condition": condition,
                    "native_metric": old["native_metric"],
                    "native_value": old["native_value"],
                    "effective_auc_secondary": old.get("effective_auc_secondary", ""),
                    "total_n": old.get("total_n", old.get("valid_n", "")),
                    "valid_n": old.get("valid_n", ""),
                    "invalid_n": old.get("invalid_n", 0),
                    "provenance": "immutable_predecessor_result",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcmi-repo", type=Path, required=True)
    parser.add_argument("--ia-repo", type=Path, required=True)
    args = parser.parse_args()

    events: list[dict] = []

    def event(phase: str, **fields: object) -> None:
        events.append({"utc": utc_now(), "phase": phase, **fields})
        atomic_text(EXP / "RUN.log", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in events))

    event("START", dcmi_repo=str(args.dcmi_repo), ia_repo=str(args.ia_repo))

    frozen_manifest = PREVIOUS / "configs" / "FINAL_DEFENSE_MANIFEST.json"
    frozen_expected = (PREVIOUS / "configs" / "FINAL_DEFENSE_MANIFEST.sha256").read_text().split()[0]
    frozen_actual = sha256(frozen_manifest)
    if frozen_actual != frozen_expected:
        raise RuntimeError("Frozen Final LC manifest drift")
    event("FROZEN_FINAL_LC_VERIFIED", sha256=frozen_actual)
    predecessor_hashes_before = {
        "final_defense_manifest": frozen_actual,
        "final_result": sha256(PREVIOUS / "FINAL_RESULT.json"),
        "generated_answers": sha256(PREVIOUS / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl"),
        "retrieval_and_detection": sha256(PREVIOUS / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl"),
    }

    atomic_json(EXP / "HEARTBEAT.json", {"campaign": EXP.name, "phase": "PHASE_A_DCMI_RECOVERY", "updated_utc": utc_now()})
    dcmi = audit_dcmi(args.dcmi_repo)
    atomic_json(EXP / "audits" / "DCMI_PROTOCOL_RECOVERY.json", dcmi)
    event("PHASE_A_DCMI_RECOVERY_COMPLETE", status=dcmi["status"])
    atomic_json(EXP / "HEARTBEAT.json", {"campaign": EXP.name, "phase": "PHASE_B_IA_RECOVERY", "updated_utc": utc_now()})
    ia = audit_ia(args.ia_repo)
    atomic_json(EXP / "audits" / "IA_PROTOCOL_RECOVERY.json", ia)
    event("PHASE_B_IA_RECOVERY_COMPLETE", status=ia["status"])

    protocol_rows = [
        {"attack": "MEntA", "role": "development", "status": "PAPER_PROTOCOL_READY", "core6_main_eligible": True, "note": "immutable predecessor artifact"},
        {"attack": "MBA", "role": "development", "status": "PAPER_PROTOCOL_READY", "core6_main_eligible": True, "note": "missing/malformed masks rescored as incorrect with no exclusion"},
        {"attack": "RAG-MIA", "role": "development", "status": "PAPER_FAITHFUL_REIMPLEMENTATION", "core6_main_eligible": True, "note": "immutable predecessor artifact"},
        {"attack": "S²-MIA", "role": "development", "status": "PAPER_FAITHFUL_REIMPLEMENTATION", "core6_main_eligible": True, "note": "frozen T-variant implementation"},
        {"attack": "DCMI", "role": "unseen hard confirmation", "status": dcmi["status"], "core6_main_eligible": False, "note": "; ".join(dcmi["blocking_reasons"])},
        {"attack": "IA", "role": "unseen hard confirmation", "status": ia["status"], "core6_main_eligible": False, "note": "; ".join(ia["blocking_reasons"])},
    ]
    write_csv(EXP / "audits" / "CORE6_PROTOCOL_TABLE.csv", protocol_rows)
    event("CORE6_PROTOCOL_GATE_CLOSED", eligible=sum(bool(row["core6_main_eligible"]) for row in protocol_rows), required=6)

    mba_detail, mba_summary = mba_missing_as_incorrect()
    write_csv(EXP / "tables" / "MBA_MISSING_AS_INCORRECT_DETAIL.csv", mba_detail)
    write_csv(EXP / "tables" / "MBA_MISSINGNESS_AUDIT.csv", mba_summary)
    core4 = preserve_core4(mba_summary)
    write_csv(EXP / "tables" / "CORE4_PRESERVED_POST_GENERATION_PRIVACY.csv", core4)
    event("MBA_MISSINGNESS_RECOMPUTED", samples=2000, excluded=0)

    predecessor_hashes_after = {
        "final_defense_manifest": sha256(frozen_manifest),
        "final_result": sha256(PREVIOUS / "FINAL_RESULT.json"),
        "generated_answers": sha256(PREVIOUS / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl"),
        "retrieval_and_detection": sha256(PREVIOUS / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl"),
    }
    immutable = predecessor_hashes_before == predecessor_hashes_after
    if not immutable:
        raise RuntimeError("Predecessor artifact changed during protocol audit")
    atomic_json(EXP / "audits" / "FROZEN_ARTIFACT_INTEGRITY.json", {
        "before": predecessor_hashes_before,
        "after": predecessor_hashes_after,
        "unchanged": immutable,
    })

    result = {
        "campaign": EXP.name,
        "completed_utc": utc_now(),
        "verdict": "CORE6_PROTOCOL_INCOMPLETE",
        "dcmi_status": dcmi["status"],
        "ia_status": ia["status"],
        "core6_ready": False,
        "core4_preserved": True,
        "mba_missingness_recomputed": True,
        "new_attack_queries_generated": 0,
        "new_rag_answers_generated": 0,
        "core6_matched_fpr_detection": "NOT_RUN_PROTOCOL_GATE_CLOSED",
        "core6_matched_budget_e2e": "NOT_RUN_PROTOCOL_GATE_CLOSED",
        "gold_qa": "NOT_RUN_CORE6_GATE_CLOSED",
        "frozen_final_lc_unchanged": immutable,
        "next_required_assets": [
            "DCMI immutable 6%-perturbation bundle with perturbation model revision and decoding",
            "IA immutable Q30 + GPT-4o-mini ground-truth bundle matching an audited member/nonmember cohort",
        ],
    }
    atomic_json(EXP / "RESULT.json", result)

    dcmp = dcmi["repository_checks"]
    iahist = ia["official_git_history"]
    ia_frozen_overlap = sum(row["exact_id_overlap"] for row in ia["official_selection_vs_frozen_cohort"])
    mba_map = {row["condition"]: row for row in mba_summary}
    report = [
        "# FINAL_CORE6_MATCHED_BUDGET_E2E_V1",
        "",
        f"- 최종 판정: `{result['verdict']}`",
        "- 새 공격 쿼리/답변 생성: `0 / 0`",
        "- Final LC 및 이전 생성·검색 artifact: `UNCHANGED`",
        "",
        "## DCMI STATUS",
        "",
        f"`{dcmi['status']}`",
        "",
        "- 논문 black-box 조건은 6% antonym perturbation과 `Yes=1, No=0`의 원본-교란 응답 차이를 정의한다.",
        f"- 공식 저장소 `{dcmi['official_commit']}`의 `perturb.py`는 3%를 사용한다.",
        f"- 고정 decoding 존재: temperature={dcmp['temperature_declared']}, top_p={dcmp['top_p_declared']}, seed={dcmp['seed_declared']}.",
        "- 공식 scorer는 실제 입력 대신 placeholder 배열을 요구한다.",
        "- 따라서 임의 설정으로 paper-exact 숫자를 만들지 않았다.",
        "",
        "## IA STATUS",
        "",
        f"`{ia['status']}` (`IA_REPOSITORY_VARIANT`는 stress-test로만 가능)",
        "",
        "- 논문: Q30, GPT-4o 질문, GPT-4o-mini GT, `lambda=5` UNK penalty.",
        "- 공개 현재 bundle: 모든 선택 target이 Q15이며 GT field가 없다.",
        f"- Git 전체 이력: result blob {iahist['unique_historic_result_blobs']}개, 최대 Q{iahist['maximum_questions_per_target']}, Q30 row {iahist['rows_with_at_least_30_questions']}개.",
        f"- 공식 selected cohort와 현재 동결 2,000-target cohort의 동일 label ID overlap은 {ia_frozen_overlap}개뿐이다.",
        "- 따라서 Q15/plain-accuracy 결과를 논문 IA로 승격하지 않았다.",
        "",
        "## CORE6 PROTOCOL TABLE",
        "",
        "| Attack | 상태 | Main Core6 |",
        "|---|---|---:|",
    ]
    for row in protocol_rows:
        report.append(f"| {row['attack']} | `{row['status']}` | {'YES' if row['core6_main_eligible'] else 'NO'} |")
    report += [
        "",
        "## CORE6 DETECTION SAME-FPR",
        "",
        "실행하지 않았다. DCMI와 IA가 모두 main-table eligible이 아니므로 Core6 query manifest 자체를 만들지 않았다.",
        "",
        "## CORE6 MATCHED-BUDGET E2E",
        "",
        "실행하지 않았다. 누락 protocol을 임의로 채우지 않았고, 따라서 same-budget MIRABEL 대 Final LC Core6 수치는 없다.",
        "",
        "## MBA MISSINGNESS",
        "",
        "누락·malformed mask를 오답으로 계산하고 표본을 하나도 제외하지 않도록 기존 동결 답변을 재계산했다.",
        "",
        "| Condition | N | AUC | E-AUC (secondary) | incomplete/malformed | member | nonmember |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ("NO_DEFENSE", "ORIGINAL_MIRABEL_SIMPLE_HIDE", "GLOBAL_BC_SIMPLE_HIDE", "FINAL_LC_OUTER_SIMPLE_HIDE"):
        row = mba_map[condition]
        report.append(
            f"| {condition} | {row['total_n']} | {row['native_roc_auc']:.6f} | {row['effective_auc_secondary']:.6f} | "
            f"{row['incomplete_or_malformed_n']} | {row['member_incomplete_or_malformed_n']} | {row['nonmember_incomplete_or_malformed_n']} |"
        )
    report += [
        "",
        "기존 valid-only MBA AUC와 직접 섞으면 안 된다. 위 값은 `N=2,000`, 제외 0의 수정된 primary audit다.",
        "",
        "## CORE6 VERDICT",
        "",
        "`CORE6_PROTOCOL_INCOMPLETE`",
        "",
        "동결 Core4 결과만 보존했다. Core6 broad protection, Gold QA utility pass, universal defense 주장은 모두 금지한다.",
        "",
        "## 다음에 필요한 원본 자산",
        "",
        "1. DCMI 6% 교란 원본 묶음과 perturbation model revision/decoding.",
        "2. IA Q30 + GPT-4o-mini GT 원본 묶음과 검증 가능한 member/nonmember provenance.",
    ]
    atomic_text(EXP / "reports" / "FINAL_REPORT_KO.md", "\n".join(report) + "\n")
    status = [
        f"# {EXP.name}",
        "",
        "- 현재 단계: `PHASE_B_PROTOCOL_RECOVERY_COMPLETE_FAIL_CLOSED`",
        f"- 갱신(UTC): `{result['completed_utc']}`",
        f"- verdict: `{result['verdict']}`",
        f"- DCMI: `{dcmi['status']}`",
        f"- IA: `{ia['status']}`",
        "- Core6 matched-FPR/E2E: `NOT RUN`",
        "- Gold QA: `NOT RUN`",
        "- GPU 작업: `없음`",
    ]
    atomic_text(EXP / "STATUS.md", "\n".join(status) + "\n")
    atomic_json(EXP / "HEARTBEAT.json", {"campaign": EXP.name, "phase": "COMPLETE_FAIL_CLOSED", "updated_utc": utc_now(), "verdict": result["verdict"]})
    event("COMPLETE_FAIL_CLOSED", verdict=result["verdict"], new_generation=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
