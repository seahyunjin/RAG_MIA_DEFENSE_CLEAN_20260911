#!/usr/bin/env python3
"""CPU-only, artifact-only characterization of frozen Final LC.

This program intentionally performs no model loading, retrieval, answer
generation, detector training, or threshold tuning.  Its deterministic rules
are frozen in SIDECAR_PRECOMMIT.json before execution.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "FINAL_LC_CPU_ANALYSIS_SIDECAR_V1"
LC = ROOT / "experiments" / "LC_MIRABEL_LARGE_V1"
FINAL8 = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1"
CORE = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
RECAL = ROOT / "experiments" / "FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1"
CLEAN3 = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"

SEED = 20260912
BOOT = 2000
BUDGETS = (0.01, 0.025, 0.03, 0.05)
ATTACKS = ("MEntA", "MBA", "RAG-MIA", "S²-MIA", "DCMI-Std-Q2")
RANDOM_RULE = "PCG64(first64(SHA256('FINAL_LC_CPU_ANALYSIS_SIDECAR_V1|RANDOM_LC_V1|' + query_id))); choose 200 of 1000 without replacement"


def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    temporary.replace(path)


def checkpoint(stage: str, **extra) -> None:
    value = {"campaign": EXP.name, "stage": stage, "updated_utc": now(), **extra}
    write_json(EXP / "STATUS.json", value)
    (EXP / "STATUS.md").write_text(
        f"# {EXP.name}\n\n- Stage: `{stage}`\n- Updated: `{value['updated_utc']}`\n" +
        "".join(f"- {key}: `{item}`\n" for key, item in extra.items()), encoding="utf-8")


def verify_precommit() -> dict:
    pre_path = EXP / "configs" / "SIDECAR_PRECOMMIT.json"
    side_path = EXP / "configs" / "SIDECAR_PRECOMMIT.sha256"
    expected = side_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha_file(pre_path)
    if actual != expected:
        raise RuntimeError(f"precommit hash mismatch: {actual} != {expected}")
    pre = read_json(pre_path)
    for item in pre["frozen_inputs"]:
        path = Path(item["path"])
        if sha_file(path) != item["sha256"]:
            raise RuntimeError(f"frozen input drift: {path}")
    return pre


def wilson(success: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = success / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - radius, center + radius


def threshold_exact(scores: np.ndarray, budget: float) -> float:
    """Frozen rule: strict > the next descending score after floor(N*FPR)."""
    n_alarm = int(math.floor(len(scores) * budget + 1e-12))
    ordered = np.sort(np.asarray(scores, dtype=float))[::-1]
    if n_alarm >= len(ordered):
        return float("-inf")
    return float(ordered[n_alarm])


def attack_rows(final_rows: list[dict], dcmi_rows: list[dict]) -> dict[str, list[dict]]:
    result = {attack: [] for attack in ATTACKS}
    for row in final_rows:
        attack = row.get("attack")
        if attack in result:
            result[attack].append(row)
    result["DCMI-Std-Q2"] = dcmi_rows
    return result


def unit_id(row: dict) -> str:
    return str(row.get("session_id") or row.get("target_id") or row["query_id"])


def positive_evaluation_rows(attack: str, rows: list[dict]) -> list[dict]:
    """Use the exact frozen Phase-A positive cohort.

    S²-MIA has a separate 402-query scorer-reference split, which must never be
    mixed into detection evaluation.  The resulting member N is 799.
    """
    positives = [row for row in rows if row.get("membership") == "member"]
    if attack == "S²-MIA":
        positives = [row for row in positives if row.get("evaluation_split") == "S2_EVALUATION"]
    return positives


def cluster_boot_delta(rows: list[dict], mir_t: float, lc_t: float, seed: int) -> np.ndarray:
    grouped = defaultdict(list)
    for row in rows:
        grouped[unit_id(row)].append(row)
    units = sorted(grouped)
    mir = np.asarray([np.mean([float(r["M"]) > mir_t for r in grouped[u]]) for u in units])
    lc = np.asarray([np.mean([float(r["R_LC"]) > lc_t for r in grouped[u]]) for u in units])
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(units), size=(BOOT, len(units)))
    return np.mean(lc[chosen] - mir[chosen], axis=1)


def detection_statistics(benign: list[dict], by_attack: dict[str, list[dict]], pre: dict) -> None:
    checkpoint("CORE5_STATISTICS")
    primary = pre["matched_thresholds"]
    mir_t = float(primary["MIRABEL@0.025"])
    lc_t = float(primary["Final LC@0.025"])
    summary, recovered_detail = [], []
    for attack_index, attack in enumerate(ATTACKS):
        positives = positive_evaluation_rows(attack, by_attack[attack])
        mir = np.asarray([float(row["M"]) > mir_t for row in positives], dtype=bool)
        lc = np.asarray([float(row["R_LC"]) > lc_t for row in positives], dtype=bool)
        grouped = defaultdict(list)
        for index, row in enumerate(positives):
            grouped[unit_id(row)].append(index)
        recovered_units = 0
        lost_units = 0
        for uid, indices in grouped.items():
            m_any, l_any = bool(mir[indices].any()), bool(lc[indices].any())
            recovered_units += int((not m_any) and l_any)
            lost_units += int(m_any and (not l_any))
        boot = cluster_boot_delta(positives, mir_t, lc_t, SEED + attack_index)
        lo, hi = np.quantile(boot, (0.025, 0.975))
        row = {
            "attack": attack, "positive_queries": len(positives), "clusters": len(grouped),
            "mirabel_tpr": float(mir.mean()), "final_lc_tpr": float(lc.mean()),
            "delta_tpr": float(lc.mean() - mir.mean()), "delta_ci95_low": float(lo),
            "delta_ci95_high": float(hi), "bootstrap_iterations": BOOT,
            "bootstrap_unit": "session/target", "recovered_queries": int((~mir & lc).sum()),
            "lost_queries": int((mir & ~lc).sum()), "recovered_sessions_targets": recovered_units,
            "lost_sessions_targets": lost_units,
        }
        summary.append(row)
        for index, source in enumerate(positives):
            if mir[index] != lc[index]:
                recovered_detail.append({
                    "attack": attack, "query_id": source["query_id"], "unit_id": unit_id(source),
                    "mirabel_alarm": int(mir[index]), "final_lc_alarm": int(lc[index]),
                    "status": "RECOVERED" if lc[index] else "LOST",
                    "M": source["M"], "R_LC": source["R_LC"],
                })
    write_csv(EXP / "tables" / "CORE5_DETECTION_STATISTICS.csv", summary)
    write_csv(EXP / "tables" / "CORE5_RECOVERED_LOST_QUERIES.csv", recovered_detail)


def e2e_statistics() -> None:
    checkpoint("CORE5_E2E_CI")
    all_path = CORE / "tables" / "CORE5_E2E_ALL_BUDGETS.csv"
    matched_path = CORE / "tables" / "CORE5_MATCHED_BUDGET_E2E_PRIVACY.csv"
    phase = read_json(CORE / "PHASE_B_RESULT.json")
    with all_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    conditions = ("NO_DEFENSE", "MIRABEL_MATCHED_2_5", "FINAL_LC_MATCHED_2_5")
    for row in rows:
        if row["condition"] in conditions:
            selected[(row["attack"], row["condition"])] = row
    output = []
    for attack in ATTACKS:
        values = {condition: selected[(attack, condition)] for condition in conditions}
        nd = float(values["NO_DEFENSE"]["native_value"])
        mir = float(values["MIRABEL_MATCHED_2_5"]["native_value"])
        lc = float(values["FINAL_LC_MATCHED_2_5"]["native_value"])
        for condition in conditions:
            item = values[condition]
            output.append({
                "attack": attack, "condition": condition,
                "native_metric": item["native_metric"], "point_estimate": float(item["native_value"]),
                "ci95_low": float(item["ci95_low"]), "ci95_high": float(item["ci95_high"]),
                "valid_n": int(item["valid_n"]), "raw_auc": float(item["raw_auc"]),
                "e_auc_secondary": float(item["e_auc_secondary"]),
                "delta_vs_no_defense": float(item["native_value"]) - nd,
                "delta_vs_matched_mirabel": float(item["native_value"]) - mir,
                "ci_lineage": "frozen scorer target-stratified bootstrap, 2000 iterations",
            })
    # Cross-artifact arithmetic verification, not merely copying one table.
    with matched_path.open(encoding="utf-8", newline="") as handle:
        matched = {row["attack"]: row for row in csv.DictReader(handle)}
    phase_map = {row["attack"]: row for row in phase["primary"]}
    checks = []
    for attack in ATTACKS:
        lc = next(r for r in output if r["attack"] == attack and r["condition"] == "FINAL_LC_MATCHED_2_5")
        checks.append({
            "attack": attack,
            "all_budget_vs_matched_table": abs(lc["point_estimate"] - float(matched[attack]["final_lc_matched_2_5"])) < 1e-12,
            "matched_table_vs_phase_result": abs(lc["point_estimate"] - float(phase_map[attack]["final_lc_matched_2_5"])) < 1e-12,
            "delta_arithmetic_valid": abs((lc["point_estimate"] - float(matched[attack]["mirabel_matched_2_5"])) - float(matched[attack]["delta_lc_vs_mirabel"])) < 1e-12,
        })
    write_csv(EXP / "tables" / "CORE5_E2E_NATIVE_CI.csv", output)
    write_json(EXP / "audits" / "CORE5_E2E_LINEAGE_VERIFICATION.json", {
        "status": "VERIFIED" if all(all(v for k, v in row.items() if k != "attack") for row in checks) else "MISMATCH",
        "checks": checks,
        "note": "Point estimates and frozen 2000-bootstrap CIs were cross-checked across the full scorer table, matched-budget table, and PHASE_B_RESULT. No answers or model were regenerated.",
        "input_hashes": {str(p): sha_file(p) for p in (all_path, matched_path, CORE / "PHASE_B_RESULT.json")},
    })


def benign_fpr_and_ablation(benign: list[dict], by_attack: dict[str, list[dict]], pre: dict) -> None:
    checkpoint("DOMAIN_FPR_AND_ABLATION")
    fixed = {
        "Original MIRABEL": ("M", 0.0),
        "Global BC": ("R_GLOBAL", float(pre["fixed_thresholds"]["Global BC"])),
        "Final LC": ("R_LC", float(pre["fixed_thresholds"]["Final LC"])),
    }
    fpr_rows = []
    domains = sorted({row["domain"] for row in benign})
    for method, (key, threshold) in fixed.items():
        for domain in ["ALL", *domains]:
            subset = benign if domain == "ALL" else [row for row in benign if row["domain"] == domain]
            hits = sum(float(row[key]) > threshold for row in subset)
            lo, hi = wilson(hits, len(subset))
            fpr_rows.append({"method": method, "domain": domain, "n": len(subset), "alarms": hits,
                             "fpr": hits / len(subset), "wilson95_low": lo, "wilson95_high": hi,
                             "threshold": threshold, "operator": "strict >"})
    for method in fixed:
        group = [row for row in fpr_rows if row["method"] == method and row["domain"] != "ALL"]
        worst = max(group, key=lambda row: row["fpr"])
        fpr_rows.append({"method": method, "domain": "WORST_DOMAIN", "n": worst["n"],
                         "alarms": worst["alarms"], "fpr": worst["fpr"],
                         "wilson95_low": worst["wilson95_low"], "wilson95_high": worst["wilson95_high"],
                         "threshold": worst["threshold"], "operator": "strict >", "worst_domain": worst["domain"]})
    write_csv(EXP / "tables" / "BENIGN_DOMAIN_FPR_WILSON.csv", fpr_rows)

    methods = {
        "Raw MIRABEL": "M",
        "Global benign calibration": "R_GLOBAL",
        "LC semantic-local calibration": "R_LC",
        "Final LC / outer deployment ranking": "R_LC",
    }
    ablation = []
    for method, key in methods.items():
        b_scores = np.asarray([float(row[key]) for row in benign])
        for budget in BUDGETS:
            threshold = threshold_exact(b_scores, budget)
            actual = float(np.mean(b_scores > threshold))
            for attack in ATTACKS:
                positives = positive_evaluation_rows(attack, by_attack[attack])
                tpr = float(np.mean([float(row[key]) > threshold for row in positives]))
                ablation.append({"method": method, "score": key, "nominal_fpr": budget,
                                 "actual_benign_fpr": actual, "threshold": threshold,
                                 "attack": attack, "positive_queries": len(positives), "tpr": tpr,
                                 "note": "Final LC and semantic-local LC share R_LC ranking; outer threshold affects fixed deployment, not same-FPR ordering." if method.startswith("Final LC") else ""})
    write_csv(EXP / "tables" / "METHOD_ABLATION_LOW_FPR.csv", ablation)


def random_neighbor_control(benign: list[dict], by_attack: dict[str, list[dict]]) -> None:
    checkpoint("RANDOM_NEIGHBOR_CONTROL")
    large_path = LC / "cache" / "LARGE_DETECTION_SCORES.jsonl"
    reference = [row for row in read_jsonl(large_path) if row.get("split") == "REFERENCE"]
    if len(reference) != 1000:
        raise RuntimeError(f"reference N drift: {len(reference)}")
    ref_scores = np.asarray([float(row["M"]) for row in reference])
    all_rows = list(benign)
    for attack in ATTACKS:
        all_rows.extend(by_attack[attack])
    random_score = {}
    for row in all_rows:
        query_id = str(row["query_id"])
        seed = int(sha_text("FINAL_LC_CPU_ANALYSIS_SIDECAR_V1|RANDOM_LC_V1|" + query_id)[:16], 16)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(ref_scores), 200, replace=False)
        p = (1 + int(np.sum(ref_scores[chosen] >= float(row["M"])))) / 201
        random_score[query_id] = -math.log(p)
    b_random = np.asarray([random_score[row["query_id"]] for row in benign])
    random_rows = []
    for budget in BUDGETS:
        threshold = threshold_exact(b_random, budget)
        actual = float(np.mean(b_random > threshold))
        for attack in ATTACKS:
            positives = positive_evaluation_rows(attack, by_attack[attack])
            tpr = float(np.mean([random_score[row["query_id"]] > threshold for row in positives]))
            random_rows.append({"method": "Random-neighbor LC", "nominal_fpr": budget,
                                "actual_benign_fpr": actual, "threshold": threshold,
                                "attack": attack, "tpr": tpr, "reference_n": 1000,
                                "k": 200, "selection_rule": RANDOM_RULE})
    write_csv(EXP / "tables" / "RANDOM_NEIGHBOR_LOW_FPR.csv", random_rows)

    correlation = []
    for attack in ["BENIGN", *ATTACKS]:
        rows = benign if attack == "BENIGN" else by_attack[attack]
        semantic = np.asarray([float(row["R_LC"]) for row in rows])
        random = np.asarray([random_score[row["query_id"]] for row in rows])
        pearson = float(np.corrcoef(semantic, random)[0, 1]) if np.std(semantic) and np.std(random) else math.nan
        # Average-rank Spearman without scipy.
        def ranks(x):
            order = np.argsort(x, kind="stable")
            out = np.empty(len(x), dtype=float)
            start = 0
            while start < len(x):
                end = start + 1
                while end < len(x) and x[order[end]] == x[order[start]]:
                    end += 1
                out[order[start:end]] = (start + end - 1) / 2
                start = end
            return out
        sr, rr = ranks(semantic), ranks(random)
        spearman = float(np.corrcoef(sr, rr)[0, 1]) if np.std(sr) and np.std(rr) else math.nan
        correlation.append({"cohort": attack, "n": len(rows), "pearson": pearson, "spearman": spearman})
    write_csv(EXP / "tables" / "RANDOM_NEIGHBOR_SCORE_CORRELATION.csv", correlation)

    budget = 0.025
    sem_t = threshold_exact(np.asarray([float(row["R_LC"]) for row in benign]), budget)
    rnd_t = threshold_exact(b_random, budget)
    recovery = []
    for attack in ("MEntA", "RAG-MIA"):
        positives = positive_evaluation_rows(attack, by_attack[attack])
        semantic = np.asarray([float(row["R_LC"]) > sem_t for row in positives])
        random = np.asarray([random_score[row["query_id"]] > rnd_t for row in positives])
        recovery.append({"attack": attack, "semantic_tpr": float(semantic.mean()),
                         "random_tpr": float(random.mean()), "semantic_only_recovered": int((semantic & ~random).sum()),
                         "random_only_recovered": int((random & ~semantic).sum()), "positive_queries": len(positives)})
    write_csv(EXP / "tables" / "RANDOM_NEIGHBOR_RECOVERY.csv", recovery)

    domain = []
    rnd_t = threshold_exact(b_random, 0.025)
    for name in sorted({row["domain"] for row in benign}):
        subset = [row for row in benign if row["domain"] == name]
        hits = sum(random_score[row["query_id"]] > rnd_t for row in subset)
        lo, hi = wilson(hits, len(subset))
        domain.append({"domain": name, "n": len(subset), "alarms": hits, "fpr": hits / len(subset),
                       "wilson95_low": lo, "wilson95_high": hi, "threshold": rnd_t})
    write_csv(EXP / "tables" / "RANDOM_NEIGHBOR_DOMAIN_FPR.csv", domain)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def is_refusal(answer: str) -> bool:
    # Exact frozen Gold scorer definition; do not broaden after observing data.
    value = " ".join(answer.casefold().split())
    return any(marker in value for marker in
               ("i don't know", "i do not know", "cannot determine",
                "insufficient information", "not enough information"))


def gold_and_fp_audit() -> None:
    checkpoint("GOLD_STRICT_VS_REFRESH_AND_FP_AUDIT")
    strict_detail = list(read_jsonl(CORE / "runtime" / "GOLD_QA_DETAIL.jsonl"))
    strict_summary = {row["condition"]: row for row in read_json(CORE / "PHASE_C_RESULT.json")["summary"]}
    refresh_result = read_json(RECAL / "GOLD_RECALIBRATION_RESULT.json")
    refresh_answers = list(read_jsonl(RECAL / "runtime" / "GOLD_REFRESHED_ANSWERS.jsonl"))
    refresh_by_id = {row["query_id"]: row for row in refresh_answers}
    no_defense = [row for row in strict_detail if row["condition"] == "NO_DEFENSE"]
    strict_lc = [row for row in strict_detail if row["condition"] == "FINAL_LC_FIXED"]
    def avg_len(rows):
        return statistics.fmean(len(re.findall(r"\S+", row["answer"])) for row in rows)
    nd_refusal = {row["query_id"]: is_refusal(row["answer"]) for row in no_defense}
    gold_rows = [
        {"condition": "No Defense", "n": 1000, "intervention": 0.0,
         "gold_f1": strict_summary["NO_DEFENSE"]["gold_f1"], "delta_f1": 0.0,
         "refusal": statistics.fmean(is_refusal(row["answer"]) for row in no_defense), "new_refusal": 0.0,
         "empty": statistics.fmean(not row["answer"].strip() for row in no_defense), "mean_answer_words": avg_len(no_defense)},
        {"condition": "Final LC strict transfer", "n": 1000,
         "intervention": strict_summary["FINAL_LC_FIXED"]["intervention_rate"],
         "gold_f1": strict_summary["FINAL_LC_FIXED"]["gold_f1"],
         "delta_f1": strict_summary["FINAL_LC_FIXED"]["gold_f1"] - strict_summary["NO_DEFENSE"]["gold_f1"],
         "refusal": statistics.fmean(is_refusal(row["answer"]) for row in strict_lc),
         "new_refusal": statistics.fmean(is_refusal(row["answer"]) and not nd_refusal[row["query_id"]] for row in strict_lc),
         "empty": statistics.fmean(not row["answer"].strip() for row in strict_lc), "mean_answer_words": avg_len(strict_lc)},
        {"condition": "Final LC benign refresh", "n": 1000,
         "intervention": refresh_result["locked_intervention"], "gold_f1": refresh_result["refreshed_gold_f1"],
         "delta_f1": refresh_result["gold_f1_delta"],
         "refusal": statistics.fmean(is_refusal(row["answer"]) for row in refresh_answers),
         "new_refusal": statistics.fmean(is_refusal(row["answer"]) and not nd_refusal[row["query_id"]] for row in refresh_answers),
         "empty": statistics.fmean(bool(row["empty"]) for row in refresh_answers),
         "mean_answer_words": statistics.fmean(int(row["answer_words"]) for row in refresh_answers)},
    ]
    write_csv(EXP / "tables" / "GOLD_STRICT_VS_REFRESH.csv", gold_rows)
    lineage_paths = [CORE / "PHASE_C_RESULT.json", CORE / "runtime" / "GOLD_QA_DETAIL.jsonl",
                     RECAL / "GOLD_RECALIBRATION_RESULT.json", RECAL / "runtime" / "GOLD_REFRESHED_ANSWERS.jsonl"]
    write_json(EXP / "audits" / "GOLD_LINEAGE_HASHES.json", {str(path): sha_file(path) for path in lineage_paths})

    eval_map = {row["query_id"]: row for row in read_jsonl(CORE / "inputs" / "TOPIOCQA_GOLD_EVAL_1000.jsonl")}
    corpus_map = {row["document_id"]: row for row in read_jsonl(CORE / "inputs" / "TOPIOCQA_GOLD_CORPUS.jsonl")}
    scores = {row["query_id"]: row for row in read_jsonl(RECAL / "cache" / "GOLD_REFRESHED_SCORES.jsonl")}
    nd_map = {row["query_id"]: row for row in no_defense}
    fp = [row for row in refresh_answers if row["intervened"]]
    if len(fp) != 27:
        raise RuntimeError(f"refreshed FP count drift: {len(fp)}")
    detailed = []
    for answer in sorted(fp, key=lambda row: row["query_id"]):
        qid = answer["query_id"]
        item, score = eval_map[qid], scores[qid]
        top_ids = list(score["top_document_ids"])
        removed = str(answer["removed_source"])
        remaining = [doc for doc in top_ids if doc != removed]
        gold_answers = [str(x) for x in item["gold_answers"]]
        gold_norm = [normalize_text(x) for x in gold_answers if normalize_text(x)]
        def present(doc_id: str) -> bool:
            if doc_id not in corpus_map:
                return False
            text = normalize_text(corpus_map[doc_id].get("source_text") or
                                  (corpus_map[doc_id].get("title", "") + " " + corpus_map[doc_id].get("text", "")))
            return any(value in text for value in gold_norm)
        removed_present = present(removed)
        remaining_hits = [doc for doc in remaining if present(doc)]
        refusal = is_refusal(answer["answer"]) or bool(answer["empty"])
        if refusal:
            category = "REFUSAL_DRIVEN"
        elif removed_present and not remaining_hits:
            category = "UNIQUE_EVIDENCE_REMOVED"
        elif remaining_hits:
            category = "REDUNDANT_EVIDENCE_AVAILABLE"
        else:
            category = "UNRESOLVED"
        nd = nd_map[qid]
        detailed.append({
            "query_id": qid, "query": item["query"], "gold_answers_json": json.dumps(gold_answers, ensure_ascii=False),
            "gold_document_id": item.get("gold_document_id"), "top_document_ids_json": json.dumps(top_ids),
            "removed_source_id": removed, "removed_rank": top_ids.index(removed) + 1 if removed in top_ids else 0,
            "no_defense_answer": nd["answer"], "defended_answer": answer["answer"],
            "gold_f1_before": float(nd["gold_f1"]), "gold_f1_after": float(answer["gold_f1"]),
            "delta_gold_f1": float(answer["gold_f1"]) - float(nd["gold_f1"]),
            "refusal": refusal, "empty": bool(answer["empty"]),
            "answer_words_before": len(re.findall(r"\S+", nd["answer"])),
            "answer_words_after": int(answer["answer_words"]),
            "answer_length_change": int(answer["answer_words"]) - len(re.findall(r"\S+", nd["answer"])),
            "gold_text_in_removed_source": removed_present,
            "gold_text_in_any_remaining_top3": bool(remaining_hits),
            "gold_text_only_in_removed_source": bool(removed_present and not remaining_hits),
            "gold_text_in_multiple_remaining_sources": len(remaining_hits) >= 2,
            "remaining_evidence_source_ids_json": json.dumps(remaining_hits),
            "removed_is_qrels_gold": removed == item.get("gold_document_id"),
            "qrels_gold_in_top4": item.get("gold_document_id") in top_ids,
            "harm_class": category,
            "audit_boundary": "exact normalized gold-answer substring only; no semantic/NLI inference",
        })
    write_csv(EXP / "tables" / "FP_27_CASE_DAMAGE_AUDIT.csv", detailed)
    mechanism = []
    for category in ("UNIQUE_EVIDENCE_REMOVED", "REDUNDANT_EVIDENCE_AVAILABLE", "REFUSAL_DRIVEN", "UNRESOLVED"):
        group = [row for row in detailed if row["harm_class"] == category]
        mechanism.append({"harm_class": category, "count": len(group),
                          "mean_delta_gold_f1": statistics.fmean(row["delta_gold_f1"] for row in group) if group else math.nan})
    write_csv(EXP / "tables" / "FP_HARM_MECHANISM.csv", mechanism)


def db_churn_manifests() -> None:
    checkpoint("DB_CHURN_MANIFEST")
    manifest = read_json(CLEAN3 / "manifests" / "CLEAN_CORE3_DB_MANIFEST.json")
    db_rows = list(read_jsonl(CLEAN3 / "inputs" / "CLEAN_CORE3_PROTECTED_DB.jsonl"))
    db_by_domain = defaultdict(list)
    for row in db_rows:
        db_by_domain[row["domain"]].append(row)
    member = set(manifest["member_target_ids"])
    nonmember = set(manifest["nonmember_target_ids"])
    benign_gold = set()
    for filename in ("BENIGN_CALIBRATION.jsonl", "BENIGN_HOLDOUT.jsonl"):
        for row in read_jsonl(CLEAN3 / "inputs" / filename):
            benign_gold.add(str(row["primary_gold_document_id"]))
    protected = member | benign_gold
    original_ids = {row["document_id"] for row in db_rows}
    original_hashes = {row["normalized_text_hash"] for row in db_rows}
    target_ids = member | nonmember
    candidates = defaultdict(list)
    with gzip.open(CLEAN3 / "inputs" / "COMMON_ELIGIBLE_POOL.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["document_id"] in original_ids or row["document_id"] in target_ids or row["normalized_text_hash"] in original_hashes:
                continue
            candidates[row["domain"]].append(row)
    versions = []
    for label, fraction in (("V0", 0.0), ("V10", 0.10), ("V25", 0.25), ("V50", 0.50)):
        removed, added, final_rows = [], [], []
        for domain in sorted(db_by_domain):
            rows = db_by_domain[domain]
            background = [row for row in rows if row["document_id"] not in protected]
            count = int(round(len(background) * fraction))
            remove = sorted(background, key=lambda row: sha_text(f"{EXP.name}|{label}|REMOVE|{row['document_id']}"))[:count]
            remove_ids = {row["document_id"] for row in remove}
            available = sorted(candidates[domain], key=lambda row: sha_text(f"{EXP.name}|{label}|ADD|{row['document_id']}|{row['normalized_text_hash']}"))
            add = available[:count]
            if len(add) != count:
                raise RuntimeError(f"insufficient churn pool {domain} {label}")
            removed.extend({"document_id": row["document_id"], "domain": domain,
                            "normalized_text_hash": row["normalized_text_hash"]} for row in remove)
            added.extend({"document_id": row["document_id"], "domain": domain,
                          "normalized_text_hash": row["normalized_text_hash"]} for row in add)
            final_rows.extend({"document_id": row["document_id"], "domain": domain,
                               "normalized_text_hash": row["normalized_text_hash"]} for row in rows if row["document_id"] not in remove_ids)
            final_rows.extend({"document_id": row["document_id"], "domain": domain,
                               "normalized_text_hash": row["normalized_text_hash"]} for row in add)
        final_rows.sort(key=lambda row: (row["domain"], row["document_id"]))
        final_ids = {row["document_id"] for row in final_rows}
        ordered_ids = [row["document_id"] for row in final_rows]
        ordered_hashes = [row["normalized_text_hash"] for row in final_rows]
        out = {
            "campaign": EXP.name, "version": label, "background_replace_fraction": fraction,
            "construction_rule": "replace a deterministic SHA256-ranked fraction of background documents per domain; protect member targets and all benign primary-qrels documents",
            "documents": len(final_rows), "domain_counts": dict(Counter(row["domain"] for row in final_rows)),
            "protected_ids_count": len(protected), "removed_count": len(removed), "added_count": len(added),
            "removed": removed, "added": added, "ordered_document_ids": ordered_ids,
            "ordered_document_id_sha256": sha_text("\n".join(ordered_ids)),
            "ordered_normalized_text_hash_sha256": sha_text("\n".join(ordered_hashes)),
            "membership_audit": {"all_member_targets_present": member <= final_ids,
                                  "nonmember_targets_present": sorted(nonmember & final_ids),
                                  "member_count": len(member & final_ids), "nonmember_count": len(nonmember & final_ids)},
            "overlap_audit": {"added_overlap_original_ids": len({x["document_id"] for x in added} & original_ids),
                              "added_overlap_original_text_hashes": len({x["normalized_text_hash"] for x in added} & original_hashes),
                              "added_removed_id_overlap": len({x["document_id"] for x in added} & {x["document_id"] for x in removed})},
            "embedding_status": "NOT_BUILT_MANIFEST_ONLY",
        }
        path = EXP / "manifests" / f"DB_CHURN_{label}.json"
        write_json(path, out)
        versions.append({"version": label, "documents": len(final_rows), "removed": len(removed), "added": len(added),
                         "member_ok": out["membership_audit"]["all_member_targets_present"],
                         "nonmember_present": len(out["membership_audit"]["nonmember_targets_present"]),
                         "ordered_id_sha256": out["ordered_document_id_sha256"], "manifest_sha256": sha_file(path)})
    write_csv(EXP / "tables" / "DB_CHURN_MANIFEST_SUMMARY.csv", versions)


def readiness_and_protocol() -> None:
    checkpoint("READINESS_AND_PROTOCOL_AUDIT")
    hf = Path("/home/traffic_3/workspace/.cache/huggingface/hub")
    candidates = [
        {"dataset": "TopiOCQA", "local": True, "rag_corpus": True, "benign_gold_qa": True,
         "member_nonmember_feasible": "PENDING_FORMAL_AUDIT", "development_overlap": "YES",
         "eligibility": "INELIGIBLE_USED_FOR_GOLD_RECALIBRATION"},
        {"dataset": "QuAC", "local": (hf / "datasets--allenai--quac").exists(), "rag_corpus": True,
         "benign_gold_qa": True, "member_nonmember_feasible": "LIKELY", "development_overlap": "YES_HISTORICAL_PROJECT_USE",
         "eligibility": "INELIGIBLE_NOT_UNTOUCHED"},
        {"dataset": "QReCC", "local": (hf / "datasets--slupart--qrecc").exists(), "rag_corpus": (hf / "datasets--slupart--qrecc-passages").exists(),
         "benign_gold_qa": True, "member_nonmember_feasible": "LIKELY", "development_overlap": "YES_HISTORICAL_PROJECT_USE",
         "eligibility": "INELIGIBLE_NOT_UNTOUCHED"},
        {"dataset": "PubMedQA", "local": (hf / "datasets--qiaojin--PUBMEDQA").exists(), "rag_corpus": "REQUIRES_IMMUTABLE_CORPUS_CONSTRUCTION",
         "benign_gold_qa": True, "member_nonmember_feasible": "PENDING", "development_overlap": "NO_MATCH_FOUND_IN_CLEAN_CAMPAIGN",
         "eligibility": "CANDIDATE_PENDING_DATA_AND_PROVENANCE"},
        {"dataset": "HotpotQA", "local": any(hf.glob("datasets--*hotpot*")), "rag_corpus": "REQUIRES_WIKIPEDIA_SNAPSHOT",
         "benign_gold_qa": True, "member_nonmember_feasible": "PENDING", "development_overlap": "NO_MATCH_FOUND_IN_CLEAN_CAMPAIGN",
         "eligibility": "CANDIDATE_PENDING_DATA_AND_PROVENANCE"},
    ]
    write_csv(EXP / "tables" / "NEW_DOMAIN_CANDIDATE_AUDIT.csv", candidates)
    write_json(EXP / "audits" / "NEW_DOMAIN_READINESS.json", {
        "status": "NO_READY_UNTOUCHED_DOMAIN_LOCAL_INPUT",
        "selection_made": False, "answer_generation": 0, "attack_examples_used": 0,
        "note": "This stage audits candidates only. PubMedQA/HotpotQA remain candidates, not selected blind datasets, until immutable corpus and overlap hashes exist.",
        "candidates": candidates,
    })

    protocol_source = FINAL8 / "reports" / "PHASE_A_PROTOCOL_AUDIT_KO.md"
    protocols = [
        {"attack": "RAGLeak", "status": "UNAVAILABLE", "target_member_definition": "member target text is present in protected RAG database; nonmember counterpart absent",
         "query_protocol": "paper uses target crops as query/reference components; exact immutable crop/prompt bundle not recovered",
         "scorer": "black-box response/reference similarity and gray-box likelihood variants reported historically; exact author scorer unavailable",
         "native_metric": "paper-native attack success/accuracy variants; exact frozen implementation unavailable",
         "generation_requirements": "fresh RAG answer under exact crop prompt", "output_budgets": "not recovered as immutable bundle",
         "missing_artifact": "exact crops, prompt, scorer implementation, immutable target pairing"},
        {"attack": "BudgetLeak", "status": "UNAVAILABLE", "target_member_definition": "paired QA target in/out of RAG database",
         "query_protocol": "repeat same QA query across multiple output budgets and classify the score sequence",
         "scorer": "author multi-budget sequence scorer (including semantic/lexical variants) not recovered",
         "native_metric": "author attack AUC/accuracy/TPR operating points; exact frozen scorer unavailable",
         "generation_requirements": "fresh outputs at budgets 10..270 step 20 for paired QA/reference targets",
         "output_budgets": "10,30,50,...,270 (14 budgets; historical audit)",
         "missing_artifact": "author scorer and paired QA/reference bundle on current Core5 target universe"},
    ]
    write_csv(EXP / "tables" / "RAGLEAK_BUDGETLEAK_PROTOCOL_AUDIT.csv", protocols)
    write_json(EXP / "audits" / "PROTOCOL_AUDIT_PROVENANCE.json", {
        "source": str(protocol_source), "source_sha256": sha_file(protocol_source),
        "boundary": "No Final LC performance is attached to these unavailable protocols; historical substitutes are not mixed into Core5.",
        "protocols": protocols,
    })


def cost_latency(started: float) -> None:
    checkpoint("COST_LATENCY")
    phase_a = read_json(CORE / "PHASE_A_RESULT.json")["runtime"]
    final_manifest = read_json(FINAL8 / "runtime" / "FINAL_GENERATION_MANIFEST.json")
    branch_manifest = read_json(CORE / "runtime" / "PHASE_B_GENERATION_MANIFEST.json")
    generation_paths = [FINAL8 / "runtime" / "FINAL_GENERATED_ANSWERS.jsonl",
                        CORE / "runtime" / "STANDARDIZED_BRANCH_ANSWERS.jsonl"]
    physical_wall = 0.0
    physical_rows = 0
    for path in generation_paths:
        for row in read_jsonl(path):
            if float(row.get("wall_seconds") or 0) > 0:
                physical_wall += float(row["wall_seconds"])
                physical_rows += 1
    cache_paths = [FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl",
                   CORE / "cache" / "CORE5_DCMI_RETRIEVAL_AND_DETECTION.jsonl",
                   LC / "cache" / "QUERY_EMBEDDINGS.float16.npy",
                   CORE / "cache" / "STANDARDIZED_QUERY_EMBEDDINGS.float16.npy"]
    rows = [
        {"metric": "DCMI encoding_and_scoring_seconds", "value": phase_a["encoding_and_scoring_seconds"], "unit": "seconds", "scope": "4000 new queries; includes encoder + LC scoring"},
        {"metric": "DCMI retrieval_seconds", "value": phase_a["retrieval_seconds"], "unit": "seconds", "scope": "4000 new queries"},
        {"metric": "detector_pipeline_per_query", "value": (phase_a["encoding_and_scoring_seconds"] + phase_a["retrieval_seconds"]) / phase_a["new_queries"], "unit": "seconds/query", "scope": "BGE encode + retrieval + MIRABEL/LC scoring; lookup alone not separately instrumented"},
        {"metric": "LC_kNN_lookup_time", "value": "NOT_SEPARATELY_INSTRUMENTED", "unit": "", "scope": "contained in encoding_and_scoring_seconds"},
        {"metric": "MIRABEL_scoring_time", "value": "NOT_SEPARATELY_INSTRUMENTED", "unit": "", "scope": "contained in encoding_and_scoring_seconds"},
        {"metric": "frozen_cache_size", "value": sum(path.stat().st_size for path in cache_paths), "unit": "bytes", "scope": "two detection caches + two query-embedding arrays"},
        {"metric": "Final8_logical_answers", "value": final_manifest["logical_rows"], "unit": "rows", "scope": "four conditions"},
        {"metric": "Final8_physical_generations", "value": final_manifest["physical_generations"], "unit": "rows", "scope": "Qwen"},
        {"metric": "Final8_cache_hit_rate", "value": final_manifest["exact_reuses"] / final_manifest["logical_rows"], "unit": "fraction", "scope": "exact answer reuse"},
        {"metric": "DCMI_branch_answers", "value": branch_manifest["answer_rows"], "unit": "rows", "scope": "A0 + required A_HIDE"},
        {"metric": "observed_unique_generation_wall", "value": physical_wall, "unit": "GPU-seconds", "scope": f"sum of positive wall_seconds over {physical_rows} stored answer rows"},
        {"metric": "observed_unique_generation_GPU_hours", "value": physical_wall / 3600, "unit": "GPU-hours", "scope": "lower-bound/accounted artifact time, not whole-project historical total"},
        {"metric": "training_steps", "value": 0, "unit": "steps", "scope": "Final LC"},
        {"metric": "trainable_parameter_updates", "value": 0, "unit": "updates", "scope": "Final LC"},
        {"metric": "sidecar_runtime", "value": time.monotonic() - started, "unit": "seconds", "scope": "CPU artifact analysis"},
    ]
    write_csv(EXP / "tables" / "COST_LATENCY.csv", rows)


def render_report(pre: dict) -> None:
    def rows(name):
        with (EXP / "tables" / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    detection = rows("CORE5_DETECTION_STATISTICS.csv")
    e2e = [r for r in rows("CORE5_E2E_NATIVE_CI.csv") if r["condition"] in ("NO_DEFENSE", "MIRABEL_MATCHED_2_5", "FINAL_LC_MATCHED_2_5")]
    domain = rows("BENIGN_DOMAIN_FPR_WILSON.csv")
    random = rows("RANDOM_NEIGHBOR_RECOVERY.csv")
    gold = rows("GOLD_STRICT_VS_REFRESH.csv")
    harm = rows("FP_HARM_MECHANISM.csv")
    churn = rows("DB_CHURN_MANIFEST_SUMMARY.csv")
    protocol = rows("RAGLEAK_BUDGETLEAK_PROTOCOL_AUDIT.csv")
    lines = [
        "# Final LC CPU-only Analysis Sidecar", "",
        f"- Completed UTC: `{now()}`", f"- Precommit SHA-256: `{sha_file(EXP/'configs'/'SIDECAR_PRECOMMIT.json')}`",
        "- IA GPU generation was not stopped, signalled, or modified.",
        "- No GPU/model/retrieval/generation/training was used by this sidecar.", "",
        "## CORE5 STATISTICS", "",
        "| Attack | MIRABEL TPR | Final LC TPR | Delta (95% cluster CI) | Recovered/lost queries | Recovered/lost units |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in detection:
        lines.append(f"| {r['attack']} | {float(r['mirabel_tpr']):.4f} | {float(r['final_lc_tpr']):.4f} | {float(r['delta_tpr']):+.4f} [{float(r['delta_ci95_low']):+.4f}, {float(r['delta_ci95_high']):+.4f}] | {r['recovered_queries']}/{r['lost_queries']} | {r['recovered_sessions_targets']}/{r['lost_sessions_targets']} |")
    lines += ["", "## CORE5 E2E CI", "",
              "Native metrics are primary. E-AUC is retained only as a secondary diagnostic where defined.", "",
              "| Attack | Condition | Native point | 95% CI | Delta vs No Defense | Delta vs matched MIRABEL |",
              "|---|---|---:|---:|---:|---:|"]
    for r in e2e:
        lines.append(f"| {r['attack']} | {r['condition']} | {float(r['point_estimate']):.4f} | [{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}] | {float(r['delta_vs_no_defense']):+.4f} | {float(r['delta_vs_matched_mirabel']):+.4f} |")
    lines += ["", "## DOMAIN FPR", "", "| Method | Domain | FPR | Wilson 95% CI |", "|---|---|---:|---:|"]
    for r in domain:
        lines.append(f"| {r['method']} | {r['domain']} | {float(r['fpr']):.4f} | [{float(r['wilson95_low']):.4f}, {float(r['wilson95_high']):.4f}] |")
    lines += ["", "## ABLATION", "",
              "Low-FPR curves for Raw MIRABEL, global benign calibration, semantic-local LC, Final LC, and random-neighbor LC are saved as CSV. Final LC and LC have identical same-FPR ranking because both use R_LC; the outer threshold only defines the fixed deployment point.", "",
              "Random-neighbor semantic recovery check:", "",
              "| Attack | Semantic TPR | Random TPR | Semantic-only recovered |", "|---|---:|---:|---:|"]
    for r in random:
        lines.append(f"| {r['attack']} | {float(r['semantic_tpr']):.4f} | {float(r['random_tpr']):.4f} | {r['semantic_only_recovered']} |")
    lines += ["", "Optional k sensitivity was not run: S² query embeddings are not frozen, and recomputing/model-loading was forbidden. The current k=200 was not reselected.",
              "", "## GOLD STRICT VS REFRESH", "", "| Condition | Intervention | Gold F1 | Delta F1 | Refusal | New refusal | Empty | Mean words |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in gold:
        lines.append(f"| {r['condition']} | {float(r['intervention']):.3f} | {float(r['gold_f1']):.4f} | {float(r['delta_f1']):+.4f} | {float(r['refusal']):.3f} | {float(r['new_refusal']):.3f} | {float(r['empty']):.3f} | {float(r['mean_answer_words']):.1f} |")
    lines += ["", "## FP 27 DAMAGE AUDIT", "",
              "All 27 refreshed false-positive interventions are in `tables/FP_27_CASE_DAMAGE_AUDIT.csv`, including answers, rank, qrels, score damage, refusal, and exact lexical evidence flags.", "",
              "## FP HARM MECHANISM", "", "| Class | N | Mean Delta Gold F1 |", "|---|---:|---:|"]
    for r in harm:
        lines.append(f"| {r['harm_class']} | {r['count']} | {float(r['mean_delta_gold_f1']):+.4f} |")
    lines += ["", "This is an exact normalized-string diagnostic, not semantic evidence or NLI.",
              "", "## DB CHURN MANIFEST", "", "| Version | Removed/added | Member audit | Nonmember present | Manifest hash |", "|---|---:|---:|---:|---|"]
    for r in churn:
        lines.append(f"| {r['version']} | {r['removed']}/{r['added']} | {r['member_ok']} | {r['nonmember_present']} | `{r['manifest_sha256']}` |")
    lines += ["", "Only deterministic manifests were created; no index or embeddings were built.",
              "", "## NEW DOMAIN READINESS", "",
              "No locally complete untouched-domain substrate is READY. PubMedQA and HotpotQA remain candidates pending immutable corpus construction and formal overlap hashes; TopiOCQA/QuAC/QReCC are not untouched.",
              "", "## RAGLEAK/BUDGETLEAK STATUS", "", "| Attack | Status | Missing critical artifact |", "|---|---|---|"]
    for r in protocol:
        lines.append(f"| {r['attack']} | {r['status']} | {r['missing_artifact']} |")
    lines += ["", "Neither unavailable protocol is mixed with Final LC Core5 results.",
              "", "## COST/LATENCY", "",
              "See `tables/COST_LATENCY.csv`. Per-query detector timing is reported only for the instrumented combined BGE encoding + retrieval + MIRABEL/LC scoring path; kNN and MIRABEL subcomponents were not separately timed.",
              "", "## Scientific boundary", "",
              "This sidecar confirms frozen Core5 statistics and diagnoses calibration/generalization preparation. It does not add IA, RAGLeak, BudgetLeak, a new domain result, or a new detector. Those claims remain pending their protocol/input gates."]
    report = EXP / "reports" / "FINAL_LC_CPU_ANALYSIS_SIDECAR_REPORT_KO.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.monotonic()
    # Force CPU libraries to stay small even if caller forgot the environment.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4"
    pre = verify_precommit()
    checkpoint("INPUT_LOADING", precommit_sha256=sha_file(EXP / "configs" / "SIDECAR_PRECOMMIT.json"))
    final_rows = list(read_jsonl(FINAL8 / "cache" / "FINAL_RETRIEVAL_AND_DETECTION.jsonl"))
    dcmi_rows = list(read_jsonl(CORE / "cache" / "CORE5_DCMI_RETRIEVAL_AND_DETECTION.jsonl"))
    benign = [row for row in final_rows if row.get("attack") == "BENIGN"]
    if len(benign) != 1000:
        raise RuntimeError(f"benign N drift: {len(benign)}")
    by_attack = attack_rows(final_rows, dcmi_rows)
    detection_statistics(benign, by_attack, pre)
    e2e_statistics()
    benign_fpr_and_ablation(benign, by_attack, pre)
    random_neighbor_control(benign, by_attack)
    gold_and_fp_audit()
    db_churn_manifests()
    readiness_and_protocol()
    cost_latency(started)
    render_report(pre)
    result = {
        "campaign": EXP.name, "verdict": "FINAL_LC_CPU_ANALYSIS_SIDECAR_COMPLETE",
        "completed_utc": now(), "runtime_seconds": time.monotonic() - started,
        "gpu_used": False, "model_loaded": False, "retrieval_run": False,
        "answers_generated": 0, "thresholds_retuned": 0, "training_steps": 0,
        "report": str(EXP / "reports" / "FINAL_LC_CPU_ANALYSIS_SIDECAR_REPORT_KO.md"),
        "report_sha256": sha_file(EXP / "reports" / "FINAL_LC_CPU_ANALYSIS_SIDECAR_REPORT_KO.md"),
    }
    write_json(EXP / "FINAL_RESULT.json", result)
    checkpoint("COMPLETE", verdict=result["verdict"], runtime_seconds=round(result["runtime_seconds"], 3))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
