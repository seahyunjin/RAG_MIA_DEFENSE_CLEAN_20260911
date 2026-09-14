#!/usr/bin/env python3
"""Score frozen IA-ST1 answers and append the result to the preserved Core5."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "IA_STEALTH_CONFIRMATION_V1"
POST = EXP / "post_ready"
PRECOMMIT = POST / "configs" / "IA_ST1_E2E_PRECOMMIT.json"
CORE = ROOT / "experiments" / "FINAL_LC_CORE6_TO_DOMAIN_CHAIN_V1"
SEED = 20260913


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def normalize_answer(value: str) -> str:
    lowered = " ".join(value.casefold().split())
    if any(term in lowered for term in ("i don't know", "i do not know", "not sure", "unable to")):
        return "UNK"
    found = set(re.findall(r"\b(?:yes|no)\b", lowered))
    if found == {"yes"}: return "Yes"
    if found == {"no"}: return "No"
    return "UNK"


def verify_precommit() -> dict:
    expected = PRECOMMIT.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if sha(PRECOMMIT) != expected: raise RuntimeError("E2E precommit hash mismatch")
    pre = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    for path, expected_hash in pre["code"].items():
        if sha(Path(path)) != expected_hash: raise RuntimeError(f"E2E code drift: {path}")
    for key in ("detection_result", "detection_cache", "queries", "gt", "corpus", "targets", "final_lc_manifest", "generator_config"):
        if sha(Path(pre[key]["path"])) != pre[key]["sha256"]: raise RuntimeError(f"frozen E2E input drift: {key}")
    return pre


def condition_branch(row: dict, condition: str, pre: dict) -> str:
    if condition == "NO_DEFENSE": return "A0"
    if condition == "MIRABEL_MATCHED_2_5": return "A_HIDE" if float(row["M"]) > float(pre["thresholds"]["MIRABEL"]["threshold"]) else "A0"
    if condition == "FINAL_LC_MATCHED_2_5": return "A_HIDE" if float(row["R_LC"]) > float(pre["thresholds"]["Final LC"]["threshold"]) else "A0"
    if condition == "ORIGINAL_MIRABEL_FIXED_REFERENCE": return "A_HIDE" if float(row["M"]) > 0.0 else "A0"
    raise KeyError(condition)


def score_sessions(pre: dict, answers: dict[tuple[str, str], dict]) -> tuple[list[dict], list[dict]]:
    detection = {row["query_id"]: row for row in read_jsonl(Path(pre["detection_cache"]["path"]))}
    gt = {row["_id"]: row["ground_truth_label"] for row in read_jsonl(Path(pre["gt"]["path"]))}
    details, summaries = [], []
    conditions = pre["conditions"]
    for condition in conditions:
        grouped = {}
        for query_id, row in detection.items():
            branch = condition_branch(row, condition, pre)
            if (query_id, branch) not in answers: raise RuntimeError(f"missing answer branch: {query_id}/{branch}")
            response = normalize_answer(answers[(query_id, branch)]["answer"])
            grouped.setdefault(row["session_id"], []).append((int(row["query_index"]), query_id, response, gt[query_id], row["membership"], branch, answers[(query_id, branch)]["answer"]))
        session_values = []
        for session_id, group in sorted(grouped.items()):
            group.sort()
            if len(group) != 15 or [item[0] for item in group] != list(range(1, 16)): raise RuntimeError(f"Q15 session drift: {session_id}")
            total = sum(-5.0 if response == "UNK" else float(response == truth) for _, _, response, truth, _, _, _ in group)
            score = total / 15.0
            session_values.append({"session_id": session_id, "membership": group[0][4], "score": score})
            details.append({"condition": condition, "session_id": session_id, "membership": group[0][4], "score": score, "correct_nonunk": sum(response != "UNK" and response == truth for _, _, response, truth, _, _, _ in group), "unknown_responses": sum(response == "UNK" for _, _, response, _, _, _, _ in group), "hidden_queries": sum(branch == "A_HIDE" for _, _, _, _, _, branch, _ in group), "mean_answer_length": float(np.mean([len(answer.split()) for _, _, _, _, _, _, answer in group]))})
        labels = np.asarray([int(row["membership"] == "member") for row in session_values])
        values = np.asarray([row["score"] for row in session_values], dtype=float)
        auc = float(roc_auc_score(labels, values))
        member = values[labels == 1]; nonmember = values[labels == 0]
        rng = np.random.default_rng(SEED + sum(map(ord, condition)))
        boot = np.empty(2000)
        mi = np.where(labels == 1)[0]; ni = np.where(labels == 0)[0]
        for index in range(2000):
            chosen = np.concatenate((rng.choice(mi, len(mi), replace=True), rng.choice(ni, len(ni), replace=True)))
            boot[index] = roc_auc_score(labels[chosen], values[chosen])
        summaries.append({"condition": condition, "native_metric": "IA-ST1 Q15 fixed-polarity session ROC-AUC", "native_auc": auc, "e_auc_secondary": max(auc, 1 - auc), "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)), "sessions": len(values), "member_sessions": len(member), "nonmember_sessions": len(nonmember), "member_score_mean": float(member.mean()), "member_score_std": float(member.std()), "nonmember_score_mean": float(nonmember.mean()), "nonmember_score_std": float(nonmember.std())})
    return summaries, details


def paired_bootstrap_delta(details: list[dict], left: str, right: str) -> tuple[float, float, float]:
    by = {(row["condition"], row["session_id"]): row for row in details}
    sessions = sorted({row["session_id"] for row in details if row["condition"] == left})
    labels = np.asarray([int(by[(left, session)]["membership"] == "member") for session in sessions])
    left_values = np.asarray([by[(left, session)]["score"] for session in sessions])
    right_values = np.asarray([by[(right, session)]["score"] for session in sessions])
    member = np.where(labels == 1)[0]; nonmember = np.where(labels == 0)[0]
    rng = np.random.default_rng(SEED + sum(map(ord, left + right)))
    values = np.empty(2000)
    for index in range(2000):
        chosen = np.concatenate((rng.choice(member, len(member), replace=True), rng.choice(nonmember, len(nonmember), replace=True)))
        values[index] = roc_auc_score(labels[chosen], right_values[chosen]) - roc_auc_score(labels[chosen], left_values[chosen])
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    pre = verify_precommit()
    generation = json.loads((POST / "runtime" / "IA_ST1_GENERATION_MANIFEST.json").read_text(encoding="utf-8"))
    answer_path = Path(generation["answers"]["path"])
    if sha(answer_path) != generation["answers"]["sha256"]: raise RuntimeError("answer artifact drift")
    answers = {(row["query_id"], row["branch"]): row for row in read_jsonl(answer_path)}
    summaries, details = score_sessions(pre, answers)
    write_csv(POST / "tables" / "IA_ST1_E2E_PRIVACY.csv", summaries)
    write_csv(POST / "tables" / "IA_ST1_E2E_SESSION_SCORES.csv", details)
    by = {row["condition"]: row for row in summaries}
    nd = by["NO_DEFENSE"]["native_auc"]
    mirabel = by["MIRABEL_MATCHED_2_5"]["native_auc"]
    final_lc = by["FINAL_LC_MATCHED_2_5"]["native_auc"]
    vs_nd = paired_bootstrap_delta(details, "NO_DEFENSE", "FINAL_LC_MATCHED_2_5")
    vs_mirabel = paired_bootstrap_delta(details, "MIRABEL_MATCHED_2_5", "FINAL_LC_MATCHED_2_5")
    checks = {"clear_privacy_reduction_vs_no_defense_3pp": final_lc <= nd - 0.03 + 1e-12, "noninferior_to_same_budget_mirabel_3pp": final_lc <= mirabel + 0.03 + 1e-12}
    verdict = "IA_STEALTH_E2E_PASS" if all(checks.values()) else "IA_STEALTH_E2E_FAILED"
    core_a = json.loads((CORE / "PHASE_A_RESULT.json").read_text(encoding="utf-8"))
    core_b = json.loads((CORE / "PHASE_B_RESULT.json").read_text(encoding="utf-8"))
    core5_detection = {key: value for key, value in core_a["primary"].items() if key != "IA-Std-Q15"}
    core5_e2e = [row for row in core_b["primary"] if row["attack"] != "IA-Std-Q15"]
    result = {"campaign": "IA_STEALTH_CONFIRMATION_V1", "phase": "IA_MATCHED_BUDGET_E2E", "completed_utc": now(), "verdict": verdict, "primary_budget": 0.025, "conditions": summaries, "primary": {"no_defense": nd, "mirabel_matched_2_5": mirabel, "final_lc_matched_2_5": final_lc, "delta_final_lc_vs_no_defense": final_lc - nd, "delta_final_lc_vs_mirabel": final_lc - mirabel, "final_lc_e_auc_secondary": by["FINAL_LC_MATCHED_2_5"]["e_auc_secondary"], "bootstrap_delta_vs_no_defense": {"mean": vs_nd[0], "ci95": [vs_nd[1], vs_nd[2]]}, "bootstrap_delta_vs_mirabel": {"mean": vs_mirabel[0], "ci95": [vs_mirabel[1], vs_mirabel[2]]}}, "checks": checks, "preferred_at_or_below_0_60": final_lc <= 0.60, "strong_preferred_at_or_below_0_55": final_lc <= 0.55, "core6_eval": {"claim": "4 paper-faithful attacks plus 2 standardized hard attacks; never 6 paper-exact attacks", "core5_detection": core5_detection, "core5_e2e": core5_e2e, "ia_st1_detection": json.loads((POST / "IA_ST1_DETECTION_RESULT.json").read_text(encoding="utf-8"))["primary"], "ia_st1_e2e": {"attack": "IA-Std-Q15-ST1", "no_defense": nd, "mirabel_matched_2_5": mirabel, "final_lc_matched_2_5": final_lc}}, "final_lc_modified": False, "next": "CROSS_DOMAIN_PRIVACY_GENERALIZATION", "ia_q30": "FUTURE_ONLY_NOT_AUTOMATIC"}
    atomic_json(POST / "IA_ST1_E2E_RESULT.json", result)
    atomic_json(EXP / "FINAL_REPORT.json", result)
    lines = ["# IA stealth confirmation", "", f"- Verdict: `{verdict}`", "- Attack: `IA-Std-Q15-ST1` (standardized stress test; not Original/paper-exact IA)", "- Benign intervention budget: `2.5%`", "", "| Condition | Native session AUC | E-AUC | 95% CI |", "|---|---:|---:|---:|"]
    for row in summaries: lines.append(f"| {row['condition']} | {row['native_auc']:.4f} | {row['e_auc_secondary']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |")
    lines.extend(["", f"- Final LC vs No Defense delta: `{final_lc - nd:+.4f}`; paired bootstrap 95% CI `[{vs_nd[1]:.4f}, {vs_nd[2]:.4f}]`", f"- Final LC vs same-budget MIRABEL delta: `{final_lc - mirabel:+.4f}`; paired bootstrap 95% CI `[{vs_mirabel[1]:.4f}, {vs_mirabel[2]:.4f}]`", "- IA-Q30 was not run automatically.", "- Next main experiment: cross-domain privacy generalization without changing Final LC."])
    atomic_text(POST / "reports" / "IA_ST1_FINAL_KO.md", "\n".join(lines) + "\n")
    atomic_text(POST / "STATUS.md", f"# IA-ST1 post-ready evaluation\n\n- stage: `{verdict}`\n- next: `CROSS_DOMAIN_PRIVACY_GENERALIZATION`\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
