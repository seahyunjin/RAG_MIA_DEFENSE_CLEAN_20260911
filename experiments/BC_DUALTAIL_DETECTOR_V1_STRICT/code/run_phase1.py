#!/usr/bin/env python3
"""Precommitted score-only evaluation for BC DualTail Detector V1 Strict.

This script deliberately has two invocations:

  python run_phase1.py precommit
  python run_phase1.py run

The first invocation freezes all IDs, formulas, thresholds, comparison units,
and the script hash before any DualTail outcome is calculated.  The second
verifies that precommit and then performs the score-only Phase-1 screen.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
PARENT = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
EXP = ROOT / "experiments" / "BC_DUALTAIL_DETECTOR_V1_STRICT"
CACHE = PARENT / "cache" / "CLEAN_CORE3_RETRIEVAL_CACHE.jsonl"
BENIGN_CAL = PARENT / "inputs" / "BENIGN_CALIBRATION.jsonl"
BENIGN_HOLD = PARENT / "inputs" / "BENIGN_HOLDOUT.jsonl"
PARENT_PRECOMMIT = PARENT / "configs" / "CLEAN_CORE3_DEV_V1_PRECOMMIT.json"
SCRIPT = Path(__file__).resolve()
PRECOMMIT = EXP / "configs" / "BC_DUALTAIL_V1_STRICT_PRECOMMIT.json"
PRECOMMIT_SHA = EXP / "configs" / "BC_DUALTAIL_V1_STRICT_PRECOMMIT.sha256"

ATTACKS = ("MEntA", "MBA", "RAG-MIA")
ALPHAS = (0.01, 0.03, 0.05)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260911


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def empirical_cutoff(values: list[float], alpha: float) -> float:
    """Frozen deployment rule: ceil(alpha*N)-th largest, alarm strict >."""
    descending = sorted((float(x) for x in values), reverse=True)
    k = math.ceil(alpha * len(descending))
    if k < 1:
        raise ValueError("calibration sample too small")
    return descending[k - 1]


def empirical_upper_tail(reference_sorted: list[float], value: float) -> float:
    count_ge = len(reference_sorted) - bisect.bisect_left(reference_sorted, float(value))
    return (1.0 + count_ge) / (len(reference_sorted) + 1.0)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if den == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / den


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def roc_points(positive: list[float], negative: list[float]) -> list[tuple[float, float]]:
    """ROC vertices for high-score-is-positive; duplicates are collapsed."""
    if not positive or not negative:
        raise ValueError("ROC requires non-empty positive and negative samples")
    groups: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for score in positive:
        groups[float(score)][0] += 1
    for score in negative:
        groups[float(score)][1] += 1
    tp = fp = 0
    raw = [(0.0, 0.0)]
    for score in sorted(groups, reverse=True):
        tp += groups[score][0]
        fp += groups[score][1]
        raw.append((fp / len(negative), tp / len(positive)))
    # At a duplicated FPR, the top of the vertical segment is the useful ROC point.
    best: dict[float, float] = {}
    for fpr, tpr in raw:
        best[fpr] = max(best.get(fpr, 0.0), tpr)
    return sorted(best.items())


def tpr_at_fpr(positive: list[float], negative: list[float], target: float) -> float:
    points = roc_points(positive, negative)
    if target <= points[0][0]:
        return points[0][1]
    if target >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= target <= x1:
            if x1 == x0:
                return max(y0, y1)
            weight = (target - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    raise AssertionError("target FPR was not bracketed")


def matched_binary_threshold(negative: list[float], alpha: float) -> tuple[float, int]:
    """Largest strict-> alarm set not exceeding floor(alpha*N), midpoint ties safe.

    This threshold is only for B/C case identities. The headline matched-FPR
    result comes from interpolated ROC, as precommitted.
    """
    budget = math.floor(alpha * len(negative) + 1e-12)
    unique = sorted(set(float(x) for x in negative), reverse=True)
    if budget <= 0:
        return max(unique), 0
    best_threshold = max(unique)
    best_count = 0
    for i in range(len(unique) - 1):
        threshold = (unique[i] + unique[i + 1]) / 2.0
        count = sum(v > threshold for v in negative)
        if count <= budget and count >= best_count:
            best_threshold, best_count = threshold, count
    return best_threshold, best_count


def bootstrap_delta(
    rows: list[dict], negative_m: list[float], negative_s: list[float],
    iterations: int, seed: int,
) -> tuple[float, float, float]:
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_unit[row["session_id"]].append(row)
    units = sorted(by_unit)
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sample = [rng.choice(units) for _ in units]
        sampled_rows = [row for unit in sample for row in by_unit[unit]]
        m = [row["M"] for row in sampled_rows]
        s = [row["S"] for row in sampled_rows]
        deltas.append(tpr_at_fpr(s, negative_s, 0.03) - tpr_at_fpr(m, negative_m, 0.03))
    return statistics.fmean(deltas), percentile(deltas, 0.025), percentile(deltas, 0.975)


def input_audit(rows: list[dict]) -> dict:
    cohorts = Counter((r["cohort"], r["split"]) for r in rows)
    attack_counts = {
        attack: {
            "queries": sum(r.get("attack") == attack for r in rows),
            "sessions": len({r["session_id"] for r in rows if r.get("attack") == attack}),
            "member_queries": sum(r.get("attack") == attack and r.get("membership") == "member" for r in rows),
            "nonmember_queries": sum(r.get("attack") == attack and r.get("membership") == "nonmember" for r in rows),
        }
        for attack in ATTACKS
    }
    required_keys = {
        "query_id", "session_id", "query_index", "cohort", "split", "attack", "domain",
        "membership", "target_id", "top_document_ids", "top_scores", "mirabel_margin",
        "selected_source_id", "target_rank",
    }
    missing = [r.get("query_id", f"ROW::{i}") for i, r in enumerate(rows) if not required_keys.issubset(r)]
    bad_top4 = [r["query_id"] for r in rows if len(r.get("top_scores", [])) != 4 or len(r.get("top_document_ids", [])) != 4]
    bad_sort = [r["query_id"] for r in rows if any(r["top_scores"][i] < r["top_scores"][i + 1] for i in range(3))]
    return {
        "rows": len(rows),
        "unique_query_ids": len({r["query_id"] for r in rows}),
        "cohort_split": {f"{k[0]}::{k[1]}": v for k, v in cohorts.items()},
        "attack_counts": attack_counts,
        "missing_required_fields": missing,
        "bad_top4": bad_top4,
        "bad_score_order": bad_sort,
        "valid": (
            len(rows) == 1280
            and len({r["query_id"] for r in rows}) == 1280
            and cohorts[("BENIGN", "CALIBRATION")] == 500
            and cohorts[("BENIGN", "HOLDOUT")] == 500
            and not missing and not bad_top4 and not bad_sort
            and attack_counts["MEntA"] == {"queries": 200, "sessions": 40, "member_queries": 100, "nonmember_queries": 100}
            and attack_counts["MBA"] == {"queries": 40, "sessions": 40, "member_queries": 20, "nonmember_queries": 20}
            and attack_counts["RAG-MIA"] == {"queries": 40, "sessions": 40, "member_queries": 20, "nonmember_queries": 20}
        ),
    }


def make_precommit() -> None:
    for directory in ("configs", "audits", "tables", "reports", "logs", "checkpoints"):
        (EXP / directory).mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(CACHE)
    audit = input_audit(rows)
    write_json(EXP / "audits" / "INPUT_AUDIT_PRECOMMIT.json", audit)
    if not audit["valid"]:
        raise RuntimeError("input audit failed; precommit not written")
    cal_ids = [r["query_id"] for r in rows if r["cohort"] == "BENIGN" and r["split"] == "CALIBRATION"]
    ordered = sorted(cal_ids, key=lambda query_id: (sha_text(query_id), query_id))
    tail_ids, threshold_ids = ordered[:250], ordered[250:]
    if len(set(tail_ids) & set(threshold_ids)) or len(tail_ids) != 250 or len(threshold_ids) != 250:
        raise RuntimeError("invalid deterministic benign split")
    holdout_rows = [r for r in rows if r["cohort"] == "BENIGN" and r["split"] == "HOLDOUT"]
    actual_q5 = all(
        sorted(rr["query_index"] for rr in holdout_rows if rr["session_id"] == sid) == [1, 2, 3, 4, 5]
        for sid in {r["session_id"] for r in holdout_rows}
    ) and len({r["session_id"] for r in holdout_rows}) * 5 == len(holdout_rows)
    precommit = {
        "campaign": "BC_DUALTAIL_DETECTOR_V1_STRICT",
        "created_utc": utcnow(),
        "development_only": True,
        "parent": {
            "campaign": "CLEAN_CORE3_DEV_V1",
            "cache_path": str(CACHE),
            "cache_sha256": sha_file(CACHE),
            "parent_precommit_sha256": sha_file(PARENT_PRECOMMIT),
            "rows": len(rows),
        },
        "implementation": {"script": str(SCRIPT), "script_sha256": sha_file(SCRIPT)},
        "inputs": {
            "benign_calibration_sha256": sha_file(BENIGN_CAL),
            "benign_holdout_sha256": sha_file(BENIGN_HOLD),
            "tail_reference_n": 250,
            "threshold_calibration_n": 250,
            "locked_holdout_n": 500,
            "tail_reference_ids": tail_ids,
            "threshold_calibration_ids": threshold_ids,
            "tail_reference_ids_sha256": sha_text("\n".join(tail_ids) + "\n"),
            "threshold_calibration_ids_sha256": sha_text("\n".join(threshold_ids) + "\n"),
            "split_rule": "ascending (SHA256(query_id), query_id); first 250 TAIL_REFERENCE, remaining 250 THRESHOLD_CALIBRATION",
        },
        "scores": {
            "M": "frozen canonical MIRABEL full-corpus mirabel_margin = s1 - tau_q",
            "C": "s1 - (s2+s3+s4)/3",
            "p_X": "(1 + count_tail_reference[X_b >= X_q]) / 251; ties >=",
            "S": "max(-ln(p_M), -ln(p_C))",
            "attack_labels_used": False,
            "extra_features": False,
        },
        "deployment": {
            "alphas": list(ALPHAS),
            "threshold_source": "THRESHOLD_CALIBRATION only",
            "threshold_rule": "ceil(alpha*N)-th largest score",
            "comparison_operator": "strict >",
            "locked_holdout_not_used_for_threshold": True,
        },
        "matched_fpr_diagnostic": {
            "negative_rows": "locked benign holdout 500",
            "positive_rows": "all attack queries in each frozen development family",
            "unit": "query",
            "targets": list(ALPHAS),
            "method": "empirical ROC with linear interpolation; marked INTERPOLATED",
            "not_a_deployment_threshold": True,
            "case_identity_threshold": "midpoint threshold giving largest strict-> benign alarm count <= floor(alpha*N); only for paired B/C identities",
        },
        "menta": {
            "primary": "query-level TPR at matched query-level benign FPR",
            "session_diagnostics": ["alarmed_queries/5", "target_removed_queries/5", "any_alarm", "all5_alarm", "any_target_removed", "all5_target_removed", "first_alarm_index"],
            "session_any_alarm_not_called_TPR_at_FPR": True,
        },
        "benign_session_fpr": "SUPPORTED" if actual_q5 else "SESSION_LEVEL_BENIGN_FPR_NOT_SUPPORTED",
        "domain_fpr": {"domains": ["nfcorpus", "scidocs", "trec-covid"], "interval": "Wilson 95%", "worst_domain_reported": True},
        "attacks": list(ATTACKS),
        "excluded": ["S²-MIA", "DCMI", "IA"],
        "phase1": "score-only; no retrieval, model loading, or generation",
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED, "unit": "target/session; all five MEntA queries move together", "benign_holdout": "fixed"},
        "screening_gate_at_query_fpr_3pct": {
            "MEntA": "DualTail - MIRABEL >= 0.05",
            "MBA": "DualTail - MIRABEL >= -0.05",
            "RAG-MIA": "DualTail - MIRABEL >= -0.05",
            "Core3_macro": "DualTail > MIRABEL",
            "small_screen_only": True,
        },
        "phase1_stop": "failure => BC_DUALTAIL_V1_NO_TPR_GAIN and no automatic handcrafted retrieval fusion search",
        "phase2_if_pass": {
            "action": "MIRABEL top1 source hide, no replacement, same Qwen generator",
            "conditions": ["NO_DEFENSE", "BC_MIRABEL_SIMPLE_HIDE", "BC_DUALTAIL_SIMPLE_HIDE"],
            "native_scorers": ["MEntA", "MBA", "RAG-MIA"],
            "adaptive_eauc": "secondary diagnostic only",
        },
    }
    write_json(PRECOMMIT, precommit)
    digest = sha_file(PRECOMMIT)
    PRECOMMIT_SHA.write_text(f"{digest}  {PRECOMMIT.name}\n", encoding="utf-8")
    write_json(EXP / "checkpoints" / "PRECOMMIT_WRITTEN.json", {"stage": "PRECOMMIT_WRITTEN", "created_utc": utcnow(), "sha256": digest})
    (EXP / "STATUS.md").write_text(
        "# BC DualTail Detector V1 Strict\n\n"
        "- 상태: `PRECOMMIT_WRITTEN`\n"
        f"- PRECOMMIT SHA-256: `{digest}`\n"
        "- 다음 단계: score-only Phase 1\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "PRECOMMIT_WRITTEN", "sha256": digest, "input_audit": audit["valid"]}, indent=2))


def verify_precommit() -> dict:
    if not PRECOMMIT.is_file() or not PRECOMMIT_SHA.is_file():
        raise RuntimeError("precommit missing")
    expected = PRECOMMIT_SHA.read_text(encoding="utf-8").split()[0]
    if sha_file(PRECOMMIT) != expected:
        raise RuntimeError("precommit checksum mismatch")
    pre = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    if sha_file(SCRIPT) != pre["implementation"]["script_sha256"]:
        raise RuntimeError("implementation changed after precommit")
    if sha_file(CACHE) != pre["parent"]["cache_sha256"]:
        raise RuntimeError("frozen retrieval cache changed")
    if sha_file(PARENT_PRECOMMIT) != pre["parent"]["parent_precommit_sha256"]:
        raise RuntimeError("parent precommit changed")
    return pre


def prepare_scores(rows: list[dict], pre: dict) -> list[dict]:
    tail_ids = set(pre["inputs"]["tail_reference_ids"])
    tail_rows = [r for r in rows if r["query_id"] in tail_ids]
    if len(tail_rows) != 250:
        raise RuntimeError(f"tail reference mismatch: {len(tail_rows)}")
    for row in rows:
        scores = [float(x) for x in row["top_scores"]]
        row["M"] = float(row["mirabel_margin"])
        row["C"] = scores[0] - statistics.fmean(scores[1:4])
    ref_m = sorted(r["M"] for r in tail_rows)
    ref_c = sorted(r["C"] for r in tail_rows)
    scored = []
    for source in rows:
        row = dict(source)
        p_m = empirical_upper_tail(ref_m, row["M"])
        p_c = empirical_upper_tail(ref_c, row["C"])
        row["p_M"] = p_m
        row["p_C"] = p_c
        row["S"] = max(-math.log(p_m), -math.log(p_c))
        row["calibration_partition"] = (
            "TAIL_REFERENCE" if row["query_id"] in tail_ids
            else "THRESHOLD_CALIBRATION" if row["query_id"] in set(pre["inputs"]["threshold_calibration_ids"])
            else "LOCKED_HOLDOUT" if row["cohort"] == "BENIGN"
            else "DEVELOPMENT_ATTACK"
        )
        scored.append(row)
    return scored


def deployment_analysis(scored: list[dict], pre: dict) -> tuple[dict, list[dict]]:
    threshold_ids = set(pre["inputs"]["threshold_calibration_ids"])
    threshold_rows = [r for r in scored if r["query_id"] in threshold_ids]
    holdout = [r for r in scored if r["cohort"] == "BENIGN" and r["split"] == "HOLDOUT"]
    thresholds = {
        score_name: {str(alpha): empirical_cutoff([r[score_name] for r in threshold_rows], alpha) for alpha in ALPHAS}
        for score_name in ("M", "S")
    }
    rows = []
    for score_name, detector in (("M", "MIRABEL"), ("S", "DUALTAIL")):
        for alpha in ALPHAS:
            threshold = thresholds[score_name][str(alpha)]
            cal_alarm = sum(r[score_name] > threshold for r in threshold_rows)
            for domain in ("ALL", "nfcorpus", "scidocs", "trec-covid"):
                subset = holdout if domain == "ALL" else [r for r in holdout if r["domain"] == domain]
                alarm = sum(r[score_name] > threshold for r in subset)
                lo, hi = wilson(alarm, len(subset))
                rows.append({
                    "detector": detector,
                    "score": score_name,
                    "nominal_fpr": alpha,
                    "threshold": threshold,
                    "operator": ">",
                    "threshold_calibration_n": len(threshold_rows),
                    "threshold_calibration_alarms": cal_alarm,
                    "threshold_calibration_rate": cal_alarm / len(threshold_rows),
                    "domain": domain,
                    "holdout_n": len(subset),
                    "holdout_false_positives": alarm,
                    "measured_holdout_fpr": alarm / len(subset),
                    "wilson95_low": lo,
                    "wilson95_high": hi,
                })
    return thresholds, rows


def matched_analysis(scored: list[dict]) -> tuple[list[dict], dict, dict]:
    holdout = [r for r in scored if r["cohort"] == "BENIGN" and r["split"] == "HOLDOUT"]
    negatives = {"M": [r["M"] for r in holdout], "S": [r["S"] for r in holdout]}
    rows = []
    tpr3: dict[str, dict[str, float]] = {attack: {} for attack in ATTACKS}
    for attack in ATTACKS:
        positives = [r for r in scored if r.get("attack") == attack]
        for alpha in ALPHAS:
            m_tpr = tpr_at_fpr([r["M"] for r in positives], negatives["M"], alpha)
            s_tpr = tpr_at_fpr([r["S"] for r in positives], negatives["S"], alpha)
            rows.append({
                "attack": attack,
                "positive_unit": "attack_query",
                "positive_queries": len(positives),
                "benign_negative_queries": len(holdout),
                "target_query_fpr": alpha,
                "interpolation": "LINEAR_INTERPOLATED_EMPIRICAL_ROC",
                "mirabel_tpr": m_tpr,
                "dualtail_tpr": s_tpr,
                "delta_dualtail_minus_mirabel": s_tpr - m_tpr,
            })
            if alpha == 0.03:
                tpr3[attack] = {"M": m_tpr, "S": s_tpr, "delta": s_tpr - m_tpr}
    thresholds = {}
    for score_name in ("M", "S"):
        threshold, alarms = matched_binary_threshold(negatives[score_name], 0.03)
        thresholds[score_name] = {"threshold": threshold, "holdout_alarms": alarms, "holdout_fpr": alarms / len(holdout)}
    return rows, tpr3, thresholds


def correlation_analysis(scored: list[dict]) -> list[dict]:
    groups = [
        ("BENIGN_CALIBRATION", [r for r in scored if r["cohort"] == "BENIGN" and r["split"] == "CALIBRATION"]),
        ("BENIGN_HOLDOUT", [r for r in scored if r["cohort"] == "BENIGN" and r["split"] == "HOLDOUT"]),
    ] + [(attack, [r for r in scored if r.get("attack") == attack]) for attack in ATTACKS]
    return [
        {"group": name, "n": len(rows), "spearman_rho_M_C": spearman([r["M"] for r in rows], [r["C"] for r in rows])}
        for name, rows in groups
    ]


def membership_diagnostic(scored: list[dict]) -> list[dict]:
    holdout = [r for r in scored if r["cohort"] == "BENIGN" and r["split"] == "HOLDOUT"]
    result = []
    for attack in ATTACKS:
        for membership in ("member", "nonmember", "all"):
            pos = [r for r in scored if r.get("attack") == attack and (membership == "all" or r.get("membership") == membership)]
            result.append({
                "attack": attack,
                "membership": membership,
                "query_n": len(pos),
                "mirabel_tpr_at_query_fpr_3pct": tpr_at_fpr([r["M"] for r in pos], [r["M"] for r in holdout], 0.03),
                "dualtail_tpr_at_query_fpr_3pct": tpr_at_fpr([r["S"] for r in pos], [r["S"] for r in holdout], 0.03),
            })
    return result


def menta_session_and_recovery(scored: list[dict], binary_thresholds: dict) -> tuple[list[dict], list[dict], dict]:
    menta = [r for r in scored if r.get("attack") == "MEntA"]
    m_threshold = binary_thresholds["M"]["threshold"]
    s_threshold = binary_thresholds["S"]["threshold"]
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in menta:
        by_session[row["session_id"]].append(row)
    session_rows = []
    recovery_rows = []
    categories = Counter()
    recovered_sessions = set()
    lost_sessions = set()
    for sid in sorted(by_session):
        rows = sorted(by_session[sid], key=lambda r: r["query_index"])
        if [r["query_index"] for r in rows] != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"invalid MEntA Q1-Q5 session: {sid}")
        m_alarm = [r["M"] > m_threshold for r in rows]
        s_alarm = [r["S"] > s_threshold for r in rows]
        # Locator is frozen similarity Top1. Removal reaches target iff alarm and target rank=1.
        m_removed = [alarm and r["target_rank"] == 1 for alarm, r in zip(m_alarm, rows)]
        s_removed = [alarm and r["target_rank"] == 1 for alarm, r in zip(s_alarm, rows)]
        session_rows.append({
            "session_id": sid,
            "membership": rows[0]["membership"],
            "queries": 5,
            "mirabel_alarmed_queries": sum(m_alarm),
            "dualtail_alarmed_queries": sum(s_alarm),
            "mirabel_target_removed_queries": sum(m_removed),
            "dualtail_target_removed_queries": sum(s_removed),
            "mirabel_any_alarm": any(m_alarm),
            "dualtail_any_alarm": any(s_alarm),
            "mirabel_all5_alarm": all(m_alarm),
            "dualtail_all5_alarm": all(s_alarm),
            "mirabel_any_target_removed": any(m_removed),
            "dualtail_any_target_removed": any(s_removed),
            "mirabel_all5_target_removed": all(m_removed),
            "dualtail_all5_target_removed": all(s_removed),
            "mirabel_first_alarm_index": next((r["query_index"] for r, a in zip(rows, m_alarm) if a), None),
            "dualtail_first_alarm_index": next((r["query_index"] for r, a in zip(rows, s_alarm) if a), None),
        })
        for row, ma, sa in zip(rows, m_alarm, s_alarm):
            category = "A_BOTH_HIT" if ma and sa else "B_MIRABEL_MISS_DUALTAIL_HIT" if (not ma and sa) else "C_MIRABEL_HIT_DUALTAIL_MISS" if (ma and not sa) else "D_BOTH_MISS"
            categories[category] += 1
            if category == "B_MIRABEL_MISS_DUALTAIL_HIT":
                recovered_sessions.add(sid)
            if category == "C_MIRABEL_HIT_DUALTAIL_MISS":
                lost_sessions.add(sid)
            recovery_rows.append({
                "category": category,
                "session_id": sid,
                "membership": row["membership"],
                "target_id": row["target_id"],
                "query_id": row["query_id"],
                "query_index": row["query_index"],
                "M": row["M"], "C": row["C"], "p_M": row["p_M"], "p_C": row["p_C"], "S": row["S"],
                "s1": row["top_scores"][0], "s2": row["top_scores"][1], "s3": row["top_scores"][2], "s4": row["top_scores"][3],
                "target_rank": row["target_rank"],
            })
    summary = {
        "binary_case_thresholds": binary_thresholds,
        "counts": dict(categories),
        "mirabel_miss_queries": categories["B_MIRABEL_MISS_DUALTAIL_HIT"] + categories["D_BOTH_MISS"],
        "newly_recovered_queries": categories["B_MIRABEL_MISS_DUALTAIL_HIT"],
        "newly_recovered_sessions": len(recovered_sessions),
        "lost_mirabel_true_positive_queries": categories["C_MIRABEL_HIT_DUALTAIL_MISS"],
        "sessions_with_lost_tp": len(lost_sessions),
        "largest_recovery_queries_in_one_session": max((sum(r["category"] == "B_MIRABEL_MISS_DUALTAIL_HIT" and r["session_id"] == sid for r in recovery_rows) for sid in recovered_sessions), default=0),
    }
    return session_rows, recovery_rows, summary


def phase1_run() -> None:
    pre = verify_precommit()
    rows = read_jsonl(CACHE)
    audit = input_audit(rows)
    if not audit["valid"]:
        raise RuntimeError("input audit no longer passes")
    scored = prepare_scores(rows, pre)
    thresholds, deployment_rows = deployment_analysis(scored, pre)
    matched_rows, tpr3, binary_thresholds = matched_analysis(scored)
    correlations = correlation_analysis(scored)
    membership_rows = membership_diagnostic(scored)
    session_rows, recovery_rows, recovery_summary = menta_session_and_recovery(scored, binary_thresholds)

    holdout = [r for r in scored if r["cohort"] == "BENIGN" and r["split"] == "HOLDOUT"]
    bootstrap_rows = []
    for attack_index, attack in enumerate(ATTACKS):
        attack_rows = [r for r in scored if r.get("attack") == attack]
        mean_delta, low, high = bootstrap_delta(
            attack_rows,
            [r["M"] for r in holdout],
            [r["S"] for r in holdout],
            BOOTSTRAP_ITERATIONS,
            BOOTSTRAP_SEED + attack_index,
        )
        bootstrap_rows.append({
            "attack": attack,
            "resampling_unit": "session/target",
            "iterations": BOOTSTRAP_ITERATIONS,
            "delta_tpr_at_query_fpr_3pct_mean": mean_delta,
            "bootstrap95_low": low,
            "bootstrap95_high": high,
        })

    mean_m = statistics.fmean(tpr3[a]["M"] for a in ATTACKS)
    mean_s = statistics.fmean(tpr3[a]["S"] for a in ATTACKS)
    checks = {
        "menta_gain_at_least_5pp": tpr3["MEntA"]["delta"] >= 0.05 - 1e-12,
        "mba_noninferior_within_5pp": tpr3["MBA"]["delta"] >= -0.05 - 1e-12,
        "rag_mia_noninferior_within_5pp": tpr3["RAG-MIA"]["delta"] >= -0.05 - 1e-12,
        "core3_macro_tpr_improved": mean_s > mean_m + 1e-12,
    }
    passed = all(checks.values())
    verdict = "BC_DUALTAIL_V1_PHASE1_PASS" if passed else "BC_DUALTAIL_V1_NO_TPR_GAIN"

    score_rows = []
    for r in scored:
        score_rows.append({
            "query_id": r["query_id"], "session_id": r["session_id"], "query_index": r["query_index"],
            "cohort": r["cohort"], "split": r["split"], "calibration_partition": r["calibration_partition"],
            "attack": r.get("attack"), "domain": r["domain"], "membership": r.get("membership"),
            "target_id": r.get("target_id"), "target_rank": r.get("target_rank"),
            "s1": r["top_scores"][0], "s2": r["top_scores"][1], "s3": r["top_scores"][2], "s4": r["top_scores"][3],
            "M": r["M"], "C": r["C"], "p_M": r["p_M"], "p_C": r["p_C"], "S": r["S"],
        })
    write_csv(EXP / "tables" / "SCORE_ROWS.csv", score_rows)
    write_csv(EXP / "tables" / "DEPLOYMENT_FPR.csv", deployment_rows)
    write_csv(EXP / "tables" / "MATCHED_QUERY_FPR_TPR.csv", matched_rows)
    write_csv(EXP / "tables" / "M_C_CORRELATION.csv", correlations)
    write_csv(EXP / "tables" / "MEMBERSHIP_STRATIFIED_TPR_3PCT.csv", membership_rows)
    write_csv(EXP / "tables" / "MENTA_SESSION_DIAGNOSTIC.csv", session_rows)
    write_csv(EXP / "tables" / "MENTA_MISS_RECOVERY_CASES.csv", recovery_rows)
    write_csv(EXP / "tables" / "PAIRED_BOOTSTRAP_DELTA.csv", bootstrap_rows)
    write_json(EXP / "configs" / "FROZEN_DEPLOYMENT_THRESHOLDS.json", {
        "created_utc": utcnow(), "precommit_sha256": sha_file(PRECOMMIT), "thresholds": thresholds,
    })

    deployment_primary = [r for r in deployment_rows if r["nominal_fpr"] == 0.03 and r["domain"] in ("ALL", "nfcorpus", "scidocs", "trec-covid")]
    result = {
        "verdict": verdict,
        "completed_utc": utcnow(),
        "precommit_sha256": sha_file(PRECOMMIT),
        "input_audit": audit,
        "session_level_benign_fpr": pre["benign_session_fpr"],
        "deployment_thresholds": thresholds,
        "deployment_primary_rows": deployment_primary,
        "matched_tpr_at_query_fpr_3pct": tpr3,
        "macro_tpr_at_query_fpr_3pct": {"MIRABEL": mean_m, "DUALTAIL": mean_s, "delta": mean_s - mean_m},
        "screening_checks": checks,
        "menta_recovery": recovery_summary,
        "bootstrap": bootstrap_rows,
        "correlations": correlations,
        "phase2_allowed": passed,
        "next_step": "PHASE2_E2E" if passed else "STOP_RETRIEVER_SCORE_HANDCRAFTED_FUSION",
    }
    write_json(EXP / "PHASE1_RESULT.json", result)
    write_json(EXP / "checkpoints" / f"{verdict}.json", {"stage": verdict, "created_utc": utcnow(), "precommit_sha256": sha_file(PRECOMMIT)})

    report = [
        "# BC DualTail Detector V1 Strict — Phase 1 결과",
        "",
        f"- 최종 판정: `{verdict}`",
        "- 성격: development score-only screening; 논문 최종 증거가 아님",
        f"- PRECOMMIT SHA-256: `{sha_file(PRECOMMIT)}`",
        f"- 정상 Q5 session GT: `{pre['benign_session_fpr']}`",
        "",
        "## 동일 query-level FPR 비교",
        "",
        "| Attack | MIRABEL TPR@3% | DualTail TPR@3% | Delta |",
        "|---|---:|---:|---:|",
    ]
    for attack in ATTACKS:
        report.append(f"| {attack} | {tpr3[attack]['M']:.3f} | {tpr3[attack]['S']:.3f} | {tpr3[attack]['delta']:+.3f} |")
    report += [
        f"| Macro | {mean_m:.3f} | {mean_s:.3f} | {mean_s-mean_m:+.3f} |",
        "",
        "위 TPR은 locked benign holdout과 attack query의 empirical ROC를 3%에서 선형 보간한 ranking diagnostic이다. deployment threshold 결과가 아니다.",
        "",
        "## Screening gate",
        "",
    ]
    report.extend(f"- {name}: `{'PASS' if value else 'FAIL'}`" for name, value in checks.items())
    report += [
        "",
        "## MEntA miss recovery (matched 3% binary case audit)",
        "",
        f"- MIRABEL miss query: {recovery_summary['mirabel_miss_queries']}",
        f"- DualTail 신규 회복 query: {recovery_summary['newly_recovered_queries']}",
        f"- 신규 회복 session: {recovery_summary['newly_recovered_sessions']}",
        f"- DualTail이 잃은 MIRABEL TP query: {recovery_summary['lost_mirabel_true_positive_queries']}",
        f"- 한 session이 만든 최대 신규 회복 query: {recovery_summary['largest_recovery_queries_in_one_session']}",
        "",
        "## 해석",
        "",
        "Phase 1 실패 시 명세에 따라 generation을 실행하지 않고 retriever-score handcrafted fusion 탐색을 종료한다. 통과 시에만 동결된 MIRABEL Top-1 hide와 Qwen으로 Phase 2를 진행한다.",
    ]
    (EXP / "reports" / "PHASE1_REPORT_KO.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (EXP / "STATUS.md").write_text(
        "# BC DualTail Detector V1 Strict\n\n"
        f"- 상태: `{verdict}`\n"
        f"- PRECOMMIT SHA-256: `{sha_file(PRECOMMIT)}`\n"
        f"- Phase 2 허용: `{passed}`\n"
        f"- 다음 단계: `{result['next_step']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"precommit", "run"}:
        raise SystemExit("usage: run_phase1.py {precommit|run}")
    if sys.argv[1] == "precommit":
        make_precommit()
    else:
        phase1_run()


if __name__ == "__main__":
    main()
