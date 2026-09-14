#!/usr/bin/env python3
"""Independent arithmetic and provenance checks for final CLEAN_CORE3 outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score


BASE = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911/experiments/CLEAN_CORE3_DEV_V1")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    checks = []
    def add(name: str, passed: bool, evidence: object) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    precommit = BASE / "configs/CLEAN_CORE3_DEV_V1_PRECOMMIT.json"
    precommit_expected = (BASE / "configs/CLEAN_CORE3_DEV_V1_PRECOMMIT.sha256").read_text().split()[0]
    add("precommit hash", sha(precommit) == precommit_expected, sha(precommit))

    cache = BASE / "cache/CLEAN_CORE3_RETRIEVAL_CACHE.jsonl"
    cache_manifest = json.loads((BASE / "cache/CLEAN_CORE3_RETRIEVAL_CACHE_MANIFEST.json").read_text())
    retrieval = jsonl(cache)
    add("retrieval hash", sha(cache) == cache_manifest["cache_sha256"], sha(cache))
    add("retrieval population", len(retrieval) == 1280, len(retrieval))

    answers_path = BASE / "runtime/CLEAN_CORE3_GENERATED_ANSWERS.jsonl"
    answer_manifest = json.loads((BASE / "runtime/CLEAN_CORE3_GENERATION_MANIFEST.json").read_text())
    answers = jsonl(answers_path)
    add("generated answer hash", sha(answers_path) == answer_manifest["output_sha256"], sha(answers_path))
    add("generated condition rows", len(answers) == 897, len(answers))
    counts = defaultdict(int)
    for row in answers:
        counts[row["condition"]] += 1
    add("three balanced answer conditions", set(counts.values()) == {299} and len(counts) == 3, dict(counts))

    amap = {(row["query_id"], row["condition"]): row for row in answers}
    bc_alarm_rows = [row for row in answers if row["condition"] == "BC_MIRABEL_Q97_TOP1_HIDE" and row["detector_alarm"]]
    identical = all(
        row["answer_sha256"] == amap[(row["query_id"], "ORIGINAL_MIRABEL_TOP1_HIDE")]["answer_sha256"] and
        row["prompt_sha256"] == amap[(row["query_id"], "ORIGINAL_MIRABEL_TOP1_HIDE")]["prompt_sha256"]
        for row in bc_alarm_rows
    )
    add("BC alarm identical-prompt reuse", identical, len(bc_alarm_rows))

    menta_queries = list(csv.DictReader((BASE / "tables/MENTA_QUERY_EVIDENCE.csv").open(encoding="utf-8")))
    menta_sessions = list(csv.DictReader((BASE / "tables/MENTA_SESSION_SCORES.csv").open(encoding="utf-8")))
    groups = defaultdict(list)
    for row in menta_queries:
        groups[(row["session_id"], row["condition"])].append(row)
    aggregation_ok = True
    for session in menta_sessions:
        group = groups[(session["session_id"], session["condition"])]
        expected = sum(int(row["contribution"]) for row in group) / 5
        aggregation_ok &= len(group) == 5 and abs(expected - float(session["native_score"])) < 1e-12
    add("MEntA exact five-query aggregation", aggregation_ok and len(menta_sessions) == 120,
        {"query_rows": len(menta_queries), "session_rows": len(menta_sessions)})

    discrete = list(csv.DictReader((BASE / "tables/DISCRETE_ATTACK_QUERY_SCORES.csv").open(encoding="utf-8")))
    privacy = list(csv.DictReader((BASE / "tables/POST_GENERATION_PRIVACY.csv").open(encoding="utf-8")))
    sources = discrete + menta_sessions
    auc_ok = True
    recomputed = {}
    for summary in privacy:
        subset = [row for row in sources if row["attack"] == summary["attack"] and
                  row["condition"] == summary["condition"] and row["valid"] in (True, "True")]
        labels = [int(row["membership"] == "member") for row in subset]
        scores = [float(row["native_score"]) for row in subset]
        value = float(roc_auc_score(labels, scores))
        recomputed[f"{summary['attack']}::{summary['condition']}"] = value
        auc_ok &= abs(value - float(summary["native_roc_auc"])) < 1e-12
    add("native AUC arithmetic", auc_ok, recomputed)

    fp_rows = list(csv.DictReader((BASE / "tables/BC_Q97_FALSE_POSITIVE_AUDIT.csv").open(encoding="utf-8")))
    expected_fp = {row["query_id"] for row in retrieval if row["cohort"] == "BENIGN" and
                   row["split"] == "HOLDOUT" and row["bc_q97_alarm"]}
    add("BC q97 false-positive census", {row["query_id"] for row in fp_rows} == expected_fp,
        {"audit": len(fp_rows), "retrieval": len(expected_fp)})
    add("excluded attacks absent", not ({"S²-MIA", "DCMI", "IA"} & {row.get("attack") for row in retrieval}),
        sorted({row.get("attack") for row in retrieval if row.get("attack")}))

    failed = [row for row in checks if row["status"] != "PASS"]
    result = {"verdict": "FINAL_INTEGRITY_AUDIT_PASS" if not failed else "FINAL_INTEGRITY_AUDIT_FAIL",
              "checks": checks, "failures": failed}
    (BASE / "audits/FINAL_INTEGRITY_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
