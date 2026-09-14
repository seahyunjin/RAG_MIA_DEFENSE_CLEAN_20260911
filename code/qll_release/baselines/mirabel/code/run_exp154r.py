#!/usr/bin/env python3
"""Exp154-R: repair the canonical conditional Original Mirabel baseline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp154r_canonical_mirabel_repair_20260826"
EXP154 = PROJECT / "exp154_selective_action_mirabel_20260826"
OFFICIAL = PROJECT.parent / "AD-test-LLM3/references/menta_official_code/defense/mirabel.py"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


base = load_module("exp154_base_for_repair", EXP154 / "code/run_exp154_screen.py")
base.ROOT = ROOT
# Reuse the exact frozen Exp154 protocol during its input-contract audit.  The
# Exp154-R repair precommit is validated separately below and never replaces
# the candidate grid or selection gates.
base.PRECOMMIT = EXP154 / "configs/PRECOMMIT.json"
base.PARTITION = "EXP154R_QWEN_MPNET_DEVELOPMENT"


def repaired_responses(protocol: dict, panel: pd.DataFrame, generations: pd.DataFrame):
    responses, candidates = base.build_responses(protocol, panel, generations)
    responses.loc[responses.condition.eq("ORIGINAL_MIRABEL"), "condition"] = "ALWAYS_HIDE_RANK1_ORACLE"
    canonical = []
    for row in panel.itertuples(index=False):
        alarm = float(row.mirabel_margin) > 0.0
        canonical.append({"exp87_row_id": str(row.exp87_row_id), "condition": "ORIGINAL_MIRABEL",
                          "response": str(row.rank1_hidden_response if alarm else row.response),
                          "A_R": str(row.A_full), "action": "MIRABEL" if alarm else "PASS",
                          "lola_alarm": alarm, "lola_score": float(row.mirabel_margin), "beta": float("nan")})
    responses = pd.concat([responses, pd.DataFrame(canonical)], ignore_index=True)
    if responses.duplicated(["exp87_row_id", "condition"]).any():
        raise RuntimeError("duplicate repaired response")
    base.atomic_csv(responses, ROOT / "private/EXP154R_RESPONSES.private.csv.gz", "gzip")
    base.checkpoint("STAGE_02_CANONICAL_BASELINE_REPAIRED", candidates=len(candidates),
                    canonical_alarm_rows=int(sum(row["lola_alarm"] for row in canonical)),
                    always_hide_relabelled=True)
    return responses, candidates


def write_audit(panel: pd.DataFrame) -> None:
    payload = {
        "official_local_source": str(OFFICIAL),
        "official_local_sha256": base.sha256_file(OFFICIAL),
        "strict_rule": "mirabel_margin > 0",
        "rows": len(panel),
        "normal_rows": int(panel.kind.eq("benign").sum()),
        "attack_rows": int(panel.kind.eq("attack").sum()),
        "overall_alarm_rate": float((panel.mirabel_margin.astype(float) > 0).mean()),
        "normal_alarm_rate": float((panel.loc[panel.kind.eq("benign"), "mirabel_margin"].astype(float) > 0).mean()),
        "attack_alarm_rate": float((panel.loc[panel.kind.eq("attack"), "mirabel_margin"].astype(float) > 0).mean()),
        "previous_mislabeled_semantics": "always hide rank-1 on every row",
        "repaired_semantics": "hide rank-1 only on canonical alarm",
    }
    base.atomic_json(ROOT / "audits/CANONICAL_MIRABEL_BASELINE_AUDIT.json", payload)
    base.atomic_csv(pd.DataFrame([
        {"path": str(ROOT / "configs/PRECOMMIT.json"),
         "sha256": base.sha256_file(ROOT / "configs/PRECOMMIT.json"), "access": "READ_ONLY"},
        {"path": str(OFFICIAL), "sha256": base.sha256_file(OFFICIAL), "access": "READ_ONLY"},
    ]), ROOT / "provenance/REPAIR_INPUTS.csv")


def main() -> None:
    protocol = json.loads((EXP154 / "configs/PRECOMMIT.json").read_text(encoding="utf-8"))
    repair = json.loads((ROOT / "configs/PRECOMMIT.json").read_text(encoding="utf-8"))
    if not repair.get("written_before_exp154r_metrics") or repair.get("candidate_changed"):
        raise RuntimeError("invalid repair precommit")
    _, panel, generations = base.initialize()
    write_audit(panel)
    responses, candidates = repaired_responses(protocol, panel, generations)
    utility, detail, privacy = base.evaluate(panel, responses)
    base.summarize(protocol, candidates, utility, detail, privacy)
    result_path = ROOT / "FINAL_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["experiment"] = "Exp154-R"
    result["baseline_repair"] = "CANONICAL_CONDITIONAL_MIRABEL"
    result["always_hide_condition"] = "ALWAYS_HIDE_RANK1_ORACLE"
    base.atomic_json(result_path, result)
    summary = pd.read_csv(ROOT / "tables/TABLE_154_03_SUMMARY_AND_GATES.csv")
    base.atomic_csv(summary, ROOT / "tables/TABLE_154R_03_SUMMARY_AND_GATES.csv")
    report = ["# Exp154-R 공식 Mirabel 기준 복구", "",
              f"- 판정: **{result['verdict']}**", f"- 선택 후보: **{result.get('selected_candidate') or '없음'}**",
              "- `ORIGINAL_MIRABEL`: margin > 0에서만 rank-1 hide/backfill", "- `ALWAYS_HIDE_RANK1_ORACLE`: 모든 요청에서 rank-1 제거하는 진단 상한",
              "- 후보 구조·cap grid·선택 gate 변경: 없음", "- 이 결과는 632행 development이며 일반화 주장이 아님", "",
              "기존 Exp128~154 일부 E2E 표의 `ORIGINAL_MIRABEL` 표기는 always-hide 응답을 사용했다. 탐지 FPR 표는 별개의 canonical gate였으므로 정확했지만, E2E privacy/utility baseline 이름은 수정이 필요하다.", ""]
    base.atomic_text(ROOT / "reports/EXP154R_FINAL_REPORT_KO.md", "\n".join(report))
    base.checkpoint("COMPLETE", verdict=result["verdict"], selected_candidate=result.get("selected_candidate"),
                    baseline="CANONICAL_CONDITIONAL_MIRABEL")


if __name__ == "__main__":
    main()
