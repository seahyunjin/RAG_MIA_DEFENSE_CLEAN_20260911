#!/usr/bin/env python3
"""Repair only the post-failure report for the frozen IA-v4 preflight.

The original run correctly reached its fail-closed terminal verdict, then its
report formatter dereferenced ``full=None``.  This script reads the already
frozen results and writes the missing report/result; it performs no generation,
parsing, scoring, thresholding, or model loading.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CHILD = Path(__file__).resolve().parents[1]
PARENT = CHILD.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


v3 = json.loads((PARENT / "IA_V3_EARLY_STOP_RESULT.json").read_text(encoding="utf-8"))
pf_path = CHILD / "IA_V4_PREFLIGHT_RESULT.json"
pf = json.loads(pf_path.read_text(encoding="utf-8"))
if pf.get("verdict") != "IA_STD_Q15_V4_FORMAT_PREFLIGHT_FAILED":
    raise RuntimeError("This reporter is valid only for the frozen failed preflight")

lines = [
    "# IA-Std-Q15-v4-TwoStage Final Summary", "",
    f"1. v3 mathematical failure: generated={v3['generated']}, valid={v3['valid']}, invalid={v3['invalid']}",
    f"2. v4 preflight validity: {pf['valid']}/100 ({pf['coverage']:.1%})",
    "3. v4 full validity: NOT RUN",
    f"4. member/nonmember validity gap: {pf['validity_gap']:.4f}",
    "5. IA Final LC TPR @2.5%: NOT RUN",
    "6. MIRABEL TPR @2.5%: NOT RUN",
    "7. IA E2E No Defense: NOT RUN",
    "8. IA E2E MIRABEL: NOT RUN",
    "9. IA E2E Final LC: NOT RUN",
    "10. Final scientific status: `STANDARDIZED_ATTACK_UNAVAILABLE`", "",
    "## Failure categories", "",
]
for name, count in sorted(pf["failure_reasons"].items()):
    lines.append(f"- `{name}`: {count}")
lines += ["", "The frozen format preflight failed before any detector/privacy result was calculated.",
          "No full-2,000 generation, retry, prompt/parser revision, attack-semantic change, or model change was performed.",
          "Core5 matched-FPR detection, Core5 matched-budget E2E, and benign-only Gold recalibration were not rerun or modified.",
          "", "The original GPU process then raised a report-only `NoneType.get` exception after writing the correct terminal heartbeat and preflight result. This file is a CPU-only reporting repair from those frozen artifacts."]
report = CHILD / "reports" / "FINAL_SUMMARY.md"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
result = {
    "campaign": "IA_STD_Q15_V4_TWO_STAGE",
    "verdict": "IA_STD_Q15_V4_FORMAT_PREFLIGHT_FAILED",
    "scientific_status": "STANDARDIZED_ATTACK_UNAVAILABLE",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "preflight": pf,
    "full_generation_started": False,
    "full_generation_sessions": 0,
    "detection_run": False,
    "e2e_run": False,
    "reporting_repair_only": True,
    "original_terminal_heartbeat_sha256": sha(CHILD / "HEARTBEAT.json"),
    "preflight_result_sha256": sha(pf_path),
    "report_sha256": sha(report),
}
(CHILD / "FINAL_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
