#!/usr/bin/env python3
"""Assemble the frozen A/B/C outcomes without inventing missing metrics."""
from __future__ import annotations

import json
from pathlib import Path

from common import EXP, atomic_json, atomic_text, now, sha_file


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def ref(path: Path):
    return {"path": str(path), "sha256": sha_file(path)} if path.is_file() else None


def main() -> None:
    ia = EXP / "IA_STD_Q15_API1"
    a1 = load(EXP / "PHASE_A1_RESULT.json")
    a2 = load(EXP / "PHASE_A2_RESULT.json")
    b = load(EXP / "PHASE_B_RESULT.json")
    c = load(ia / "FINAL_RESULT.json")
    detection = load(ia / "post_ready" / "IA_API1_DETECTION_RESULT.json")
    e2e = load(ia / "post_ready" / "IA_API1_E2E_RESULT.json")
    result = {
        "campaign": EXP.name, "completed_utc": now(), "final_lc_modified": False,
        "automatic_followup_executed": False,
        "phase_a1": ref(EXP / "PHASE_A1_RESULT.json"), "phase_a2": ref(EXP / "PHASE_A2_RESULT.json"),
        "phase_b": ref(EXP / "PHASE_B_RESULT.json"), "phase_c_validity": ref(ia / "FINAL_RESULT.json"),
        "phase_c_detection": ref(ia / "post_ready" / "IA_API1_DETECTION_RESULT.json"),
        "phase_c_e2e": ref(ia / "post_ready" / "IA_API1_E2E_RESULT.json"),
        "verdicts": {
            "churn": a2.get("verdict") if a2 else None,
            "fp_closed_book": b.get("verdict") if b else None,
            "ia_validity": c.get("verdict") if c else None,
            "ia_detection": detection.get("verdict") if detection else None,
            "ia_e2e": e2e.get("verdict") if e2e else None,
        },
        "scientific_boundary": "No new detector, LC feature, or protection family was selected; this campaign only characterized frozen Final LC and one precommitted fallback.",
    }
    atomic_json(EXP / "FINAL_RESULT.json", result)
    lines = [
        "# Final LC deployment gaps and IA recovery", "",
        f"- Campaign: `{EXP.name}`", "- Final LC modified: `False`", "",
        "| Stage | Verdict |", "|---|---|",
        f"| DB churn causal decomposition | {result['verdicts']['churn'] or 'not available'} |",
        f"| False-positive closed-book fallback | {result['verdicts']['fp_closed_book'] or 'not available'} |",
        f"| IA API1 validity | {result['verdicts']['ia_validity'] or 'not available'} |",
        f"| IA same-FPR detection | {result['verdicts']['ia_detection'] or 'not run'} |",
        f"| IA E2E privacy | {result['verdicts']['ia_e2e'] or 'not run'} |", "",
        "The campaign stops here. No new detector, protection action, transfer, RAGLeak, or BudgetLeak experiment was launched automatically.",
    ]
    atomic_text(EXP / "FINAL_REPORT_KO.md", "\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
