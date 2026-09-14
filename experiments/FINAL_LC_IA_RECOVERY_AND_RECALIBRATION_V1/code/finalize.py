#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from common import EXP, SOURCE, atomic_json, atomic_text, now, sha_file


def load(name: str):
    path = EXP / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    gold = load("GOLD_RECALIBRATION_RESULT.json")
    validity = load("IA_V3_VALIDITY_RESULT.json")
    detection = load("IA_V3_DETECTION_RESULT.json")
    e2e = load("IA_V3_E2E_RESULT.json")
    gt = load("IA_V3_GT_RESULT.json")
    lines = ["# FINAL_LC_IA_RECOVERY_AND_RECALIBRATION_V1", "",
             f"1. IA-v3 valid coverage: `{validity.get('coverage') if validity else 'PENDING'}`",
             f"2. IA same-budget detection: `{detection.get('primary') if detection else 'NOT_AVAILABLE'}`",
             f"3. IA E2E privacy: `{e2e.get('conditions') if e2e else 'NOT_AVAILABLE'}`",
             "4. strict-transfer intervention: `0.692` (`STRICT_TRANSFER_FAILED` preserved)",
             f"5. benign-refresh intervention: `{gold.get('locked_intervention') if gold else 'PENDING'}`",
             f"6. No Defense Gold F1: `{gold.get('no_defense_gold_f1') if gold else 0.20894493709122594}`",
             f"7. refreshed Final LC Gold F1: `{gold.get('refreshed_gold_f1') if gold else 'PENDING'}`",
             f"8. refusal delta: `{gold.get('new_refusal') if gold else 'PENDING'}`",
             f"9. FP-subset Gold F1 before/after: `{(gold.get('fp_subset_gold_f1_before'), gold.get('fp_subset_gold_f1_after')) if gold else 'PENDING'}`",
             "10. retraining/gradient steps: `0 / 0`", "11. attack samples used for recalibration: `0`",
             f"12. next stage: `{gold.get('next_stage') if gold else 'PENDING_GOLD'}; {e2e.get('verdict') if e2e else (gt or detection or validity or {}).get('verdict', 'PENDING_IA')}`"]
    atomic_text(EXP / "reports" / "FINAL_REPORT_KO.md", "\n".join(lines) + "\n")
    atomic_json(EXP / "FINAL_CAMPAIGN_RESULT.json", {"campaign": EXP.name, "completed_utc": now(),
                "final_lc_frozen_sha256": sha_file(SOURCE / "configs" / "FINAL_LC_FROZEN_MANIFEST.json"),
                "gold": gold, "ia_validity": validity, "ia_detection": detection, "ia_gt": gt, "ia_e2e": e2e})


if __name__ == "__main__": main()
