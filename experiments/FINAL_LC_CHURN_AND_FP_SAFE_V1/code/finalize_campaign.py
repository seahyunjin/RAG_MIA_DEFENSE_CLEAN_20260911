#!/usr/bin/env python3
"""Combine the authorized IA, churn, and conditional FP-safe outcomes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from common import ATTACKS, EXP, V6, VERSIONS, atomic_json, atomic_text, now, sha_file


def main() -> None:
    ia = json.loads((V6 / "FINAL_RESULT.json").read_text())
    churn = json.loads((EXP / "DB_CHURN_V2_RESULT.json").read_text())
    packing = json.loads((EXP / "audits/CURRENT_HIDE_PACKING_POLICY.json").read_text())
    rows = list(csv.DictReader((EXP / "tables/DB_CHURN_V2_SUMMARY.csv").open(encoding="utf-8")))
    by_key = {(row["db"], row["method"]): row for row in rows}
    combined = {
        "campaign": EXP.name, "completed_utc": now(),
        "ia": {"verdict": ia["verdict"], "scientific_status": ia["scientific_status"],
               "partial_database_sha256": ia["partial_artifact"]["database_sha256"]},
        "db_churn": {"verdict": churn["verdict"],
                     "strict_transfer_verdict": churn["strict_transfer_verdict"]},
        "fp_safe": {"verdict": packing["verdict"],
                    "current_policy": packing["CURRENT_HIDE_PACKING_POLICY"],
                    "candidate_generation_run": packing["candidate_opened"]},
        "final_lc_modified": False, "training_steps": 0,
        "new_qwen_answer_generations": 0,
    }
    atomic_json(EXP / "FINAL_RESULT.json", combined)

    lines = [
        "# Final LC DB Churn and FP-Safe Validation", "",
        "## IA FINAL STATUS", "",
        f"- `{ia['verdict']}` / `{ia['scientific_status']}`",
        f"- 보존된 question/judgment rows: {ia['partial_artifact']['question_rows']} / {ia['partial_artifact']['judgment_rows']}",
        "- 형식 게이트 실패이므로 탐지·AUC·privacy 성능은 계산하지 않았다.", "",
        "## DB CHURN V2 PRECOMMIT", "",
        f"- SHA-256: `{sha_file(EXP / 'configs/DB_CHURN_V2_PRECOMMIT.json')}`",
        "- Final LC/BGE/k/수식/2.5% budget/strict `>`는 모두 동결했다.", "",
        "## INHERITED DUPLICATE AUDIT", "",
        "- V0의 기존 normalized-text duplicate excess 2개는 baseline provenance로 유지했다.",
        "- V10/V25/V50에서 새 duplicate group은 0개다.", "",
        "## MISSING EMBEDDING UNION", "",
        f"- 고유 추가 문서: {json.loads((EXP / 'cache/BGE_EMBEDDING_UNION_MANIFEST.json').read_text())['documents']}개",
        "- frozen BGE-M3로 각 문서를 한 번만 임베딩했고 학습은 하지 않았다.", "",
        "## CHURN STRICT TRANSFER / BENIGN REFRESH", "",
        "| DB | Original MIRABEL FPR | Final LC strict FPR | Final LC refresh FPR | MEntA TPR | MBA TPR | RAG-MIA TPR | S² TPR | DCMI TPR | Core5 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for version in VERSIONS:
        original = by_key[(version, "Original MIRABEL")]
        strict = by_key[(version, "Final LC Strict")]
        refresh = by_key[(version, "Final LC Refresh")]
        lines.append(
            f"| {version} | {float(original['benign_fpr']):.3%} | {float(strict['benign_fpr']):.3%} | "
            f"{float(refresh['benign_fpr']):.3%} | {float(refresh['MEntA']):.4f} | {float(refresh['MBA']):.4f} | "
            f"{float(refresh['RAG-MIA']):.4f} | {float(refresh['S²-MIA']):.4f} | "
            f"{float(refresh['DCMI-Std-Q2']):.4f} | {float(refresh['core5_mean_tpr']):.4f} |")
    lines += [
        "", "## DB CHURN VERDICT", "",
        f"- Benign refresh: `{churn['verdict']}`",
        f"- Strict transfer: `{churn['strict_transfer_verdict']}`",
        "- 주의: refresh FPR은 상속된 protocol과 같이 1,000-query benign holdout으로 outer threshold를 정하고 그 empirical exceedance를 보고한 값이다.",
        "- retraining/gradient update/attack calibration: 0 / 0 / 0", "",
        "## CURRENT HIDE PACKING POLICY", "",
        f"- `{packing['CURRENT_HIDE_PACKING_POLICY']}`",
        f"- 판정: `{packing['verdict']}`",
        "- 이미 남은 Top-3로 2,048-token water-fill을 수행하므로 같은 재배분 후보는 열지 않았다.", "",
        "## FP 27 UTILITY / CORE5 PRIVACY SCREEN", "",
        "- 조건부 후보 미개방: 새 답변 생성 0개, FP utility 비교 미실행, Core5 100/100 privacy screen 미실행.",
        "- 기존 Simple-Hide FP subset F1 약 0.218은 동결 limitation으로 유지한다.", "",
        "## 12-line summary", "",
        f"1. IA: {ia['verdict']} / {ia['scientific_status']}",
        f"2. Churn embedding union: {json.loads((EXP / 'cache/BGE_EMBEDDING_UNION_MANIFEST.json').read_text())['documents']}",
        f"3. V10 strict/refresh FPR: {float(by_key[('V10','Final LC Strict')]['benign_fpr']):.3%} / {float(by_key[('V10','Final LC Refresh')]['benign_fpr']):.3%}",
        f"4. V25 strict/refresh FPR: {float(by_key[('V25','Final LC Strict')]['benign_fpr']):.3%} / {float(by_key[('V25','Final LC Refresh')]['benign_fpr']):.3%}",
        f"5. V50 strict/refresh FPR: {float(by_key[('V50','Final LC Strict')]['benign_fpr']):.3%} / {float(by_key[('V50','Final LC Refresh')]['benign_fpr']):.3%}",
        f"6. Core5 mean TPR degradation at V50 refresh: {churn['refresh_gate']['mean_degradation']['V50']:+.3%}",
        "7. Retraining steps: 0",
        "8. Current hide packing: remaining Top-3 water-fill to 2,048 tokens",
        "9. FP Simple-Hide F1: ~0.218 (frozen prior result)",
        "10. FP candidate F1: N/A (candidate not applicable)",
        "11. Privacy screen: N/A (candidate not opened)",
        "12. Next: do not duplicate redistribution; retain conditional FP harm as a limitation or request a separately precommitted action study.",
    ]
    atomic_text(EXP / "reports/FINAL_REPORT_KO.md", "\n".join(lines) + "\n")
    atomic_text(EXP / "STATUS.md", "# FINAL_LC_CHURN_AND_FP_SAFE_V1\n\n"
                f"- Stage: `COMPLETE`\n- Churn: `{churn['verdict']}`\n"
                f"- Strict: `{churn['strict_transfer_verdict']}`\n- FP-safe: `{packing['verdict']}`\n")
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
