#!/usr/bin/env python3
"""Write a single evidence-limited report from whatever gated phases completed."""
from __future__ import annotations

import json

from common import EXP, atomic_json, atomic_text, now, sha_file


def load(name):
    path=EXP/name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main():
    a=load("PHASE_A_RESULT.json");b=load("PHASE_B_RESULT.json");c=load("PHASE_C_RESULT.json");d=load("PHASE_D_RESULT.json")
    frozen=EXP/"configs"/"FINAL_LC_FROZEN_MANIFEST.json"
    lines=["# Final LC Core6 → Gold QA → New Domain", "",
        f"- Final LC frozen manifest SHA-256: `{sha_file(frozen) if frozen.is_file() else 'UNAVAILABLE'}`",
        "- DCMI-Std-Q2 and IA-Std-Q15 are frozen standardized variants, not Original DCMI/IA claims.",
        f"- Phase A Core6 detection: `{a['verdict'] if a else 'NOT_COMPLETED'}`",
        f"- Phase B matched-budget E2E: `{b['verdict'] if b else 'NOT_RUN_BY_GATE'}`",
        f"- Phase C Gold QA: `{c['verdict'] if c else 'NOT_RUN_BY_GATE'}`",
        f"- Phase D finance transfer: `{d['verdict'] if d else 'NOT_RUN_BY_GATE'}`",""]
    if a:
        lines += ["## Core6 Detection", "", "| Attack | MIRABEL TPR@2.5% | Final LC TPR@2.5% | Delta |", "|---|---:|---:|---:|"]
        for attack,row in a["primary"].items():lines.append(f"| {attack} | {row['MIRABEL']:.4f} | {row['Final LC']:.4f} | {row['delta']:+.4f} |")
    if b:
        lines += ["", "## Core6 Matched-Budget Privacy", "", "| Attack | No Defense | MIRABEL@2.5% | Final LC@2.5% |", "|---|---:|---:|---:|"]
        for row in b["primary"]:lines.append(f"| {row['attack']} | {row['no_defense']:.4f} | {row['mirabel_matched_2_5']:.4f} | {row['final_lc_matched_2_5']:.4f} |")
    if c:
        lines += ["", "## Gold QA", "", f"- Verdict: `{c['verdict']}`", f"- Final LC Gold F1: `{c.get('summary',[])}`"]
    if d:
        lines += ["", "## New Finance Domain", "", f"- Verdict: `{d['verdict']}`", f"- Transfer interpretation: `{d['transfer_label']}`"]
    summary=["## 15줄 이내 최종 요약", "",
        f"1. Core6 탐지: {a['verdict'] if a else '미완료'}.",
        f"2. Core6 E2E: {b['verdict'] if b else '선행 게이트로 미실행'}.",
        f"3. DCMI-Q2: {a['primary']['DCMI-Std-Q2'] if a else '미완료'}.",
        f"4. IA-Q15: {a['primary']['IA-Std-Q15'] if a else '미완료'}.",
        f"5. Gold QA loss: {c.get('checks') if c else '미실행'}.",
        f"6. FP harm: {'GOLD_FALSE_POSITIVE_HARM.csv 저장' if c else '미실행'}.",
        f"7. Strict domain transfer: {d.get('strict_checks') if d else '미실행'}.",
        f"8. Benign-refresh transfer: {d.get('benign_refresh_checks') if d else '미실행'}.",
        "9. Claim boundary: 통과한 마지막 게이트까지만 주장하며 standardized DCMI/IA를 원 공격 재현으로 부르지 않는다."]
    lines += [""]+summary
    atomic_text(EXP/"reports"/"FINAL_CHAIN_REPORT_KO.md","\n".join(lines)+"\n")
    atomic_json(EXP/"FINAL_CHAIN_RESULT.json",{"campaign":EXP.name,"completed_utc":now(),"phase_a":a,"phase_b":b,"phase_c":c,"phase_d":d,"claim_boundary":"last passed gate only"})


if __name__=="__main__":main()
