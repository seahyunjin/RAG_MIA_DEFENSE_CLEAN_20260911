#!/usr/bin/env python3
"""Build one compact roadmap before deleting bulky experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


WORKSPACE = Path("/home/traffic_3/workspace/workspace/SH")
LORA = WORKSPACE / "LoRA-mirabel-Decter"
AD3 = WORKSPACE / "AD-test-LLM3"
OUTPUT = WORKSPACE / "RAG_MIA_DEFENSE_CLEAN_20260911" / "EXPERIMENT_ROADMAP_ALL_20260911_KO.md"

METRIC_PRIORITY = (
    "effective_auc", "e_auc", "e-auc", "auc", "tpr", "fpr", "utility",
    "token_f1", "f1", "grounded", "unsupported", "refusal", "intervention",
    "hide_rate", "member", "nonmember", "normal", "passed",
)
VERDICT_KEYS = ("verdict", "final_verdict", "campaign_verdict", "status", "decision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def size_text(path: Path) -> str:
    completed = subprocess.run(
        ["du", "-sh", str(path)], check=False, capture_output=True, text=True
    )
    return completed.stdout.split("\t", 1)[0] if completed.stdout else "?"


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten(child, name)
    elif isinstance(value, list):
        # Avoid dumping large per-row arrays into the roadmap.
        if len(value) <= 8 and all(not isinstance(item, (dict, list)) for item in value):
            yield prefix, value
    elif isinstance(value, (str, int, float, bool)) or value is None:
        yield prefix, value


def select_result(unit: Path) -> Path | None:
    preferred = [
        unit / "FINAL_RESULT.json",
        unit / "FINAL_VERDICT.json",
        unit / "RESULT.json",
        unit / "final_result.json",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = []
    for pattern in ("FINAL_RESULT.json", "FINAL_VERDICT.json"):
        candidates.extend(unit.glob(f"*/{pattern}"))
        candidates.extend(unit.glob(f"*/*/{pattern}"))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def status_fallback(unit: Path) -> str:
    for name in ("STATUS.md", "FINAL_REPORT.md", "README_KO.md", "README.md"):
        path = unit / name
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")[:12000]
        for line in text.splitlines():
            if re.search(r"verdict|판정|status|결론", line, re.I):
                return re.sub(r"\s+", " ", line.strip("# -*"))[:160]
    return "최종 판정 파일 없음"


def summarize(unit: Path) -> tuple[str, str, str]:
    result = select_result(unit)
    if result is None:
        return status_fallback(unit), "-", "-"
    try:
        payload = json.loads(result.read_text(errors="ignore"))
    except Exception as error:
        return f"JSON 읽기 실패: {type(error).__name__}", result.name, sha256(result)
    flat = list(flatten(payload))
    verdict = ""
    for wanted in VERDICT_KEYS:
        hits = [(key, value) for key, value in flat if key.casefold().split(".")[-1] == wanted]
        if hits:
            verdict = str(hits[0][1])
            break
    if not verdict:
        verdict = status_fallback(unit)
    metrics = []
    seen = set()
    for priority in METRIC_PRIORITY:
        for key, value in flat:
            low = key.casefold()
            if priority not in low or key in seen or key.casefold().split(".")[-1] in VERDICT_KEYS:
                continue
            if isinstance(value, float):
                rendered = f"{value:.5g}"
            elif isinstance(value, list):
                rendered = str(value)
            else:
                rendered = str(value)
            metrics.append(f"{key}={rendered}")
            seen.add(key)
            if len(metrics) >= 8:
                break
        if len(metrics) >= 8:
            break
    return verdict[:180], "; ".join(metrics)[:600] or "-", f"{result.relative_to(unit)} ({sha256(result)})"


def units():
    for child in sorted(LORA.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            yield "LoRA/생성 방어", child
    for parent, phase in ((AD3 / "experiments", "AD-Mirabel/초기 실험"),
                          (AD3 / "artifacts", "AD-Mirabel/후기 실험")):
        if parent.is_dir():
            for child in sorted(parent.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    yield phase, child


def esc(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    rows = []
    for phase, unit in units():
        verdict, metrics, source = summarize(unit)
        rows.append((phase, unit.name, size_text(unit), verdict, metrics, source))

    total_before = f"LoRA={size_text(LORA)}, AD-test-LLM3={size_text(AD3)}"
    lines = [
        "# RAG Membership Privacy Defense 전체 실험 로드맵",
        "",
        "- 작성일: 2026-09-11",
        f"- 정리 전 용량: {total_before}",
        "- 목적: 대용량 response/cache/checkpoint/figure를 삭제하기 전 실험명, 최종 판정, 핵심 지표와 최종 JSON 해시를 한 파일에 보존",
        "- 주의: 이 문서는 결과 요약이지 삭제된 raw response와 checkpoint를 복원하는 백업은 아니다.",
        "",
        "## 연구 흐름 요약",
        "",
        "1. Mirabel 재현과 Early/AD/CCT: 동일 FPR 비교와 세션 누적 검정을 시도했으나 정상·stealth query 중첩과 도메인 이동이 병목이었다.",
        "2. V16/LDF/LoRA 계열: 개발 cohort에서는 높은 탐지율을 보였지만 hard-benign, LOFO/LODO, retriever 전이에서 일반 탐지기 주장이 무너졌다.",
        "3. GlobalCap/Prefix64 계열: 여러 native MIA의 생성 후 E-AUC를 낮췄지만 rank-1 근거 손실, 정상 정답률 저하와 BudgetLeak/rank 이동 문제가 나타났다.",
        "4. QLL source influence 계열: stateless source hide가 core attacks와 BudgetLeak 및 Llama에서 가장 유망했으나 BGE 동일 3K 재평가, untouched domain, 개입 정상 답변 손상이 남았다.",
        "5. claim/counterfactual/dual-view/fusion 계열: 근거 보존 또는 탐지 결합을 시도했지만 정상 utility, MEntA, S²/MBA 또는 fresh-blind gate를 동시에 통과하지 못했다.",
        "6. 2026-09-11 결론: 완전한 최종 모델은 아직 없다. 가장 가까운 재검증 후보는 정상-only 보정 stateless QLL source hide이며, IA provenance와 동일 3K 재현 및 정상 factuality 감사가 필요하다.",
        "",
        "## 최종적으로 보존한 코드",
        "",
        "- `code/qll_final_model`: 동결 QLL source-hide inference/calibration 코드",
        "- `code/qll_release`: QLL 코드 전용 release 패키지",
        "- `code/paper_evaluation`: 동일 cohort Mirabel/QLL, core-6, BudgetLeak 평가 코드",
        "- `code/ad_mirabel_core`: AD-Mirabel detector와 후기 실험의 핵심 Python 모듈",
        "- `code/menta_official`: MEntA가 공개한 원 공격·Mirabel 코드",
        "",
        f"## 실험 인덱스 ({len(rows)}개 디렉터리)",
        "",
        "| 계열 | 실험/디렉터리 | 정리 전 용량 | 최종 판정 | 핵심 지표(최대 8개) | 판정 파일과 SHA-256 |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(esc(item) for item in row) + " |")
    lines.extend([
        "",
        "## 해석 원칙",
        "",
        "- 서로 다른 cohort/retriever/generator lineage의 숫자를 같은 표의 직접 우열로 사용하지 않는다.",
        "- `PASS`는 해당 실험이 사전 정의한 gate 통과를 의미할 뿐 universal claim을 뜻하지 않는다.",
        "- member/nonmember label, query budget, native scorer와 FPR operating point가 같을 때만 apples-to-apples 비교로 사용한다.",
        "- E-AUC는 `max(AUC, 1-AUC)`이며 0.5에 가까울수록 생성 답변으로 membership을 구분하기 어렵다.",
        "- 정상 답변 보존율은 gold accuracy와 다르므로 factuality/groundedness 결과가 없으면 환각 없음으로 해석하지 않는다.",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT} rows={len(rows)} bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
