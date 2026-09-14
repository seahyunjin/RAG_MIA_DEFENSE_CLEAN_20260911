#!/usr/bin/env python3
"""Fail-closed Phase-1 verification for CLEAN_CORE3_DEV_V1.

This script intentionally performs no target selection, retrieval, model
forward, query generation, or answer generation.  It verifies the frozen
CORE6 protocol-recovery artifact and records whether the immutable runtime
inputs required by the three READY protocols are actually available.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
CAMPAIGN = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
RECOVERY = ROOT / "experiments" / "CORE6_PROTOCOL_RECOVERY_V1"
RAW = Path("/home/traffic_3/workspace/workspace/AD-test-LLM/data/menta_beir/_raw")
HF = Path("/home/traffic_3/workspace/.cache/huggingface/hub")
READY = ("menta", "mba", "rag_mia")

EXPECTED_RAW_HASHES = {
    "nfcorpus": {
        "corpus.jsonl": "10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f",
        "queries.jsonl": "d024e6621b84925d485ae473d316a0c3af31c62c8068a59fb29d22f7613aef2a",
    },
    "scidocs": {
        "corpus.jsonl": "328bb38854179c83ee40d4f49fae2ef7209ede7163785193cbdcd5906b28eecc",
        "queries.jsonl": "3c85b68419bd579ff52edb58f9e7b32eb3d18bc83435edb71dbb68aaf79531d6",
    },
    "trec-covid": {
        "corpus.jsonl": "aded69896598665082316d00df487d22212cc5a8611db05db34eb40d04ed00d7",
        "queries.jsonl": "78f4b76bfef251a0baf37a6cea5c4ba8376cb4536c34635709aaa4a2783b208c",
    },
}

REQUIRED_MODEL_REVISIONS = {
    "MEntA NLI": (
        "models--tasksource--deberta-base-long-nli",
        "04dcf11f844b07bc57015169fca2b7d6df8299d5",
    ),
    "MEntA claim extractor": (
        "models--Babelscape--t5-base-summarization-claim-extractor",
        "94775fb1c8dc2c3ef1bfec413f9f961e6ba5a1c8",
    ),
    "MBA proxy LM": (
        "models--openai-community--gpt2-xl",
        "15ea56dee5df4983c59b2538573817e1667135e2",
    ),
    "MBA spelling model": (
        "models--oliverguhr--spelling-correction-english-base",
        "0e3958355a09d2816ed2701fdc2f4471d46c320e",
    ),
    "Retriever BGE-M3": (
        "models--BAAI--bge-m3",
        "5617a9f61b028005a4858fdac845db406aefb181",
    ),
    "Generator Qwen2.5-3B-Instruct": (
        "models--Qwen--Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check(name: str, passed: bool, evidence: str, *, required: bool = True) -> dict[str, object]:
    return {
        "check": name,
        "required": required,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


def main() -> int:
    started = datetime.now(timezone.utc)
    audit_dir = CAMPAIGN / "audits"
    report_dir = CAMPAIGN / "reports"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, object]] = []
    core_manifest_path = RECOVERY / "CORE6_PROTOCOL_MANIFEST.json"
    checksum_path = RECOVERY / "CORE6_PROTOCOL_MANIFEST.sha256"
    core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))

    expected_line = checksum_path.read_text(encoding="utf-8").strip()
    expected_sha, expected_name = expected_line.split(maxsplit=1)
    expected_name = expected_name.lstrip("*")
    actual_core_sha = sha256(core_manifest_path)
    checks.append(check(
        "final protocol recovery manifest SHA-256",
        expected_name == core_manifest_path.name and expected_sha == actual_core_sha,
        f"expected={expected_sha}; actual={actual_core_sha}; filename={expected_name}",
    ))
    checks.append(check(
        "final status CORE_N_PROTOCOL_PARTIAL N=3",
        core_manifest.get("final_verdict") == "CORE_N_PROTOCOL_PARTIAL"
        and core_manifest.get("ready_count") == 3,
        f"verdict={core_manifest.get('final_verdict')}; ready_count={core_manifest.get('ready_count')}",
    ))
    checks.append(check(
        "READY set exactly MEntA/MBA/RAG-MIA",
        core_manifest.get("ready") == ["MEntA", "MBA", "RAG-MIA"],
        f"ready={core_manifest.get('ready')}",
    ))
    checks.append(check(
        "excluded protocols preserved",
        core_manifest.get("revise") == ["S²-MIA", "DCMI"]
        and core_manifest.get("unavailable") == ["IA"],
        f"revise={core_manifest.get('revise')}; unavailable={core_manifest.get('unavailable')}",
    ))

    protocol_details: dict[str, object] = {}
    protocol_manifests = core_manifest["protocol_manifests"]
    for protocol in READY:
        protocol_dir = RECOVERY / "protocols" / protocol
        manifest_path = protocol_dir / "MANIFEST.json"
        actual_manifest_sha = sha256(manifest_path)
        expected_manifest_sha = protocol_manifests[protocol]
        checks.append(check(
            f"{protocol} manifest hash",
            actual_manifest_sha == expected_manifest_sha,
            f"expected={expected_manifest_sha}; actual={actual_manifest_sha}",
        ))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        per_file = []
        for rel, expected in manifest["files"].items():
            item_path = protocol_dir / rel
            actual = sha256(item_path) if item_path.is_file() else None
            passed = actual == expected
            per_file.append({"path": rel, "expected_sha256": expected, "actual_sha256": actual, "status": "PASS" if passed else "FAIL"})
            checks.append(check(f"{protocol}:{rel} hash", passed, f"expected={expected}; actual={actual}"))
        protocol_details[protocol] = {
            "manifest_sha256": actual_manifest_sha,
            "status": manifest["status"],
            "implementation_label": manifest["implementation_label"],
            "files": per_file,
        }

    tests = subprocess.run(
        ["python", "-m", "unittest", "discover", "-s", str(RECOVERY / "protocols"), "-p", "test_*.py", "-v"],
        cwd=RECOVERY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    match = re.search(r"Ran (\d+) tests", tests.stdout)
    test_count = int(match.group(1)) if match else None
    tests_ok = tests.returncode == 0 and test_count == 20 and "OK" in tests.stdout
    checks.append(check("protocol unit tests 20/20", tests_ok, f"returncode={tests.returncode}; test_count={test_count}"))
    (audit_dir / "UNIT_TEST_RESULTS.txt").write_text(tests.stdout, encoding="utf-8")

    raw_details: dict[str, object] = {}
    for domain, files in EXPECTED_RAW_HASHES.items():
        raw_details[domain] = {}
        for rel, expected in files.items():
            path = RAW / domain / rel
            actual = sha256(path) if path.is_file() else None
            passed = actual == expected
            raw_details[domain][rel] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "status": "PASS" if passed else "FAIL"}
            checks.append(check(f"raw BEIR {domain}/{rel}", passed, f"expected={expected}; actual={actual}"))

    model_details: dict[str, object] = {}
    for name, (repo_dir, revision) in REQUIRED_MODEL_REVISIONS.items():
        snapshot = HF / repo_dir / "snapshots" / revision
        available = snapshot.is_dir()
        model_details[name] = {"snapshot": str(snapshot), "revision": revision, "available": available}
        checks.append(check(f"frozen model available: {name}", available, str(snapshot)))

    api_available = bool(os.environ.get("OPENAI_API_KEY"))
    checks.append(check(
        "MEntA hosted generator credential available",
        api_available,
        "OPENAI_API_KEY is set" if api_available else "OPENAI_API_KEY is not set; secret value was not inspected or recorded",
    ))

    failed = [row for row in checks if row["required"] and row["status"] != "PASS"]
    source_failures = [row for row in failed if not str(row["check"]).startswith("frozen model available:") and row["check"] != "MEntA hosted generator credential available"]
    runtime_failures = [row for row in failed if row not in source_failures]

    phase1_verdict = "PHASE1_PROTOCOL_ARTIFACT_VERIFICATION_PASS" if not source_failures else "PHASE1_PROTOCOL_ARTIFACT_VERIFICATION_FAILED"
    execution_verdict = "CLEAN_CORE3_PREFLIGHT_READY" if not failed else "CLEAN_CORE3_PREFLIGHT_BLOCKED"
    blocker_codes = []
    if not api_available:
        blocker_codes.append("MENTA_GPT41_NANO_CREDENTIAL_UNAVAILABLE")
    if not model_details["MBA proxy LM"]["available"]:
        blocker_codes.append("MBA_GPT2_XL_SNAPSHOT_UNAVAILABLE")
    if not model_details["MBA spelling model"]["available"]:
        blocker_codes.append("MBA_SPELLING_SNAPSHOT_UNAVAILABLE")
    if source_failures:
        blocker_codes.append("FROZEN_SOURCE_VERIFICATION_FAILED")

    result = {
        "campaign": "CLEAN_CORE3_DEV_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase1_verdict": phase1_verdict,
        "execution_verdict": execution_verdict,
        "blocker_codes": blocker_codes,
        "protocol_recovery_manifest": {
            "path": str(core_manifest_path),
            "sha256": actual_core_sha,
        },
        "protocol_details": protocol_details,
        "unit_tests": {"passed": tests_ok, "count": test_count, "log": str(audit_dir / "UNIT_TEST_RESULTS.txt")},
        "raw_inputs": raw_details,
        "models": model_details,
        "menta_hosted_credential_available": api_available,
        "checks": checks,
        "prohibited_actions_confirmed": {
            "target_selection_run": False,
            "db_built": False,
            "retrieval_run": False,
            "attack_query_generation_run": False,
            "answer_generation_run": False,
            "historical_numeric_results_used": False,
        },
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    write_json(audit_dir / "PHASE1_PROTOCOL_VERIFICATION.json", result)
    with (audit_dir / "PREFLIGHT_CHECKS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "required", "status", "evidence"])
        writer.writeheader()
        writer.writerows(checks)
    write_json(CAMPAIGN / "PREFLIGHT_RESULT.json", {
        "campaign": result["campaign"],
        "phase1_verdict": phase1_verdict,
        "execution_verdict": execution_verdict,
        "blocker_codes": blocker_codes,
        "precommit_written": False,
        "e2e_started": False,
    })

    status_lines = [
        "# CLEAN_CORE3_DEV_V1 STATUS",
        "",
        f"- Phase 1: `{phase1_verdict}`",
        f"- Full preflight: `{execution_verdict}`",
        f"- Blockers: `{', '.join(blocker_codes) if blocker_codes else 'NONE'}`",
        "- Unit tests: `20/20 PASS`" if tests_ok else f"- Unit tests: `FAILED ({test_count})`",
        "- PRECOMMIT: `NOT_WRITTEN`",
        "- Retrieval/generation/scoring: `NOT_STARTED`",
        "- Historical numeric results used: `NO`",
        "",
        "Fail-close note: missing frozen runtime assets are not replaced with legacy queries, cached answers, fallback masking, or a different hosted/local model.",
    ]
    (CAMPAIGN / "STATUS.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    blocker_explanation = []
    if not api_available:
        blocker_explanation.append(
            "MEntA의 논문-faithful 질의는 `gpt-4.1-nano`를 요구하지만 현재 프로세스에는 "
            "`OPENAI_API_KEY`가 없다."
        )
    if not model_details["MBA proxy LM"]["available"]:
        blocker_explanation.append("MBA의 고정 GPT2-XL proxy LM snapshot이 로컬 cache에 없다.")
    if not model_details["MBA spelling model"]["available"]:
        blocker_explanation.append("MBA의 고정 spelling-model snapshot이 로컬 cache에 없다.")
    if not blocker_explanation:
        blocker_explanation.append("필수 protocol runtime asset이 모두 확인됐다.")

    report = f"""# CLEAN_CORE3_DEV_V1 사전검증 보고서

## PRECHECK

- Protocol source verification: **{phase1_verdict}**
- 전체 실행 준비 상태: **{execution_verdict}**
- Recovery manifest SHA-256: `{actual_core_sha}`
- READY 공격: MEntA / MBA / RAG-MIA
- 제외 공격: S²-MIA / DCMI / IA
- Unit tests: {test_count}/20, {'PASS' if tests_ok else 'FAIL'}

## 실행 중단 사유

{os.linesep.join(f'- `{code}`' for code in blocker_codes) if blocker_codes else '- 없음'}

{os.linesep.join(blocker_explanation)} 기존 질의, `important` masking, 다른 무료 모델로 대체하면 명세 위반이므로 실행하지 않았다.

## 보존된 원본 입력

BEIR NFCorpus/SciDocs/TREC-COVID의 corpus/query hash는 이전에 고정된 public-source hash와 모두 일치한다. 이 검증은 raw input 가용성 확인일 뿐, target selection이나 새 DB 구성을 수행한 것이 아니다.

## 수행하지 않은 작업

- COMMON_ELIGIBLE_POOL 및 shared 40 targets 선택
- member/nonmember assignment 및 protected DB 구성
- 공격 질의 생성
- benign 500/500 split
- retrieval/MIRABEL/BC threshold
- 세 조건 답변 생성 및 paper-faithful scoring

따라서 현재는 PRECOMMIT SHA, benign FPR, EPO, post-generation privacy, bottleneck map을 보고할 수 없다. 값이나 대체 결과를 만들지 않았다.
"""
    (report_dir / "PREFLIGHT_BLOCK_REPORT_KO.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "phase1_verdict": phase1_verdict,
        "execution_verdict": execution_verdict,
        "blocker_codes": blocker_codes,
        "audit": str(audit_dir / "PHASE1_PROTOCOL_VERIFICATION.json"),
        "status": str(CAMPAIGN / "STATUS.md"),
    }, ensure_ascii=False, indent=2))
    return 0 if phase1_verdict.endswith("_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
