#!/usr/bin/env python3
"""Generate frozen fresh MEntA, MBA, and RAG-MIA query artifacts.

MEntA uses GPT-4.1-nano exactly once per target for the summary and Q5 packet;
responses are cached by immutable target ID for safe resume.  MBA and RAG-MIA
reuse the recovered paper-aligned local implementations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2"
RECOVERY = ROOT / "experiments" / "CORE6_PROTOCOL_RECOVERY_V1"
INPUTS = EXP / "inputs"
AUDITS = EXP / "audits"
RUNTIME = EXP / "runtime" / "menta_api"
HF = Path("/home/traffic_3/workspace/.cache/huggingface/hub")
GPT2_PATH = HF / "models--openai-community--gpt2-xl" / "snapshots" / "15ea56dee5df4983c59b2538573817e1667135e2"
SPELL_PATH = HF / "models--oliverguhr--spelling-correction-english-base" / "snapshots" / "0e3958355a09d2816ed2701fdc2f4471d46c320e"
MODEL = "gpt-4.1-nano"

sys.path.insert(0, str(RECOVERY))
from protocols.menta.query_generator import build_prompt, build_summary_prompt, build_system_prompt, make_session, parse_five_queries  # noqa: E402
from protocols.rag_mia.query_generator import make_query as make_rag_mia_query  # noqa: E402


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def targets() -> list[dict]:
    return list(csv.DictReader((INPUTS / "FRESH_SHARED_TARGETS.csv").open(encoding="utf-8")))


def verify_precommit_code() -> dict:
    path = EXP / "configs" / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2_PRECOMMIT.json"
    digest_path = EXP / "configs" / "DUALTAIL_MEMBER_EXPOSURE_REPLICATION_V2_PRECOMMIT.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("precommit missing")
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    if sha_file(path) != expected:
        raise RuntimeError("precommit checksum mismatch")
    pre = json.loads(path.read_text(encoding="utf-8"))
    expected_code = pre["code_sha256_at_precommit"].get(Path(__file__).name)
    if expected_code != sha_file(Path(__file__)):
        raise RuntimeError("query generation implementation changed after precommit")
    if sha_file(INPUTS / "FRESH_SHARED_TARGETS.csv") != pre["fresh_targets"]["sha256"]:
        raise RuntimeError("fresh targets changed after precommit")
    return pre


def extract_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"].strip()
    parts = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def call_responses(api_key: str, input_text: str, instructions: str, temperature: float, max_output_tokens: int) -> dict:
    payload = {
        "model": MODEL,
        "instructions": instructions,
        "input": input_text,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def cached_call(path: Path, **kwargs) -> dict:
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("model") == MODEL and stored.get("text"):
            return stored
    last_error = None
    for attempt in range(1, 7):
        try:
            raw = call_responses(**kwargs)
            text = extract_text(raw)
            if not text:
                raise RuntimeError("empty output_text")
            stored = {
                "created_utc": utcnow(), "model": MODEL,
                "response_id": raw.get("id"), "reported_model": raw.get("model"),
                "usage": raw.get("usage", {}), "text": text, "text_sha256": sha(text),
            }
            write_json(path, stored)
            return stored
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    body = json.loads(exc.read().decode("utf-8"))
                    last_error = f"HTTP {exc.code}: {body.get('error', {}).get('message', str(exc))}"
                except Exception:
                    last_error = f"HTTP {exc.code}"
            else:
                last_error = str(exc)
            if attempt < 6:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def generate_menta() -> None:
    verify_precommit_code()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key.startswith("sk-"):
        write_json(AUDITS / "MENTA_GENERATION_BLOCK.json", {
            "verdict": "OPENAI_API_KEY_REQUIRED", "created_utc": utcnow(),
            "completed_cached_targets": len(list(RUNTIME.glob("*_questions.json"))),
        })
        raise RuntimeError("OPENAI_API_KEY unavailable; no API request made")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    output_rows = []
    invalid = []
    usage = CounterLike()
    all_targets = targets()
    for index, target in enumerate(all_targets, start=1):
        tag = sha(target["document_id"])[:16]
        try:
            summary_result = cached_call(
                RUNTIME / f"{tag}_summary.json", api_key=api_key,
                input_text=build_summary_prompt(target["source_text"]),
                instructions="Return only the requested one-sentence topic-focused description.",
                temperature=0.3, max_output_tokens=256,
            )
            question_result = cached_call(
                RUNTIME / f"{tag}_questions.json", api_key=api_key,
                input_text=build_prompt(target["source_text"]), instructions=build_system_prompt(),
                temperature=0.7, max_output_tokens=1500,
            )
            questions = parse_five_queries(question_result["text"])
            session = make_session(target["document_id"], target["membership"], summary_result["text"].strip(), questions)
            for row in session:
                row.update({
                    "attack": "MEntA", "domain": target["domain"], "generator_model": MODEL,
                    "generator_config_sha256": sha(json.dumps({"model": MODEL, "summary_temperature": 0.3, "question_temperature": 0.7, "max_question_tokens": 1500}, sort_keys=True)),
                    "output_sha256": sha(row["query"]), "summary_response_sha256": summary_result["text_sha256"], "questions_response_sha256": question_result["text_sha256"],
                })
                output_rows.append(row)
            usage.add(summary_result.get("usage", {})); usage.add(question_result.get("usage", {}))
        except Exception as exc:
            invalid.append({"target_id": target["document_id"], "membership": target["membership"], "error_type": type(exc).__name__, "error": str(exc)})
        write_json(EXP / "HEARTBEAT.json", {"stage": "MENTA_FRESH_QUERY_GENERATION", "completed_targets": index, "total_targets": len(all_targets), "valid_sessions": len(output_rows)//5, "invalid_sessions": len(invalid), "updated_utc": utcnow()})
        print(f"MENTA_PROGRESS {index}/{len(all_targets)} valid={len(output_rows)//5} invalid={len(invalid)}", flush=True)
    output = INPUTS / "FRESH_MENTA_ATTACK_QUERIES.jsonl"
    write_jsonl(output, output_rows)
    audit = {
        "verdict": "FRESH_MENTA_INPUT_PASS" if len(output_rows) == 1000 and not invalid else "FRESH_MENTA_INPUT_INCOMPATIBLE",
        "targets": len(all_targets), "valid_sessions": len(output_rows)//5, "valid_queries": len(output_rows), "invalid": invalid,
        "usage": usage.values, "output_sha256": sha_file(output),
        "old_query_reuse": False,
    }
    write_json(AUDITS / "FRESH_MENTA_INPUT_AUDIT.json", audit)
    if audit["verdict"] != "FRESH_MENTA_INPUT_PASS":
        raise RuntimeError(audit["verdict"])
    print(json.dumps(audit, ensure_ascii=False, indent=2))


class CounterLike:
    def __init__(self) -> None:
        self.values = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def add(self, value: dict) -> None:
        for key in self.values:
            self.values[key] += int(value.get(key, 0) or 0)


def generate_local() -> None:
    verify_precommit_code()
    # Import the already-audited implementation and redirect its immutable I/O.
    source = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1" / "code" / "generate_mba_ragmia_queries.py"
    spec = importlib.util.spec_from_file_location("fresh_mba_impl", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load recovered MBA implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CAMPAIGN = EXP
    module.INPUTS = INPUTS
    module.AUDITS = AUDITS
    module.main()
    mba_path = INPUTS / "MBA_ATTACK_QUERIES.jsonl"
    rag_path = INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl"
    mba_rows = [json.loads(line) for line in mba_path.open(encoding="utf-8") if line.strip()]
    rag_rows = [json.loads(line) for line in rag_path.open(encoding="utf-8") if line.strip()]
    if len(mba_rows) != 200 or len(rag_rows) != 200:
        raise RuntimeError(f"fresh local query count mismatch: MBA={len(mba_rows)} RAG-MIA={len(rag_rows)}")
    target_ids = {r["document_id"] for r in targets()}
    for label, rows in (("MBA", mba_rows), ("RAG-MIA", rag_rows)):
        if {r["target_id"] for r in rows} != target_ids:
            raise RuntimeError(f"{label} target set mismatch")
    mba_fresh = INPUTS / "FRESH_MBA_ATTACK_QUERIES.jsonl"
    rag_fresh = INPUTS / "FRESH_RAG_MIA_ATTACK_QUERIES.jsonl"
    mba_path.replace(mba_fresh)
    rag_path.replace(rag_fresh)
    write_json(AUDITS / "FRESH_LOCAL_ATTACK_INPUT_AUDIT.json", {
        "verdict": "FRESH_MBA_RAG_MIA_INPUT_PASS", "mba_queries": 200, "rag_mia_queries": 200,
        "mba_sha256": sha_file(mba_fresh), "rag_mia_sha256": sha_file(rag_fresh), "target_set_exact": True,
    })
    print(json.dumps({"verdict": "FRESH_MBA_RAG_MIA_INPUT_PASS", "mba": 200, "rag_mia": 200}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("menta", "local"))
    args = parser.parse_args()
    if args.mode == "menta":
        generate_menta()
    else:
        generate_local()


if __name__ == "__main__":
    main()
