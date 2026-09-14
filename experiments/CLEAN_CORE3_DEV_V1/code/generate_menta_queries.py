#!/usr/bin/env python3
"""Generate fresh paper-faithful MEntA summary+question sessions via API."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
CAMPAIGN = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
RECOVERY = ROOT / "experiments" / "CORE6_PROTOCOL_RECOVERY_V1"
INPUTS = CAMPAIGN / "inputs"
RUNTIME = CAMPAIGN / "runtime" / "menta_api"
OUTPUT = INPUTS / "MENTA_ATTACK_QUERIES.jsonl"
MODEL = "gpt-4.1-nano"

sys.path.insert(0, str(RECOVERY))
from protocols.menta.query_generator import (  # noqa: E402
    build_prompt,
    build_summary_prompt,
    build_system_prompt,
    make_session,
    parse_five_queries,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"].strip()
    parts = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def call_responses(*, api_key: str, input_text: str, instructions: str, temperature: float, max_output_tokens: int) -> dict:
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
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": MODEL,
                "response_id": raw.get("id"),
                "reported_model": raw.get("model"),
                "usage": raw.get("usage", {}),
                "text": text,
                "text_sha256": sha(text),
            }
            write_json(path, stored)
            return stored
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    body = json.loads(exc.read().decode("utf-8"))
                    safe_message = body.get("error", {}).get("message", str(exc))
                except Exception:
                    safe_message = str(exc)
                last_error = f"HTTP {exc.code}: {safe_message}"
            else:
                last_error = str(exc)
            if attempt < 6:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY unavailable or malformed")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    targets = list(csv.DictReader((INPUTS / "SHARED_TARGETS.csv").open(encoding="utf-8")))
    results = []
    invalid = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for index, target in enumerate(targets, start=1):
        tag = sha(target["document_id"])[:16]
        document = target["source_text"]
        try:
            summary_result = cached_call(
                RUNTIME / f"{tag}_summary.json",
                api_key=api_key,
                input_text=build_summary_prompt(document),
                instructions="Return only the requested one-sentence topic-focused description.",
                temperature=0.3,
                max_output_tokens=256,
            )
            question_result = cached_call(
                RUNTIME / f"{tag}_questions.json",
                api_key=api_key,
                input_text=build_prompt(document),
                instructions=build_system_prompt(),
                temperature=0.7,
                max_output_tokens=1500,
            )
            summary = summary_result["text"].strip()
            questions = parse_five_queries(question_result["text"])
            session = make_session(target["document_id"], target["membership"], summary, questions)
            for row in session:
                row.update({
                    "attack": "MEntA",
                    "domain": target["domain"],
                    "generator_model": MODEL,
                    "generator_config_sha256": sha(json.dumps({"model": MODEL, "summary_temperature": 0.3, "question_temperature": 0.7, "max_question_tokens": 1500}, sort_keys=True)),
                    "output_sha256": sha(row["query"]),
                    "summary_response_sha256": summary_result["text_sha256"],
                    "questions_response_sha256": question_result["text_sha256"],
                })
                results.append(row)
            for result in (summary_result, question_result):
                for key in usage:
                    usage[key] += int(result.get("usage", {}).get(key, 0) or 0)
        except Exception as exc:
            invalid.append({"target_id": target["document_id"], "membership": target["membership"], "error_type": type(exc).__name__, "error": str(exc)})
        write_json(CAMPAIGN / "HEARTBEAT.json", {
            "stage": "MENTA_QUERY_GENERATION",
            "completed_targets": index,
            "total_targets": len(targets),
            "valid_sessions": len(results) // 5,
            "invalid_sessions": len(invalid),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"MENTA_PROGRESS {index}/{len(targets)} valid={len(results)//5} invalid={len(invalid)}", flush=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(CAMPAIGN / "audits" / "MENTA_INPUT_VALIDITY.json", {
        "status": "PASS" if len(results) == 200 else "ATTACK_INPUT_INCOMPATIBLE",
        "target_sessions": 40,
        "valid_sessions": len(results) // 5,
        "valid_queries": len(results),
        "invalid": invalid,
        "usage": usage,
        "output_file": str(OUTPUT),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    })
    if not results:
        raise RuntimeError("no valid MEntA session generated")
    print(json.dumps({"valid_sessions": len(results)//5, "invalid_sessions": len(invalid), "queries": len(results), "usage": usage}, indent=2))


if __name__ == "__main__":
    main()
