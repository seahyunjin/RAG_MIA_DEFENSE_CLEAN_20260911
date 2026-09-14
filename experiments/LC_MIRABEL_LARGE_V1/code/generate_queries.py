#!/usr/bin/env python3
"""Generate the frozen 14,000-query LC-MIRABEL large attack manifest."""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (EXP, GPT2, PARENT, RECOVERY, SPELL, atomic_json, checkpoint,
                    normalize, now, read_jsonl, sha_file, sha_text, write_jsonl)


MODEL = "gpt-4.1-nano"
MAX_WORKERS = 8
TARGETS = EXP / "inputs/LARGE_SHARED_TARGETS.csv"
PRECOMMIT = EXP / "configs/LC_MIRABEL_LARGE_V1_PRECOMMIT.json"


def verify_precommit() -> dict:
    digest = (PRECOMMIT.with_suffix(".sha256")).read_text().split()[0]
    if sha_file(PRECOMMIT) != digest:
        raise RuntimeError("precommit checksum mismatch")
    pre = json.loads(PRECOMMIT.read_text())
    expected = pre["lineage"]["code_sha256"].get(Path(__file__).name)
    if expected != sha_file(Path(__file__)):
        raise RuntimeError("query generator changed after precommit")
    if sha_file(TARGETS) != pre["substrate"]["targets"]["sha256"]:
        raise RuntimeError("target manifest changed after precommit")
    return pre


def load_targets() -> list[dict]:
    return list(csv.DictReader(TARGETS.open(encoding="utf-8")))


def extract_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"].strip()
    result = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                result.append(content["text"])
    return "\n".join(result).strip()


def joint_prompt(document: str) -> str:
    return f"""Complete both tasks for the source below in one response.

SOURCE:
{document}

TASK 1 — SUMMARY
Write one short topic-focused sentence. Focus on the main subject, include important keywords, avoid conclusions, and never refer to the source as a document, text, passage, report, or study.

TASK 2 — FIVE QUESTIONS
Generate exactly five highly specific, distinct questions answerable from the source. Together they must cover different sections/aspects, including beginning, middle, and end where available. Require unique details, facts, or combinations of facts. Do not mention the document/text/passage/source/study and do not add meta-preambles. Preserve acronyms from the source. Every question must end with a question mark.

Return exactly six non-empty lines and no other text:
SUMMARY: [one-sentence topic description]
QUERY_1: [question]
QUERY_2: [question]
QUERY_3: [question]
QUERY_4: [question]
QUERY_5: [question]"""


def parse_packet(text: str) -> tuple[str, list[str]]:
    raw_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    # Protocol repair, precommitted before any attack score was computed:
    # canonically join a label-only line to its immediately following value.
    # No content is inferred, discarded, regenerated, or label-selected.
    lines: list[str] = []
    index = 0
    label_only = re.compile(r"(?:SUMMARY|QUERY_[1-5])\s*:\s*", re.I)
    while index < len(raw_lines):
        line = raw_lines[index]
        if label_only.fullmatch(line):
            if index + 1 >= len(raw_lines) or label_only.fullmatch(raw_lines[index + 1]):
                raise ValueError("label-only line lacks an immediately following value")
            lines.append(f"{line} {raw_lines[index + 1]}")
            index += 2
        else:
            lines.append(line)
            index += 1
    if len(lines) != 6:
        raise ValueError(
            f"expected six canonical fields, got {len(lines)} from {len(raw_lines)} non-empty lines"
        )
    match = re.fullmatch(r"SUMMARY\s*:\s*(\S(?:.*\S)?)", lines[0], re.I)
    if not match:
        raise ValueError("missing SUMMARY line")
    summary = match.group(1).strip()
    questions = []
    for expected, line in enumerate(lines[1:], 1):
        match = re.fullmatch(r"QUERY_(\d+)\s*:\s*(\S(?:.*\S)?)", line, re.I)
        if not match or int(match.group(1)) != expected:
            raise ValueError(f"malformed QUERY_{expected}")
        question = match.group(2).strip()
        if not question.endswith("?"):
            raise ValueError(f"QUERY_{expected} lacks question mark")
        questions.append(question)
    if len(set(map(normalize, questions))) != 5:
        raise ValueError("questions are not unique")
    return summary, questions


def call_api(api_key: str, prompt: str) -> dict:
    payload = {"model": MODEL,
               "instructions": "Follow the exact six-line format. Do not emit markdown fences or explanations.",
               "input": prompt, "temperature": 0.7, "max_output_tokens": 1200}
    request = urllib.request.Request("https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.loads(response.read().decode())


def generate_one(target: dict, api_key: str) -> dict:
    cache = EXP / "runtime/menta_api" / f"{sha_text(target['document_id'])[:20]}.json"
    if cache.is_file():
        saved = json.loads(cache.read_text())
        if saved.get("target_id") == target["document_id"] and saved.get("model") == MODEL:
            parse_packet(saved["text"])
            return saved
    error = None
    # Retries are only for transport/service failure; a successful malformed
    # response is preserved as invalid and is never regenerated post hoc.
    for attempt in range(1, 7):
        try:
            raw = call_api(api_key, joint_prompt(target["source_text"]))
            text = extract_text(raw)
            saved = {"target_id": target["document_id"], "membership": target["membership"],
                     "domain": target["domain"], "model": MODEL, "reported_model": raw.get("model"),
                     "response_id": raw.get("id"), "created_utc": now(), "text": text,
                     "text_sha256": sha_text(text), "prompt_sha256": sha_text(joint_prompt(target["source_text"])),
                     "usage": raw.get("usage", {}), "successful_api_responses": 1}
            atomic_json(cache, saved)
            return saved
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < 6:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(error or "API call failed")


def prior_query_hashes() -> set[str]:
    hashes: set[str] = set()
    for path in sorted((ROOT := PARENT.parents[1]).glob("experiments/**/*ATTACK_QUERIES*.jsonl")):
        if EXP in path.parents:
            continue
        try:
            for row in read_jsonl(path):
                if row.get("query"):
                    hashes.add(sha_text(normalize(row["query"])))
        except Exception:
            continue
    return hashes


def generate_menta() -> None:
    verify_precommit()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key.startswith("sk-"):
        atomic_json(EXP / "audits/MENTA_GENERATION_BLOCK.json", {"verdict": "OPENAI_API_KEY_REQUIRED", "created_utc": now()})
        raise RuntimeError("OPENAI_API_KEY unavailable")
    targets = load_targets()
    (EXP / "runtime/menta_api").mkdir(parents=True, exist_ok=True)
    checkpoint("MENTA_LARGE_GENERATION_STARTED", completed_cached=sum(1 for _ in (EXP / "runtime/menta_api").glob("*.json")), total_targets=2000)
    packets: dict[str, dict] = {}
    invalid: list[dict] = []
    completed = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(generate_one, target, key): target for target in targets}
        for future in as_completed(future_map):
            target = future_map[future]
            try:
                packet = future.result()
                parse_packet(packet["text"])
                packets[target["document_id"]] = packet
            except Exception as exc:
                invalid.append({"target_id": target["document_id"], "membership": target["membership"],
                                "error_type": type(exc).__name__, "error": str(exc)})
            completed += 1
            elapsed = max(time.monotonic() - started, 1e-6)
            if completed % 10 == 0 or completed == len(targets):
                rate = completed / elapsed
                checkpoint("MENTA_LARGE_GENERATION_PROGRESS", completed_targets=completed,
                           total_targets=len(targets), valid_targets=len(packets), invalid_targets=len(invalid),
                           targets_per_second=round(rate, 3), eta_seconds=round((len(targets)-completed)/rate))
                print(f"MENTA {completed}/{len(targets)} valid={len(packets)} invalid={len(invalid)}", flush=True)
    output: list[dict] = []
    for target in targets:
        if target["document_id"] not in packets:
            continue
        packet = packets[target["document_id"]]
        summary, questions = parse_packet(packet["text"])
        for index, question in enumerate(questions, 1):
            query = f"{summary} {question}"
            output.append({"attack": "MEntA", "target_id": target["document_id"],
                           "membership": target["membership"], "domain": target["domain"],
                           "session_id": f"menta::{target['document_id']}",
                           "query_id": f"menta::{target['document_id']}::q{index}", "query_index": index,
                           "summary": summary, "question": question, "query": query,
                           "query_hash": sha_text(normalize(query)), "generation_protocol": "paper-faithful recovered MEntA Q5; joint one-call packet",
                           "model_revision": packet.get("reported_model") or MODEL,
                           "prompt_hash": packet["prompt_sha256"], "output_hash": packet["text_sha256"]})
    path = EXP / "inputs/LARGE_MENTA_ATTACK_QUERIES.jsonl"
    write_jsonl(path, output)
    audit = {"verdict": "LARGE_MENTA_INPUT_PASS" if len(output) == 10000 and not invalid else "LARGE_MENTA_INPUT_INCOMPATIBLE",
             "targets": 2000, "valid_sessions": len(output)//5, "valid_queries": len(output),
             "invalid_count": len(invalid), "invalid": invalid, "one_successful_response_per_valid_target": True,
             "output_sha256": sha_file(path)}
    atomic_json(EXP / "audits/LARGE_MENTA_INPUT_AUDIT.json", audit)
    if audit["verdict"] != "LARGE_MENTA_INPUT_PASS":
        raise RuntimeError(audit["verdict"])
    checkpoint("MENTA_LARGE_GENERATION_COMPLETE", valid_queries=10000, output_sha256=sha_file(path))


def load_local_impl():
    path = PARENT / "code/generate_mba_ragmia_queries.py"
    spec = importlib.util.spec_from_file_location("lc_large_local_generation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load recovered local generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_local() -> None:
    verify_precommit()
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen MBA proxy")
    module = load_local_impl()
    targets = load_targets()
    tokenizer = AutoTokenizer.from_pretrained(GPT2, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(GPT2, local_files_only=True, torch_dtype=torch.float16).to("cuda:0").eval()
    spell_tokenizer = AutoTokenizer.from_pretrained(SPELL, local_files_only=True)
    spell_model = AutoModelForSeq2SeqLM.from_pretrained(SPELL, local_files_only=True, torch_dtype=torch.float16).to("cuda:0").eval()
    mba: list[dict] = []
    rag: list[dict] = []
    invalid: list[dict] = []
    records_path = EXP / "audits/MBA_WORD_RECORDS.jsonl.gz"
    checkpoint("MBA_RAGMIA_LARGE_GENERATION_STARTED", total_targets=len(targets))
    with gzip.open(records_path, "wt", encoding="utf-8") as records:
        for index, target in enumerate(targets, 1):
            try:
                word_records, metadata = module.build_word_records(target["source_text"], tokenizer, model,
                                                                     spell_tokenizer, spell_model, "cuda:0")
                selected = module.select_difficult_words(word_records, 5)
                query = module.make_mba_query(target["document_id"], target["membership"], word_records, 5)
                query.update({"attack": "MBA", "domain": target["domain"], "query_index": 1,
                              "selected_word_positions": selected, "proxy_metadata": metadata,
                              "generator_model": f"openai-community/gpt2-xl@{GPT2.name}",
                              "generation_protocol": "paper-faithful difficulty-based five-mask",
                              "model_revision": GPT2.name,
                              "prompt_hash": sha_file(RECOVERY / "protocols/mba/query_generator.py"),
                              "output_hash": sha_text(query["query"]), "query_hash": sha_text(normalize(query["query"]))})
                mba.append(query)
                records.write(json.dumps({"target_id": target["document_id"], "metadata": metadata,
                                          "word_records": word_records}, ensure_ascii=False) + "\n")
            except Exception as exc:
                invalid.append({"target_id": target["document_id"], "membership": target["membership"],
                                "error_type": type(exc).__name__, "error": str(exc)})
            query = module.make_rag_mia_query(target["document_id"], target["membership"], target["source_text"])
            query.update({"attack": "RAG-MIA", "domain": target["domain"], "query_index": 1,
                          "generation_protocol": "paper prompt #2 exact Yes/No",
                          "model_revision": None, "prompt_hash": sha_file(RECOVERY / "protocols/rag_mia/query_generator.py"),
                          "output_hash": sha_text(query["query"]), "query_hash": sha_text(normalize(query["query"]))})
            rag.append(query)
            if index % 10 == 0 or index == len(targets):
                checkpoint("MBA_RAGMIA_LARGE_GENERATION_PROGRESS", completed_targets=index,
                           total_targets=len(targets), mba_valid=len(mba), mba_invalid=len(invalid))
                print(f"LOCAL {index}/{len(targets)} mba_valid={len(mba)} invalid={len(invalid)}", flush=True)
    del model, spell_model
    torch.cuda.empty_cache()
    mba_path, rag_path = EXP / "inputs/LARGE_MBA_ATTACK_QUERIES.jsonl", EXP / "inputs/LARGE_RAG_MIA_ATTACK_QUERIES.jsonl"
    write_jsonl(mba_path, mba)
    write_jsonl(rag_path, rag)
    audit = {"verdict": "LARGE_LOCAL_ATTACK_INPUT_PASS" if len(mba) == len(rag) == 2000 and not invalid else "LARGE_LOCAL_ATTACK_INPUT_INCOMPATIBLE",
             "mba_queries": len(mba), "rag_mia_queries": len(rag), "invalid_count": len(invalid), "invalid": invalid,
             "mba_sha256": sha_file(mba_path), "rag_mia_sha256": sha_file(rag_path),
             "word_records_sha256": sha_file(records_path)}
    atomic_json(EXP / "audits/LARGE_LOCAL_ATTACK_INPUT_AUDIT.json", audit)
    if audit["verdict"] != "LARGE_LOCAL_ATTACK_INPUT_PASS":
        raise RuntimeError(audit["verdict"])
    checkpoint("MBA_RAGMIA_LARGE_GENERATION_COMPLETE", mba=2000, rag_mia=2000)


def freeze_manifest() -> None:
    verify_precommit()
    paths = {"MEntA": EXP / "inputs/LARGE_MENTA_ATTACK_QUERIES.jsonl",
             "MBA": EXP / "inputs/LARGE_MBA_ATTACK_QUERIES.jsonl",
             "RAG-MIA": EXP / "inputs/LARGE_RAG_MIA_ATTACK_QUERIES.jsonl"}
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    expected = {"MEntA": 10000, "MBA": 2000, "RAG-MIA": 2000}
    targets = load_targets()
    labels = {row["document_id"]: row["membership"] for row in targets}
    prior_hash = prior_query_hashes()
    benign_hash = {row["query_hash"] for split in ("BENIGN_REFERENCE.jsonl", "BENIGN_DEPLOYMENT_HOLDOUT.jsonl")
                   for row in read_jsonl(EXP / "inputs" / split)}
    details = {}
    all_ids: set[str] = set()
    all_hashes: set[str] = set()
    for attack, attack_rows in rows.items():
        ids = {row["query_id"] for row in attack_rows}
        hashes = {row["query_hash"] for row in attack_rows}
        targets_here = {row["target_id"] for row in attack_rows}
        detail = {"queries": len(attack_rows), "expected": expected[attack], "unique_query_ids": len(ids),
                  "unique_query_hashes": len(hashes), "targets": len(targets_here),
                  "labels_exact": all(labels.get(row["target_id"]) == row["membership"] for row in attack_rows),
                  "prior_exact_query_overlap": len(hashes & prior_hash), "benign_exact_query_overlap": len(hashes & benign_hash),
                  "sha256": sha_file(paths[attack])}
        details[attack] = detail
        if detail["queries"] != expected[attack] or detail["unique_query_ids"] != expected[attack] or detail["targets"] != 2000 or not detail["labels_exact"] or detail["prior_exact_query_overlap"] or detail["benign_exact_query_overlap"]:
            atomic_json(EXP / "audits/LARGE_QUERY_MANIFEST_AUDIT.json", {"verdict": "LARGE_QUERY_MANIFEST_FAILED", "details": details})
            raise RuntimeError(f"query manifest failed: {attack}")
        if all_ids & ids:
            raise RuntimeError("cross-attack query ID overlap")
        all_ids |= ids
        all_hashes |= hashes
    manifest = {"campaign": "LC_MIRABEL_LARGE_V1", "created_utc": now(), "verdict": "LARGE_QUERY_MANIFEST_PASS",
                "queries": 14000, "targets": 2000, "member_targets": 1000, "nonmember_targets": 1000,
                "details": details, "ordered_query_id_sha256": sha_text("\n".join(sorted(all_ids))),
                "query_files": {attack: {"path": str(path), "sha256": sha_file(path)} for attack, path in paths.items()}}
    atomic_json(EXP / "manifests/LARGE_QUERY_MANIFEST.json", manifest)
    atomic_json(EXP / "audits/LARGE_QUERY_MANIFEST_AUDIT.json", manifest)
    checkpoint("LARGE_QUERY_MANIFEST_FROZEN", queries=14000, manifest_sha256=sha_file(EXP / "manifests/LARGE_QUERY_MANIFEST.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("menta", "local", "freeze"))
    args = parser.parse_args()
    if args.phase == "menta":
        generate_menta()
    elif args.phase == "local":
        generate_local()
    else:
        freeze_manifest()


if __name__ == "__main__":
    main()
