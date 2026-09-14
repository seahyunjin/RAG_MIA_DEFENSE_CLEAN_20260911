#!/usr/bin/env python3
"""Generate fresh MBA (GPT2-XL difficult masks) and RAG-MIA queries."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
CAMPAIGN = ROOT / "experiments" / "CLEAN_CORE3_DEV_V1"
RECOVERY = ROOT / "experiments" / "CORE6_PROTOCOL_RECOVERY_V1"
INPUTS = CAMPAIGN / "inputs"
AUDITS = CAMPAIGN / "audits"
HF = Path("/home/traffic_3/workspace/.cache/huggingface/hub")
GPT2_PATH = HF / "models--openai-community--gpt2-xl" / "snapshots" / "15ea56dee5df4983c59b2538573817e1667135e2"
SPELL_PATH = HF / "models--oliverguhr--spelling-correction-english-base" / "snapshots" / "0e3958355a09d2816ed2701fdc2f4471d46c320e"
MASK_COUNT = 5
MAX_PROXY_TOKENS = 512

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "having", "he", "her", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "may", "might", "more", "most", "no", "not", "of",
    "on", "or", "our", "ours", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these", "they",
    "this", "those", "to", "too", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "you", "your",
}
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")

sys.path.insert(0, str(RECOVERY))
from protocols.mba.query_generator import make_query as make_mba_query, select_difficult_words  # noqa: E402
from protocols.rag_mia.query_generator import make_query as make_rag_mia_query  # noqa: E402


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rank_of_token(logits: torch.Tensor, token_id: int) -> float:
    actual = logits[token_id]
    return float((logits > actual).sum().item() + 1)


def spelling_corrections(words: list[str], fragmented: list[bool], tokenizer, model, device: str) -> dict[int, str]:
    indexes = [i for i, flag in enumerate(fragmented) if flag]
    corrections: dict[int, str] = {}
    for start in range(0, len(indexes), 32):
        batch_indices = indexes[start:start + 32]
        texts = [" ".join(words[max(0, i - 2): i + 1]) for i in batch_indices]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=24)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for i, original_input, corrected_phrase in zip(batch_indices, texts, decoded):
            original = words[i]
            corrected_words = WORD_RE.findall(corrected_phrase)
            corrected = corrected_words[-1] if corrected_words else original
            # The spelling model is only allowed to add an accepted variant.
            # Empty or nonlexical output cannot invalidate the source word.
            corrections[i] = corrected
    return corrections


def build_word_records(text: str, tokenizer, model, spell_tokenizer, spell_model, device: str) -> tuple[list[dict], dict]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_PROXY_TOKENS,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]

    max_char = max((end for start, end in offsets if end > start), default=0)
    word_matches = [m for m in WORD_RE.finditer(text[:max_char])]
    words = [m.group(0) for m in word_matches]
    token_positions_by_word: list[list[int]] = []
    for match in word_matches:
        positions = [i for i, (start, end) in enumerate(offsets) if end > start and start < match.end() and end > match.start()]
        token_positions_by_word.append(positions)
    fragmented = [len(pos) > 1 for pos in token_positions_by_word]
    corrections = spelling_corrections(words, fragmented, spell_tokenizer, spell_model, device)

    records = []
    for index, (word, positions) in enumerate(zip(words, token_positions_by_word)):
        ranks = [rank_of_token(logits[pos - 1], int(input_ids[0, pos])) for pos in positions if pos > 0]
        difficulty = max(ranks) if ranks else -1.0
        corrected = corrections.get(index, word)
        accepted = [word]
        if corrected.lower() != word.lower():
            accepted.append(corrected)
        records.append({
            "word": word,
            "eligible": bool(difficulty >= 0 and word.lower() not in STOPWORDS and len(word) > 1),
            "difficulty_rank": difficulty,
            "accepted_answers": accepted,
            "fragmented": fragmented[index],
            "corrected_variant": corrected if corrected.lower() != word.lower() else None,
            "proxy_token_positions": positions,
        })
    metadata = {
        "source_chars": len(text),
        "proxy_visible_chars": max_char,
        "proxy_tokens": int(input_ids.shape[1]),
        "word_records": len(records),
        "fragmented_words": sum(fragmented),
        "corrected_variants": sum(r["corrected_variant"] is not None for r in records),
    }
    return records, metadata


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen GPT2-XL query generator")
    device = "cuda:0"
    targets = list(csv.DictReader((INPUTS / "SHARED_TARGETS.csv").open(encoding="utf-8")))
    tokenizer = AutoTokenizer.from_pretrained(GPT2_PATH, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(GPT2_PATH, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
    spell_tokenizer = AutoTokenizer.from_pretrained(SPELL_PATH, local_files_only=True)
    spell_model = AutoModelForSeq2SeqLM.from_pretrained(SPELL_PATH, local_files_only=True, torch_dtype=torch.float16).to(device).eval()

    mba_rows = []
    rag_rows = []
    invalid = []
    word_records_path = AUDITS / "MBA_WORD_RECORDS.jsonl.gz"
    with gzip.open(word_records_path, "wt", encoding="utf-8") as records_out:
        for index, target in enumerate(targets, start=1):
            try:
                records, metadata = build_word_records(target["source_text"], tokenizer, model, spell_tokenizer, spell_model, device)
                selected = select_difficult_words(records, MASK_COUNT)
                query = make_mba_query(target["document_id"], target["membership"], records, MASK_COUNT)
                query.update({
                    "attack": "MBA",
                    "domain": target["domain"],
                    "query_index": 1,
                    "selected_word_positions": selected,
                    "generator_model": "openai-community/gpt2-xl@15ea56dee5df4983c59b2538573817e1667135e2",
                    "spelling_model": "oliverguhr/spelling-correction-english-base@0e3958355a09d2816ed2701fdc2f4471d46c320e",
                    "generator_config_sha256": sha(json.dumps({"mask_count": MASK_COUNT, "max_proxy_tokens": MAX_PROXY_TOKENS, "dtype": "float16", "difficulty": "full-vocabulary ordinal rank; max rank for fragmented word"}, sort_keys=True)),
                    "output_sha256": sha(query["query"]),
                    "proxy_metadata": metadata,
                })
                mba_rows.append(query)
                records_out.write(json.dumps({"target_id": target["document_id"], "metadata": metadata, "word_records": records}, ensure_ascii=False) + "\n")
            except Exception as exc:
                invalid.append({"target_id": target["document_id"], "membership": target["membership"], "error_type": type(exc).__name__, "error": str(exc)})

            rag = make_rag_mia_query(target["document_id"], target["membership"], target["source_text"])
            rag.update({
                "attack": "RAG-MIA",
                "domain": target["domain"],
                "query_index": 1,
                "generator_model": None,
                "generator_config_sha256": sha_file(RECOVERY / "protocols" / "rag_mia" / "query_generator.py"),
                "output_sha256": sha(rag["query"]),
            })
            rag_rows.append(rag)
            write_json(CAMPAIGN / "HEARTBEAT.json", {
                "stage": "MBA_RAGMIA_QUERY_GENERATION",
                "completed_targets": index,
                "total_targets": len(targets),
                "mba_valid": len(mba_rows),
                "mba_invalid": len(invalid),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(f"MBA_PROGRESS {index}/{len(targets)} valid={len(mba_rows)} invalid={len(invalid)}", flush=True)

    for path, rows in ((INPUTS / "MBA_ATTACK_QUERIES.jsonl", mba_rows), (INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl", rag_rows)):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(AUDITS / "MBA_INPUT_VALIDITY.json", {
        "status": "PASS" if len(mba_rows) == 40 else "ATTACK_INPUT_INCOMPATIBLE",
        "target_sessions": 40,
        "valid_sessions": len(mba_rows),
        "invalid": invalid,
        "mask_count": MASK_COUNT,
        "word_records_sha256": sha_file(word_records_path),
        "output_sha256": sha_file(INPUTS / "MBA_ATTACK_QUERIES.jsonl"),
    })
    write_json(AUDITS / "RAG_MIA_INPUT_VALIDITY.json", {
        "status": "PASS" if len(rag_rows) == 40 else "ATTACK_INPUT_INCOMPATIBLE",
        "target_sessions": 40,
        "valid_sessions": len(rag_rows),
        "output_sha256": sha_file(INPUTS / "RAG_MIA_ATTACK_QUERIES.jsonl"),
    })
    print(json.dumps({"mba_valid": len(mba_rows), "mba_invalid": len(invalid), "rag_mia_valid": len(rag_rows)}, indent=2))


if __name__ == "__main__":
    main()
