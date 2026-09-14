import json
import os
import argparse
from pathlib import Path
from typing import Callable, List, Dict, Optional

def read_jsonl(file_path: str) -> List[dict]:
    """Read JSONL file and return list of records."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def read_json(file_path: str) -> dict:
    """Read JSON file and return dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(data: dict, file_path: str):
    """Write a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(data: List[dict], file_path: str):
    """Write list of records to JSONL file."""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def append_jsonl_records(records: List[dict], file_path: str):
    """Append records to a JSONL file without rewriting existing content."""
    if not records:
        return
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def merge_jsonl_by_key(
    file_path: str,
    new_records: List[dict],
    key: str = '_id',
    sort_key: Optional[Callable] = None,
):
    """Merge records into a JSONL file, replacing entries that share the same key."""
    if not new_records:
        return

    existing = {}
    if os.path.exists(file_path):
        for record in read_jsonl(file_path):
            existing[record.get(key)] = record

    for record in new_records:
        existing[record.get(key)] = record

    merged = list(existing.values())
    if sort_key is not None:
        merged.sort(key=sort_key)
    write_jsonl(merged, file_path)


def merge_query_variation_records(output_path: str, new_records: List[dict]):
    """Merge MEntA-style query records, replacing all variations for each target_doc_id."""
    if not new_records:
        return

    if os.path.exists(output_path):
        all_records = read_jsonl(output_path)
    else:
        all_records = []

    regenerated_doc_ids = {record.get('target_doc_id') for record in new_records}
    all_records = [
        record for record in all_records
        if record.get('target_doc_id') not in regenerated_doc_ids
    ]
    all_records.extend(new_records)
    all_records.sort(
        key=lambda record: (record.get('target_doc_id', ''), record.get('variation_index', 0))
    )
    write_jsonl(all_records, output_path)