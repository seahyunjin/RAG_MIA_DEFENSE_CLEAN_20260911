#!/usr/bin/env python3
"""
Prepare Detection Dataset
Extracts 1000 benign queries and 1000 attack queries from each attack type
for creating a balanced detection dataset.
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_jsonl, write_jsonl


def extract_query_text(record: Dict, attack_type: str) -> str:
    """Extract the query text from different attack type formats."""
    if attack_type == 'IA-MIA':
        return record.get('text', '')
    elif attack_type == 'MBA':
        return record.get('query_prompt', '')
    elif attack_type == 'MEntA':
        return record.get('text', '')
    elif attack_type == 'S2-MIA':
        return record.get('query_prompt', '')
    elif attack_type == 'DCMI':
        return record.get('query_prompt', '')
    else:
        return record.get('text', '') or record.get('query_text', '') or record.get('masked_text', '')


def get_record_id(record: Dict, attack_type: str) -> str:
    """Get unique identifier for a record."""
    return record.get('_id', '') or record.get('target_doc_id', '')


def load_attack_queries(
    attack_file: Path,
    attack_type: str,
    n_samples: int = 1000,
    existing_keys: set = None
) -> List[Dict]:
    """Load attack queries from file, skipping already-present entries.

    n_samples <= 0 means load all available queries.
    """
    if existing_keys is None:
        existing_keys = set()

    if not attack_file.exists():
        print(f"  Warning: Attack queries file not found: {attack_file}")
        return []

    records = read_jsonl(str(attack_file))
    random.shuffle(records)

    queries = []
    for record in records:
        query_text = extract_query_text(record, attack_type)
        record_id = get_record_id(record, attack_type)

        if not query_text or not record_id:
            continue

        key = (record_id, attack_type)
        if key in existing_keys:
            continue

        entry = {
            '_id': record_id,
            'text': query_text,
            'attack_type': attack_type,
            'is_attack': True,
        }

        # Preserve DCMI-specific fields
        if attack_type == 'DCMI':
            entry['is_perturbed'] = record.get('is_perturbed', False)
            entry['membership_label'] = record.get('membership_label', None)
            if record.get('is_perturbed'):
                entry['original_id'] = record.get('original_id', '')

        queries.append(entry)
        existing_keys.add(key)

        if n_samples > 0 and len(queries) >= n_samples:
            break

    return queries


def load_existing_dataset(output_file: Path) -> tuple:
    """Load existing dataset and return (records, existing_keys set)."""
    if not output_file.exists():
        return [], set()

    records = read_jsonl(str(output_file))
    existing_keys = {(r.get('_id', ''), r.get('attack_type', '')) for r in records}
    print(f"  Found existing dataset with {len(records)} entries ({len(existing_keys)} unique keys)")
    return records, existing_keys


def clean_dcmi_entries(existing_queries: List[Dict]) -> tuple:
    """
    Remove all DCMI-related entries (DCMI, DCMI_original, DCMI_perturbed).
    Returns (cleaned_queries, cleaned_existing_keys).
    """
    dcmi_types = {'DCMI', 'DCMI_original', 'DCMI_perturbed'}
    cleaned = [q for q in existing_queries if q.get('attack_type') not in dcmi_types]
    removed = len(existing_queries) - len(cleaned)
    if removed > 0:
        print(f"  Removed {removed} stale DCMI entries (DCMI / DCMI_original / DCMI_perturbed)")
    cleaned_keys = {(r.get('_id', ''), r.get('attack_type', '')) for r in cleaned}
    return cleaned, cleaned_keys



def load_member_doc_ids(member_file: Path) -> set:
    """Load document ids from corpus_member.jsonl for benign-query filtering."""
    if not member_file.exists():
        print(f"  Warning: member corpus not found for benign filtering: {member_file}")
        return set()

    member_ids = {row.get('_id', '') for row in read_jsonl(str(member_file))}
    member_ids.discard('')
    print(f"  Loaded {len(member_ids)} member document ids for benign filtering")
    return member_ids


def get_benign_source_id(record: Dict) -> str:
    """Return the document id that a benign query targets."""
    source_id = record.get('source_id') or record.get('_id') or record.get('query_id') or ''
    return str(source_id).split('::benign_', 1)[0]


def clean_nonmember_benign_entries(existing_queries: List[Dict], member_ids: set) -> tuple:
    """Remove cached benign rows whose source document is not in corpus_member."""
    cleaned = []
    removed = 0
    for query in existing_queries:
        if query.get('attack_type') == 'benign' and get_benign_source_id(query) not in member_ids:
            removed += 1
            continue
        cleaned.append(query)

    if removed > 0:
        print(f"  Removed {removed} benign entries not targeting corpus_member documents")

    cleaned_keys = {(r.get('_id', ''), r.get('attack_type', '')) for r in cleaned}
    return cleaned, cleaned_keys


def main():
    parser = argparse.ArgumentParser(
        description='Prepare detection dataset with benign and attack queries'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['BeIR_nfcorpus', 'BeIR_scidocs', 'BeIR_trec-covid'],
        help='Datasets to process'
    )
    parser.add_argument(
        '--attack_types',
        type=str,
        nargs='+',
        default=['IA-MIA', 'MBA', 'MEntA', 'S2-MIA', 'DCMI'],
        help='Attack types to include'
    )
    parser.add_argument(
        '--data_base',
        type=str,
        default='data',
        help='Base path for data'
    )
    parser.add_argument(
        '--results_base',
        type=str,
        default='results',
        help='Base path for results'
    )
    parser.add_argument(
        '--benign_data_base',
        type=str,
        default=None,
        help='Base path for benign_queries.jsonl; defaults to data_base'
    )
    parser.add_argument(
        '--n_benign',
        type=int,
        default=1000,
        help='Number of benign queries to sample'
    )
    parser.add_argument(
        '--n_attack_per_type',
        type=int,
        default=1000,
        help='Number of attack queries to sample per attack type; 0 means all available'
    )
    parser.add_argument(
        '--all_attack_queries',
        action='store_true',
        help='Use all available generated attack queries for each attack type'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--ia_queries_file',
        type=str,
        default='IA-MIA/{dataset}/queries/queries_5.jsonl',
        help='Path template (relative to results_base) for IA-MIA query files'
    )
    parser.add_argument(
        '--menta_queries_file',
        type=str,
        default='MEntA/{dataset}/queries/queries_5v.jsonl',
        help='Path template (relative to results_base) for MEntA query files'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    attack_file_templates = {
        'IA-MIA':  args.ia_queries_file,
        'MBA':     'MBA/{dataset}/queries/queries.jsonl',
        'MEntA':   args.menta_queries_file,
        'S2-MIA':  'S2-MIA/{dataset}/queries/queries.jsonl',
        'DCMI':    'DCMI/{dataset}/queries/queries.jsonl',
    }

    print(f"\n{'#'*60}")
    print("PREPARING DETECTION DATASET")
    benign_data_base = args.benign_data_base or args.data_base

    print(f"Datasets: {args.datasets}")
    print(f"Attack types: {args.attack_types}")
    print(f"Data base: {args.data_base}")
    print(f"Benign query base: {benign_data_base}")
    print(f"Benign queries per dataset: {args.n_benign}")
    attack_limit = 0 if args.all_attack_queries else args.n_attack_per_type
    attack_limit_label = 'all available' if attack_limit <= 0 else str(attack_limit)
    print(f"Attack queries per type: {attack_limit_label}")
    print(f"Random seed: {args.seed}")
    print(f"{'#'*60}")
    
    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print(f"{'='*60}")
        
        output_file = Path(args.data_base) / dataset / 'queries_dataset.jsonl'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nChecking for existing dataset: {output_file}")
        existing_queries, _ = load_existing_dataset(output_file)

        member_ids = load_member_doc_ids(Path(args.data_base) / dataset / 'corpus_member.jsonl')

        # Always purge all DCMI variants and re-sample fresh
        existing_queries, existing_keys = clean_dcmi_entries(existing_queries)
        existing_queries, existing_keys = clean_nonmember_benign_entries(existing_queries, member_ids)

        new_queries = []
        stats = defaultdict(int)
        for q in existing_queries:
            stats[q.get('attack_type', 'unknown')] += 1

        # --- Benign queries ---
        benign_file = Path(benign_data_base) / dataset / 'benign_queries.jsonl'
        print(f"\nLoading benign queries from: {benign_file}")

        benign_needed = max(0, args.n_benign - stats.get('benign', 0))
        if benign_needed == 0:
            print(f"  Already have {stats['benign']} benign queries, skipping")
        else:
            if benign_file.exists():
                all_records = list(enumerate(read_jsonl(str(benign_file))))
                random.shuffle(all_records)
                added = 0
                skipped_nonmember = 0
                for source_idx, record in all_records:
                    if added >= benign_needed:
                        break
                    raw_id = get_benign_source_id(record)
                    if raw_id not in member_ids:
                        skipped_nonmember += 1
                        continue
                    rid = f"{raw_id}::benign_{source_idx}"
                    key = (rid, 'benign')
                    if key in existing_keys:
                        continue
                    entry = {
                        '_id': rid,
                        'source_id': raw_id,
                        'text': record.get('text', record.get('query', '')),
                        'attack_type': 'benign',
                        'is_attack': False,
                    }
                    new_queries.append(entry)
                    existing_keys.add(key)
                    added += 1
                print(f"  Added {added} new benign queries (had {stats.get('benign', 0)})")
                if skipped_nonmember > 0:
                    print(f"  Skipped {skipped_nonmember} benign queries outside corpus_member")
                stats['benign'] = stats.get('benign', 0) + added
            else:
                print(f"  Warning: Benign queries file not found: {benign_file}")

        # --- Attack queries ---
        for attack_type in args.attack_types:
            if attack_type not in attack_file_templates:
                print(f"Warning: Unknown attack type: {attack_type}")
                continue

            attack_file_path = attack_file_templates[attack_type].format(dataset=dataset)
            attack_file = Path(args.results_base) / attack_file_path

            already_have = stats.get(attack_type, 0)
            needed = 0 if attack_limit <= 0 else max(0, attack_limit - already_have)

            print(f"\nLoading {attack_type} queries from: {attack_file}")
            if attack_limit > 0 and needed == 0:
                print(f"  Already have {already_have} {attack_type} queries, skipping")
                continue

            attack_queries = load_attack_queries(
                attack_file,
                attack_type,
                n_samples=needed,
                existing_keys=existing_keys,
            )

            print(f"  Added {len(attack_queries)} new {attack_type} queries (had {already_have})")
            new_queries.extend(attack_queries)
            stats[attack_type] = already_have + len(attack_queries)

        # Combine, shuffle, and save
        random.shuffle(new_queries)
        all_queries = existing_queries + new_queries
        random.shuffle(all_queries)

        write_jsonl(all_queries, str(output_file))
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        print(f"Total queries: {len(all_queries)}")
        print(f"\nBreakdown:")
        for query_type, count in sorted(stats.items()):
            percentage = (count / len(all_queries) * 100) if all_queries else 0
            print(f"  {query_type}: {count} ({percentage:.1f}%)")
        print(f"\nSaved to: {output_file}")
    
    print(f"\n{'='*60}")
    print("PREPARATION COMPLETE")
    print(f"Output directory: {args.data_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()