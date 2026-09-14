#!/usr/bin/env python3
"""
S²MIA Query Generation Script
Splits member and non-member documents in half to create queries for membership inference attacks.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_jsonl, write_jsonl


def split_text_in_half(text: str) -> tuple[str, str]:
    """
    Split text into two halves by character count.
    Returns (first_half, second_half).
    """
    mid_point = len(text) // 2
    # Find the nearest space to avoid splitting words
    while mid_point < len(text) and text[mid_point] != ' ':
        mid_point += 1
    
    first_half = text[:mid_point].strip()
    second_half = text[mid_point:].strip()
    
    return first_half, second_half


def create_query_template(query_text: str) -> str:
    """
    Create query using the S²MIA template format.
    """
    template = (
        f'Given the [Query]: "{query_text}".\n'
        "Do not include any introductory or explanatory text, "
        "use the following format for output: {[Response]: 'Provide a concise response directly "
        "addressing the [Query] by using the most relevant and matching text in the prompt.'}."
    )
    return template


def process_corpus_file(
    input_file: Path,
    membership_label: int
) -> List[Dict]:
    """
    Process a corpus file and generate queries.
    
    Args:
        input_file: Path to input JSONL file (corpus_member.jsonl or corpus_nonmember.jsonl)
        membership_label: 1 for member, 0 for non-member
    
    Returns:
        List of query records
    """
    print(f"Processing {input_file}...")
    
    # Read input corpus
    corpus_data = read_jsonl(str(input_file))
    
    queries = []
    for doc in corpus_data:
        doc_id = doc["_id"]
        title = doc.get("title", "")
        text = doc.get("text", "")
        
        # Combine title and text for complete document
        full_text = f"{title} {text}".strip() if title else text
        
        # Split document in half
        first_half, second_half = split_text_in_half(full_text)
        
        # Create query from first half
        query_text = first_half
        query_prompt = create_query_template(query_text)
        
        # Store query information
        query_record = {
            "_id": doc_id,
            "query_text": query_text,  # First half (x_t^q)
            "query_prompt": query_prompt,  # Full prompt to send to RAG
            "knowledge_text": second_half,  # Second half (x_t^k) for similarity comparison
            "membership_label": membership_label,  # Ground truth
            "original_title": title,
            "full_text": full_text
        }
        
        queries.append(query_record)
    
    print(f"Generated {len(queries)} queries from {input_file.name}")
    return queries


def main():
    """Main function to process selected datasets."""
    parser = argparse.ArgumentParser(
        description='Generate S2-MIA queries for selected datasets'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=["BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid"],
        help='Datasets to process'
    )
    parser.add_argument(
        '--data_base',
        default='data',
        help='Base directory for corpus files'
    )
    parser.add_argument(
        '--results_base',
        default='results/S2-MIA',
        help='Base directory for query outputs'
    )
    args = parser.parse_args()

    data_base = Path(args.data_base)
    results_base = Path(args.results_base)

    total_queries = 0

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*60}")

        member_file = data_base / dataset / "corpus_member.jsonl"
        nonmember_file = data_base / dataset / "corpus_nonmember.jsonl"

        output_dir = results_base / dataset / "queries"
        queries_file = output_dir / "queries.jsonl"

        all_queries = []

        if member_file.exists():
            member_queries = process_corpus_file(
                member_file,
                membership_label=1
            )
            all_queries.extend(member_queries)
            total_queries += len(member_queries)
        else:
            print(f"Warning: {member_file} not found!")

        if nonmember_file.exists():
            nonmember_queries = process_corpus_file(
                nonmember_file,
                membership_label=0
            )
            all_queries.extend(nonmember_queries)
            total_queries += len(nonmember_queries)
        else:
            print(f"Warning: {nonmember_file} not found!")

        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(all_queries, str(queries_file))

        print(f"\nSaved {len(all_queries)} queries -> {queries_file}")
        print(f"  - Members: {sum(1 for q in all_queries if q['membership_label'] == 1)}")
        print(f"  - Non-members: {sum(1 for q in all_queries if q['membership_label'] == 0)}")

    print(f"\n{'='*60}")
    print(f"Total queries generated: {total_queries}")
    print(f"Query files saved to: {results_base}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
