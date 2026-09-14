#!/usr/bin/env python3
"""
DCMI Query Generation Script
Generates queries for DCMI (Differential Calibration Membership Inference) attack.
Target sample is the complete document (title + text) to be tested for membership.
Generates queries for both original and perturbed documents.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_jsonl, write_jsonl


def create_query_template(query_text: str) -> str:
    """
    Create query using the DCMI template format.
    The query asks if the target sample is in the prompt/corpus.
    """
    template = (
        f'Is this: "{query_text}" right? Answer with Yes or No.'
    )
    return template


def process_corpus_file(
    input_file: Path, 
    perturbed_file: Path,
    membership_label: int
) -> List[Dict]:
    """
    Process original and perturbed corpus files and generate queries for DCMI attack.
    
    Args:
        input_file: Path to original corpus file
        perturbed_file: Path to perturbed corpus file
        membership_label: 1 for member, 0 for non-member
    
    Returns:
        List of query records for both original and perturbed
    """
    print(f"Processing {input_file.name}...")
    
    # Read original corpus
    corpus_data = read_jsonl(str(input_file))
    print(f"  - Loaded {len(corpus_data)} original documents")
    
    # Read perturbed corpus
    perturbed_data = []
    if perturbed_file.exists():
        perturbed_data = read_jsonl(str(perturbed_file))
        print(f"  - Loaded {len(perturbed_data)} perturbed documents")
    else:
        print(f"  - Warning: {perturbed_file} not found! Skipping perturbed queries.")
    
    # Create lookup for perturbed documents by ID
    perturbed_lookup = {doc["_id"]: doc for doc in perturbed_data}
    
    queries = []
    
    # Process each original document
    for doc in corpus_data:
        doc_id = doc["_id"]
        title = doc.get("title", "")
        text = doc.get("text", "")
        
        # Original target sample: complete document (title + text)
        original_target_sample = f"{title} {text}".strip() if title else text
        
        # Create query for original document
        original_query_prompt = create_query_template(original_target_sample)
        
        # Store original query information
        original_query_record = {
            "_id": doc_id,
            "target_sample": original_target_sample,
            "query_prompt": original_query_prompt,
            "membership_label": membership_label,
            "is_perturbed": False,
            "title": title,
            "text": text
        }
        
        queries.append(original_query_record)
        
        # Process perturbed version if it exists
        if doc_id in perturbed_lookup:
            perturbed_doc = perturbed_lookup[doc_id]
            perturbed_title = perturbed_doc.get("title", "")
            perturbed_text = perturbed_doc.get("text", "")
            
            # Perturbed target sample: complete perturbed document
            perturbed_target_sample = f"{perturbed_title} {perturbed_text}".strip() if perturbed_title else perturbed_text
            
            # Create query for perturbed document
            perturbed_query_prompt = create_query_template(perturbed_target_sample)
            
            # Store perturbed query information
            perturbed_query_record = {
                "_id": f"{doc_id}_perturbed",
                "original_id": doc_id,
                "target_sample": perturbed_target_sample,
                "query_prompt": perturbed_query_prompt,
                "membership_label": membership_label,
                "is_perturbed": True,
                "title": perturbed_title,
                "text": perturbed_text,
                "original_title": perturbed_doc.get("original_title", ""),
                "original_text": perturbed_doc.get("original_text", ""),
                "perturbation_magnitude": perturbed_doc.get("perturbation_magnitude", 0.0)
            }
            
            queries.append(perturbed_query_record)
    
    print(f"  - Generated {len(queries)} queries ({len(corpus_data)} original + {len(perturbed_lookup)} perturbed)")
    return queries


def main():
    """Main function to process selected datasets."""
    parser = argparse.ArgumentParser(
        description='Generate DCMI queries for selected datasets'
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
        help='Base directory for original corpus files'
    )
    parser.add_argument(
        '--perturbed_data_base',
        default=None,
        help='Base directory for perturbed corpus files; defaults to --data_base'
    )
    parser.add_argument(
        '--results_base',
        default='results/DCMI',
        help='Base directory for query outputs'
    )
    args = parser.parse_args()

    data_base = Path(args.data_base)
    perturbed_data_base = Path(args.perturbed_data_base) if args.perturbed_data_base else data_base
    results_base = Path(args.results_base)

    total_queries = 0
    total_original = 0
    total_perturbed = 0

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*60}")

        member_file = data_base / dataset / "corpus_member.jsonl"
        nonmember_file = data_base / dataset / "corpus_nonmember.jsonl"
        perturbed_member_file = perturbed_data_base / dataset / "perturbed_corpus_member.jsonl"
        perturbed_nonmember_file = perturbed_data_base / dataset / "perturbed_corpus_nonmember.jsonl"

        output_dir = results_base / dataset / "queries"
        queries_file = output_dir / "queries.jsonl"

        all_queries = []

        if member_file.exists():
            member_queries = process_corpus_file(
                member_file,
                perturbed_member_file,
                membership_label=1
            )
            all_queries.extend(member_queries)
            total_queries += len(member_queries)
            total_original += sum(1 for q in member_queries if not q['is_perturbed'])
            total_perturbed += sum(1 for q in member_queries if q['is_perturbed'])
        else:
            print(f"Warning: {member_file} not found!")

        if nonmember_file.exists():
            nonmember_queries = process_corpus_file(
                nonmember_file,
                perturbed_nonmember_file,
                membership_label=0
            )
            all_queries.extend(nonmember_queries)
            total_queries += len(nonmember_queries)
            total_original += sum(1 for q in nonmember_queries if not q['is_perturbed'])
            total_perturbed += sum(1 for q in nonmember_queries if q['is_perturbed'])
        else:
            print(f"Warning: {nonmember_file} not found!")

        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(all_queries, str(queries_file))

        print(f"\nSaved {len(all_queries)} queries -> {queries_file}")
        print(f"  - Members: {sum(1 for q in all_queries if q['membership_label'] == 1)}")
        print(f"  - Non-members: {sum(1 for q in all_queries if q['membership_label'] == 0)}")
        print(f"  - Original: {sum(1 for q in all_queries if not q['is_perturbed'])}")
        print(f"  - Perturbed: {sum(1 for q in all_queries if q['is_perturbed'])}")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries generated: {total_queries}")
    print(f"  - Original queries: {total_original}")
    print(f"  - Perturbed queries: {total_perturbed}")
    print(f"Query files saved to: {results_base}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
