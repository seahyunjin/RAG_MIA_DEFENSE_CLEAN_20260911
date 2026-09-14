#!/usr/bin/env python3
"""
DCMI Corpus Perturbation Script
Perturbs target samples (title + text) using GPT-4o for DCMI attack.
Generates perturbed versions of corpus documents for calibration.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_jsonl, write_json, write_jsonl, merge_jsonl_by_key
from utils.openai_gen import (
    initialize_openai_client,
    split_into_batches,
    run_batch_generation,
    create_batch_request,
    estimate_tokens_accurate
)


def create_perturbation_prompt(text: str, perturbation_magnitude: float = 0.03) -> str:
    """
    Create perturbation prompt for GPT-4o.
    
    Args:
        text: Original text (title + text combined)
        perturbation_magnitude: Percentage of words to perturb (default 3%)
    
    Returns:
        Perturbation prompt
    """
    word_count = len(text.split())
    replace_count = int(perturbation_magnitude * word_count) or 1
    
    prompt = (
        f"Replace {replace_count} key adjectives or adverbs in noticeable positions "
        f"with their antonyms in the following text, ensuring the modified text remains logically correct:\n"
        f"{text}\n"
        f"Return only the modified text."
    )
    
    return prompt


def main():
    """Main function to process all datasets."""
    parser = argparse.ArgumentParser(
        description='Perturb corpus documents using GPT-4o for DCMI'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=["BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid"],
        help='List of datasets to process'
    )
    parser.add_argument(
        '--data_base',
        type=str,
        default='data',
        help='Base directory for data files'
    )
    parser.add_argument(
        '--output_data_base',
        type=str,
        default=None,
        help='Where perturbed corpus files are written; defaults to --data_base'
    )
    
    # OpenAI arguments
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4.1-nano',
        help='OpenAI model name'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature'
    )
    parser.add_argument(
        '--env_path',
        type=str,
        default='.env',
        help='Path to .env file with OPENAI_API_KEY'
    )
    parser.add_argument(
        '--max_tokens_per_batch',
        type=int,
        default=2_000_000,
        help='Maximum tokens per OpenAI batch'
    )
    parser.add_argument(
        '--check_interval',
        type=int,
        default=120,
        help='Seconds between batch status checks'
    )
    
    # Perturbation arguments
    parser.add_argument(
        '--perturbation_magnitude',
        type=float,
        default=0.06,
        help='Percentage of words to perturb (default: 0.06 = 6%%)'
    )
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = initialize_openai_client(args.env_path)
    
    # Base paths
    data_base = Path(args.data_base)
    output_data_base = Path(args.output_data_base) if args.output_data_base else data_base
    
    print(f"\n{'#'*60}")
    print(f"DCMI CORPUS PERTURBATION")
    print(f"{'#'*60}")
    print(f"Model: {args.model}")
    print(f"Perturbation magnitude: {args.perturbation_magnitude * 100}%")
    print(f"Max tokens per batch: {args.max_tokens_per_batch:,}")
    print(f"Output data base: {output_data_base}")
    print(f"{'#'*60}\n")
    
    # Collect ALL documents from all datasets
    all_items = []
    doc_metadata = {}  # Track which dataset/membership each doc belongs to
    
    for dataset in args.datasets:
        print(f"Loading dataset: {dataset}")
        
        # Input paths
        member_file = data_base / dataset / "corpus_member.jsonl"
        nonmember_file = data_base / dataset / "corpus_nonmember.jsonl"
        
        # Load member corpus
        if member_file.exists():
            corpus_member = read_jsonl(str(member_file))
            print(f"  - Loaded {len(corpus_member)} member documents")
            
            for doc in corpus_member:
                doc_id = doc["_id"]
                title = doc.get("title", "")
                text = doc.get("text", "")
                
                # Combine title and text as target sample
                target_sample = f"{title} {text}".strip() if title else text
                
                # Create perturbation prompt
                perturbation_prompt = create_perturbation_prompt(
                    target_sample, 
                    args.perturbation_magnitude
                )
                
                all_items.append({
                    'doc': doc,
                    'target_sample': target_sample,
                    'prompt': perturbation_prompt
                })
                
                # Store metadata
                doc_metadata[doc_id] = {
                    'dataset': dataset,
                    'membership': 'member'
                }
        else:
            print(f"  - Warning: {member_file} not found!")
        
        # Load non-member corpus
        if nonmember_file.exists():
            corpus_nonmember = read_jsonl(str(nonmember_file))
            print(f"  - Loaded {len(corpus_nonmember)} non-member documents")
            
            for doc in corpus_nonmember:
                doc_id = doc["_id"]
                title = doc.get("title", "")
                text = doc.get("text", "")
                
                # Combine title and text as target sample
                target_sample = f"{title} {text}".strip() if title else text
                
                # Create perturbation prompt
                perturbation_prompt = create_perturbation_prompt(
                    target_sample, 
                    args.perturbation_magnitude
                )
                
                all_items.append({
                    'doc': doc,
                    'target_sample': target_sample,
                    'prompt': perturbation_prompt
                })
                
                # Store metadata
                doc_metadata[doc_id] = {
                    'dataset': dataset,
                    'membership': 'nonmember'
                }
        else:
            print(f"  - Warning: {nonmember_file} not found!")
    
    print(f"\n{'='*60}")
    print(f"Total documents collected: {len(all_items)}")
    print(f"{'='*60}\n")
    
    # Split into batches respecting token limit
    print("Splitting into batches based on token limit...")

    temp_output_dir = output_data_base / "temp_perturbed"
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    partial_results_path = temp_output_dir / "perturbed_corpus_partial.jsonl"

    completed_doc_ids = set()
    if partial_results_path.exists():
        for record in read_jsonl(str(partial_results_path)):
            completed_doc_ids.add(record['_id'])
        print(f"Resuming perturbation with {len(completed_doc_ids)} documents already saved")

    pending_items = [
        item for item in all_items
        if item['doc']['_id'] not in completed_doc_ids
    ]

    all_results = []
    all_failures = []
    if not pending_items:
        print("All documents already perturbed in checkpoint file.")
        if partial_results_path.exists():
            all_results = read_jsonl(str(partial_results_path))
    else:
        print(f"Documents remaining after resume: {len(pending_items)}")

        def estimate_pending_tokens(item):
            return estimate_tokens_accurate(item['prompt'], 1, args.model)

        batches = split_into_batches(pending_items, estimate_pending_tokens, args.max_tokens_per_batch)
        print(f"Created {len(batches)} pending batches\n")

        def create_request(item):
            doc_id = item['doc']['_id']
            prompt = item['prompt']

            messages = [
                {
                    "role": "system",
                    "content": "You are a text perturbation assistant. Replace specified adjectives/adverbs with antonyms while maintaining logical correctness."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            return create_batch_request(
                custom_id=doc_id,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=2048
            )

        def process_result(custom_id, generated_text):
            matching_doc = None
            for item in all_items:
                if item['doc']['_id'] == custom_id:
                    matching_doc = item['doc']
                    break

            if matching_doc:
                perturbed_full = generated_text.strip()
                original_title = matching_doc.get("title", "")
                original_text = matching_doc.get("text", "")

                if original_title:
                    original_title_len = len(original_title.split())
                    perturbed_words = perturbed_full.split()
                    perturbed_title = ' '.join(perturbed_words[:original_title_len])
                    perturbed_text = ' '.join(perturbed_words[original_title_len:])
                else:
                    perturbed_title = ""
                    perturbed_text = perturbed_full

                perturbed_doc = {
                    "_id": matching_doc["_id"],
                    "title": perturbed_title,
                    "text": perturbed_text,
                    "original_title": original_title,
                    "original_text": original_text,
                    "perturbation_magnitude": args.perturbation_magnitude
                }

                return perturbed_doc, None

            return None, {'doc_id': custom_id, 'reason': 'Document not found'}

        def save_chunk_perturbations(successful, failed, batch_num):
            if successful:
                merge_jsonl_by_key(str(partial_results_path), successful, key='_id')
                print(
                    f"  [checkpoint] Saved {len(successful)} perturbed documents after batch "
                    f"{batch_num} -> {partial_results_path}"
                )

        print("Starting batch generation...")
        _, batch_failures = run_batch_generation(
            client,
            batches,
            create_request,
            process_result,
            temp_output_dir,
            "perturbed_corpus",
            args.check_interval,
            metadata_base={
                "description": "Perturbed corpus for DCMI (all datasets)",
                "perturbation_magnitude": str(args.perturbation_magnitude),
                "model": args.model
            },
            on_chunk_complete=save_chunk_perturbations,
            resume=True,
        )
        all_failures.extend(batch_failures)

    if partial_results_path.exists():
        all_results = read_jsonl(str(partial_results_path))
    
    print(f"\n{'='*60}")
    print(f"Batch generation completed")
    print(f"Total successful: {len(all_results)}")
    print(f"Total failed: {len(all_failures)}")
    print(f"{'='*60}\n")
    
    # Organize results by dataset and membership
    print("Organizing results by dataset and membership...")
    results_by_category = {}
    
    for result in all_results:
        doc_id = result["_id"]
        if doc_id in doc_metadata:
            dataset = doc_metadata[doc_id]['dataset']
            membership = doc_metadata[doc_id]['membership']
            
            key = (dataset, membership)
            if key not in results_by_category:
                results_by_category[key] = []
            
            results_by_category[key].append(result)
    
    # Save results to appropriate files
    total_saved = 0
    for (dataset, membership), results in results_by_category.items():
        output_file = output_data_base / dataset / f"perturbed_corpus_{membership}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        write_jsonl(results, str(output_file))
        print(f"✓ Saved {len(results)} documents -> {output_file}")
        total_saved += len(results)
    
    # Save failures
    if all_failures:
        failure_path = temp_output_dir / "perturbed_corpus_failures.json"
        write_json({
            'failures': all_failures, 
            'total_failed': len(all_failures)
        }, str(failure_path))
        print(f"\n⚠ Saved {len(all_failures)} failures to: {failure_path}")
    
    # Final summary
    print(f"\n{'#'*60}")
    print(f"FINAL SUMMARY")
    print(f"{'#'*60}")
    print(f"Total documents processed: {len(all_items)}")
    print(f"Successful perturbations: {len(all_results)}")
    print(f"Failed perturbations: {len(all_failures)}")
    print(f"Success rate: {len(all_results) / len(all_items) * 100:.1f}%")
    print(f"Documents saved: {total_saved}")
    print(f"Perturbation magnitude: {args.perturbation_magnitude * 100}%")
    print(f"\nOutput files by dataset:")
    for dataset in args.datasets:
        member_file = output_data_base / dataset / "perturbed_corpus_member.jsonl"
        nonmember_file = output_data_base / dataset / "perturbed_corpus_nonmember.jsonl"
        if member_file.exists():
            print(f"  - {member_file}")
        if nonmember_file.exists():
            print(f"  - {nonmember_file}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()