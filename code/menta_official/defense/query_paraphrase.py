#!/usr/bin/env python3
"""
Query Paraphrasing Script for MIA Attacks
Paraphrases queries for IA-MIA, MEntA, Mask-based MIA, S²MIA, and DCMI attacks.
Preserves technical terminology and document-specific information while
rephrasing sentence structure.
"""

from datetime import timedelta
import os
import json
from typing import List, Dict, Optional
import argparse
from pathlib import Path
import sys
import re

# Set distributed timeout BEFORE importing torch/accelerate
os.environ["NCCL_TIMEOUT"] = "36000"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0" 
os.environ["NCCL_BLOCKING_WAIT"] = "0"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl, merge_jsonl_by_key
from utils.openai_gen import (
    initialize_openai_client,
    split_into_batches,
    run_batch_generation,
    create_batch_request,
    estimate_tokens_accurate
)
from utils.transformers_gen import generate_batch_transformers, create_chat_prompt
from utils.load_model import load_generator_model
from accelerate import Accelerator, InitProcessGroupKwargs

from dotenv import load_dotenv
load_dotenv('.env')


PARAPHRASE_PROMPT = """Paraphrase the following query while preserving all technical terminology, measurements, names, numbers, and domain-specific keywords EXACTLY as they appear. Only rephrase the sentence structure and connecting words.

Original query: {query}

Important: Output ONLY the paraphrased query text, with no additional commentary, labels, or explanations.

Paraphrased query:"""


def detect_attack_type(query: Dict) -> str:
    """
    Detect which attack type this query belongs to based on its structure.
    
    Returns: 'ia-mia', 'menta', 'mba', 's2mia', or 'dcmi'
    """
    # DCMI: has 'target_sample', 'query_prompt', 'membership_label', 'is_perturbed'
    if 'target_sample' in query and 'query_prompt' in query and \
       'membership_label' in query and 'is_perturbed' in query:
        return 'dcmi'

    # MEntA legacy format: has 'query_text', 'doc_id', 'membership'.
    # Current MEntA uses the same text/target_doc_id/_membership shape as
    # IA-MIA, so callers should pass attack_type='menta' when invoking via
    # --menta_queries.
    if 'query_text' in query and 'doc_id' in query and 'membership' in query:
        return 'menta'

    # IA-MIA/current MEntA shared shape. Without explicit CLI context this is
    # treated as IA-MIA for backward compatibility.
    if 'text' in query and 'target_doc_id' in query and '_membership' in query and 'variation_index' in query:
        return 'ia-mia'
    
    # Mask-based MIA: has 'masked_text', 'mask_values', 'membership_label'
    if 'masked_text' in query and 'mask_values' in query and 'membership_label' in query:
        return 'mba'
    
    # S²MIA: has 'query_text', 'query_prompt', 'knowledge_text', 'membership_label'
    if 'query_text' in query and 'query_prompt' in query and 'knowledge_text' in query and 'membership_label' in query:
        return 's2mia'
    
    raise ValueError(f"Cannot detect attack type from query structure: {query.keys()}")


def extract_query_text(query: Dict, attack_type: str) -> str:
    """Extract the query text to paraphrase based on attack type."""
    if attack_type == 'ia-mia':
        return query['text']
    elif attack_type == 'menta':
        return query.get('query_text') or query['text']
    elif attack_type == 'mba':
        return query['masked_text']
    elif attack_type == 's2mia':
        return query['query_text']
    elif attack_type == 'dcmi':
        # Paraphrase the target_sample text (the document excerpt being queried)
        return query['target_sample']
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def clean_paraphrased_text(text: str) -> str:
    """
    Clean paraphrased text to remove common artifacts and ensure only the query remains.
    """
    # Remove common prefixes
    prefixes_to_remove = [
        'paraphrased query:',
        'paraphrase:',
        'rephrased:',
        'rewritten:',
        'answer:',
        'response:',
        'here is the paraphrased query:',
        'here\'s the paraphrased query:',
    ]
    
    text_lower = text.lower().strip()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            text_lower = text.lower().strip()
    
    # Remove quotes if they wrap the entire text
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    
    # Remove markdown formatting
    text = text.replace('**', '').replace('__', '')
    
    # Remove model-specific tokens
    text = text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
    
    # Take only the first line if multiple lines exist (removes explanations)
    if '\n' in text:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Take the first substantial line
        for line in lines:
            if len(line) > 10:  # Avoid taking very short lines
                text = line
                break
    
    return text.strip()


def update_query_with_paraphrase(query: Dict, paraphrased_text: str, attack_type: str) -> Dict:
    """Update query with paraphrased text based on attack type."""
    updated_query = query.copy()
    
    # Clean the paraphrased text first
    paraphrased_text = clean_paraphrased_text(paraphrased_text)
    
    if attack_type == 'ia-mia':
        updated_query['text'] = paraphrased_text
    
    elif attack_type == 'menta':
        if 'query_text' in updated_query:
            updated_query['query_text'] = paraphrased_text
        else:
            updated_query['text'] = paraphrased_text
    
    elif attack_type == 'mba':
        # For mask-based MIA, paraphrase the masked text while preserving [Mask_X] tokens
        updated_query['masked_text'] = paraphrased_text
    
    elif attack_type == 's2mia':
        # For S²MIA, update both query_text and regenerate query_prompt
        updated_query['query_text'] = paraphrased_text
        # Regenerate query_prompt with new paraphrased text
        updated_query['query_prompt'] = (
            f'Given the [Query]: "{paraphrased_text}".\n'
            f'Do not include information not mentioned in contexts. '
            f'Provide a single-sentence response in the format: [Response]: "Your answer here".'
        )

    elif attack_type == 'dcmi':
        # Update target_sample with paraphrased version
        updated_query['target_sample'] = paraphrased_text
        # Regenerate query_prompt with paraphrased target_sample
        updated_query['query_prompt'] = (
            f'Is this: "{paraphrased_text}" right? Answer with Yes or No.'
        )
    
    return updated_query


def paraphrase_with_openai_batch(
    queries: List[dict],
    openai_client,
    openai_model: str,
    output_dir: Path,
    attack_type: str,
    output_path: str,
) -> List[dict]:
    """Paraphrase queries using OpenAI Batch API."""
    
    print("\n" + "="*60)
    print(f"PARAPHRASING {attack_type.upper()} QUERIES (OpenAI)")
    print("="*60)
    
    # Estimate tokens function for batching
    def estimate_tokens(query):
        query_text = extract_query_text(query, attack_type)
        prompt = PARAPHRASE_PROMPT.format(query=query_text)
        return estimate_tokens_accurate(prompt, 1, openai_model)
    
    # Split into batches
    batches = split_into_batches(queries, estimate_tokens, max_tokens_per_batch=2_000_000)
    print(f"Total batches created: {len(batches)}")
    
    # Create batch requests
    def create_paraphrase_request(query):
        query_text = extract_query_text(query, attack_type)
        messages = [{"role": "user", "content": PARAPHRASE_PROMPT.format(query=query_text)}]
        return create_batch_request(
            custom_id=query['_id'],
            model=openai_model,
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
    
    # Build query lookup for result processing
    query_lookup = {q['_id']: q for q in queries}
    
    # Process results
    def process_paraphrase_result(custom_id, generated_text):
        query = query_lookup.get(custom_id)
        if not query:
            return None, {'custom_id': custom_id, 'reason': 'Query not found'}
        
        paraphrased_text = generated_text.strip()
        
        # Update query with paraphrased text (cleaning happens inside)
        updated_query = update_query_with_paraphrase(query, paraphrased_text, attack_type)
        
        return updated_query, None

    def save_chunk_paraphrases(successful, failed, batch_num):
        if successful:
            merge_jsonl_by_key(output_path, successful, key='_id')
            print(
                f"  [checkpoint] Saved {len(successful)} paraphrased queries after batch "
                f"{batch_num} -> {output_path}"
            )

    results, failed = run_batch_generation(
        openai_client,
        batches,
        create_paraphrase_request,
        process_paraphrase_result,
        output_dir,
        f"{attack_type}_paraphrase_batch",
        metadata_base={"type": f"{attack_type}_paraphrase"},
        on_chunk_complete=save_chunk_paraphrases,
        resume=True,
    )
    
    print(f"\n✓ Successfully paraphrased {len(results)} queries")
    print(f"  Failures: {len(failed)}")
    
    return results


def paraphrase_with_transformers_batch(
    queries: List[dict],
    tokenizer,
    model,
    batch_size: int,
    accelerator,
    attack_type: str
) -> List[dict]:
    """Paraphrase queries using transformers."""
    
    print("\n" + "="*60)
    print(f"PARAPHRASING {attack_type.upper()} QUERIES (Transformers)")
    print("="*60)
    
    # Prepare prompts
    all_prompts = []
    for query in queries:
        query_text = extract_query_text(query, attack_type)
        prompt = PARAPHRASE_PROMPT.format(query=query_text)
        all_prompts.append(prompt)
    
    print(f"Total queries to process: {len(all_prompts)}")
    
    # Format prompts with chat template
    formatted_prompts = []
    for prompt in all_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = create_chat_prompt(tokenizer, messages)
        formatted_prompts.append(formatted_prompt)
    
    # Generate paraphrases
    all_paraphrases = generate_batch_transformers(
        formatted_prompts,
        tokenizer,
        model,
        accelerator=accelerator,
        temperature=0.7,
        max_new_tokens=300,
        batch_size=batch_size
    )
    
    # Post-process and build results
    results = []
    for query, paraphrased_text in zip(queries, all_paraphrases):
        # Update query with paraphrased text (cleaning happens inside)
        updated_query = update_query_with_paraphrase(query, paraphrased_text, attack_type)
        results.append(updated_query)
    
    return results


def paraphrase_queries_for_attack(
    queries_path: str,
    output_path: str,
    generation_method: str = 'openai',
    openai_client = None,
    openai_model: str = 'gpt-4.1-nano',
    tokenizer = None,
    model = None,
    generation_batch_size: int = 8,
    accelerator = None,
    attack_type: Optional[str] = None
):
    """Paraphrase queries for a specific attack."""
    
    print(f"\n{'#'*60}")
    print(f"Query Paraphrasing")
    print(f"Method: {generation_method}")
    if generation_method == 'openai':
        print(f"Model: {openai_model}")
    else:
        print(f"Batch size: {generation_batch_size}")
    print(f"Input: {queries_path}")
    print(f"Output: {output_path}")
    print(f"{'#'*60}\n")
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        existing_ids = {record['_id'] for record in read_jsonl(output_path)}
        queries = [query for query in queries if query['_id'] not in existing_ids]
        print(f"Resuming paraphrase with {len(existing_ids)} queries already saved")
        if not queries:
            print("All queries already paraphrased. Nothing to do.")
            return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load queries
    queries = read_jsonl(queries_path)
    print(f"Loaded {len(queries)} queries")
    
    # Detect attack type from first query
    if not queries:
        print("No queries found!")
        return
    
    if attack_type is None:
        attack_type = detect_attack_type(queries[0])
        print(f"Detected attack type: {attack_type.upper()}")
    else:
        print(f"Attack type: {attack_type.upper()} (from CLI argument)")
    
    # Generate paraphrases
    if generation_method == 'openai':
        output_dir = Path(output_path).parent / "openai_batches"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = paraphrase_with_openai_batch(
            queries,
            openai_client,
            openai_model,
            output_dir,
            attack_type,
            output_path,
        )
    else:
        results = paraphrase_with_transformers_batch(
            queries,
            tokenizer,
            model,
            generation_batch_size,
            accelerator,
            attack_type
        )
    
    # Write results (idempotent if incremental saves already ran)
    if generation_method != 'openai':
        write_jsonl(results, output_path)
    elif os.path.exists(output_path):
        results = read_jsonl(output_path)
    else:
        write_jsonl(results, output_path)
    
    print(f"\n{'='*60}")
    print(f"Paraphrased {len(results)} queries")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Paraphrase queries for MIA attacks (IA-MIA, MEntA, MBA, S²MIA, DCMI)'
    )
    
    # Query paths for each attack
    parser.add_argument(
        '--ia_mia_queries',
        type=str,
        default=None,
        help='Path to IA-MIA queries file'
    )
    parser.add_argument(
        '--menta_queries',
        type=str,
        default=None,
        help='Path to MEntA queries file'
    )
    parser.add_argument(
        '--mba_queries',
        type=str,
        default=None,
        help='Path to Mask-based MIA queries file'
    )
    parser.add_argument(
        '--s2mia_queries',
        type=str,
        default=None,
        help='Path to S²MIA queries file'
    )
    parser.add_argument(
        '--dcmi_queries',
        type=str,
        default="/fred/oz396/lbao/MIA/results/DCMI/BeIR_nfcorpus/queries/queries.jsonl",
        help='Path to DCMI queries file'
    )
    
    # Generation method arguments
    parser.add_argument(
        '--generation_method',
        type=str,
        choices=['transformers', 'openai'],
        default='openai',
        help='Generation method for paraphrasing'
    )
    parser.add_argument(
        '--transformers_model',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='Transformers model name'
    )
    parser.add_argument(
        '--openai_model',
        type=str,
        default='gpt-4.1-nano',
        help='OpenAI model name'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU (for transformers method)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='~/.cache/huggingface/hub',
        help='Cache directory for models'
    )
    parser.add_argument(
        '--generation_batch_size',
        type=int,
        default=32,
        help='Batch size for generation'
    )
    
    args = parser.parse_args()
    
    # Collect all query paths
    query_configs = []
    
    if args.ia_mia_queries:
        output_path = args.ia_mia_queries.replace('queries/', 'queries_paraphrased/')
        query_configs.append({
            'name': 'IA-MIA',
            'attack_type': 'ia-mia',
            'input': args.ia_mia_queries,
            'output': output_path
        })
    
    if args.menta_queries:
        output_path = args.menta_queries.replace('queries/', 'queries_paraphrased/')
        query_configs.append({
            'name': 'MEntA',
            'attack_type': 'menta',
            'input': args.menta_queries,
            'output': output_path
        })
    
    if args.mba_queries:
        output_path = args.mba_queries.replace('queries/', 'queries_paraphrased/')
        query_configs.append({
            'name': 'MBA',
            'attack_type': 'mba',
            'input': args.mba_queries,
            'output': output_path
        })
    
    if args.s2mia_queries:
        output_path = args.s2mia_queries.replace('queries/', 'queries_paraphrased/')
        query_configs.append({
            'name': 'S²MIA',
            'attack_type': 's2mia',
            'input': args.s2mia_queries,
            'output': output_path
        })

    if args.dcmi_queries:
        output_path = args.dcmi_queries.replace('queries/', 'queries_paraphrased/')
        query_configs.append({
            'name': 'DCMI',
            'attack_type': 'dcmi',
            'input': args.dcmi_queries,
            'output': output_path
        })
    
    if not query_configs:
        print("Error: No query files specified. Please provide at least one attack's queries.")
        print("Use --ia_mia_queries, --menta_queries, --mba_queries, --s2mia_queries, or --dcmi_queries")
        return
    
    print(f"\n{'='*60}")
    print(f"QUERY PARAPHRASING FOR MULTIPLE ATTACKS")
    print(f"{'='*60}")
    print(f"Attacks to process: {len(query_configs)}")
    for config in query_configs:
        print(f"  - {config['name']}: {config['input']}")
    print(f"{'='*60}\n")
    
    # Initialize generation backend
    openai_client = None
    tokenizer = None
    model = None
    accelerator = None
    
    if args.generation_method == 'openai':
        openai_client = initialize_openai_client('.env')
        print(f"Using OpenAI: {args.openai_model}\n")
    else:
        if args.cache_dir:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        accelerator = Accelerator(kwargs_handlers=[kwargs])
        tokenizer, model = load_generator_model(
            args.transformers_model,
            args.use_gpu,
            args.cache_dir,
            accelerator
        )
        print(f"Using Transformers: {args.transformers_model}\n")
    
    # Process each attack's queries
    for i, config in enumerate(query_configs, 1):
        print(f"\n{'#'*60}")
        print(f"Processing {i}/{len(query_configs)}: {config['name']}")
        print(f"{'#'*60}")
        
        paraphrase_queries_for_attack(
            config['input'],
            config['output'],
            args.generation_method,
            openai_client,
            args.openai_model,
            tokenizer,
            model,
            args.generation_batch_size,
            accelerator,
            config['attack_type']
        )
    
    print("\n" + "="*60)
    print("Query paraphrasing completed successfully for all attacks!")
    print("="*60)


if __name__ == '__main__':
    main()