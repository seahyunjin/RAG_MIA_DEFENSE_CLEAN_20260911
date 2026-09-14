from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Set
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path
import sys
# Add project root to Python path (MUST be before utils import)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl
from utils.load_model import load_generator_model, load_embedding_model
from utils.openai_gen import (
    initialize_openai_client,
    split_into_batches,
    run_batch_generation,
    create_batch_request,
    estimate_tokens_accurate
)
from utils.transformers_gen import generate_batch_transformers, create_chat_prompt

# Load environment variables
load_dotenv('.env')


# Ground truth generation prompt
PROMPT_TEMPLATE = """System Prompt:
You are a helpful assistant. Based solely on the provided contexts, generate a concise and accurate response to the question from the user.

Contexts: {context}

Question: {question}

Answer only with "Yes," "No," or "I don't know".

Answer:
"""


def load_existing_results(output_path: str) -> Dict[str, dict]:
    """Load existing results and return as dict keyed by query _id."""
    if not os.path.exists(output_path):
        return {}
    
    existing = read_jsonl(output_path)
    existing_map = {item['_id']: item for item in existing}
    print(f"Found {len(existing_map)} existing results in output file")
    return existing_map


def identify_missing_queries(all_queries: List[dict], existing_results: Dict[str, dict]) -> List[dict]:
    """Identify queries that are missing from existing results."""
    missing = []
    for query in all_queries:
        if query['_id'] not in existing_results:
            missing.append(query)
    
    print(f"Missing queries: {len(missing)} out of {len(all_queries)} total")
    return missing


def load_queries_and_corpus(
    queries_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str
) -> List[dict]:
    """
    Load queries and match them with their source documents.
    
    Returns list of dicts with query info and document text.
    """
    # Load queries
    queries = read_jsonl(queries_path)
    print(f"Loaded {len(queries)} queries")
    
    # Load corpora
    corpus_member = read_jsonl(corpus_member_path)
    corpus_nonmember = read_jsonl(corpus_nonmember_path)
    
    # Build document lookup
    doc_lookup = {}
    
    for doc in corpus_member:
        text_parts = []
        for key, value in doc.items():
            if key != '_id' and value:
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    text_parts.append(f"{key}: {str(value)}")
        combined_text = ' '.join(text_parts).strip()
        doc_lookup[doc['_id']] = combined_text
    
    for doc in corpus_nonmember:
        text_parts = []
        for key, value in doc.items():
            if key != '_id' and value:
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    text_parts.append(f"{key}: {str(value)}")
        combined_text = ' '.join(text_parts).strip()
        doc_lookup[doc['_id']] = combined_text
    
    print(f"Loaded {len(corpus_member)} member documents")
    print(f"Loaded {len(corpus_nonmember)} nonmember documents")
    
    # Enrich queries with document text
    enriched_queries = []
    for query in queries:
        doc_id = query['target_doc_id']
        if doc_id in doc_lookup:
            enriched_query = {
                '_id': query['_id'],
                'text': query['text'],
                'target_doc_id': doc_id,
                '_membership': query['_membership'],
                'variation_index': query['variation_index'],
                'context': doc_lookup[doc_id]
            }
            enriched_queries.append(enriched_query)
        else:
            print(f"Warning: Document {doc_id} not found for query {query['_id']}")
    
    return enriched_queries


def generate_ground_truth_answers_batch_transformers(
    queries: List[dict],
    tokenizer,
    model,
    batch_size: int = 64
) -> List[dict]:
    """
    Generate ground truth answers using transformers batch generation.
    
    Parameters:
    - queries: List of enriched query dictionaries
    - tokenizer: Tokenizer
    - model: Model
    - batch_size: Batch size for generation
    
    Returns:
    - List of query dictionaries with answers added
    """
    # Prepare prompts
    all_prompts = []
    for query in queries:
        prompt = PROMPT_TEMPLATE.format(context=query['context'], question=query['text'])
        all_prompts.append(prompt)
    
    print(f"Total queries to process: {len(all_prompts)}")
    
    # Format prompts with chat template
    formatted_prompts = []
    for prompt in all_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = create_chat_prompt(tokenizer, messages)
        formatted_prompts.append(formatted_prompt)
    
    # Use batch generation from transformers_gen
    all_answers = generate_batch_transformers(
        formatted_prompts,
        tokenizer,
        model,
        temperature=0.3,
        max_new_tokens=50,
        batch_size=batch_size
    )
    
    # Post-process answers and build results
    results = []
    for query, answer in zip(queries, all_answers):
        # Remove common prefixes/patterns
        if "Answer:" in answer:
            answer = answer.split("Answer:", 1)[1].strip()
        
        if "assistant" in answer.lower():
            parts = answer.split("assistant", 1)
            if len(parts) > 1:
                answer = parts[1].strip()
        
        if "?" in answer:
            parts = answer.split("?")
            if len(parts) > 1:
                potential_answer = parts[-1].strip()
                if len(potential_answer) < 50 and any(word in potential_answer.lower() for word in ["yes", "no", "don't", "do not"]):
                    answer = potential_answer
        
        answer = answer.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
        answer = answer.split('\n')[0].strip()
        answer = answer.rstrip('!?:;,')
        
        if not answer.endswith('.'):
            answer = answer + '.'
        
        # Normalize common variations
        answer_lower = answer.lower()
        if "yes" in answer_lower and "no" not in answer_lower:
            answer = "Yes."
        elif "no" in answer_lower and "yes" not in answer_lower:
            answer = "No."
        elif "don't know" in answer_lower or "do not know" in answer_lower:
            answer = "I don't know."
        
        result = {
            "_id": query['_id'],
            "text": query['text'],
            "target_doc_id": query['target_doc_id'],
            "_membership": query['_membership'],
            "variation_index": query['variation_index'],
            "ground_truth_answer": answer
        }
        results.append(result)
    
    return results


def generate_ground_truth_answers_batch_openai(
    queries: List[dict],
    client: OpenAI,
    model_name: str,
    output_dir: Path,
    output_path: str,
    existing_results: Dict[str, dict],
) -> List[dict]:
    """
    Generate ground truth answers using OpenAI batch API.
    
    Parameters:
    - queries: List of enriched query dictionaries
    - client: OpenAI client
    - model_name: OpenAI model name
    - output_dir: Directory for batch files
    
    Returns:
    - List of query dictionaries with answers added
    """
    print(f"Total queries to process: {len(queries)}")
    
    # Estimate tokens function for batching
    def estimate_tokens(query):
        prompt = PROMPT_TEMPLATE.format(context=query['context'], question=query['text'])
        return estimate_tokens_accurate(prompt, 1, model_name)
    
    # Split into batches
    batches = split_into_batches(queries, estimate_tokens, max_tokens_per_batch=2_000_000)
    print(f"Total batches created: {len(batches)}")
    
    # Create request function
    def create_request(query):
        prompt = PROMPT_TEMPLATE.format(context=query['context'], question=query['text'])
        
        return create_batch_request(
            custom_id=query['_id'],
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
    
    # Build query lookup for result processing
    query_lookup = {q['_id']: q for q in queries}
    
    # Result processor function
    def process_result(custom_id, generated_text):
        query = query_lookup.get(custom_id)
        if not query:
            return None, {'custom_id': custom_id, 'reason': 'Query not found'}
        
        answer = generated_text.strip()
        
        # Normalize answer
        answer_lower = answer.lower()
        if "yes" in answer_lower and "no" not in answer_lower:
            answer = "Yes."
        elif "no" in answer_lower and "yes" not in answer_lower:
            answer = "No."
        elif "don't know" in answer_lower or "do not know" in answer_lower:
            answer = "I don't know."
        
        success_data = {
            '_id': query['_id'],
            'text': query['text'],
            'target_doc_id': query['target_doc_id'],
            '_membership': query['_membership'],
            'variation_index': query['variation_index'],
            'ground_truth_answer': answer
        }
        
        return success_data, None

    def save_chunk_ground_truth(successful, failed, batch_num):
        if successful:
            merge_and_save_results(existing_results, successful, output_path)
            print(
                f"  [checkpoint] Saved {len(successful)} ground-truth answers after batch "
                f"{batch_num} -> {output_path}"
            )

    # Run batch generation
    successful, failed = run_batch_generation(
        client=client,
        batches=batches,
        create_request_func=create_request,
        result_processor_func=process_result,
        output_dir=output_dir,
        base_filename="ground_truth_batch",
        check_interval=60,
        metadata_base={"task": "ground_truth_generation"},
        on_chunk_complete=save_chunk_ground_truth,
        resume=True,
    )
    
    if failed:
        print(f"\n⚠️  {len(failed)} queries failed to generate")
        for failure in failed[:5]:
            print(f"  - {failure}")
    
    return successful


def merge_and_save_results(
    existing_results: Dict[str, dict],
    new_results: List[dict],
    output_path: str
):
    """Merge existing and new results, then save to file."""
    # Update existing with new results
    for result in new_results:
        existing_results[result['_id']] = result
    
    # Convert to list and sort by _id for consistency
    all_results = list(existing_results.values())
    all_results.sort(key=lambda x: x['_id'])
    
    # Save to file
    write_jsonl(all_results, output_path)
    print(f"\nSaved {len(all_results)} total results to: {output_path}")


def generate_ground_truth_for_dataset(
    queries_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    output_path: str,
    generation_method: str = 'openai',
    openai_model: str = 'gpt-4o-mini',
    transformers_model: str = 'meta-llama/Llama-3.1-8B-Instruct',
    use_gpu: bool = False,
    cache_dir: str = None,
    batch_size: int = 64,
    resume: bool = True
):
    """
    Generate ground truth answers for all queries in the dataset.
    
    If resume=True and output file exists, only generates missing queries.
    """
    print(f"\n{'#'*60}")
    print(f"Generating ground truth answers")
    print(f"Method: {generation_method}")
    if generation_method == 'openai':
        print(f"Model: {openai_model}")
    else:
        print(f"Model: {transformers_model}")
        print(f"Batch size: {batch_size}")
    print(f"Queries: {queries_path}")
    print(f"Member corpus: {corpus_member_path}")
    print(f"Nonmember corpus: {corpus_nonmember_path}")
    print(f"Output: {output_path}")
    print(f"Resume mode: {resume}")
    print(f"{'#'*60}\n")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load existing results if resume mode
    existing_results = {}
    if resume and os.path.exists(output_path):
        existing_results = load_existing_results(output_path)
    
    # Load all queries and corpus
    all_queries = load_queries_and_corpus(queries_path, corpus_member_path, corpus_nonmember_path)
    
    # Identify missing queries
    if resume and existing_results:
        queries_to_process = identify_missing_queries(all_queries, existing_results)
        if not queries_to_process:
            print("\n✓ All queries already have ground truth answers!")
            return
    else:
        queries_to_process = all_queries
    
    print(f"\nProcessing {len(queries_to_process)} queries...")
    
    # Initialize generation backend and generate
    if generation_method == 'openai':
        openai_client = initialize_openai_client('.env')
        print("✓ OpenAI client initialized\n")
        
        output_dir = Path(output_path).parent / "openai_batches"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        new_results = generate_ground_truth_answers_batch_openai(
            queries_to_process,
            openai_client,
            openai_model,
            output_dir,
            output_path,
            existing_results,
        )
    else:
        if cache_dir:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        tokenizer, model = load_generator_model(transformers_model, use_gpu, cache_dir)
        print("✓ Transformers model loaded\n")
        
        new_results = generate_ground_truth_answers_batch_transformers(
            queries_to_process,
            tokenizer,
            model,
            batch_size
        )
    
    # Merge and save results (idempotent if incremental saves already ran)
    if new_results:
        merge_and_save_results(existing_results, new_results, output_path)
    
    # Count by membership
    all_final_results = read_jsonl(output_path)
    member_count = sum(1 for r in all_final_results if r['_membership'] == 'member')
    nonmember_count = sum(1 for r in all_final_results if r['_membership'] == 'nonmember')
    
    print(f"\n{'='*60}")
    print(f"Total ground truth answers: {len(all_final_results)}")
    print(f"  Member queries: {member_count}")
    print(f"  Nonmember queries: {nonmember_count}")
    print(f"  Newly generated: {len(new_results)}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate IA-MIA ground truth answers')
    parser.add_argument(
        '--queries_file',
        type=str,
        default='results/IA-MIA/BeIR_nfcorpus/queries/queries_30.jsonl',
        help='Path to queries file generated by generate_queries.py'
    )
    parser.add_argument(
        '--corpus_member',
        type=str,
        default='data/BeIR_nfcorpus/corpus_member.jsonl',
        help='Path to corpus_member.jsonl'
    )
    parser.add_argument(
        '--corpus_nonmember',
        type=str,
        default='data/BeIR_nfcorpus/corpus_nonmember.jsonl',
        help='Path to corpus_nonmember.jsonl'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results/IA-MIA/BeIR_nfcorpus/gt/gt.jsonl',
        help='Path to output ground truth answers file'
    )
    parser.add_argument(
        '--generation_method',
        type=str,
        choices=['openai', 'transformers'],
        default='openai',
        help='Generation method'
    )
    parser.add_argument(
        '--openai_model',
        type=str,
        default='gpt-4.1-nano',
        help='OpenAI model name (for openai method)'
    )
    parser.add_argument(
        '--transformers_model',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='Transformers model name (for transformers method)'
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
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Disable resume mode - regenerate all queries'
    )
    
    args = parser.parse_args()
    
    generate_ground_truth_for_dataset(
        args.queries_file,
        args.corpus_member,
        args.corpus_nonmember,
        args.output_file,
        args.generation_method,
        args.openai_model,
        args.transformers_model,
        args.use_gpu,
        args.cache_dir,
        args.batch_size,
        resume=not args.no_resume
    )
    
    print("\n" + "="*60)
    print("Ground truth generation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()