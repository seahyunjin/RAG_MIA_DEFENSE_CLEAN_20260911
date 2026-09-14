#!/usr/bin/env python3
"""
DCMI RAG Answer Generation Script
Generates RAG system answers for DCMI queries (both original and perturbed).
Uses retrieved context to answer: "Is this: [target_sample] right? Answer with Yes or No."
Includes defense mechanisms: DP, Reranking, Prompt Instruction, Paraphrase
"""

from datetime import timedelta
import os
import json
from typing import List, Dict
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import numpy as np

# Set distributed timeout BEFORE importing torch/accelerate
os.environ["NCCL_TIMEOUT"] = "36000"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0" 
os.environ["NCCL_BLOCKING_WAIT"] = "0"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl
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

# Load environment variables
load_dotenv('.env')

# DCMI RAG prompt template (standard)
DCMI_RAG_PROMPT = """Answer the question based on the provided context.

Contexts: {context}

Query: {question}

Answer:"""

# DCMI RAG prompt with instruction-based defense
DCMI_RAG_PROMPT_WITH_INSTRUCTION = """Answer the question based on the provided context.

IMPORTANT: Do not directly repeat any retrieved content, but summarize it based on your understanding.

Contexts: {context}

Query: {question}

Answer:"""


def exponential_mechanism(scores: np.ndarray, epsilon: float, sensitivity: float = 1.0) -> int:
    """
    Implement exponential mechanism for differential privacy.
    
    Args:
        scores: Array of scores (e.g., similarity scores) for each item
        epsilon: Privacy budget
        sensitivity: Sensitivity of the scoring function
    
    Returns:
        Index of selected item
    """
    # Compute probabilities using exponential mechanism
    normalized_scores = (epsilon * scores) / (2 * sensitivity)
    # Subtract max for numerical stability
    normalized_scores = normalized_scores - np.max(normalized_scores)
    probabilities = np.exp(normalized_scores)
    probabilities = probabilities / np.sum(probabilities)
    
    # Sample according to probabilities
    selected_idx = np.random.choice(len(scores), p=probabilities)
    return selected_idx


def apply_dp_retrieval(
    retrieved_doc_ids: List[str],
    retrieval_scores: List[float],
    epsilon: float,
    top_k: int
) -> List[str]:
    """
    Apply differential privacy to retrieval results using exponential mechanism.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs
        retrieval_scores: Corresponding similarity scores
        epsilon: Privacy budget for DP
        top_k: Number of documents to select
    
    Returns:
        List of DP-selected document IDs
    """
    if not retrieved_doc_ids or epsilon <= 0:
        return retrieved_doc_ids[:top_k]
    
    scores = np.array(retrieval_scores[:len(retrieved_doc_ids)])
    selected_ids = []
    available_indices = list(range(len(retrieved_doc_ids)))
    
    # Apply exponential mechanism to select top_k documents
    for _ in range(min(top_k, len(retrieved_doc_ids))):
        if not available_indices:
            break
        
        # Get scores for available documents
        available_scores = scores[available_indices]
        
        # Select using exponential mechanism
        selected_pos = exponential_mechanism(available_scores, epsilon)
        selected_idx = available_indices[selected_pos]
        
        selected_ids.append(retrieved_doc_ids[selected_idx])
        available_indices.remove(selected_idx)
    
    return selected_ids


def rerank_retrieved_docs(
    retrieved_doc_ids: List[str],
    retrieval_scores: List[float],
    top_k: int,
    rerank_strategy: str = 'reverse'
) -> List[str]:
    """
    Rerank retrieved documents to obscure membership signals.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs
        retrieval_scores: Corresponding similarity scores
        top_k: Number of documents to return
        rerank_strategy: 'reverse' (reverse order), 'shuffle' (random), or 'score_noise' (add noise to scores)
    
    Returns:
        List of reranked document IDs
    """
    if not retrieved_doc_ids:
        return retrieved_doc_ids
    
    # Get top_k documents first
    doc_ids = retrieved_doc_ids[:top_k]
    scores = retrieval_scores[:len(doc_ids)] if retrieval_scores else [1.0] * len(doc_ids)
    
    if rerank_strategy == 'reverse':
        # Simply reverse the order
        return list(reversed(doc_ids))
    
    elif rerank_strategy == 'shuffle':
        # Random shuffle
        indices = list(range(len(doc_ids)))
        np.random.shuffle(indices)
        return [doc_ids[i] for i in indices]
    
    elif rerank_strategy == 'score_noise':
        # Add noise to scores and re-sort
        noisy_scores = np.array(scores) + np.random.normal(0, 0.1, len(scores))
        sorted_indices = np.argsort(noisy_scores)[::-1]
        return [doc_ids[i] for i in sorted_indices]
    
    else:
        return doc_ids


def add_laplace_noise(text: str, epsilon: float, sensitivity: int = 1) -> str:
    """
    Add Laplace noise to output by randomly dropping/keeping tokens (simplified DP).
    
    Args:
        text: Generated text
        epsilon: Privacy budget
        sensitivity: Sensitivity parameter
    
    Returns:
        Noised text
    """
    if epsilon <= 0:
        return text
    
    # Simple token-level DP: randomly keep tokens based on privacy budget
    tokens = text.split()
    scale = sensitivity / epsilon
    
    # Probability of keeping each token
    keep_prob = 1.0 / (1.0 + scale)
    
    noised_tokens = [token for token in tokens if np.random.random() < keep_prob]
    
    # Ensure at least some tokens remain
    if len(noised_tokens) == 0 and len(tokens) > 0:
        noised_tokens = [np.random.choice(tokens)]
    
    return ' '.join(noised_tokens) if noised_tokens else "No"


def build_context_from_retrieved_docs(retrieved_docs: List[dict]) -> str:
    """Build context text from retrieved documents."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        doc_text = []
        if 'title' in doc and doc['title']:
            doc_text.append(f"Title: {doc['title']}")
        if 'text' in doc and doc['text']:
            doc_text.append(f"Text: {doc['text']}")
        
        if doc_text:
            context_parts.append(f"[{i}] {' '.join(doc_text)}")
    
    full_context = '\n'.join(context_parts) if context_parts else "No context available."
    return full_context


def load_queries_and_retrieval(
    queries_path: str,
    retrieval_file_path: str,
    corpus: List[dict],
    max_context_docs: int = 5,
    defense_type: str = None,
    defense_params: Dict = None
) -> List[dict]:
    """
    Load DCMI queries and match them with retrieval results.
    Apply defenses at retrieval level if specified.
    
    Returns list of dicts with query info and context from retrieved documents.
    """
    # Load queries
    queries = read_jsonl(queries_path)
    print(f"Loaded {len(queries)} queries")
    
    # Count query types
    original_count = sum(1 for q in queries if not q.get('is_perturbed', False))
    perturbed_count = sum(1 for q in queries if q.get('is_perturbed', False))
    print(f"  - Original queries: {original_count}")
    print(f"  - Perturbed queries: {perturbed_count}")
    
    # Load retrieval results
    with open(retrieval_file_path, 'r') as f:
        retrieval_data = json.load(f)
    retrieval_results = retrieval_data.get('retrieval_results', {})
    print(f"Loaded retrieval results for {len(retrieval_results)} queries")
    
    # Create document map from corpus
    doc_map = {doc['_id']: doc for doc in corpus}
    print(f"Using corpus with {len(corpus)} documents")
    
    if defense_type in ['dp', 'rerank']:
        print(f"Applying {defense_type} defense at retrieval level")
        if defense_params:
            print(f"  Defense params: {defense_params}")
    
    # Enrich queries with retrieval context
    enriched_queries = []
    for query in queries:
        query_id = query['_id']
        
        # Get retrieved doc IDs for this query
        if query_id in retrieval_results:
            retrieved_doc_ids = retrieval_results[query_id].get('retrieved_doc_ids', [])
            retrieval_scores = retrieval_results[query_id].get('retrieval_scores', [1.0] * len(retrieved_doc_ids))
            is_hit = retrieval_results[query_id].get('is_hit', False)
            hit_rank = retrieval_results[query_id].get('hit_rank', -1)
            
            # Apply defense at retrieval level
            if defense_type == 'dp' and defense_params and defense_params.get('level') in ['retrieval', 'both']:
                retrieved_doc_ids = apply_dp_retrieval(
                    retrieved_doc_ids,
                    retrieval_scores,
                    defense_params.get('epsilon', 0.1),
                    max_context_docs
                )
            elif defense_type == 'rerank' and defense_params:
                retrieved_doc_ids = rerank_retrieved_docs(
                    retrieved_doc_ids,
                    retrieval_scores,
                    max_context_docs,
                    defense_params.get('strategy', 'shuffle')
                )
            else:
                retrieved_doc_ids = retrieved_doc_ids[:max_context_docs]
        else:
            print(f"Warning: Query {query_id} not found in retrieval results")
            retrieved_doc_ids = []
            is_hit = False
            hit_rank = -1
        
        # Build context from retrieved documents
        retrieved_docs = [doc_map[rid] for rid in retrieved_doc_ids if rid in doc_map]
        context = build_context_from_retrieved_docs(retrieved_docs)
        
        enriched_query = {
            '_id': query['_id'],
            'target_sample': query['target_sample'],
            'query_prompt': query['query_prompt'],
            'membership_label': query['membership_label'],
            'is_perturbed': query.get('is_perturbed', False),
            'title': query.get('title', ''),
            'text': query.get('text', ''),
            'original_id': query.get('original_id', ''),
            'original_title': query.get('original_title', ''),
            'original_text': query.get('original_text', ''),
            'perturbation_magnitude': query.get('perturbation_magnitude', 0.0),
            'context': context,
            'retrieved_doc_ids': retrieved_doc_ids,
            'is_hit': is_hit,
            'hit_rank': hit_rank
        }
        enriched_queries.append(enriched_query)
    
    return enriched_queries


def generate_with_openai_batch(
    queries: List[dict],
    openai_client: OpenAI,
    openai_model: str,
    output_dir: Path,
    defense_type: str = None,
    defense_params: Dict = None
) -> List[dict]:
    """Generate DCMI RAG answers using OpenAI Batch API with defenses."""
    
    print("\n" + "="*60)
    print("GENERATING DCMI RAG ANSWERS (OpenAI)")
    if defense_type == 'dp' and defense_params and defense_params.get('level') in ['output', 'both']:
        print(f"DP defense enabled at output level (ε={defense_params.get('epsilon', 0.1)})")
    print("="*60)
    
    # Select prompt template based on defense
    use_instruction = defense_type == 'prompt_instruction'
    prompt_template = DCMI_RAG_PROMPT_WITH_INSTRUCTION if use_instruction else DCMI_RAG_PROMPT
    
    # Estimate tokens function for batching
    def estimate_tokens(query):
        prompt = prompt_template.format(
            context=query['context'],
            question=query['query_prompt']
        )
        return estimate_tokens_accurate(prompt, 1, openai_model)
    
    # Split into batches
    batches = split_into_batches(queries, estimate_tokens, max_tokens_per_batch=2_000_000)
    print(f"Total batches created: {len(batches)}")
    
    # Create batch requests
    def create_rag_request(query):
        prompt = prompt_template.format(
            context=query['context'],
            question=query['query_prompt']
        )
        messages = [{"role": "user", "content": prompt}]
        return create_batch_request(
            custom_id=query['_id'],
            model=openai_model,
            messages=messages,
            temperature=0.7,
            max_tokens=50
        )
    
    # Build query lookup for result processing
    query_lookup = {q['_id']: q for q in queries}
    
    # Process results
    def process_rag_result(custom_id, generated_text):
        query = query_lookup.get(custom_id)
        if not query:
            return None, {'custom_id': custom_id, 'reason': 'Query not found'}
        
        answer = generated_text.strip()
        
        # Apply DP at output level if enabled
        if defense_type == 'dp' and defense_params and defense_params.get('level') in ['output', 'both']:
            answer = add_laplace_noise(answer, defense_params.get('epsilon', 0.1))
        
        # Normalize answer to Yes/No
        answer_lower = answer.lower()
        if "yes" in answer_lower and "no" not in answer_lower:
            answer = "Yes"
        elif "no" in answer_lower and "yes" not in answer_lower:
            answer = "No"
        else:
            # If unclear, default to "No"
            answer = "No"
        
        success_data = {
            '_id': query['_id'],
            'target_sample': query['target_sample'],
            'query_prompt': query['query_prompt'],
            'membership_label': query['membership_label'],
            'is_perturbed': query['is_perturbed'],
            'title': query['title'],
            'text': query['text'],
            'original_id': query['original_id'],
            'original_title': query['original_title'],
            'original_text': query['original_text'],
            'perturbation_magnitude': query['perturbation_magnitude'],
            'retrieved_doc_ids': query['retrieved_doc_ids'],
            'is_hit': query['is_hit'],
            'hit_rank': query['hit_rank'],
            'rag_answer': answer,
            'raw_answer': generated_text.strip()
        }
        
        return success_data, None
    
    # Add defense metadata
    metadata_base = {"type": "dcmi_rag_answers"}
    if defense_type:
        metadata_base["defense"] = {'type': defense_type, **(defense_params or {})}
    
    results, failed = run_batch_generation(
        openai_client,
        batches,
        create_rag_request,
        process_rag_result,
        output_dir,
        "dcmi_rag_answers_batch",
        metadata_base=metadata_base
    )
    
    print(f"\n✓ Successfully generated {len(results)} RAG answers")
    print(f"  Failures: {len(failed)}")
    
    return results


def generate_with_transformers_batch(
    queries: List[dict],
    tokenizer,
    model,
    batch_size: int = 8,
    accelerator = None,
    defense_type: str = None,
    defense_params: Dict = None
) -> List[dict]:
    """Generate DCMI RAG answers using transformers with defenses."""
    
    print("\n" + "="*60)
    print("GENERATING DCMI RAG ANSWERS (Transformers)")
    if defense_type == 'dp' and defense_params and defense_params.get('level') in ['output', 'both']:
        print(f"DP defense enabled at output level (ε={defense_params.get('epsilon', 0.1)})")
    print("="*60)
    
    # Select prompt template based on defense
    use_instruction = defense_type == 'prompt_instruction'
    prompt_template = DCMI_RAG_PROMPT_WITH_INSTRUCTION if use_instruction else DCMI_RAG_PROMPT
    
    # Prepare prompts
    all_prompts = []
    for query in queries:
        prompt = prompt_template.format(
            context=query['context'],
            question=query['query_prompt']
        )
        all_prompts.append(prompt)
    
    print(f"Total queries to process: {len(all_prompts)}")
    
    # Format prompts with chat template
    formatted_prompts = []
    for prompt in all_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = create_chat_prompt(tokenizer, messages)
        formatted_prompts.append(formatted_prompt)
    
    # Generate answers
    all_answers = generate_batch_transformers(
        formatted_prompts,
        tokenizer,
        model,
        accelerator=accelerator,
        temperature=0.7,
        max_new_tokens=10,
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
        
        answer = answer.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
        answer = answer.split('\n')[0].strip()
        answer = answer.rstrip('!?:;,.')
        
        raw_answer = answer
        
        # Apply DP at output level if enabled
        if defense_type == 'dp' and defense_params and defense_params.get('level') in ['output', 'both']:
            answer = add_laplace_noise(answer, defense_params.get('epsilon', 0.1))
        
        # Normalize answer to Yes/No
        answer_lower = answer.lower()
        if "yes" in answer_lower and "no" not in answer_lower:
            answer = "Yes"
        elif "no" in answer_lower and "yes" not in answer_lower:
            answer = "No"
        else:
            # If unclear, default to "No"
            answer = "No"
        
        result = {
            "_id": query['_id'],
            "target_sample": query['target_sample'],
            "query_prompt": query['query_prompt'],
            "membership_label": query['membership_label'],
            "is_perturbed": query['is_perturbed'],
            "title": query['title'],
            "text": query['text'],
            "original_id": query['original_id'],
            "original_title": query['original_title'],
            "original_text": query['original_text'],
            "perturbation_magnitude": query['perturbation_magnitude'],
            "retrieved_doc_ids": query['retrieved_doc_ids'],
            "is_hit": query['is_hit'],
            "hit_rank": query['hit_rank'],
            "rag_answer": answer,
            "raw_answer": raw_answer
        }
        results.append(result)
    
    return results


def generate_rag_answers_for_dataset(
    queries_path: str,
    output_path: str,
    retrieval_file_path: str,
    corpus: List[dict],
    generation_method: str = 'openai',
    openai_client: OpenAI = None,
    openai_model: str = 'gpt-4.1-nano',
    tokenizer = None,
    model = None,
    max_context_docs: int = 5,
    generation_batch_size: int = 8,
    accelerator = None,
    defense_type: str = None,
    defense_params: Dict = None
):
    """Generate DCMI RAG answers using retrieval results with defenses."""
    
    print(f"\n{'#'*60}")
    print(f"Generating DCMI RAG answers")
    print(f"Method: {generation_method}")
    if generation_method == 'openai':
        print(f"Model: {openai_model}")
    else:
        print(f"Batch size: {generation_batch_size}")
    
    # Display defense configuration
    print(f"\n--- Defense Configuration ---")
    if defense_type:
        print(f"✓ Defense Type: {defense_type}")
        if defense_params:
            for key, value in defense_params.items():
                print(f"  {key}: {value}")
    else:
        print(f"✗ No Defense Applied")
    print(f"-----------------------------")
    
    print(f"\nQueries: {queries_path}")
    print(f"Retrieval: {retrieval_file_path}")
    print(f"Corpus documents: {len(corpus)}")
    print(f"Output: {output_path}")
    print(f"{'#'*60}\n")
    
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping.")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load queries and retrieval results (with optional defense at retrieval)
    queries = load_queries_and_retrieval(
        queries_path,
        retrieval_file_path,
        corpus,
        max_context_docs,
        defense_type,
        defense_params
    )
    
    print(f"\nProcessing {len(queries)} queries...")
    
    # Generate based on method
    if generation_method == 'openai':
        output_dir = Path(output_path).parent / "openai_batches"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = generate_with_openai_batch(
            queries,
            openai_client,
            openai_model,
            output_dir,
            defense_type,
            defense_params
        )
    else:
        results = generate_with_transformers_batch(
            queries,
            tokenizer,
            model,
            generation_batch_size,
            accelerator=accelerator,
            defense_type=defense_type,
            defense_params=defense_params
        )
    
    # Write results
    write_jsonl(results, output_path)
    
    # Calculate statistics
    member_count = sum(1 for r in results if r['membership_label'] == 1)
    nonmember_count = sum(1 for r in results if r['membership_label'] == 0)
    original_count = sum(1 for r in results if not r['is_perturbed'])
    perturbed_count = sum(1 for r in results if r['is_perturbed'])
    
    yes_count = sum(1 for r in results if r['rag_answer'] == 'Yes')
    no_count = sum(1 for r in results if r['rag_answer'] == 'No')
    
    # Calculate Yes rates by category
    member_yes = sum(1 for r in results if r['membership_label'] == 1 and r['rag_answer'] == 'Yes')
    nonmember_yes = sum(1 for r in results if r['membership_label'] == 0 and r['rag_answer'] == 'Yes')
    original_yes = sum(1 for r in results if not r['is_perturbed'] and r['rag_answer'] == 'Yes')
    perturbed_yes = sum(1 for r in results if r['is_perturbed'] and r['rag_answer'] == 'Yes')
    
    print(f"\n{'='*60}")
    print(f"Generated RAG answers for {len(results)} queries")
    print(f"\nQuery Distribution:")
    print(f"  Member queries: {member_count}")
    print(f"  Nonmember queries: {nonmember_count}")
    print(f"  Original queries: {original_count}")
    print(f"  Perturbed queries: {perturbed_count}")
    print(f"\nAnswer Distribution:")
    print(f"  Yes: {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"  No: {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"\nYes Rate by Category:")
    print(f"  Member: {member_yes}/{member_count} ({member_yes/member_count*100:.1f}%)" if member_count > 0 else "  Member: N/A")
    print(f"  Nonmember: {nonmember_yes}/{nonmember_count} ({nonmember_yes/nonmember_count*100:.1f}%)" if nonmember_count > 0 else "  Nonmember: N/A")
    print(f"  Original: {original_yes}/{original_count} ({original_yes/original_count*100:.1f}%)" if original_count > 0 else "  Original: N/A")
    print(f"  Perturbed: {perturbed_yes}/{perturbed_count} ({perturbed_yes/perturbed_count*100:.1f}%)" if perturbed_count > 0 else "  Perturbed: N/A")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate DCMI RAG answers with defense mechanisms')
    
    # Basic arguments
    parser.add_argument(
        '--queries_file',
        type=str,
        default='results/DCMI/BeIR_nfcorpus/queries/queries.jsonl',
        help='Path to queries file generated by generate_queries.py'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results/DCMI/BeIR_nfcorpus/rag/rag_answers.jsonl',
        help='Path to output RAG answers file'
    )
    parser.add_argument(
        '--retrieval_file',
        type=str,
        default='results/DCMI/BeIR_nfcorpus/retrieval/topk5_results.json',
        help='Path to retrieval results file'
    )
    parser.add_argument(
        '--corpus_member_file',
        type=str,
        default='data/BeIR_nfcorpus/corpus_member.jsonl',
        help='Path to member corpus file'
    )
    parser.add_argument(
        '--corpus_nonmember_file',
        type=str,
        default='data/BeIR_nfcorpus/corpus_nonmember.jsonl',
        help='Path to nonmember corpus file'
    )
    
    # Generation method arguments
    parser.add_argument(
        '--generation_method',
        type=str,
        choices=['transformers', 'openai'],
        default='transformers',
        help='Generation method'
    )
    parser.add_argument(
        '--transformers_model',
        type=str,
        default='microsoft/phi-4',
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
        default="../hf_cache/hub",
        help='Cache directory for models'
    )
    parser.add_argument(
        '--max_context_docs',
        type=int,
        default=5,
        help='Maximum number of retrieved documents to use as context'
    )
    parser.add_argument(
        '--generation_batch_size',
        type=int,
        default=64,
        help='Batch size for generation'
    )
    
    # Defense mechanisms (ONLY ONE at a time)
    parser.add_argument(
        '--defense',
        type=str,
        choices=['none', 'dp', 'rerank', 'prompt_instruction', 'paraphrase'],
        default='none',
        help='Defense mechanism to apply (ONLY ONE)'
    )
    
    # DP-specific parameters
    parser.add_argument(
        '--dp_epsilon',
        type=float,
        default=0.1,
        help='Per-query privacy budget epsilon for DP defense (default: 0.1 for ε_total=2)'
    )
    parser.add_argument(
        '--dp_level',
        type=str,
        choices=['retrieval', 'output', 'both'],
        default='output',
        help='Where to apply DP: retrieval, output (standard for DP-RAG), or both'
    )
    
    # Reranking-specific parameters
    parser.add_argument(
        '--rerank_strategy',  
        type=str,
        choices=['reverse', 'shuffle', 'score_noise'],
        default='shuffle',
        help='Reranking strategy for rerank defense'
    )
    
    args = parser.parse_args()

    from utils.env_config import normalize_cache_dir

    args.cache_dir = normalize_cache_dir(args.cache_dir)

    # Prepare defense configuration
    defense_type = None
    defense_params = {}
    
    if args.defense != 'none':
        defense_type = args.defense
        
        if args.defense == 'dp':
            defense_params = {
                'epsilon': args.dp_epsilon,
                'level': args.dp_level
            }
        elif args.defense == 'rerank':
            defense_params = {
                'strategy': args.rerank_strategy
            }
    
    # -------------------------------------------------------
    # Resolve queries_file and retrieval_file for paraphrase
    # defense if not explicitly overridden by the user
    # -------------------------------------------------------
    queries_file = args.queries_file
    retrieval_file = args.retrieval_file
    output_file = args.output_file

    if args.defense == 'paraphrase':
        queries_path = Path(args.queries_file)
        dataset_root = queries_path.parents[1]

        queries_file = str(dataset_root / "queries_paraphrased" / queries_path.name)

        retrieval_path = Path(args.retrieval_file)
        retrieval_file = str(dataset_root / "retrieval_paraphrased" / retrieval_path.name)

        # Do NOT re-derive output_file — bash already passes the correct path
        output_file = args.output_file

        print(f"\n[Paraphrase Defense]")
        print(f"  Queries  : {queries_file}")
        print(f"  Retrieval: {retrieval_file}")
        print(f"  Output   : {output_file}")

    # Load corpus from both member and nonmember files
    print(f"\nLoading member corpus: {args.corpus_member_file}")
    corpus_member = read_jsonl(args.corpus_member_file)
    print(f"Loaded {len(corpus_member)} member documents")
    
    print(f"Loading nonmember corpus: {args.corpus_nonmember_file}")
    corpus_nonmember = read_jsonl(args.corpus_nonmember_file)
    print(f"Loaded {len(corpus_nonmember)} nonmember documents")
    
    # Combine corpora
    corpus_combined = corpus_member + corpus_nonmember
    print(f"Total corpus size: {len(corpus_combined)} documents")
    
    # Initialize generation backend
    openai_client = None
    tokenizer = None
    model = None
    accelerator = None
    
    if args.generation_method == 'openai':
        openai_client = initialize_openai_client('.env')
        print(f"\nUsing OpenAI: {args.openai_model}")
    else:
        if args.cache_dir:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        accelerator = Accelerator(kwargs_handlers=[kwargs])
        tokenizer, model = load_generator_model(args.transformers_model, args.use_gpu, args.cache_dir, accelerator)
        print(f"\nUsing Transformers: {args.transformers_model}")
        print(f"Batch size: {args.generation_batch_size}")
    
    generate_rag_answers_for_dataset(
        queries_file,       # resolved path
        output_file,        # resolved path
        retrieval_file,     # resolved path
        corpus_combined,
        args.generation_method,
        openai_client,
        args.openai_model,
        tokenizer,
        model,
        args.max_context_docs,
        args.generation_batch_size,
        accelerator=accelerator,
        defense_type=defense_type,
        defense_params=defense_params
    )
    
    print("\n" + "="*60)
    print("DCMI RAG answer generation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()