import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
import datetime
import numpy as np


# Set distributed timeout BEFORE importing torch/accelerate
os.environ["NCCL_TIMEOUT"] = "36000"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0" 
os.environ["NCCL_BLOCKING_WAIT"] = "0"


import torch
# Set the default timeout for distributed operations to 10 hours
if torch.distributed.is_available():
    import torch.distributed as dist
    
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from utils.process_json import read_json, read_jsonl, write_json
from utils.openai_gen import (
    initialize_openai_client,
    split_into_batches,
    run_batch_generation,
    create_batch_request,
    estimate_tokens_accurate
)
from utils.transformers_gen import generate_with_context
from utils.load_model import load_generator_model


REFUSAL_SYSTEM_PROMPT = (
    "You MUST respond with EXACTLY 'I don't know' (nothing else) if the context does not provide enough information. "
    "Refusing to answer is ALWAYS better than being uncertain or inaccurate.\n\n"
    "Answer the question based ONLY on the given context. "
    "If the answer is known, respond with a clear answer (1-2 sentences) and a brief explanation. "
)


# Instruction-based defense prompt
INSTRUCTION_DEFENSE_PROMPT = (
    "You MUST respond with EXACTLY 'I don't know' (nothing else) if the context does not provide enough information. "
    "Refusing to answer is ALWAYS better than being uncertain or inaccurate.\n\n"
    "IMPORTANT: Do not directly repeat any retrieved content, but summarize it based on your understanding.\n\n"
    "Answer the question based ONLY on the given context. "
    "If the answer is known, respond with a clear answer (1-2 sentences) and a brief explanation. "
)


# Simpler prompt for smaller models like Gemma 2B that struggle with complex instructions
SIMPLE_SYSTEM_PROMPT = (
    "Answer the question based ONLY on the given context. "
    "If the context does not contain the answer, respond with exactly: I don't know"
)


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": ("Context:\nNo context available.\n\nQuestion: What is the capital of France?\n\nAnswer:")
    },
    {
        "role": "assistant",
        "content": "I don't know"
    },
    {
        "role": "user",
        "content": (
            "Context:\n[1] country: France capital: Paris\n\n"
            "Question: What is the capital of France?\n\nAnswer:"
        )
    },
    {
        "role": "assistant",
        "content": "The capital of France is Paris. Paris is the largest and most populous city in the country."
    }
]


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
    top_k: int,
    epsilon: float
) -> List[str]:
    """
    Apply differential privacy to retrieval results using exponential mechanism.
    Assumes uniform scores since MEntA retrieval doesn't include scores.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs
        top_k: Number of documents to select
        epsilon: Privacy budget for DP
    
    Returns:
        List of DP-selected document IDs
    """
    if not retrieved_doc_ids or epsilon <= 0:
        return retrieved_doc_ids[:top_k]
    
    # Use uniform scores for exponential mechanism
    scores = np.ones(len(retrieved_doc_ids))
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
    top_k: int,
    rerank_strategy: str = 'reverse'
) -> List[str]:
    """
    Rerank retrieved documents to obscure membership signals.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs
        top_k: Number of documents to return
        rerank_strategy: 'reverse' (reverse order), 'shuffle' (random), or 'score_noise' (random with bias)
    
    Returns:
        List of reranked document IDs
    """
    if not retrieved_doc_ids:
        return retrieved_doc_ids
    
    # Get top_k documents first
    doc_ids = retrieved_doc_ids[:top_k]
    
    if rerank_strategy == 'reverse':
        # Simply reverse the order
        return list(reversed(doc_ids))
    
    elif rerank_strategy == 'shuffle':
        # Random shuffle
        indices = list(range(len(doc_ids)))
        np.random.shuffle(indices)
        return [doc_ids[i] for i in indices]
    
    elif rerank_strategy == 'score_noise':
        # Create pseudo-scores and add noise
        scores = np.linspace(1.0, 0.5, len(doc_ids))  # Decreasing scores
        noisy_scores = scores + np.random.normal(0, 0.1, len(scores))
        sorted_indices = np.argsort(noisy_scores)[::-1]
        return [doc_ids[i] for i in sorted_indices]
    
    else:
        return doc_ids


def load_summary_map(summary_path: Path) -> Dict[str, dict]:
    """Load summaries keyed by document id."""
    summaries = {}
    for record in read_jsonl(str(summary_path)):
        doc_id = record.get('_id')
        if doc_id:
            summaries[doc_id] = record
    return summaries


def get_answers_dirname(
    defense_type: Optional[str],
    generic_queries: bool,
    disable_retrieval: bool
) -> str:
    """Build answers directory name from defense/query mode flags."""
    if defense_type == 'paraphrase':
        dirname = 'answers_paraphrased'
    elif defense_type and defense_type != 'none':
        dirname = f'answers_{defense_type}'
    else:
        dirname = 'answers'

    if generic_queries:
        dirname = f'{dirname}_generic'
    if disable_retrieval:
        dirname = f'{dirname}_disable_retrieval'

    return dirname


def infer_generic_queries_mode(
    retrieval_data: Dict,
    retrieval_path: Path,
    generic_queries: bool
) -> bool:
    """Infer whether retrieval was built from generic queries."""
    query_file = retrieval_data.get('metadata', {}).get('query_file', '')
    inferred = (
        'queries_generic' in str(query_file) or
        any(part.endswith('_generic') and part.startswith('retrieval') for part in retrieval_path.parts)
    )
    return generic_queries or inferred


def resolve_query_records(
    retrieval_data: Dict,
    generic_queries: bool
) -> Tuple[List[Dict], Optional[str]]:
    """Resolve the query records to use for answer generation."""
    queries = retrieval_data['queries']
    if not generic_queries:
        return queries, retrieval_data.get('metadata', {}).get('query_file')

    query_file = retrieval_data.get('metadata', {}).get('query_file')
    if not query_file:
        return queries, None

    query_path = Path(query_file)
    generic_query_path = query_path
    if 'queries_generic' not in query_path.parts:
        generic_query_path = Path(str(query_path).replace('/queries/', '/queries_generic/'))

    if not generic_query_path.exists():
        print(f"Warning: Generic query file not found, falling back to retrieval queries: {generic_query_path}")
        return queries, str(query_path)

    generic_query_records = read_jsonl(str(generic_query_path))
    generic_query_map = {record.get('_id'): record for record in generic_query_records}

    resolved_queries = []
    missing_query_ids = []
    for query in queries:
        generic_query = generic_query_map.get(query.get('_id'))
        if generic_query:
            resolved_queries.append(generic_query)
        else:
            resolved_queries.append(query)
            missing_query_ids.append(query.get('_id'))

    if missing_query_ids:
        print(f"Warning: {len(missing_query_ids)} generic queries missing by id; kept original retrieval queries for those entries")

    return resolved_queries, str(generic_query_path)


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
    
    return ' '.join(noised_tokens) if noised_tokens else "I don't know."


def generate_answers_openai(
    queries_list: List[str],
    context_docs_list: List[List[dict]],
    query_metadata: List[Dict],
    model: str,
    temperature: float,
    max_context_docs: int,
    output_path: Path,
    env_path: str,
    max_tokens_per_batch: int,
    check_interval: int,
    defense_config: Dict = None
):
    """Generate answers using OpenAI Batch API with optional defenses."""
    client = initialize_openai_client(env_path)
    
    # Select system prompt based on defense
    if defense_config and defense_config.get('type') == 'prompt_instruction':
        system_prompt = INSTRUCTION_DEFENSE_PROMPT
        print("Using instruction-based defense prompt")
    else:
        system_prompt = REFUSAL_SYSTEM_PROMPT
    
    # Prepare items
    items = list(zip(queries_list, context_docs_list, query_metadata))
    
    # Estimate tokens function
    def estimate_item_tokens(item):
        query, context_docs, _ = item
        
        context_texts = []
        for i, doc in enumerate(context_docs[:max_context_docs]):
            text_parts = []
            for key, value in doc.items():
                if key != '_id' and value:
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, (list, dict)):
                        text_parts.append(f"{key}: {str(value)}")
            doc_text = ' '.join(text_parts).strip()
            context_texts.append(f"[{i+1}] {doc_text}")
        
        context = "\n\n".join(context_texts)
        full_text = f"{context}\n\nQuestion: {query}"
        
        return estimate_tokens_accurate(full_text, 1, model)
    
    # Split into batches
    batches = split_into_batches(items, estimate_item_tokens, max_tokens_per_batch)
    
    # Create request function
    def create_request(item):
        query, context_docs, metadata = item
        
        context_texts = []
        for i, doc in enumerate(context_docs[:max_context_docs]):
            text_parts = []
            for key, value in doc.items():
                if key != '_id' and value:
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, (list, dict)):
                        text_parts.append(f"{key}: {str(value)}")
            doc_text = ' '.join(text_parts).strip()
            context_texts.append(f"[{i+1}] {doc_text}")
    
        context = "\n\n".join(context_texts) if context_texts else "No context available."
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""Based on the following context, answer the question. If the context does not contain enough information to answer the question, respond only with 'I don't know'.

Context:
{context}

Question: {query}

Answer:"""
            }
        ]
        
        return create_batch_request(
            custom_id=metadata['query_id'],
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=512
        )
    
    # Result processor function
    def process_result(custom_id, generated_text):
        # Find matching metadata
        matching_metadata = None
        for metadata in query_metadata:
            if metadata['query_id'] == custom_id:
                matching_metadata = metadata
                break
        
        if matching_metadata:
            answer = generated_text
            
            # Apply output-level DP if enabled
            if defense_config and defense_config.get('type') == 'dp' and defense_config.get('level') in ['output', 'both']:
                answer = add_laplace_noise(answer, defense_config.get('epsilon', 1.0))
            
            matching_metadata['generated_answer'] = answer
            return matching_metadata, None
        else:
            return None, {'query_id': custom_id, 'reason': 'Metadata not found'}
    
    # Run batch generation
    all_results, all_failures = run_batch_generation(
        client,
        batches,
        create_request,
        process_result,
        output_path.parent,
        output_path.stem,
        check_interval,
        metadata_base={
            "description": "RAG answer generation with defenses",
            "defense": defense_config
        }
    )
    
    return all_results, all_failures


def generate_answers_transformers(
    queries_list: List[str],
    context_docs_list: List[List[dict]],
    query_metadata: List[Dict],
    tokenizer,
    model,
    max_context_docs: int,
    max_new_tokens: int,
    batch_size: int,
    model_name: str = None,
    accelerator: Accelerator = None,
    defense_config: Dict = None
):
    """Generate answers using transformers with optional defenses."""
    # Select system prompt based on defense
    if defense_config and defense_config.get('type') == 'prompt_instruction':
        system_prompt = INSTRUCTION_DEFENSE_PROMPT
        print("Using instruction-based defense prompt")
    else:
        system_prompt = REFUSAL_SYSTEM_PROMPT
    
    generated_answers = generate_with_context(
        queries_list,
        context_docs_list,
        tokenizer,
        model,
        accelerator=accelerator,
        max_new_tokens=max_new_tokens,
        max_context_docs=max_context_docs,
        batch_size=batch_size,
        system_prompt=system_prompt,
        temperature=0.4,
        top_p=0.9,
        top_k=50,
        model_name=model_name
    )
    
    all_results = []
    all_failures = []
    
    for answer, metadata in zip(generated_answers, query_metadata):
        if answer and answer.strip():
            # Apply output-level DP if enabled
            if defense_config and defense_config.get('type') == 'dp' and defense_config.get('level') in ['output', 'both']:
                answer = add_laplace_noise(answer, defense_config.get('epsilon', 1.0))
            
            metadata['generated_answer'] = answer
            all_results.append(metadata)
        else:
            all_failures.append({
                'query_id': metadata['query_id'],
                'reason': 'Empty or None answer'
            })
    
    return all_results, all_failures


def generate_with_retrieval(
    retrieval_file_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    summary_path: Optional[str] = None,
    generation_method: str = 'openai',
    generator_name: str = 'meta-llama/Llama-3.1-8B-Instruct',
    use_gpu: bool = False,
    cache_dir: str = None,
    max_context_docs: int = 5,
    max_new_tokens: int = 512,
    generation_batch_size: int = 8,
    openai_model: str = 'gpt-4.1-nano',
    env_path: str = '.env',
    max_tokens_per_batch: int = 1_200_000,
    check_interval: int = 120,
    generic_queries: bool = False,
    disable_retrieval: bool = False,
    defense_type: str = None,
    defense_params: Dict = None
):
    """Generate answers using retrieval results with optional defenses."""
    print(f"\n{'#'*60}")
    print(f"RAG ANSWER GENERATION (MEntA)")
    print(f"{'#'*60}")
    print(f"Retrieval file: {retrieval_file_path}")
    print(f"Corpus member: {corpus_member_path}")
    print(f"Corpus nonmember: {corpus_nonmember_path}")
    print(f"Generic queries: {generic_queries}")
    print(f"Summary-only mode: {disable_retrieval}")
    if summary_path:
        print(f"Summary file: {summary_path}")
    print(f"Generation method: {generation_method}")
    
    # Display defense configuration
    if defense_type:
        print(f"\n--- Defense Configuration ---")
        print(f"✓ Defense Type: {defense_type}")
        if defense_params:
            for key, value in defense_params.items():
                print(f"  {key}: {value}")
        print(f"-----------------------------")
    else:
        print(f"\n✗ No Defense Applied")
    
    print(f"{'#'*60}\n")
    
    retrieval_path = Path(retrieval_file_path)
    corpus_member_path_obj = Path(corpus_member_path)
    corpus_nonmember_path_obj = Path(corpus_nonmember_path)
    summary_path_obj = Path(summary_path).expanduser() if summary_path else corpus_member_path_obj.parent / 'summary.jsonl'
    
    if not retrieval_path.exists():
        print(f"Error: Retrieval file not found: {retrieval_file_path}")
        return
    
    if disable_retrieval:
        if not summary_path_obj.exists():
            print(f"Error: Summary file not found: {summary_path_obj}")
            return
    else:
        if not corpus_member_path_obj.exists():
            print(f"Error: Corpus member file not found: {corpus_member_path}")
            return
        
        if not corpus_nonmember_path_obj.exists():
            print(f"Error: Corpus nonmember file not found: {corpus_nonmember_path}")
            return
    
    # Load data
    retrieval_data = read_json(str(retrieval_path))
    generic_queries = infer_generic_queries_mode(retrieval_data, retrieval_path, generic_queries)
    
    doc_map = {}
    summary_map = {}
    corpus_member = []
    corpus_nonmember = []
    if disable_retrieval:
        summary_map = load_summary_map(summary_path_obj)
    else:
        corpus_member = read_jsonl(str(corpus_member_path_obj))
        corpus_nonmember = read_jsonl(str(corpus_nonmember_path_obj))
        
        # Combine both corpora into single doc_map
        for doc in corpus_member:
            doc_map[doc['_id']] = doc
        for doc in corpus_nonmember:
            doc_map[doc['_id']] = doc
    
    queries, resolved_query_file = resolve_query_records(retrieval_data, generic_queries)

    print(f"Loaded retrieval results for {len(retrieval_data['queries'])} queries")
    print(f"Using generic queries: {generic_queries}")
    if resolved_query_file:
        print(f"Query file used for generation: {resolved_query_file}")
    if disable_retrieval:
        print(f"Loaded {len(summary_map)} summaries")
    else:
        print(f"Loaded {len(corpus_member)} member documents")
        print(f"Loaded {len(corpus_nonmember)} nonmember documents")
        print(f"Total corpus size: {len(doc_map)} documents")
    
    # Setup output path
    import re
    topk_match = re.search(r'topk(\d+)', str(retrieval_path))
    version_match = re.search(r'retrieval_(\d+)v\.json', retrieval_path.name)
    
    if topk_match and version_match:
        topk_value = topk_match.group(1)
        num_queries = version_match.group(1)
        
        # Build output path with defense suffix
        generator_name_safe = generator_name.replace('/', '--') if generation_method == 'transformers' else openai_model.replace('/', '--')
        
        # Get base directory from retrieval path, replace retrieval folder with answers variant
        parts = list(retrieval_path.parts)
        answers_dirname = get_answers_dirname(defense_type, generic_queries, disable_retrieval)
        base_idx = -1
        for idx, part in enumerate(parts[:-1]):
            if part == 'retrieval' or part.startswith('retrieval_'):
                base_idx = idx
                break
        if base_idx != -1:
            parts[base_idx] = answers_dirname
        
        base_dir = Path(*parts[:-1])
        output_path = base_dir / generator_name_safe / f"answers_{num_queries}v.json"
    else:
        print("Warning: Could not extract topk/version from path, using fallback output path")
        generator_name_safe = generator_name.replace('/', '--') if generation_method == 'transformers' else openai_model.replace('/', '--')
        filename = retrieval_path.name.replace('retrieval', 'answers')
        output_parent = retrieval_path.parent.parent / get_answers_dirname(defense_type, generic_queries, disable_retrieval) / generator_name_safe
        output_path = output_parent / filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output: {output_path}\n")
    
    # Prepare data
    retrieval_results = retrieval_data['retrieval_results']
    
    queries_list = []
    context_docs_list = []
    query_metadata = []
    
    for query in queries:
        query_id = query['_id']
        query_text = query['text']
        retrieved_doc_ids = retrieval_results.get(query_id, [])
        
        if disable_retrieval:
            target_doc_id = query.get('target_doc_id')
            summary_record = summary_map.get(target_doc_id, {})
            summary_text = summary_record.get('summary', '')
            context_docs = [{
                '_id': target_doc_id,
                'title': summary_record.get('title'),
                'summary': summary_text
            }] if summary_text else []
            retrieved_doc_ids = []
        else:
            # Apply defense at retrieval level if needed
            if defense_type == 'dp' and defense_params.get('level') in ['retrieval', 'both']:
                retrieved_doc_ids = apply_dp_retrieval(
                    retrieved_doc_ids,
                    max_context_docs,
                    defense_params.get('epsilon', 1.0)
                )
            elif defense_type == 'rerank':
                retrieved_doc_ids = rerank_retrieved_docs(
                    retrieved_doc_ids,
                    max_context_docs,
                    defense_params.get('strategy', 'reverse')
                )
            else:
                retrieved_doc_ids = retrieved_doc_ids[:max_context_docs]
            
            context_docs = [doc_map[doc_id] for doc_id in retrieved_doc_ids if doc_id in doc_map]
        
        queries_list.append(query_text)
        context_docs_list.append(context_docs)
        query_metadata.append({
            'query_id': query_id,
            'query_text': query_text,
            'doc_id': query.get('target_doc_id'),
            'membership': query.get('_membership'),
            'retrieved_doc_ids': retrieved_doc_ids,
            'generated_answer': ''
        })
    
    print(f"Generating answers for {len(queries_list)} queries...\n")
    
    # Prepare defense config for generation
    defense_config = None
    if defense_type:
        defense_config = {
            'type': defense_type,
            **defense_params
        }
    
    # Generate
    if generation_method == 'openai':
        all_results, all_failures = generate_answers_openai(
            queries_list,
            context_docs_list,
            query_metadata,
            openai_model,
            0.7,
            max_context_docs,
            output_path,
            env_path,
            max_tokens_per_batch,
            check_interval,
            defense_config
        )
    else:
        if cache_dir:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        print(f"Loading generator model: {generator_name}")

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        accelerator = Accelerator(kwargs_handlers=[kwargs])
        tokenizer, model = load_generator_model(generator_name, use_gpu, cache_dir, accelerator)
        
        all_results, all_failures = generate_answers_transformers(
            queries_list,
            context_docs_list,
            query_metadata,
            tokenizer,
            model,
            max_context_docs,
            max_new_tokens,
            generation_batch_size,
            model_name=generator_name,
            accelerator=accelerator,
            defense_config=defense_config
        )
    
    # Save results
    if all_results:
        metadata_dict = {
            'retrieval_file': str(retrieval_path),
            'corpus_member_file': corpus_member_path,
            'corpus_nonmember_file': corpus_nonmember_path,
            'generation_method': generation_method,
            'generator_model': generator_name if generation_method == 'transformers' else openai_model,
            'retrieval_model': retrieval_data['metadata'].get('retrieval_model', 'unknown'),
            'query_file': resolved_query_file,
            'summary_file': str(summary_path_obj) if disable_retrieval else None,
            'generic_queries': generic_queries,
            'query_mode': 'generic' if generic_queries else 'specific',
            'disable_retrieval': disable_retrieval,
            'context_source': 'target_summary' if disable_retrieval else 'retrieval',
            'max_context_docs': max_context_docs,
            'max_new_tokens': max_new_tokens if generation_method == 'transformers' else 512,
            'total_queries': len(queries),
            'successful_generations': len(all_results),
            'failed_generations': len(all_failures),
            'defense': defense_config
        }
        
        output_data = {
            'metadata': metadata_dict,
            'results': all_results
        }
        write_json(output_data, str(output_path))
        print(f"\n✓ Saved {len(all_results)} results to: {output_path}")
    else:
        print(f"\n⚠ No successful results to save")
    
    if all_failures:
        failure_path = output_path.parent / f"{output_path.stem}_failures.json"
        write_json({'failures': all_failures, 'total_failed': len(all_failures)}, str(failure_path))
        print(f"⚠ Saved {len(all_failures)} failures to: {failure_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries: {len(queries)}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {len(all_failures)}")
    print(f"Success rate: {len(all_results) / len(queries) * 100:.1f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate answers using retrieval results (MEntA) with defenses')
    
    # Basic arguments
    parser.add_argument(
        '--retrieval_file',
        type=str,
        default="results/MEntA/BeIR_nfcorpus/retrieval/target_summary/topk3/retrieval_5v.json",
        help='Path to retrieval results JSON'
    )
    parser.add_argument(
        '--corpus_member_path',
        type=str,
        default="data/BeIR_nfcorpus/corpus_member.jsonl",
        help='Path to corpus member JSONL'
    )
    parser.add_argument(
        '--corpus_nonmember_path',
        type=str,
        default="data/BeIR_nfcorpus/corpus_nonmember.jsonl",
        help='Path to corpus nonmember JSONL'
    )
    parser.add_argument(
        '--summary_path',
        type=str,
        default=None,
        help='Path to summary JSONL file used for summary-only generation (defaults to sibling summary.jsonl)'
    )
    
    # Generation method arguments
    parser.add_argument(
        '--generation_method',
        type=str,
        choices=['transformers', 'openai'],
        default='transformers',
        help='Generation method (openai uses Batch API only)'
    )
    parser.add_argument(
        '--generator_model',
        type=str,
        default='google/gemma-2b-it',
        help='Generator model name (for transformers method)'
    )
    parser.add_argument(
        '--openai_model',
        type=str,
        default='gpt-4.1-nano',
        help='OpenAI model name (for openai method)'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU for transformers generation'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='../hf_cache/hub/',
        help='HuggingFace cache directory'
    )
    parser.add_argument(
        '--max_context_docs',
        type=int,
        default=5,
        help='Maximum number of context documents to use'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Maximum tokens to generate (transformers only)'
    )
    parser.add_argument(
        '--generation_batch_size',
        type=int,
        default=32,
        help='Batch size for transformers generation'
    )
    parser.add_argument(
        '--env_path',
        type=str,
        default='../env',
        help='Path to .env file with OPENAI_API_KEY'
    )
    parser.add_argument(
        '--max_tokens_per_batch',
        type=int,
        default=2_000_000,
        help='Maximum tokens per OpenAI batch (default: 2,000,000)'
    )
    parser.add_argument(
        '--check_interval',
        type=int,
        default=120,
        help='Seconds between batch status checks (default: 120)'
    )
    parser.add_argument(
        '--generic_queries',
        action='store_true',
        help='Use retrieval results produced from generic topical queries and save under *_generic folders'
    )
    parser.add_argument(
        '--disable_retrieval',
        action='store_true',
        help='Skip retrieved documents and answer using only the target document summary'
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
        help='Privacy budget epsilon for DP defense'
    )
    parser.add_argument(
        '--dp_level',
        type=str,
        choices=['retrieval', 'output', 'both'],
        default='output',
        help='Where to apply DP: retrieval, output, or both'
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
    
    generate_with_retrieval(
        retrieval_file_path=args.retrieval_file,
        corpus_member_path=args.corpus_member_path,
        corpus_nonmember_path=args.corpus_nonmember_path,
        summary_path=args.summary_path,
        generation_method=args.generation_method,
        generator_name=args.generator_model,
        use_gpu=args.use_gpu,
        cache_dir=args.cache_dir,
        max_context_docs=args.max_context_docs,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
        openai_model=args.openai_model,
        env_path=args.env_path,
        max_tokens_per_batch=args.max_tokens_per_batch,
        check_interval=args.check_interval,
        generic_queries=args.generic_queries,
        disable_retrieval=args.disable_retrieval,
        defense_type=defense_type,
        defense_params=defense_params
    )


if __name__ == '__main__':
    main()
