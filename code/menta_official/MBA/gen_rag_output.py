#!/usr/bin/env python3
"""
Mask-based MIA RAG Output Generation Script
Generates RAG system outputs for mask prediction using retrieved context.
Saves log probabilities for perplexity calculation.
Includes defense mechanisms: DP, Reranking, Prompt Instruction, Paraphrasing
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
from utils.transformers_gen import create_chat_prompt
from utils.load_model import load_generator_model
from accelerate import Accelerator, InitProcessGroupKwargs


# Load environment variables
load_dotenv('.env')


# Standard prompt template
RAG_PROMPT_TEMPLATE = """Predict the answers of the masks in the given texts:

{masked_document}

You should strictly follow the contexts below to answer, and provide your answers in the specified format: "[Mask_i]: answer_i\\n" for each mask:

{retrieved_documents}"""


# Instruction-based defense prompt
RAG_PROMPT_TEMPLATE_INSTRUCTION = """Predict the answers of the masks in the given texts:

{masked_document}

IMPORTANT: Do not directly repeat any retrieved content, but summarize it based on your understanding.

You should strictly follow the contexts below to answer, and provide your answers in the specified format: "[Mask_i]: answer_i\\n" for each mask:

{retrieved_documents}"""


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

    return ' '.join(noised_tokens) if noised_tokens else "[Mask_1]: unknown"


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
            context_parts.append(' '.join(doc_text))

    full_context = '\n\n'.join(context_parts)
    return full_context


def build_rag_prompt_mask_mia(
    masked_document: str,
    retrieved_documents: str,
    use_instruction_defense: bool = False
) -> str:
    """
    Build RAG prompt for Mask-based MIA.

    Args:
        masked_document: Text with masks
        retrieved_documents: Retrieved context
        use_instruction_defense: Whether to use instruction-based defense
    """
    template = RAG_PROMPT_TEMPLATE_INSTRUCTION if use_instruction_defense else RAG_PROMPT_TEMPLATE

    prompt = template.format(
        masked_document=masked_document,
        retrieved_documents=retrieved_documents
    )

    return prompt


def load_queries_and_retrieval(
    queries_path: str,
    retrieval_file_path: str,
    corpus: List[dict],
    max_context_docs: int = 5,
    defense_type: str = None,
    defense_params: Dict = None
) -> List[dict]:
    """
    Load Mask-based MIA queries and match them with retrieval results.
    Apply defenses at retrieval level if specified.

    Returns list of dicts with query info and context from retrieved documents.
    """
    # Load queries
    queries = read_jsonl(queries_path)
    print(f"Loaded {len(queries)} queries")

    # Load retrieval results
    with open(retrieval_file_path, 'r') as f:
        retrieval_data = json.load(f)
    retrieval_results = retrieval_data.get('retrieval_results', retrieval_data)
    print(f"Loaded retrieval results for {len(retrieval_results)} queries")

    # Use provided corpus
    doc_map = {doc['_id']: doc for doc in corpus}
    print(f"Using corpus with {len(corpus)} documents")

    if defense_type in ['dp', 'rerank']:
        print(f"Applying {defense_type} defense at retrieval level")

    # Check if using instruction-based defense
    use_instruction = defense_type == 'prompt_instruction'

    # Enrich queries with retrieval context
    enriched_queries = []
    for query in queries:
        doc_id = query['_id']
        masked_text = query['masked_text']
        mask_values = query['mask_values']
        mask_positions = query.get('mask_positions', [])
        membership_label = query['membership_label']
        original_text = query.get('original_text', '')
        masking_strategy = query.get('masking_strategy', '')

        # Get retrieved doc IDs for this query
        if doc_id in retrieval_results:
            retrieved_doc_ids = retrieval_results[doc_id].get('retrieved_doc_ids', [])
            retrieval_scores = retrieval_results[doc_id].get('scores', [1.0] * len(retrieved_doc_ids))

            # Apply defense at retrieval level
            if defense_type == 'dp' and defense_params.get('level') in ['retrieval', 'both']:
                retrieved_doc_ids = apply_dp_retrieval(
                    retrieved_doc_ids,
                    retrieval_scores,
                    defense_params.get('epsilon', 1.0),
                    max_context_docs
                )
            elif defense_type == 'rerank':
                retrieved_doc_ids = rerank_retrieved_docs(
                    retrieved_doc_ids,
                    retrieval_scores,
                    max_context_docs,
                    defense_params.get('strategy', 'reverse')
                )
            else:
                retrieved_doc_ids = retrieved_doc_ids[:max_context_docs]
        else:
            print(f"Warning: Query {doc_id} not found in retrieval results")
            retrieved_doc_ids = []

        # Build context from retrieved documents
        retrieved_docs = [doc_map[rid] for rid in retrieved_doc_ids if rid in doc_map]
        context = build_context_from_retrieved_docs(retrieved_docs) if retrieved_docs else ""

        # Build full RAG prompt with masked document + retrieved context
        full_prompt = build_rag_prompt_mask_mia(masked_text, context, use_instruction)

        enriched_query = {
            '_id': doc_id,
            'masked_text': masked_text,
            'mask_values': mask_values,
            'mask_positions': mask_positions,
            'membership_label': membership_label,
            'original_text': original_text,
            'masking_strategy': masking_strategy,
            'context': context,
            'full_prompt': full_prompt,
            'retrieved_doc_ids': retrieved_doc_ids
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
    """Generate Mask-based MIA RAG outputs using OpenAI Batch API with logprobs and defenses."""

    print("\n" + "="*60)
    print("GENERATING MASK-BASED MIA RAG OUTPUTS (OpenAI)")
    if defense_type == 'dp' and defense_params.get('level') in ['output', 'both']:
        print(f"DP defense enabled at output level (ε={defense_params.get('epsilon', 1.0)})")
    print("="*60)

    # Estimate tokens function for batching
    def estimate_tokens(query):
        return estimate_tokens_accurate(query['full_prompt'], 1, openai_model)

    # Split into batches
    batches = split_into_batches(queries, estimate_tokens, max_tokens_per_batch=2_000_000)
    print(f"Total batches created: {len(batches)}")

    # Create batch requests with logprobs
    def create_rag_request(query):
        messages = [{"role": "user", "content": query['full_prompt']}]
        request = create_batch_request(
            custom_id=query['_id'],
            model=openai_model,
            messages=messages,
            temperature=0.0,  # Deterministic for consistency
            max_tokens=300
        )
        # Add logprobs parameter
        request['body']['logprobs'] = True
        request['body']['top_logprobs'] = 1
        return request

    # Build query lookup for result processing
    query_lookup = {q['_id']: q for q in queries}

    # Process results
    def process_rag_result(custom_id, generated_text, raw_response=None):
        query = query_lookup.get(custom_id)
        if not query:
            return None, {'custom_id': custom_id, 'reason': 'Query not found'}

        generated_output = generated_text.strip()

        # Apply DP at output level if enabled
        if defense_type == 'dp' and defense_params.get('level') in ['output', 'both']:
            generated_output = add_laplace_noise(generated_output, defense_params.get('epsilon', 1.0))

        # Extract logprobs if available
        logprobs = None
        if raw_response and 'choices' in raw_response:
            choice = raw_response['choices'][0]
            if 'logprobs' in choice and choice['logprobs']:
                content_logprobs = choice['logprobs'].get('content', [])
                logprobs = [token_data['logprob'] for token_data in content_logprobs]

        success_data = {
            '_id': query['_id'],
            'masked_text': query['masked_text'],
            'mask_values': query['mask_values'],
            'mask_positions': query['mask_positions'],
            'membership_label': query['membership_label'],
            'original_text': query['original_text'],
            'masking_strategy': query['masking_strategy'],
            'generated_output': generated_output,
            'logprobs': logprobs,
            'retrieved_doc_ids': query['retrieved_doc_ids']
        }

        return success_data, None

    results, failed = run_batch_generation(
        openai_client,
        batches,
        create_rag_request,
        process_rag_result,
        output_dir,
        "mask_mia_rag_outputs_batch",
        metadata_base={
            "type": "mask_mia_rag_outputs",
            "defense": {'type': defense_type, **(defense_params or {})}
        }
    )

    print(f"\n✓ Successfully generated {len(results)} RAG outputs")
    print(f"  Failures: {len(failed)}")

    return results


def generate_batch_transformers_with_logprobs(
    prompts: List[str],
    tokenizer,
    model,
    accelerator=None,
    temperature: float = 0.0,
    max_new_tokens: int = 300,
    batch_size: int = 8
) -> List[Dict]:
    """
    Generate text with transformers and return both text and logprobs.
    Returns list of dicts with 'text' and 'logprobs' keys.
    """
    all_results = []
    device = accelerator.device if accelerator else model.device

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)

        # Generate with output_scores=True to get logits
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_ids = outputs.sequences
        scores = outputs.scores  # List of tensors, one per generated token

        # Process each sample in batch
        for j, gen_ids in enumerate(generated_ids):
            # Get input length for this sample
            input_len = inputs['input_ids'][j].shape[0]

            # Extract only generated tokens (remove input)
            generated_tokens = gen_ids[input_len:]

            # Decode generated text
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Extract logprobs for generated tokens
            logprobs = []
            for token_idx, token_id in enumerate(generated_tokens):
                if token_idx < len(scores):
                    # Get logits for this position
                    token_logits = scores[token_idx][j]  # Shape: [vocab_size]

                    # Convert to log probabilities
                    log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)

                    # Get logprob for the actual generated token
                    token_logprob = log_probs[token_id].item()
                    logprobs.append(token_logprob)

            all_results.append({
                'text': generated_text,
                'logprobs': logprobs
            })

    return all_results


def generate_with_transformers_batch(
    queries: List[dict],
    tokenizer,
    model,
    batch_size: int = 8,
    accelerator = None,
    defense_type: str = None,
    defense_params: Dict = None
) -> List[dict]:
    """Generate Mask-based MIA RAG outputs using transformers with logprobs and defenses."""

    print("\n" + "="*60)
    print("GENERATING MASK-BASED MIA RAG OUTPUTS (Transformers)")
    if defense_type == 'dp' and defense_params.get('level') in ['output', 'both']:
        print(f"DP defense enabled at output level (ε={defense_params.get('epsilon', 1.0)})")
    print("="*60)

    # Prepare prompts
    all_prompts = []
    for query in queries:
        all_prompts.append(query['full_prompt'])

    print(f"Total queries to process: {len(all_prompts)}")

    # Format prompts with chat template
    formatted_prompts = []
    for prompt in all_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = create_chat_prompt(tokenizer, messages)
        formatted_prompts.append(formatted_prompt)

    # Generate outputs with logprobs
    all_outputs = generate_batch_transformers_with_logprobs(
        formatted_prompts,
        tokenizer,
        model,
        accelerator=accelerator,
        temperature=0.0,
        max_new_tokens=300,
        batch_size=batch_size
    )

    # Post-process outputs and build results
    results = []
    for query, output_dict in zip(queries, all_outputs):
        output = output_dict['text']
        logprobs = output_dict['logprobs']

        # Clean up output - keep the mask predictions
        output = output.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

        # Apply DP at output level if enabled
        if defense_type == 'dp' and defense_params.get('level') in ['output', 'both']:
            output = add_laplace_noise(output, defense_params.get('epsilon', 1.0))

        result = {
            "_id": query['_id'],
            "masked_text": query['masked_text'],
            "mask_values": query['mask_values'],
            "mask_positions": query['mask_positions'],
            "membership_label": query['membership_label'],
            "original_text": query['original_text'],
            "masking_strategy": query['masking_strategy'],
            "generated_output": output,
            "logprobs": logprobs,
            "retrieved_doc_ids": query['retrieved_doc_ids']
        }
        results.append(result)

    return results


def generate_rag_outputs_for_dataset(
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
    """Generate Mask-based MIA RAG outputs using retrieval results with defenses."""

    print(f"\n{'#'*60}")
    print(f"Generating Mask-based MIA RAG outputs")
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

    # Count by membership
    member_count = sum(1 for r in results if r['membership_label'] == 1)
    nonmember_count = sum(1 for r in results if r['membership_label'] == 0)

    print(f"\n{'='*60}")
    print(f"Generated RAG outputs for {len(results)} queries")
    print(f"  Member queries: {member_count}")
    print(f"  Nonmember queries: {nonmember_count}")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate Mask-based MIA RAG outputs with logprobs and defenses')

    # Basic arguments
    parser.add_argument(
        '--queries_file',
        type=str,
        default='results/MBA/BeIR_nfcorpus/queries/queries.jsonl',
        help='Path to queries file generated by mask_gen.py'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results/MBA/BeIR_nfcorpus/rag/rag_outputs.jsonl',
        help='Path to output RAG outputs file'
    )
    parser.add_argument(
        '--retrieval_file',
        type=str,
        default='results/MBA/BeIR_nfcorpus/retrieval/topk3_results.json',
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
        default='~/.cache/huggingface/hub',
        help='Cache directory for models'
    )
    parser.add_argument(
        '--max_context_docs',
        type=int,
        default=3,
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

    generate_rag_outputs_for_dataset(
        args.queries_file,
        args.output_file,
        args.retrieval_file,
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
    print("Mask-based MIA RAG output generation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
