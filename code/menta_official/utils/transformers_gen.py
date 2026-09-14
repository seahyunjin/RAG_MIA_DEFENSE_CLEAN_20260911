import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
import datetime


# Set distributed timeout to 10 hours before any distributed operations
os.environ["NCCL_TIMEOUT"] = "36000"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
if torch.distributed.is_available():
    torch.distributed.init_process_group_timeout = datetime.timedelta(hours=10)


# Model context window sizes (only your 4 models)
MODEL_CONTEXT_WINDOWS = {
    'llama-3.1-8b-instruct': 131072,  # meta-llama/Llama-3.1-8B-Instruct
    'phi-4': 16384,                   # microsoft/phi-4
    'gemma-2b-it': 8192,              # google/gemma-2b-it
    'command-r7b': 131072,            # CohereLabs/c4ai-command-r7b-12-2024
    'default': 8192                   # Conservative default
}


def get_model_max_length(model_name: str = None, tokenizer=None) -> int:
    """
    Get the maximum context length for a model.
    Prioritizes: 1) Tokenizer config, 2) Model name lookup, 3) Default
    
    Args:
        model_name: Name of the model
        tokenizer: Tokenizer object
    
    Returns:
        Maximum context length
    """
    # Try to get from tokenizer first (most reliable)
    if tokenizer is not None:
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 1e10:
            return tokenizer.model_max_length
    
    # Fall back to model name lookup
    if model_name:
        model_name_lower = model_name.lower()
        for key, value in MODEL_CONTEXT_WINDOWS.items():
            if key in model_name_lower:
                return value
    
    # Last resort: conservative default
    return MODEL_CONTEXT_WINDOWS['default']


def calculate_safe_max_length(
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    model_name: str = None,
    safety_margin: int = 200,
    use_full_context: bool = True
) -> Tuple[int, List[int]]:
    """
    Calculate safe max_length for tokenization.
    
    Args:
        tokenizer: The tokenizer
        prompts: List of prompts to process
        max_new_tokens: Number of tokens to generate
        model_name: Model name for context window lookup
        safety_margin: Extra tokens to leave as buffer
        use_full_context: If True, use model's full context window; if False, use actual prompt length
    
    Returns:
        Tuple of (safe_max_length, list of prompt token counts)
    """
    model_max_length = get_model_max_length(model_name, tokenizer)
    
    # Calculate token counts for all prompts
    prompt_token_counts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_token_counts.append(len(tokens))
    
    max_prompt_tokens = max(prompt_token_counts) if prompt_token_counts else 0
    
    # Calculate maximum allowed input length
    max_allowed_input = model_max_length - max_new_tokens - safety_margin
    
    if use_full_context:
        # Use the model's full context window (up to the limit)
        safe_max_length = max_allowed_input
    else:
        # Use actual prompt length + small buffer
        safe_max_length = min(max_prompt_tokens + safety_margin, max_allowed_input)
    
    # Ensure it's at least reasonable
    safe_max_length = max(safe_max_length, 512)
    
    return safe_max_length, prompt_token_counts


def generate_batch_transformers(
    prompts: List[str],
    tokenizer,
    model,
    accelerator: Accelerator = None,
    temperature: float = 0.4,
    max_new_tokens: int = 128,
    batch_size: int = 8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    model_name: str = None,
    use_full_context: bool = True
) -> List[str]:
    """
    Generate text for a batch of prompts using transformers with Multi-GPU support.
    
    Args:
        prompts: List of prompts to generate from
        tokenizer: The tokenizer
        model: The model
        accelerator: Accelerator instance for distributed processing
        temperature: Sampling temperature
        max_new_tokens: Maximum new tokens to generate
        batch_size: Batch size for processing
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        no_repeat_ngram_size: No repeat n-gram size
        model_name: Model name for context window detection
        use_full_context: If True, use model's full context window (default: True)
    
    Returns:
        List of generated texts
    """
    if accelerator is None:
        accelerator = Accelerator()

    # Calculate safe max_length - now uses full context by default
    safe_max_length, prompt_token_counts = calculate_safe_max_length(
        tokenizer, prompts, max_new_tokens, model_name, use_full_context=use_full_context
    )
    model_max_length = get_model_max_length(model_name, tokenizer)
    
    # Log statistics on main process
    if accelerator.is_main_process:
        max_tokens = max(prompt_token_counts) if prompt_token_counts else 0
        min_tokens = min(prompt_token_counts) if prompt_token_counts else 0
        avg_tokens = sum(prompt_token_counts) / len(prompt_token_counts) if prompt_token_counts else 0
        
        print(f"\n{'='*60}")
        print(f"TOKEN ANALYSIS")
        print(f"{'='*60}")
        print(f"Model: {model_name or 'Unknown'}")
        print(f"Model context window: {model_max_length:,}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Prompt token stats:")
        print(f"  - Min: {min_tokens:,}")
        print(f"  - Max: {max_tokens:,}")
        print(f"  - Avg: {avg_tokens:,.1f}")
        print(f"Max length for tokenization: {safe_max_length:,}")
        print(f"Using full context window: {use_full_context}")
        
        # Warn about truncation
        truncation_count = sum(1 for c in prompt_token_counts if c > safe_max_length)
        if truncation_count > 0:
            print(f"\n⚠ WARNING: {truncation_count}/{len(prompts)} prompts will be truncated!")
            print(f"  Longest prompt: {max_tokens:,} tokens")
            print(f"  Model limit: {safe_max_length:,} tokens")
            print(f"  → {max_tokens - safe_max_length:,} tokens will be cut")
        else:
            print(f"\n✓ All prompts fit within context window")
        print(f"{'='*60}\n")

    # Print distribution information on each process
    print(f"[Process {accelerator.process_index}/{accelerator.num_processes}] "
          f"Device: {accelerator.device} | "
          f"Total prompts: {len(prompts)} | "
          f"Max length: {safe_max_length:,}")

    local_outputs = []
    local_truncation_warnings = []

    # Split the prompts across all available GPUs
    with accelerator.split_between_processes(list(enumerate(prompts))) as indexed_prompts_split:
        
        # Extract indices and prompts
        indices_split = [item[0] for item in indexed_prompts_split]
        prompts_split = [item[1] for item in indexed_prompts_split]
        
        # Print split information
        print(f"[Process {accelerator.process_index}/{accelerator.num_processes}] "
              f"Received {len(prompts_split)} prompts to process")
        
        num_batches = (len(prompts_split) + batch_size - 1) // batch_size
        
        # Only show progress bar on the main process
        disable_tqdm = not accelerator.is_local_main_process
        
        for batch_idx, batch_start in enumerate(tqdm(
            range(0, len(prompts_split), batch_size), 
            total=num_batches, 
            desc=f"GPU {accelerator.process_index}/{accelerator.num_processes}", 
            disable=disable_tqdm
        )):
            
            batch_end = min(batch_start + batch_size, len(prompts_split))
            batch_prompts = prompts_split[batch_start:batch_end]
            batch_indices = indices_split[batch_start:batch_end]

            # Tokenize with dynamically calculated max_length
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=safe_max_length
            ).to(accelerator.device)
            
            # Check for truncation in this batch
            for i, (idx, prompt) in enumerate(zip(batch_indices, batch_prompts)):
                input_len = inputs['input_ids'][i].shape[0]
                original_len = len(tokenizer.encode(prompt, add_special_tokens=True))
                if original_len > safe_max_length:
                    local_truncation_warnings.append({
                        'index': idx,
                        'original_tokens': original_len,
                        'truncated_to': input_len
                    })
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            for i, output in enumerate(outputs):
                input_length = inputs['input_ids'][i].shape[0]
                generated_tokens = output[input_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                local_outputs.append((batch_indices[i], generated_text))

    # Print completion status
    print(f"[Process {accelerator.process_index}/{accelerator.num_processes}] "
          f"Finished processing {len(local_outputs)} prompts. "
          f"Truncation warnings: {len(local_truncation_warnings)}. "
          f"Waiting for other processes...")

    # Gather results from all GPUs
    all_outputs_gathered = gather_object(local_outputs)
    all_truncation_warnings = gather_object(local_truncation_warnings)
    
    # Sort by original index to maintain order
    all_outputs_gathered.sort(key=lambda x: x[0])
    final_outputs = [text for idx, text in all_outputs_gathered]
    
    # Print final gathering status (only on main process)
    if accelerator.is_main_process:
        print(f"\n[Main Process] Gathered {len(all_outputs_gathered)} total outputs from all {accelerator.num_processes} processes")
        
        if all_truncation_warnings:
            print(f"\n⚠ TRUNCATION SUMMARY:")
            print(f"  Total prompts truncated: {len(all_truncation_warnings)}")
            if len(all_truncation_warnings) <= 10:
                for warn in all_truncation_warnings:
                    print(f"  - Index {warn['index']}: {warn['original_tokens']:,} -> {warn['truncated_to']:,} tokens")
            else:
                print(f"  (Showing first 5 and last 5)")
                for warn in all_truncation_warnings[:5]:
                    print(f"  - Index {warn['index']}: {warn['original_tokens']:,} -> {warn['truncated_to']:,} tokens")
                print(f"  ...")
                for warn in all_truncation_warnings[-5:]:
                    print(f"  - Index {warn['index']}: {warn['original_tokens']:,} -> {warn['truncated_to']:,} tokens")
        
        print(f"[Main Process] Returning {len(final_outputs)} outputs")
    
    return final_outputs[:len(prompts)]


def postprocess_refusal(answer: str) -> str:
    """Convert hedging/uncertain responses to 'I don't know'."""
    if not answer or not answer.strip():
        return "I don't know"
    
    answer_stripped = answer.strip()
    lower = answer_stripped.lower()
    
    hedges = [
        "i am not sure", "i'm not sure", "not sure",
        "i think", "i believe", "possibly", "might be", 
        "could be", "maybe", "perhaps",
        "no context", "context does not", "insufficient",
        "cannot determine", "can't determine",
        "not enough information", "doesn't say",
        "unclear", "uncertain"
    ]
    
    if lower == "i don't know" or lower == "idk":
        return "I don't know"
    
    if lower.startswith("i don't know") or lower.startswith("idk"):
        return "I don't know"
    
    if any(hedge in lower for hedge in hedges):
        return "I don't know"
    
    if len(answer_stripped) < 10 and any(word in lower for word in ["not", "no", "none"]):
        return "I don't know"
    
    return answer_stripped


def create_chat_prompt(
    tokenizer,
    messages: List[Dict[str, str]],
    model_name: str = None
) -> str:
    """Create a chat prompt from messages."""
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            if 'system' in str(e).lower() or 'role not supported' in str(e).lower():
                if messages and messages[0].get('role') == 'system':
                    system_content = messages[0]['content']
                    modified_messages = messages[1:]
                    for i, msg in enumerate(modified_messages):
                        if msg.get('role') == 'user':
                            modified_messages[i]['content'] = f"{system_content}\n\n{msg['content']}"
                            break
                    try:
                        return tokenizer.apply_chat_template(
                            modified_messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    except Exception:
                        messages = modified_messages
    
    # Manual fallback
    prompt_parts = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            prompt_parts.append(f"System: {content}\n")
        elif role == 'user':
            prompt_parts.append(f"User: {content}\n")
        elif role == 'assistant':
            prompt_parts.append(f"Assistant: {content}\n")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def generate_with_context(
    queries: List[str],
    context_docs_list: List[List[dict]],
    tokenizer,
    model,
    accelerator: Accelerator = None,
    max_new_tokens: int = 128,
    max_context_docs: int = 5,
    batch_size: int = 8,
    system_prompt: str = None,
    temperature: float = 0.4,
    top_p: float = 0.9,
    top_k: int = 50,
    model_name: str = None
) -> List[str]:
    """
    Generate answers using context documents, distributed across GPUs.
    Now uses full model context window by default (no manual truncation needed).
    
    Args:
        queries: List of query strings
        context_docs_list: List of lists of context documents
        tokenizer: The tokenizer
        model: The model
        accelerator: Accelerator for distributed processing
        max_new_tokens: Maximum tokens to generate
        max_context_docs: Maximum context documents to include
        batch_size: Batch size for processing
        system_prompt: System prompt
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        model_name: Model name for context window detection
    
    Returns:
        List of generated answers
    """
    if accelerator is None:
        accelerator = Accelerator()
    
    # Prepare all prompts (simplified - no manual truncation)
    prompts = []
    
    for query, context_docs in zip(queries, context_docs_list):
        # Format all requested documents
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
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        })
        
        prompt = create_chat_prompt(tokenizer, messages, model_name)
        prompts.append(prompt)
    
    # Generate using the distributed function with full context window
    generated_answers = generate_batch_transformers(
        prompts,
        tokenizer,
        model,
        accelerator=accelerator,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        model_name=model_name,
        use_full_context=True  # KEY: Uses model's full context window
    )
    
    return generated_answers
