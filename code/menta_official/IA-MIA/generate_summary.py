import os
import sys
import json
from pathlib import Path
from typing import List
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.openai_gen import (
    create_batch_request,
    estimate_tokens_accurate,
    initialize_openai_client,
    run_batch_generation,
    split_into_batches,
)


SUMMARY_PROMPT = """Task Description:
You are tasked with generating a concise and accurate topic-focused description of a document based on its content and title (if provided). The description should:
1. Be a single, short sentence.
2. Focus only on the main topic or subject of the document, avoiding verbs and conclusions.
3. Include important keywords from the document.
4. Avoid referencing the document itself with phrases like "The document discusses," "The report highlights," or "This paper investigates."
5. Output only a short, noun-phrase-like description or topic sentence.

Examples:
- Instead of: "The report from the Düsseldorf conference highlights advancements in green energy technologies."
- Generate: "Advancements in green energy technologies and discussions at the Düsseldorf conference."
- Instead of: "The document investigates the cyclooxygenase pathway in inflammatory responses."
- Generate: "The cyclooxygenase pathway and its role in inflammatory responses."

Ensure the description is concise, focused on the main topic, and includes relevant keywords. Avoid any extra text, explanations, or labels.

Input:
Title: {title}
Text: {text}

Output:
Provide only the one-sentence topic-focused description as the output.
"""


def load_generator_model(model_name: str, use_gpu: bool, cache_dir: str):
    """Load transformers model for generation."""
    print(f"Loading generator model: {model_name}")
    
    model_loaded = False
    if cache_dir:
        model_name_safe = model_name.replace('/', '--')
        model_cache_path = os.path.join(cache_dir, f"models--{model_name_safe}")
        
        if os.path.exists(model_cache_path):
            if os.path.exists(os.path.join(model_cache_path, 'config.json')):
                print(f"Loading from HF cache (direct): {model_cache_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_cache_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_cache_path,
                    torch_dtype=torch.float16 if use_gpu else torch.float32,
                    device_map='auto' if use_gpu else None
                )
                model_loaded = True
            else:
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshot_dirs:
                        model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                        print(f"Loading from HF cache (snapshot): {model_path}")
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if use_gpu else torch.float32,
                            device_map='auto' if use_gpu else None
                        )
                        model_loaded = True
        
        if not model_loaded:
            raise FileNotFoundError(f"Model '{model_name}' not found in cache directory: {cache_dir}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_gpu else torch.float32,
            device_map='auto' if use_gpu else None
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    tokenizer.padding_side = 'left'
    model.eval()
    
    return tokenizer, model


def read_jsonl(file_path: str) -> List[dict]:
    """Read JSONL file and return list of records."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def write_jsonl(data: List[dict], file_path: str):
    """Write list of records to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_openai_summaries(
    client,
    prompts: List[str],
    model_name: str,
    output_dir: Path,
    base_filename: str,
    offset: int = 0,
    temperature: float = 0.2,
) -> List[str]:
    """Generate summaries through the shared OpenAI batch/single switch."""
    prompt_items = [(offset + idx, prompt) for idx, prompt in enumerate(prompts)]

    def estimate_tokens(item):
        _, prompt = item
        return estimate_tokens_accurate(prompt, 1, model_name)

    batches = split_into_batches(prompt_items, estimate_tokens)

    def create_request(item):
        item_idx, prompt = item
        return create_batch_request(
            custom_id=f"summary_{item_idx}",
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200,
        )

    def process_result(custom_id, generated_text):
        try:
            item_idx = int(custom_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return None, {"custom_id": custom_id, "reason": "Invalid summary custom_id"}
        return {"index": item_idx, "summary": generated_text.strip()}, None

    records, failures = run_batch_generation(
        client,
        batches,
        create_request,
        process_result,
        output_dir,
        base_filename,
        metadata_base={"type": "ia_mia_summary"},
    )

    if failures:
        print(f"Warning: {len(failures)} OpenAI summary request(s) failed")

    summaries_by_index = {record["index"]: record["summary"] for record in records}
    return [summaries_by_index.get(offset + idx, "") for idx in range(len(prompts))]


def generate_batch_transformers(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.2,
    batch_size: int = 4
) -> List[str]:
    """Generate text in batches using transformers model."""
    all_outputs = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Format prompts with chat template if available
        formatted_prompts = []
        for prompt in batch_prompts:
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = prompt
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize batch with padding
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        
        if model.device.type == 'cuda':
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode each output
        for j, output in enumerate(outputs):
            input_length = (inputs['attention_mask'][j] == 1).sum().item()
            generated_tokens = output[input_length:]
            answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            all_outputs.append(answer)
    
    return all_outputs


def add_summaries(
    input_path: str,
    output_path: str,
    generation_method: str,
    openai_client=None,
    openai_model: str = "gpt-4.1-nano",
    tokenizer=None,
    model=None,
    batch_size: int = 4
):
    """Add summaries to existing JSONL file."""
    print(f"\n{'#'*60}")
    print(f"Adding summaries to: {input_path}")
    print(f"Method: {generation_method}")
    if generation_method == 'transformers':
        print(f"Batch size: {batch_size}")
    print(f"Output: {output_path}")
    print(f"{'#'*60}\n")
    
    # Load existing data
    data = read_jsonl(input_path)
    print(f"Loaded {len(data)} documents")
    
    # Check if summaries already exist
    has_summaries = all('summary' in doc for doc in data)
    if has_summaries:
        print("All documents already have summaries!")
        response = input("Do you want to regenerate summaries? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Process in batches
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(data), batch_size), total=num_batches, desc="Generating summaries"):
        batch_docs = data[i:i + batch_size]
        
        try:
            # Generate prompts
            prompts = []
            for doc in batch_docs:
                title = doc.get("title", "")
                text = doc.get("text", "")
                prompt = SUMMARY_PROMPT.format(title=title, text=text)
                prompts.append(prompt)
            
            # Generate summaries
            if generation_method == 'openai':
                summaries = generate_openai_summaries(
                    openai_client,
                    prompts,
                    openai_model,
                    Path(output_path).parent / "openai_batches",
                    f"{Path(output_path).stem}_summary_{i}",
                    offset=i,
                    temperature=0.2,
                )
            else:
                summaries = generate_batch_transformers(tokenizer, model, prompts, max_new_tokens=100, temperature=0.2, batch_size=batch_size)
            
            # Add summaries to documents
            for doc, summary in zip(batch_docs, summaries):
                doc['summary'] = summary
        
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            continue
    
    # Write results
    write_jsonl(data, output_path)
    print(f"\n{'='*60}")
    print(f"Added summaries to {len(data)} documents")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Add summaries to existing query file')
    parser.add_argument(
        '--input',
        type=str,
        default="data/BeIR_nfcorpus/ia_mia/corpus_30_questions.jsonl",
        help='Path to input JSONL file (without summaries)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output JSONL file (with summaries). If not specified, will overwrite input file.'
    )
    parser.add_argument(
        '--generation_method',
        type=str,
        choices=['openai', 'transformers'],
        default='transformers',
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
        help='Batch size for transformers generation'
    )
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        args.output = args.input
        print(f"\nWarning: No output path specified. Will overwrite input file: {args.input}")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Initialize generation backend
    openai_client = None
    tokenizer = None
    model = None
    
    if args.generation_method == 'openai':
        openai_client = initialize_openai_client('.env')
        print(f"\nUsing OpenAI: {args.openai_model}")
    else:
        if args.cache_dir:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        tokenizer, model = load_generator_model(args.transformers_model, args.use_gpu, args.cache_dir)
        print(f"\nUsing Transformers: {args.transformers_model}")
        print(f"Batch size: {args.batch_size}")
    
    # Add summaries
    add_summaries(
        args.input,
        args.output,
        args.generation_method,
        openai_client=openai_client,
        openai_model=args.openai_model,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("Summary generation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()