import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set
import sys

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


def create_paraphrase_prompt(text: str) -> str:
    """Create prompt for paraphrasing a document."""
    return f"""Task Description:
You are tasked with paraphrasing a document while maintaining its meaning, technical accuracy, and all domain-specific terminology.

Requirements:
1. Rewrite the text using different sentence structures and word choices while preserving the original meaning.
2. Keep ALL technical terms, specialized vocabulary, proper nouns, numerical values, and domain-specific keywords EXACTLY as they appear in the original text. Do NOT paraphrase or replace these terms.
3. Only paraphrase common connecting words, general phrases, and sentence structures.
4. Maintain the same level of detail and technical accuracy as the original.
5. Ensure the paraphrased text flows naturally and reads coherently.
6. Do NOT add information that wasn't in the original text.
7. Do NOT summarize or shorten the content - maintain similar length to the original.

Examples of what to preserve:
- Medical/scientific terms (e.g., "cyclooxygenase pathway", "inflammatory responses", "mitochondrial dysfunction")
- Specific measurements, percentages, and numerical data
- Proper nouns (names of diseases, medications, proteins, genes, locations)
- Abbreviations and acronyms
- Technical methodology terms

Input Text:
{text}

Output:
Provide only the paraphrased text without any additional explanations, labels, or commentary.
"""


def load_and_merge_corpus(member_path: str, nonmember_path: str) -> List[dict]:
    """Load and merge member and nonmember corpus files."""
    corpus = []
    
    if os.path.exists(member_path):
        members = read_jsonl(member_path)
        for doc in members:
            doc['_membership'] = 'member'
        corpus.extend(members)
        print(f"Loaded {len(members)} member documents")
    else:
        print(f"Warning: Member file not found: {member_path}")
    
    if os.path.exists(nonmember_path):
        nonmembers = read_jsonl(nonmember_path)
        for doc in nonmembers:
            doc['_membership'] = 'nonmember'
        corpus.extend(nonmembers)
        print(f"Loaded {len(nonmembers)} non-member documents")
    else:
        print(f"Warning: Non-member file not found: {nonmember_path}")
    
    print(f"Total corpus size: {len(corpus)}")
    return corpus


def sync_output_file(output_path: Path, valid_ids: Set[str], output_key: str) -> Dict[str, dict]:
    """
    Sync output file with valid IDs.
    Remove entries not in valid_ids, return existing entries.
    
    Args:
        output_path: Path to output file
        valid_ids: Set of valid document IDs
        output_key: Key for the generated content (e.g., 'summary', 'paraphrase')
    
    Returns:
        Dictionary mapping doc_id to existing record
    """
    existing_records = {}
    
    if not output_path.exists():
        return existing_records
    
    print(f"Loading existing output file: {output_path}")
    all_records = read_jsonl(str(output_path))
    
    # Filter to keep only valid IDs
    valid_records = []
    removed_count = 0
    
    for record in all_records:
        doc_id = record.get('_id')
        if doc_id in valid_ids:
            existing_records[doc_id] = record
            valid_records.append(record)
        else:
            removed_count += 1
    
    if removed_count > 0:
        print(f"Removed {removed_count} records not in current corpus")
        # Rewrite file with only valid records
        write_jsonl(valid_records, str(output_path))
        print(f"Updated output file with {len(valid_records)} valid records")
    
    print(f"Found {len(existing_records)} existing {output_key} records")
    return existing_records


def get_missing_docs(corpus: List[dict], existing_records: Dict[str, dict], output_key: str) -> List[dict]:
    """Get documents that need processing (not in existing records or missing output_key)."""
    missing = []
    
    for doc in corpus:
        doc_id = doc.get('_id')
        if doc_id not in existing_records:
            missing.append(doc)
        elif output_key not in existing_records[doc_id]:
            missing.append(doc)
    
    return missing


def paraphrase_corpus_openai(
    corpus: List[dict],
    model: str,
    temperature: float,
    output_path: Path,
    env_path: str,
    max_tokens_per_batch: int,
    check_interval: int
):
    """Paraphrase corpus using OpenAI Batch API."""
    client = initialize_openai_client(env_path)
    
    # Prepare document items
    doc_items = [(doc['_id'], doc) for doc in corpus]
    
    # Estimate tokens function
    def estimate_item_tokens(item):
        doc_id, doc = item
        text = ""
        if 'title' in doc and doc['title']:
            text += doc['title'] + ". "
        if 'text' in doc and doc['text']:
            text += doc['text']
        
        prompt = create_paraphrase_prompt(text.strip())
        return estimate_tokens_accurate(prompt, 1, model)
    
    # Split into batches
    batches = split_into_batches(doc_items, estimate_item_tokens, max_tokens_per_batch)
    
    # Create request function
    def create_request(item):
        doc_id, doc = item
        text = ""
        if 'title' in doc and doc['title']:
            text += doc['title'] + ". "
        if 'text' in doc and doc['text']:
            text += doc['text']
        
        prompt = create_paraphrase_prompt(text.strip())
        
        messages = [
            {"role": "system", "content": "You are an expert paraphrasing assistant that rewrites technical documents while preserving all specialized terminology and technical accuracy."},
            {"role": "user", "content": prompt}
        ]
        
        estimated_output_tokens = int(len(text.split()) * 2)
        
        return create_batch_request(
            custom_id=doc_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=min(estimated_output_tokens, 4000)
        )
    
    # Result processor function
    def process_result(custom_id, generated_text):
        doc_id = custom_id
        original_doc = next((doc for doc_id_check, doc in doc_items if doc_id_check == doc_id), None)
        
        if not original_doc:
            return None, [{'_id': doc_id, 'reason': 'Original document not found'}]
        
        if not generated_text or not generated_text.strip():
            return None, [{'_id': doc_id, 'reason': 'Empty paraphrase generated'}]
        
        paraphrase_record = {
            '_id': doc_id,
            'title': original_doc.get('title', ''),
            'text': original_doc.get('text', ''),
            'paraphrase': generated_text.strip(),
            '_membership': original_doc.get('_membership', '')
        }
        
        return paraphrase_record, None
    
    # Run batch generation
    all_paraphrases, all_failures = run_batch_generation(
        client,
        batches,
        create_request,
        process_result,
        output_path.parent,
        output_path.stem,
        check_interval,
        metadata_base={
            "description": "Document paraphrasing",
            "temperature": str(temperature)
        }
    )
    
    return all_paraphrases, all_failures


def paraphrase_corpus_transformers(
    corpus: List[dict],
    tokenizer,
    model,
    temperature: float,
    batch_size: int
):
    """Paraphrase corpus using transformers."""
    all_paraphrases = []
    all_failures = []
    
    prompts = []
    metadata_list = []
    
    for doc in corpus:
        text = ""
        if 'title' in doc and doc['title']:
            text += doc['title'] + ". "
        if 'text' in doc and doc['text']:
            text += doc['text']
        
        text = text.strip()
        
        if not text:
            all_failures.append({'_id': doc['_id'], 'reason': 'Empty document text'})
            continue
        
        user_prompt = create_paraphrase_prompt(text)
        
        messages = [
            {"role": "system", "content": "You are an expert paraphrasing assistant that rewrites technical documents while preserving all specialized terminology and technical accuracy."},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = create_chat_prompt(tokenizer, messages)
        prompts.append(prompt)
        metadata_list.append({
            'doc_id': doc['_id'],
            'title': doc.get('title', ''),
            'text': doc.get('text', ''),
            '_membership': doc.get('_membership', '')
        })
    
    # Generate paraphrases
    generated_texts = generate_batch_transformers(
        prompts,
        tokenizer,
        model,
        temperature=temperature,
        max_new_tokens=1024,
        batch_size=batch_size
    )
    
    # Process results
    for generated_text, metadata in zip(generated_texts, metadata_list):
        if not generated_text or not generated_text.strip():
            all_failures.append({'_id': metadata['doc_id'], 'reason': 'Empty paraphrase generated'})
            continue
        
        paraphrase_record = {
            '_id': metadata['doc_id'],
            'title': metadata['title'],
            'text': metadata['text'],
            'paraphrase': generated_text.strip(),
            '_membership': metadata['_membership']
        }
        all_paraphrases.append(paraphrase_record)
    
    return all_paraphrases, all_failures


def main():
    parser = argparse.ArgumentParser(description='Paraphrase documents in corpus')
    parser.add_argument(
        '--member_path',
        type=str,
        default='data/BeIR_trec-covid/corpus_member.jsonl',
        help='Path to member corpus JSONL file'
    )
    parser.add_argument(
        '--nonmember_path',
        type=str,
        default='data/BeIR_trec-covid/corpus_nonmember.jsonl',
        help='Path to non-member corpus JSONL file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/BeIR_trec-covid/paraphrase.jsonl',
        help='Path to output paraphrased JSONL file'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['openai', 'transformers'],
        default='openai',
        help='Paraphrasing method'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="gpt-4.1-nano",
        help='Model name (e.g., gpt-4o-mini, meta-llama/Llama-3.1-8B-Instruct)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='~/.cache/huggingface/hub',
        help='Cache directory for models'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU for transformers'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Temperature for generation'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for transformers'
    )
    parser.add_argument(
        '--env_path',
        type=str,
        default='.env',
        help='Path to .env file'
    )
    parser.add_argument(
        '--max_tokens_per_batch',
        type=int,
        default=2_000_000,
        help='Maximum tokens per batch for OpenAI'
    )
    parser.add_argument(
        '--check_interval',
        type=int,
        default=120,
        help='Interval to check batch status (seconds)'
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"Paraphrasing corpus")
    print(f"Member path: {args.member_path}")
    print(f"Non-member path: {args.nonmember_path}")
    print(f"Output: {output_path}")
    print(f"Method: {args.method}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"{'#'*60}\n")
    
    # Load and merge corpus
    corpus = load_and_merge_corpus(args.member_path, args.nonmember_path)
    
    if not corpus:
        print("Error: No documents loaded. Exiting.")
        return
    
    # Get valid IDs
    valid_ids = {doc['_id'] for doc in corpus}
    
    # Sync output file and get existing records
    existing_records = sync_output_file(output_path, valid_ids, 'paraphrase')
    
    # Get missing documents
    missing_docs = get_missing_docs(corpus, existing_records, 'paraphrase')
    
    print(f"Documents to process: {len(missing_docs)}")
    
    if not missing_docs:
        print("All documents already have paraphrases. Nothing to do.")
        return
    
    # Generate paraphrases for missing documents
    if args.method == 'openai':
        new_paraphrases, all_failures = paraphrase_corpus_openai(
            missing_docs,
            args.model,
            args.temperature,
            output_path,
            args.env_path,
            args.max_tokens_per_batch,
            args.check_interval
        )
    else:
        if args.cache_dir:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        tokenizer, model = load_generator_model(
            args.model,
            args.use_gpu,
            args.cache_dir
        )
        
        new_paraphrases, all_failures = paraphrase_corpus_transformers(
            missing_docs,
            tokenizer,
            model,
            args.temperature,
            args.batch_size
        )
    
    # Append new paraphrases to output file
    if new_paraphrases:
        if output_path.exists():
            all_records = read_jsonl(str(output_path))
        else:
            all_records = []
        
        all_records.extend(new_paraphrases)
        write_jsonl(all_records, str(output_path))
        print(f"\n✓ Added {len(new_paraphrases)} new paraphrases to: {output_path}")
        print(f"✓ Total records in file: {len(all_records)}")
    
    if all_failures:
        failure_report_path = output_path.parent / f"{output_path.stem}_failures.json"
        write_json({
            'failures': all_failures,
            'total_failed': len(all_failures)
        }, str(failure_report_path))
        print(f"⚠ Saved {len(all_failures)} failures to: {failure_report_path}")
    
    # Report statistics
    total_docs = len(corpus)
    total_existing = len(existing_records)
    total_new = len(new_paraphrases) if new_paraphrases else 0
    total_complete = total_existing + total_new
    success_rate = total_complete / total_docs * 100 if total_docs > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Paraphrase Statistics:")
    print(f"  Total documents: {total_docs}")
    print(f"  Already processed: {total_existing}")
    print(f"  Newly processed: {total_new}")
    print(f"  Total complete: {total_complete}/{total_docs}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()