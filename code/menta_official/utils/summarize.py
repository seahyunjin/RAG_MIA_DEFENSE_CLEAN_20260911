import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set
import sys

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


def create_summary_prompt(text: str) -> str:
    """Create prompt for summarizing a document."""
    return f"""Task Description:
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
Text: {text}

Output:
Provide only the one-sentence topic-focused description as the output.
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


def summarize_corpus_openai(
    corpus: List[dict],
    model: str,
    temperature: float,
    output_path: Path,
    env_path: str,
    max_tokens_per_batch: int,
    check_interval: int
):
    """Summarize corpus using OpenAI Batch API."""
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
        
        prompt = create_summary_prompt(text.strip())
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
        
        prompt = create_summary_prompt(text.strip())
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries of scientific and medical documents."},
            {"role": "user", "content": prompt}
        ]
        
        return create_batch_request(
            custom_id=doc_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=200
        )
    
    # Result processor function
    def process_result(custom_id, generated_text):
        doc_id = custom_id
        original_doc = next((doc for doc_id_check, doc in doc_items if doc_id_check == doc_id), None)
        
        if not original_doc:
            return None, [{'_id': doc_id, 'reason': 'Original document not found'}]
        
        if not generated_text or not generated_text.strip():
            return None, [{'_id': doc_id, 'reason': 'Empty summary generated'}]
        
        summary_record = {
            '_id': doc_id,
            'title': original_doc.get('title', ''),
            'text': original_doc.get('text', ''),
            'summary': generated_text.strip(),
            '_membership': original_doc.get('_membership', '')
        }
        
        return summary_record, None

    def save_chunk_summaries(successful, failed, batch_num):
        if successful:
            merge_jsonl_by_key(str(output_path), successful, key='_id')
            print(
                f"  [checkpoint] Saved {len(successful)} summaries after batch "
                f"{batch_num} -> {output_path}"
            )

    # Run batch generation
    all_summaries, all_failures = run_batch_generation(
        client,
        batches,
        create_request,
        process_result,
        output_path.parent,
        output_path.stem,
        check_interval,
        metadata_base={
            "description": "Document summarization",
            "temperature": str(temperature)
        },
        on_chunk_complete=save_chunk_summaries,
        resume=True,
    )
    
    return all_summaries, all_failures


def summarize_corpus_transformers(
    corpus: List[dict],
    tokenizer,
    model,
    temperature: float,
    batch_size: int
):
    """Summarize corpus using transformers."""
    all_summaries = []
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
        
        user_prompt = create_summary_prompt(text)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries of scientific and medical documents."},
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
    
    # Generate summaries
    generated_texts = generate_batch_transformers(
        prompts,
        tokenizer,
        model,
        temperature=temperature,
        max_new_tokens=200,
        batch_size=batch_size
    )
    
    # Process results
    for generated_text, metadata in zip(generated_texts, metadata_list):
        if not generated_text or not generated_text.strip():
            all_failures.append({'_id': metadata['doc_id'], 'reason': 'Empty summary generated'})
            continue
        
        summary_record = {
            '_id': metadata['doc_id'],
            'title': metadata['title'],
            'text': metadata['text'],
            'summary': generated_text.strip(),
            '_membership': metadata['_membership']
        }
        all_summaries.append(summary_record)
    
    return all_summaries, all_failures


def main():
    parser = argparse.ArgumentParser(description='Summarize documents in corpus')
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
        default='data/BeIR_trec-covid/summary.jsonl',
        help='Path to output summary JSONL file'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['openai', 'transformers'],
        default='openai',
        help='Summarization method'
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
    print(f"Summarizing corpus")
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
    existing_records = sync_output_file(output_path, valid_ids, 'summary')
    
    # Get missing documents
    missing_docs = get_missing_docs(corpus, existing_records, 'summary')
    
    print(f"Documents to process: {len(missing_docs)}")
    
    if not missing_docs:
        print("All documents already have summaries. Nothing to do.")
        return
    
    # Generate summaries for missing documents
    if args.method == 'openai':
        new_summaries, all_failures = summarize_corpus_openai(
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
        
        new_summaries, all_failures = summarize_corpus_transformers(
            missing_docs,
            tokenizer,
            model,
            args.temperature,
            args.batch_size
        )
    
    # Append new summaries to output file (idempotent if incremental saves already ran)
    if new_summaries:
        # Load existing records and append new ones
        if output_path.exists():
            all_records = read_jsonl(str(output_path))
        else:
            all_records = []
        
        all_records.extend(new_summaries)
        write_jsonl(all_records, str(output_path))
        print(f"\n✓ Added {len(new_summaries)} new summaries to: {output_path}")
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
    total_new = len(new_summaries) if new_summaries else 0
    total_complete = total_existing + total_new
    success_rate = total_complete / total_docs * 100 if total_docs > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics:")
    print(f"  Total documents: {total_docs}")
    print(f"  Already processed: {total_existing}")
    print(f"  Newly processed: {total_new}")
    print(f"  Total complete: {total_complete}/{total_docs}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()