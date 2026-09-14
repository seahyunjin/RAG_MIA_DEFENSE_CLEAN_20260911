import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set
import sys
import re

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_json, read_jsonl, write_json, write_jsonl, merge_query_variation_records
from utils.openai_gen import (
    initialize_openai_client,
    split_into_batches,
    run_batch_generation,
    create_batch_request,
    estimate_tokens_accurate
)
from utils.transformers_gen import generate_batch_transformers, create_chat_prompt
from utils.load_model import load_generator_model


GENERIC_SPECIFICITY_PATTERNS = [
    r'\baccording to\b',
    r'\breported\b',
    r'\bmeasured\b',
    r'\bspecific\b',
    r'\bexact\b',
    r'\bprecise\b',
    r'\bhow many\b',
    r'\bwhat percentage\b',
    r'\bwhat proportion\b',
    r'\bparticipants?\b',
    r'\bcohort\b',
    r'\btrial\b',
    r'\bbetween\s+\d',
    r'\b\d{4}\b',
]


def clean_generated_query(query: str) -> str:
    """Clean up generated query text."""
    lines = [line.strip() for line in query.split('\n') if line.strip()]
    
    meta_phrases = [
        "given the documents", "based on the documents", "from the documents",
        "according to the documents", "the provided documents", "these documents",
        "document context", "based on document", "from document", "(document ",
        "(based on ", "**document", "here is a question:", "here is the question:",
        "question query:", "question:", "query:", "summary and question:", "summary:",
    ]
    
    cleaned_lines = []
    question_found = False
    
    for line in lines:
        line = re.sub(r'^\d+\.\s*\*?\*?\s*', '', line)
        line = line.replace('**', '')
        line = re.sub(r'^[-*]\s+', '', line)
        
        line_lower = line.lower()
        has_meta_reference = any(phrase in line_lower for phrase in meta_phrases)
        
        if has_meta_reference and len(line) < 100:
            continue
        
        if has_meta_reference:
            line = re.sub(r'\s*\([Bb]ased on [Dd]ocument\s+\d+\)', '', line)
            line = re.sub(r'\s*\([Dd]ocument\s+\d+\)', '', line)
        
        if line and len(line) > 20:
            is_question = '?' in line
            
            if is_question and not question_found:
                cleaned_lines.append(line)
                question_found = True
            elif not question_found:
                cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines) if cleaned_lines else None


def is_valid_generic_query(query: str) -> bool:
    """Return True if a generic query looks broad enough to answer without the target document."""
    if not query:
        return False

    normalized = ' '.join(query.split()).strip()
    lowered = normalized.lower()

    if any(re.search(pattern, lowered) for pattern in GENERIC_SPECIFICITY_PATTERNS):
        return False

    return True


def create_generation_prompt_multi_queries(
    target_document: str,
    num_queries: int = 3,
    query_mode: str = 'specific'
) -> str:
    """Create prompt for generating multiple unique queries for a single document."""
    if query_mode == 'generic':
        return f"""Given the document below, generate {num_queries} EASY topical questions that are clearly relevant to the document's subject area but do NOT require access to this specific document to answer.

DOCUMENT:   
{target_document}
    
Requirements:
- Generate EXACTLY {num_queries} different questions
- The questions must stay on the same overall topic as the document
- The questions must be easy enough that a normal person could answer them or look up the answer quickly
- The questions should sound like simple everyday curiosity questions, not expert or technical questions
- Use plain language and common words whenever possible
- Ask about broad, familiar ideas such as causes, effects, symptoms, prevention, healthy habits, common ingredients, common examples, or what something means
- Avoid technical mechanisms, niche terminology, exact numbers, named cohorts, precise dates, study-specific methods, uncommon formulations, or other details that depend on this exact document
- If the document mentions obscure entities, specialized chemicals, or uncommon terms, replace them with a simpler broader topic
- Prefer questions similar in difficulty to: "What causes a healthy lifestyle?" or "How much spices are in a meal?"
- Make the questions useful, natural, and genuinely easy to understand
- Ensure the set of questions covers different topical aspects instead of repeating the same idea
- DO NOT mention "the document", "the text", "this passage", "the study", or similar references
- DO NOT add meta-preambles like "Here is a question:", "Question:", or "Query:"
- If the text uses abbreviations or acronyms that are standard for the topic, you may use them when natural
- Before finalizing each question, ask: "Could a normal person answer this or easily look it up without the target document?" If not, simplify it

Output format (IMPORTANT - follow this EXACTLY):
QUERY_1: [first question here]
QUERY_2: [second question here]
QUERY_3: [third question here]
...

Generate {num_queries} queries now:"""

    return f"""Given the document below, generate {num_queries} highly specific questions that can be answered by this document.

DOCUMENT:
{target_document}

Requirements:
- Generate EXACTLY {num_queries} different questions
- **CRITICAL: The set of questions must cover ALL different aspects/sections of the document.**
- **DISTRIBUTION: Do not focus all questions on a single fact. If the text has a beginning, middle, and end, or multiple distinct points, ensure the {num_queries} questions are distributed across these different parts.**
- Each question should require specific information from the document
- Focus on unique details, specific facts, or specific combinations of information
- Make each query DIFFERENT by using different phrasing, focusing on different aspects, varying question structure
- DO NOT mention "the document", "the text", "this passage", or similar references
- DO NOT add meta-preambles like "Here is a question:", "Question:", or "Query:"
- If the text uses any abbreviations or acronyms, use the same forms in your questions
- Avoid mentioning 'the study' or any references to the passage itself

Output format (IMPORTANT - follow this EXACTLY):
QUERY_1: [first question here]
QUERY_2: [second question here]
QUERY_3: [third question here]
...

Generate {num_queries} queries now:"""


def create_query_generation_system_prompt(num_queries: int, query_mode: str) -> str:
    """Create system prompt for query generation."""
    if query_mode == 'generic':
        return (
            "You generate easy, plain-language topical questions that are relevant to the document's subject, "
            "but must remain answerable without access to the specific document itself. "
            "The questions should feel simple enough for a normal person, not an expert. "
            "Prefer broad everyday background questions over technical or specialized ones. "
            "Do not ask for document-specific facts, exact measurements, named cohorts, technical mechanisms, or rare paper-specific entities. "
            f"Always output exactly {num_queries} questions in the format: "
            "QUERY_1: [question], QUERY_2: [question], etc."
        )

    return (
        "You are a helpful assistant that generates highly specific questions. "
        f"Always output exactly {num_queries} questions in the format: "
        "QUERY_1: [question], QUERY_2: [question], etc."
    )


def parse_multi_query_response(response_text: str, num_queries: int, query_mode: str = 'specific') -> List[str]:
    """Parse multiple queries from LLM response."""
    queries = []

    def maybe_keep(cleaned_query: str):
        if not cleaned_query:
            return
        if query_mode == 'generic' and not is_valid_generic_query(cleaned_query):
            return
        queries.append(cleaned_query)
    
    # Try QUERY_N: format
    pattern = r'QUERY_\d+:\s*(.+?)(?=QUERY_\d+:|$)'
    matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        for match in matches:
            query = match.strip()
            cleaned = clean_generated_query(query)
            maybe_keep(cleaned)
    
    # Fallback: numbered list
    if len(queries) < num_queries:
        queries = []
        lines = response_text.split('\n')
        current_query = []
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+[\.\)]\s+', line):
                if current_query:
                    query_text = ' '.join(current_query)
                    cleaned = clean_generated_query(query_text)
                    maybe_keep(cleaned)
                    current_query = []
                line = re.sub(r'^\d+[\.\)]\s+', '', line)
                current_query.append(line)
            elif current_query and line:
                current_query.append(line)
        
        if current_query:
            query_text = ' '.join(current_query)
            cleaned = clean_generated_query(query_text)
            maybe_keep(cleaned)
    
    # Fallback: blank lines
    if len(queries) < num_queries:
        queries = []
        paragraphs = response_text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:
                cleaned = clean_generated_query(para)
                maybe_keep(cleaned)
    
    return queries[:num_queries]


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


def limit_corpus_for_test(corpus: List[dict], docs_per_split: int) -> List[dict]:
    """Keep a tiny balanced member/nonmember sample for smoke tests."""
    limit = max(1, docs_per_split)
    members = [doc for doc in corpus if doc.get("_membership") == "member"][:limit]
    nonmembers = [doc for doc in corpus if doc.get("_membership") == "nonmember"][:limit]
    limited = members + nonmembers
    print(f"\nTEST MODE: Processing {len(limited)} documents ({len(members)} member, {len(nonmembers)} nonmember)")
    return limited


def sync_output_file(output_path: Path, valid_ids: Set[str], num_queries: int) -> Dict[str, dict]:
    """
    Sync output file with valid IDs.
    Remove entries not in valid_ids, return existing COMPLETE records only.
    Any document with incomplete queries will be excluded from existing_records.
    """
    existing_records = {}
    
    if not output_path.exists():
        return existing_records
    
    print(f"Loading existing output file: {output_path}")
    all_records = read_jsonl(str(output_path))
    
    # Group by target_doc_id
    doc_groups = {}
    for record in all_records:
        target_doc_id = record.get('target_doc_id', '')
        if target_doc_id not in doc_groups:
            doc_groups[target_doc_id] = []
        doc_groups[target_doc_id].append(record)
    
    # Filter: keep only valid IDs with COMPLETE query sets
    valid_records = []
    removed_count = 0
    incomplete_docs = []
    
    for doc_id, queries in doc_groups.items():
        if doc_id not in valid_ids:
            # Document no longer in corpus
            removed_count += len(queries)
            continue
        
        # Check if we have exactly num_queries variations
        variation_indices = {q.get('variation_index') for q in queries}
        expected_indices = set(range(num_queries))
        
        if variation_indices == expected_indices and len(queries) == num_queries:
            # Complete set - keep it
            existing_records[doc_id] = queries
            valid_records.extend(queries)
        else:
            # Incomplete set - discard all variations for this document
            incomplete_docs.append(doc_id)
            removed_count += len(queries)
            print(f"  ⚠ Document {doc_id}: incomplete query set ({len(queries)}/{num_queries}), will regenerate all")
    
    if removed_count > 0:
        print(f"Removed {removed_count} records ({len(incomplete_docs)} incomplete docs + invalid IDs)")
        write_jsonl(valid_records, str(output_path))
        print(f"Updated output file with {len(valid_records)} valid records")
    
    print(f"Found {len(existing_records)} documents with COMPLETE query sets")
    return existing_records


def get_missing_docs(corpus: List[dict], existing_records: Dict[str, list], num_queries: int) -> List[dict]:
    """
    Get documents that need processing.
    A document needs processing if:
    - It has no queries at all, OR
    - It has incomplete queries (< num_queries variations)
    
    Note: sync_output_file already filters out incomplete sets, so this just checks existence.
    """
    missing = []
    
    for doc in corpus:
        doc_id = doc.get('_id')
        existing_queries = existing_records.get(doc_id, [])
        
        # Since sync_output_file only returns complete sets, we just check existence
        if len(existing_queries) < num_queries:
            missing.append(doc)
    
    return missing

def get_doc_text(doc: dict) -> str:
    """Extract text from document."""
    text_parts = []
    
    # Prioritize title and text fields
    if 'title' in doc and doc['title']:
        text_parts.append(f"Title: {doc['title']}")
    if 'text' in doc and doc['text']:
        text_parts.append(f"Content: {doc['text']}")
    
    # If no title/text, use all string fields
    if not text_parts:
        for key, value in doc.items():
            if key not in ['_id', '_membership'] and value:
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
    
    return ' '.join(text_parts).strip()


def generate_queries_openai(
    corpus: List[dict],
    model: str,
    temperature: float,
    num_queries: int,
    query_mode: str,
    output_path: Path,
    env_path: str,
    max_tokens_per_batch: int,
    check_interval: int
):
    """Generate queries using OpenAI Batch API."""
    client = initialize_openai_client(env_path)
    
    # Prepare document items
    doc_items = [(doc['_id'], doc) for doc in corpus]
    
    # Estimate tokens function
    def estimate_item_tokens(item):
        doc_id, doc = item
        text = get_doc_text(doc)
        prompt = create_generation_prompt_multi_queries(text, num_queries, query_mode)
        return estimate_tokens_accurate(prompt, num_queries, model)
    
    # Split into batches
    batches = split_into_batches(doc_items, estimate_item_tokens, max_tokens_per_batch)
    
    # Create request function
    def create_request(item):
        doc_id, doc = item
        text = get_doc_text(doc)
        prompt = create_generation_prompt_multi_queries(text, num_queries, query_mode)
        
        messages = [
            {"role": "system", "content": create_query_generation_system_prompt(num_queries, query_mode)},
            {"role": "user", "content": prompt}
        ]
        
        return create_batch_request(
            custom_id=doc_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=300 * num_queries
        )
    
    # Build doc lookup for result processing
    doc_lookup = {doc['_id']: doc for doc in corpus}
    
    # Result processor function - ALL OR NOTHING approach
    def process_result(custom_id, generated_text):
        target_doc_id = custom_id
        queries = parse_multi_query_response(generated_text, num_queries, query_mode)
        doc = doc_lookup.get(target_doc_id, {})
        membership = doc.get('_membership', '')
        
        successful = []
        failed = []
        
        # ALL OR NOTHING: Only accept if we got all num_queries
        if len(queries) == num_queries:
            for q_idx, query in enumerate(queries):
                successful.append({
                    '_id': f'q_{target_doc_id}_v{q_idx}',
                    'text': query,
                    'target_doc_id': target_doc_id,
                    '_membership': membership,
                    'variation_index': q_idx,
                    'query_mode': query_mode
                })
        else:
            # Incomplete - mark ALL variations as failed
            for q_idx in range(num_queries):
                failed.append({
                    'target_doc_id': target_doc_id,
                    'variation_index': q_idx,
                    'reason': f'Incomplete generation: only {len(queries)}/{num_queries} queries generated'
                })
        
        return successful if successful else None, failed if failed else None

    def save_chunk_queries(successful, failed, batch_num):
        if successful:
            merge_query_variation_records(str(output_path), successful)
            print(
                f"  [checkpoint] Saved {len(successful)} query records after batch "
                f"{batch_num} -> {output_path}"
            )
    
    # Run batch generation
    all_queries, all_failures = run_batch_generation(
        client,
        batches,
        create_request,
        process_result,
        output_path.parent,
        output_path.stem,
        check_interval,
        metadata_base={
            "description": "Query generation",
            "num_queries": str(num_queries),
            "query_mode": query_mode
        },
        on_chunk_complete=save_chunk_queries,
        resume=True,
    )
    
    return all_queries, all_failures


def generate_queries_transformers(
    corpus: List[dict],
    tokenizer,
    model,
    temperature: float,
    num_queries: int,
    query_mode: str,
    batch_size: int
):
    """Generate queries using transformers with ALL-OR-NOTHING strategy."""
    all_queries = []
    all_failures = []
    
    prompts = []
    metadata_list = []
    
    for doc in corpus:
        text = get_doc_text(doc)
        
        if not text:
            for q_idx in range(num_queries):
                all_failures.append({
                    'target_doc_id': doc['_id'],
                    'variation_index': q_idx,
                    'reason': 'Document text not found'
                })
            continue
        
        user_prompt = create_generation_prompt_multi_queries(text, num_queries, query_mode)
        
        messages = [
            {"role": "system", "content": create_query_generation_system_prompt(num_queries, query_mode)},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = create_chat_prompt(tokenizer, messages)
        prompts.append(prompt)
        metadata_list.append({
            'doc_id': doc['_id'],
            '_membership': doc.get('_membership', '')
        })
    
    # Generate
    generated_texts = generate_batch_transformers(
        prompts,
        tokenizer,
        model,
        temperature=temperature,
        max_new_tokens=300 * num_queries,
        batch_size=batch_size
    )
    
    # Process results - ALL OR NOTHING approach
    for generated_text, metadata in zip(generated_texts, metadata_list):
        target_doc_id = metadata['doc_id']
        membership = metadata['_membership']
        
        queries = parse_multi_query_response(generated_text, num_queries, query_mode)
        
        # ALL OR NOTHING: Only accept complete sets
        if len(queries) == num_queries:
            for q_idx, query in enumerate(queries):
                all_queries.append({
                    '_id': f'q_{target_doc_id}_v{q_idx}',
                    'text': query,
                    'target_doc_id': target_doc_id,
                    '_membership': membership,
                    'variation_index': q_idx,
                    'query_mode': query_mode
                })
        else:
            # Incomplete - mark ALL variations as failed
            for q_idx in range(num_queries):
                all_failures.append({
                    'target_doc_id': target_doc_id,
                    'variation_index': q_idx,
                    'reason': f'Incomplete generation: only {len(queries)}/{num_queries} queries generated'
                })
    
    return all_queries, all_failures


def main():
    parser = argparse.ArgumentParser(description='Generate queries for member and non-member documents')
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
        '--dataset',
        type=str,
        default='BeIR_trec-covid',
        help='Dataset name for output path'
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='results/MEntA',
        help='Base directory for query outputs (default: results/MEntA)'
    )
    parser.add_argument(
        '--generation_method',
        type=str,
        choices=['openai', 'transformers'],
        default='openai',
        help='Generation method'
    )
    parser.add_argument(
        '--generation_model',
        type=str,
        default='gpt-4.1-nano',
        help='Model name for generation'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU for transformers'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='~/.cache/huggingface/hub',
        help='Cache directory for models'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for generation'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for transformers'
    )
    parser.add_argument(
        '--num_queries',
        type=int,
        default=5,
        help='Number of queries to generate per document'
    )
    parser.add_argument(
        '--query_mode',
        type=str,
        choices=['specific', 'generic'],
        default='specific',
        help='Generate document-specific queries or broader generic topical queries'
    )
    parser.add_argument(
        '--env_path',
        type=str,
        default='../.env',
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
    
    query_dir = 'queries_generic' if args.query_mode == 'generic' else 'queries'
    output_path = Path(args.output_base_dir) / args.dataset / query_dir / f'queries_{args.num_queries}v.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"Generating queries with ALL-OR-NOTHING strategy")
    print(f"Member path: {args.member_path}")
    print(f"Non-member path: {args.nonmember_path}")
    print(f"Output: {output_path}")
    print(f"Method: {args.generation_method}")
    print(f"Model: {args.generation_model}")
    print(f"Num queries per doc: {args.num_queries}")
    print(f"Query mode: {args.query_mode}")
    print(f"Temperature: {args.temperature}")
    print(f"{'#'*60}\n")
    
    # Load and merge corpus
    corpus = load_and_merge_corpus(args.member_path, args.nonmember_path)
    
    if not corpus:
        print("Error: No documents loaded. Exiting.")
        return
    
    # Get valid IDs
    valid_ids = {doc['_id'] for doc in corpus}
    
    # Sync output file and get existing COMPLETE records only
    existing_records = sync_output_file(output_path, valid_ids, args.num_queries)
    
    # Get missing/incomplete documents
    missing_docs = get_missing_docs(corpus, existing_records, args.num_queries)
    
    print(f"Documents to process: {len(missing_docs)}")
    
    if not missing_docs:
        print("All documents already have complete query sets. Nothing to do.")
        return
    
    # Generate queries for missing documents
    if args.generation_method == 'openai':
        new_queries, all_failures = generate_queries_openai(
            missing_docs,
            args.generation_model,
            args.temperature,
            args.num_queries,
            args.query_mode,
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
            args.generation_model,
            args.use_gpu,
            args.cache_dir
        )
        
        new_queries, all_failures = generate_queries_transformers(
            missing_docs,
            tokenizer,
            model,
            args.temperature,
            args.num_queries,
            args.query_mode,
            args.batch_size
        )
    
    # Merge with existing complete records (idempotent if incremental saves already ran)
    if new_queries or output_path.exists():
        # Read existing COMPLETE records
        if output_path.exists():
            all_records = read_jsonl(str(output_path))
        else:
            all_records = []
        
        # Flatten new_queries if nested
        flattened_queries = []
        for item in new_queries:
            if isinstance(item, list):
                flattened_queries.extend(item)
            elif isinstance(item, dict):
                flattened_queries.append(item)
        
        # Remove any existing records for documents we just regenerated
        regenerated_doc_ids = {q.get('target_doc_id') for q in flattened_queries}
        all_records = [r for r in all_records if r.get('target_doc_id') not in regenerated_doc_ids]
        
        # Add new complete query sets
        all_records.extend(flattened_queries)
        
        # Sort by target_doc_id and variation_index
        all_records.sort(key=lambda x: (x.get('target_doc_id', ''), x.get('variation_index', 0)))
        
        write_jsonl(all_records, str(output_path))
        print(f"\n✓ Added {len(flattened_queries)} new queries to: {output_path}")
        print(f"✓ Total records in file: {len(all_records)}")
    
    if all_failures:
        failure_report_path = output_path.parent / f"{output_path.stem}_failures.json"
        
        # Load existing failures if any
        if failure_report_path.exists():
            existing_failures = read_json(str(failure_report_path))
            existing_failure_list = existing_failures.get('failures', [])
        else:
            existing_failure_list = []
        
        # Add new failures
        existing_failure_list.extend(all_failures)
        
        write_json({
            'failures': existing_failure_list,
            'total_failed': len(existing_failure_list)
        }, str(failure_report_path))
        print(f"⚠ Saved {len(all_failures)} new failures (total: {len(existing_failure_list)}) to: {failure_report_path}")
    
    # Report statistics
    total_docs = len(corpus)
    total_expected = total_docs * args.num_queries
    
    # Count complete docs in final output
    if output_path.exists():
        final_records = read_jsonl(str(output_path))
        doc_query_counts = {}
        for r in final_records:
            doc_id = r.get('target_doc_id')
            doc_query_counts[doc_id] = doc_query_counts.get(doc_id, 0) + 1
        
        complete_docs = sum(1 for count in doc_query_counts.values() if count == args.num_queries)
        total_complete = complete_docs * args.num_queries
    else:
        complete_docs = 0
        total_complete = 0
    
    success_rate = total_complete / total_expected * 100 if total_expected > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Query Generation Statistics (ALL-OR-NOTHING):")
    print(f"  Total documents: {total_docs}")
    print(f"  Queries per document: {args.num_queries}")
    print(f"  Expected total queries: {total_expected}")
    print(f"  Documents with COMPLETE sets: {complete_docs}/{total_docs}")
    print(f"  Total complete queries: {total_complete}/{total_expected}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
