from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys
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

# Load environment variables
load_dotenv('.env')

# Prompts
SUMMARY_PROMPT = """Summarize the following document in 1-2 sentences:

Title: {title}
Text: {text}

Summary:"""

QUESTIONS_PROMPT = """Main Text:
Based on the following text corpus, generate a list of {num} specific, diverse yes/no questions as queries, along with their answer that could be used to retrieve information from this corpus. Note: If the text uses any abbreviations or acronyms, such as 'AhR' or 'IC(50)', use the same forms in your questions. Do not use the expanded version unless it is explicitly mentioned in the text. The answer must be Yes or No only, not any other extra information allowed. Here are a few examples of the type of questions we are looking for:

Example Text:
Dioxins invade the body mainly through the diet, and produce toxicity through the transformation of aryl hydrocarbon receptor (AhR). An inhibitor of the transformation should therefore protect against the toxicity and ideally be part of the diet. We examined flavonoids ubiquitously expressed in plant foods as one of the best candidates, and found that the subclasses flavones and flavonols suppressed antagonistically the transformation of AhR induced by 1 nM of 2,3,7,8-tetrachlorodibenzo-p-dioxin, without exhibiting agonistic effects that transform AhR. The antagonistic IC(50) values ranged from 0.14 to 10 microM, close to the physiological levels in human.

Example Questions:
1. Are flavones and flavonols shown to antagonistically suppress the transformation of AhR induced by dioxins? Yes
2. Do flavones and flavonols exhibit agonistic effects that transform the aryl hydrocarbon receptor? No
3. Are the antagonistic IC(50) values for flavones and flavonols between 0.14 and 10 microM? Yes

Now, based on the main corpus provided below, create questions ans answers that are specific, contain keywords from the text, and are diverse enough to cover different aspects or concepts discussed. Avoid mentioning 'the study' or any references to the passage itself, and ensure that questions do not contain general phrases that could apply to any text.

Here is the Corpus:
{text}

Generate {num} yes/no questions based on this text, along with the answer to each question. Remember, the answer must be either "Yes" or "No" only.
"""

NUM_QUESTIONS = 30


def get_existing_ia_doc_ids(output_path: str) -> set:
    """Return document IDs that already have queries in the output file."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()
    return {record['target_doc_id'] for record in read_jsonl(output_path)}


def parse_questions_from_output(output: str) -> List[str]:
    """Parse yes/no questions from model output using tolerant line formats."""
    questions = []
    seen = set()
    for raw_line in output.split('\n'):
        line = raw_line.strip()
        if not line or '?' not in line:
            continue
        line = re.sub(r"^\s*(?:\d+[\).:-]\s*|[-*]\s*)", "", line)
        line = re.sub(r"^(?:question|q)\s*[:.-]\s*", "", line, flags=re.IGNORECASE)
        question_part, _ = line.split("?", 1)
        question = question_part.strip().strip('"') + "?"
        if question and question not in seen:
            seen.add(question)
            questions.append(question)
    return questions


def format_queries_output(doc_id: str, questions: List[str], membership: str) -> List[dict]:
    """Format questions into the required output format."""
    queries = []
    for idx, question in enumerate(questions):
        query = {
            "_id": f"q_{doc_id}_v{idx}",
            "text": question,
            "target_doc_id": doc_id,
            "_membership": membership,
            "variation_index": idx
        }
        queries.append(query)
    return queries


def generate_with_openai_batch(
    documents: List[dict],
    num_questions: int,
    openai_client: OpenAI,
    openai_model: str,
    output_dir: Path,
    output_path: str,
) -> List[dict]:
    """Generate questions using OpenAI Batch API."""
    
    # Prepare items for batch generation
    items = [(doc['_id'], doc['text'], doc['membership'], doc.get('title', '')) for doc in documents]
    
    # Create batch requests for questions
    def create_questions_request(item):
        doc_id, text, membership, title = item
        messages = [{"role": "user", "content": QUESTIONS_PROMPT.format(num=num_questions, text=text)}]
        return create_batch_request(
            custom_id=f"questions_{doc_id}",
            model=openai_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
    
    # Process questions
    print("\n" + "="*60)
    print("GENERATING QUESTIONS")
    print("="*60)
    
    questions_batches = split_into_batches(
        items,
        lambda item: estimate_tokens_accurate(QUESTIONS_PROMPT.format(num=num_questions, text=item[1]), num_questions, openai_model),
        max_tokens_per_batch=2_000_000
    )
    
    # Build membership map
    membership_map = {doc['_id']: doc['membership'] for doc in documents}
    
    def process_questions_result(custom_id, generated_text):
        doc_id = custom_id.replace("questions_", "")
        questions = parse_questions_from_output(generated_text)
        if questions:
            membership = membership_map.get(doc_id, 'unknown')
            formatted_queries = format_queries_output(doc_id, questions, membership)
            return {'_id': doc_id, 'queries': formatted_queries}, None
        else:
            return None, {'custom_id': custom_id, 'reason': 'No valid questions parsed'}

    def save_chunk_queries(successful, failed, batch_num):
        flattened = []
        for result in successful:
            if isinstance(result, dict) and 'queries' in result:
                flattened.extend(result['queries'])
            elif isinstance(result, dict):
                flattened.append(result)
        if flattened:
            merge_query_variation_records(output_path, flattened)
            print(
                f"  [checkpoint] Saved {len(flattened)} IA-MIA query records after batch "
                f"{batch_num} -> {output_path}"
            )

    questions_results, questions_failed = run_batch_generation(
        openai_client,
        questions_batches,
        create_questions_request,
        process_questions_result,
        output_dir,
        "questions_batch",
        metadata_base={"type": "questions"},
        on_chunk_complete=save_chunk_queries,
        resume=True,
    )
    
    # Flatten results into individual queries
    all_queries = []
    for result in questions_results:
        all_queries.extend(result['queries'])
    
    print(f"\n✓ Successfully generated {len(all_queries)} queries from {len(questions_results)} documents")
    print(f"  Questions failures: {len(questions_failed)}")
    
    return all_queries


def generate_with_transformers_batch(
    documents: List[dict],
    num_questions: int,
    tokenizer,
    model,
    batch_size: int = 4
) -> List[dict]:
    """Generate questions using transformers."""
    
    print("\n" + "="*60)
    print("GENERATING QUESTIONS")
    print("="*60)
    
    # Generate questions
    question_prompts = []
    for doc in documents:
        messages = [{"role": "user", "content": QUESTIONS_PROMPT.format(num=num_questions, text=doc['text'])}]
        prompt = create_chat_prompt(tokenizer, messages)
        question_prompts.append(prompt)
    
    question_outputs = generate_batch_transformers(
        question_prompts,
        tokenizer,
        model,
        temperature=0.2,
        max_new_tokens=1000,
        batch_size=batch_size
    )
    
    # Parse questions and format output
    all_queries = []
    for doc, output in zip(documents, question_outputs):
        questions = parse_questions_from_output(output)
        formatted_queries = format_queries_output(doc['_id'], questions, doc['membership'])
        all_queries.extend(formatted_queries)
        print(f"Processed doc {doc['_id']} ({doc['membership']}): {len(questions)} questions")
    
    return all_queries


def generate_queries_from_corpus_files(
    corpus_member_path: str,
    corpus_nonmember_path: str,
    output_path: str,
    num_questions: int,
    generation_method: str,
    openai_client=None,
    openai_model: str = "gpt-4.1-nano",
    tokenizer=None,
    model=None,
    batch_size: int = 4
):
    """
    Generate yes/no questions for documents from member and nonmember corpus files.
    """
    print(f"\n{'#'*60}")
    print(f"Generating IA-MIA queries from corpus files")
    print(f"Method: {generation_method}")
    if generation_method == 'transformers':
        print(f"Batch size: {batch_size}")
    print(f"Member corpus: {corpus_member_path}")
    print(f"Nonmember corpus: {corpus_nonmember_path}")
    print(f"Output: {output_path}")
    print(f"{'#'*60}\n")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    existing_doc_ids = get_existing_ia_doc_ids(output_path)
    if existing_doc_ids:
        print(f"Resuming from {len(existing_doc_ids)} documents already present in {output_path}")
    
    # Load member and nonmember corpora
    corpus_member = read_jsonl(corpus_member_path)
    corpus_nonmember = read_jsonl(corpus_nonmember_path)
    
    print(f"Loaded {len(corpus_member)} member documents")
    print(f"Loaded {len(corpus_nonmember)} nonmember documents")
    
    # Prepare documents with combined text and labels
    documents = []
    
    # Process member documents
    for doc in corpus_member:
        text_parts = []
        for key, value in doc.items():
            if key != '_id' and value:
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    text_parts.append(f"{key}: {str(value)}")
        combined_text = ' '.join(text_parts).strip()
        
        documents.append({
            '_id': doc['_id'],
            'text': combined_text,
            'title': doc.get('title', ''),
            'membership': 'member'
        })
    
    # Process nonmember documents
    for doc in corpus_nonmember:
        text_parts = []
        for key, value in doc.items():
            if key != '_id' and value:
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    text_parts.append(f"{key}: {str(value)}")
        combined_text = ' '.join(text_parts).strip()
        
        documents.append({
            '_id': doc['_id'],
            'text': combined_text,
            'title': doc.get('title', ''),
            'membership': 'nonmember'
        })
    
    print(f"\nTotal documents to process: {len(documents)}")
    print(f"  Members: {len(corpus_member)}")
    print(f"  Nonmembers: {len(corpus_nonmember)}")

    pending_documents = [doc for doc in documents if doc['_id'] not in existing_doc_ids]
    print(f"Documents remaining after resume: {len(pending_documents)}")

    if not pending_documents:
        print("All documents already have IA-MIA queries. Nothing to do.")
        return
    
    # Generate based on method
    if generation_method == 'openai':
        output_dir = Path(output_path).parent
        all_queries = generate_with_openai_batch(
            pending_documents,
            num_questions,
            openai_client,
            openai_model,
            output_dir,
            output_path,
        )
    else:
        all_queries = generate_with_transformers_batch(
            pending_documents,
            num_questions,
            tokenizer,
            model,
            batch_size
        )
    
    if not all_queries and not os.path.exists(output_path):
        raise RuntimeError(
            "No IA-MIA queries were generated. Check the OpenAI batch output and "
            "question parsing before running the offline retrieval stage."
        )

    if generation_method != 'openai' and all_queries:
        if os.path.exists(output_path):
            prior_queries = read_jsonl(output_path)
            all_queries = prior_queries + all_queries
        write_jsonl(all_queries, output_path)
    elif os.path.exists(output_path):
        all_queries = read_jsonl(output_path)
    
    # Count by membership
    member_queries = sum(1 for q in all_queries if q['_membership'] == 'member')
    nonmember_queries = sum(1 for q in all_queries if q['_membership'] == 'nonmember')
    
    print(f"\n{'='*60}")
    print(f"Generated {len(all_queries)} total queries")
    print(f"  Member queries: {member_queries}")
    print(f"  Nonmember queries: {nonmember_queries}")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate IA-MIA queries for documents')
    parser.add_argument(
        '--corpus_member',
        type=str,
        default='data/BeIR_trec-covid/corpus_member.jsonl',
        help='Path to corpus_member.jsonl'
    )
    parser.add_argument(
        '--corpus_nonmember',
        type=str,
        default='data/BeIR_trec-covid/corpus_nonmember.jsonl',
        help='Path to corpus_nonmember.jsonl'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/IA-MIA/BeIR_trec-covid/queries',
        help='Output directory for IA-MIA files'
    )
    parser.add_argument(
        '--num_questions',
        type=int,
        default=30,
        help='Number of questions to generate per document'
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
        help='Batch size for transformers generation'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Generate questions from member and nonmember corpus files
    output_path = os.path.join(args.output_dir, f'queries_{args.num_questions}.jsonl')
    generate_queries_from_corpus_files(
        args.corpus_member,
        args.corpus_nonmember,
        output_path,
        args.num_questions,
        args.generation_method,
        openai_client=openai_client,
        openai_model=args.openai_model,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print("IA-MIA query generation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()