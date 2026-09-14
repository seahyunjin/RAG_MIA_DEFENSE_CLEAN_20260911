import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
import argparse
from tqdm import tqdm
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.openai_gen import initialize_openai_client


def download_batch_results(client: OpenAI, file_id: str) -> List[Dict]:
    """Download batch results from OpenAI."""
    file_response = client.files.content(file_id)
    file_content = file_response.text
    
    results = []
    for line in file_content.strip().split('\n'):
        if line:
            results.append(json.loads(line))
    
    return results


def process_rag_batch_results(batch_results: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """
    Process RAG answer batch results.
    Expects custom_id format: rag_{doc_id}_{question_idx}
    """
    successful = []
    failed = []
    
    for result in batch_results:
        custom_id = result['custom_id']
        
        if result.get('error'):
            error_info = result['error']
            failed.append({
                'custom_id': custom_id,
                'reason': f"{error_info.get('code', 'unknown')}: {error_info.get('message', 'unknown error')}"
            })
            continue
        
        response = result.get('response', {})
        body = response.get('body', {})
        choices = body.get('choices', [])
        
        if not choices:
            failed.append({
                'custom_id': custom_id,
                'reason': 'No choices in response'
            })
            continue
        
        generated_text = choices[0].get('message', {}).get('content', '').strip()
        
        if not generated_text:
            failed.append({
                'custom_id': custom_id,
                'reason': 'Empty generated text'
            })
            continue
        
        # Parse custom_id: rag_{doc_id}_{question_idx}
        try:
            parts = custom_id.split('_')
            # Handle doc IDs that might contain underscores
            doc_id = '_'.join(parts[1:-1])
            question_idx = int(parts[-1])
            
            successful.append({
                'doc_id': doc_id,
                'question_idx': question_idx,
                'answer': generated_text.strip()
            })
        except Exception as e:
            failed.append({
                'custom_id': custom_id,
                'reason': f'Failed to parse custom_id: {str(e)}'
            })
    
    return successful, failed


def get_all_batches(client: OpenAI, limit: int = 100) -> List[Dict]:
    """Get all batches from OpenAI API."""
    print(f"\nFetching batches from OpenAI (limit={limit})...")
    batches_response = client.batches.list(limit=limit)
    
    batch_list = []
    for batch in batches_response.data:
        batch_info = {
            'id': batch.id,
            'status': batch.status,
            'created_at': batch.created_at,
            'endpoint': batch.endpoint,
        }
        
        if hasattr(batch, 'metadata') and batch.metadata:
            batch_info['metadata'] = batch.metadata
        
        if hasattr(batch, 'request_counts'):
            counts = batch.request_counts
            batch_info['request_counts'] = {
                'total': counts.total,
                'completed': counts.completed,
                'failed': counts.failed
            }
        
        if hasattr(batch, 'completed_at') and batch.completed_at:
            batch_info['completed_at'] = batch.completed_at
        
        batch_list.append(batch_info)
    
    print(f"✓ Found {len(batch_list)} batches")
    return batch_list


def filter_rag_batches(batches: List[Dict], limit: int = None) -> List[str]:
    """
    Filter batches for RAG answer generation (type='rag_answers' in metadata).
    Returns batch IDs sorted by creation time.
    """
    # Filter for RAG batches with completed status
    rag_batches = [
        b for b in batches 
        if b['status'] == 'completed' 
        and b.get('metadata', {}).get('type') == 'rag_answers'
    ]
    
    # Sort by created_at (oldest first to maintain order)
    rag_batches = sorted(rag_batches, key=lambda x: x['created_at'])
    
    # Apply limit
    if limit:
        rag_batches = rag_batches[:limit]
    
    return [b['id'] for b in rag_batches]


def retrieve_rag_batches(
    client: OpenAI,
    batch_ids: List[str],
    output_dir: Path
) -> tuple[List[Dict], List[Dict]]:
    """Retrieve RAG answer batch results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_successful = []
    all_failed = []
    
    print(f"\n{'='*80}")
    print(f"RETRIEVING RAG ANSWERS FROM {len(batch_ids)} BATCHES")
    print(f"{'='*80}\n")
    
    for i, batch_id in enumerate(tqdm(batch_ids, desc="Processing batches"), 1):
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            
            print(f"\nBatch {i}/{len(batch_ids)}: {batch_id}")
            print(f"  Status: {status}")
            
            if hasattr(batch, 'request_counts'):
                counts = batch.request_counts
                print(f"  Progress: {counts.completed}/{counts.total} completed, {counts.failed} failed")
            
            if status == 'completed':
                # Download and process results
                try:
                    batch_results = download_batch_results(client, batch.output_file_id)
                    successful, failed = process_rag_batch_results(batch_results)
                    
                    all_successful.extend(successful)
                    all_failed.extend(failed)
                    
                    print(f"  ✓ Retrieved: {len(successful)} successful, {len(failed)} failed")
                    
                    # Save incremental batch file
                    batch_output = {
                        'batch_id': batch_id,
                        'batch_number': i,
                        'successful': successful,
                        'failed': failed,
                        'metadata': {
                            'status': status,
                            'request_counts': {
                                'total': counts.total,
                                'completed': counts.completed,
                                'failed': counts.failed
                            } if hasattr(batch, 'request_counts') else None
                        }
                    }
                    
                    batch_file = output_dir / "incremental" / f"batch_{i:03d}_{batch_id[:8]}.json"
                    batch_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(batch_file, 'w') as f:
                        json.dump(batch_output, f, indent=2)
                    
                    print(f"  ✓ Saved to: {batch_file.name}")
                
                except Exception as e:
                    print(f"  ✗ Error downloading results: {e}")
                    all_failed.append({
                        'batch_id': batch_id,
                        'reason': f'Download error: {str(e)}'
                    })
            else:
                print(f"  ⚠️  Batch not completed yet (status: {status})")
        
        except Exception as e:
            print(f"  ✗ Error retrieving batch: {e}")
    
    return all_successful, all_failed


def reconstruct_rag_answers(
    answers_results: List[Dict],
    questions_file: str
) -> List[Dict]:
    """
    Reconstruct RAG answers in the expected JSONL format.
    Format matches: corpus_rag_answers_gpt.jsonl
    """
    print("\nReconstructing RAG answers in final format...")
    
    # Load original questions to get document metadata
    from utils.process_json import read_jsonl
    questions_data = read_jsonl(questions_file)
    
    # Create doc_id to document mapping
    doc_map = {doc['_id']: doc for doc in questions_data}
    
    # Organize answers by doc_id
    answers_by_doc = {}
    for result in answers_results:
        doc_id = result['doc_id']
        q_idx = result['question_idx']
        answer = result['answer']
        
        if doc_id not in answers_by_doc:
            answers_by_doc[doc_id] = {}
        answers_by_doc[doc_id][q_idx] = answer
    
    # Build final output in expected format
    results = []
    for doc_id, doc_info in doc_map.items():
        if doc_id in answers_by_doc:
            qa_pairs = []
            for q_idx, question in enumerate(doc_info['questions']):
                answer = answers_by_doc[doc_id].get(q_idx, "")
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
            
            results.append({
                "_id": doc_id,
                "label": doc_info['label'],
                "text": doc_info['text'],
                "summary": doc_info['summary'],
                "result": qa_pairs
            })
    
    print(f"✓ Reconstructed RAG answers for {len(results)} documents")
    return results


def save_rag_answers(
    rag_answers: List[Dict],
    output_path: Path
):
    """Save RAG answers in JSONL format."""
    from utils.process_json import write_jsonl
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rag_answers, str(output_path))
    
    print(f"\n{'='*80}")
    print(f"RAG ANSWERS SAVED")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"Total documents: {len(rag_answers)}")
    
    # Calculate total Q&A pairs
    total_qa = sum(len(doc['result']) for doc in rag_answers)
    print(f"Total Q&A pairs: {total_qa}")


def main():
    parser = argparse.ArgumentParser(
        description='Retrieve OpenAI RAG answer batches and save in final format'
    )
    parser.add_argument('--env_path', type=str, default='.env',
                        help='Path to .env file with OPENAI_API_KEY')
    parser.add_argument('--questions_file', type=str, 
                       default='data/BeIR_nfcorpus/ia_mia/corpus_30_questions.jsonl',
                       help='Original questions file (for document metadata)')
    parser.add_argument('--output_file', type=str,
                       default='data/BeIR_nfcorpus/ia_mia/corpus_rag_answers_gpt_retrieved.jsonl',
                       help='Output file for RAG answers')
    parser.add_argument('--output_dir', type=str, 
                       default='data/BeIR_nfcorpus/ia_mia/batch_input',
                       help='Directory for incremental batch files')
    parser.add_argument('--limit', type=int, default=56,
                       help='Number of batches to retrieve (default: 56)')
    parser.add_argument('--max_fetch', type=int, default=100,
                       help='Maximum number of batches to fetch from API')
    parser.add_argument('--list_only', action='store_true',
                       help='Only list available RAG batches without retrieving')
    
    args = parser.parse_args()
    
    # Add project root to path for utils import
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Initialize client
    client = initialize_openai_client(args.env_path)
    
    # Get all batches
    all_batches = get_all_batches(client, limit=args.max_fetch)
    
    # Filter for RAG batches
    rag_batch_ids = filter_rag_batches(all_batches, limit=args.limit)
    
    print(f"\n{'='*80}")
    print(f"FOUND {len(rag_batch_ids)} COMPLETED RAG ANSWER BATCHES")
    print(f"{'='*80}")
    
    if not rag_batch_ids:
        print("\n✗ No completed RAG answer batches found!")
        print("Make sure batches have metadata: {'type': 'rag_answers'}")
        return
    
    if args.list_only:
        print("\nBatch IDs:")
        for i, batch_id in enumerate(rag_batch_ids, 1):
            print(f"  {i}. {batch_id}")
        return
    
    # Retrieve batch results
    output_dir = Path(args.output_dir)
    successful, failed = retrieve_rag_batches(client, rag_batch_ids, output_dir)
    
    print(f"\n{'='*80}")
    print(f"RETRIEVAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total successful answers: {len(successful)}")
    print(f"Total failed: {len(failed)}")
    
    if failed:
        print(f"\n⚠️  Failed items:")
        for fail in failed[:10]:  # Show first 10
            print(f"  - {fail.get('custom_id', 'unknown')}: {fail.get('reason', 'unknown')}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # Save raw results
    raw_results_file = output_dir / "raw_answers.json"
    with open(raw_results_file, 'w') as f:
        json.dump({
            'successful': successful,
            'failed': failed,
            'summary': {
                'total_successful': len(successful),
                'total_failed': len(failed),
                'batches_processed': len(rag_batch_ids)
            }
        }, f, indent=2)
    print(f"\n✓ Raw results saved to: {raw_results_file}")
    
    # Reconstruct in final format
    rag_answers = reconstruct_rag_answers(successful, args.questions_file)
    
    # Save final output
    output_file = Path(args.output_file)
    save_rag_answers(rag_answers, output_file)
    
    print(f"\n✓ Done! RAG answers retrieved and saved successfully.")


if __name__ == '__main__':
    main()
