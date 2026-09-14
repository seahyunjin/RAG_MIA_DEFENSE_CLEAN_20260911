import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional
import time
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from tqdm import tqdm


def ensure_openai_network_access() -> None:
    """
    OpenAI Chat/Batch/File APIs require outbound HTTPS.

    HuggingFace hub offline flags (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE) must not
    block OpenAI traffic. Force them off for the duration of OpenAI client usage.
    """
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"


@contextmanager
def openai_network_session():
    """Temporarily allow network access for OpenAI while preserving prior HF offline flags."""
    prev_hf = os.environ.get("HF_HUB_OFFLINE")
    prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    ensure_openai_network_access()
    try:
        yield
    finally:
        if prev_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_hf
        if prev_tf is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prev_tf


def estimate_tokens_accurate(text: str, num_queries: int, model: str = "gpt-4") -> int:
    """Accurately estimate tokens using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        input_tokens = len(encoding.encode(text))
    except:
        input_tokens = len(text) // 3.5
    
    overhead = 50
    output_tokens = 300 * num_queries
    
    return int(input_tokens + overhead + output_tokens)


def create_batch_request(
    custom_id: str,
    model: str,
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 1500
) -> Dict:
    """Create a single batch request in OpenAI Batch API format."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    }


def get_openai_inference_mode() -> str:
    """Return OpenAI inference mode from .env/environment: batch or single."""
    raw = os.getenv("OPENAI_INFERENCE_MODE", "batch").strip().lower()
    aliases = {
        "batch": "batch",
        "batches": "batch",
        "batch_api": "batch",
        "single": "single",
        "direct": "single",
        "chat": "single",
        "sequential": "single",
    }
    if raw not in aliases:
        raise ValueError(
            "OPENAI_INFERENCE_MODE must be one of: batch, single "
            f"(got {raw!r})"
        )
    return aliases[raw]


def _extract_chat_completion_text(response) -> str:
    """Extract assistant text from an OpenAI chat completion response."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "") or ""
    return content.strip()


def create_batch_input_file(
    requests: List[Dict],
    output_path: Path
) -> Path:
    """Create JSONL batch input file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    return output_path


def submit_batch_job(
    client: OpenAI,
    batch_input_path: Path,
    metadata: Dict = None
) -> str:
    """Upload batch input file and create batch job."""
    ensure_openai_network_access()
    print(f"\nSubmitting batch: {batch_input_path.name}")
    
    with open(batch_input_path, 'rb') as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata or {}
    )
    
    print(f"  ✓ Batch ID: {batch.id}")
    print(f"  Status: {batch.status}")
    
    return batch.id


def wait_for_batch_completion(
    client: OpenAI,
    batch_id: str,
    check_interval: int = 60,
    part_name: str = "Batch"
) -> Dict:
    """Wait for a batch to complete and return its status."""
    ensure_openai_network_access()
    print(f"\n⏳ Waiting for {part_name} ({batch_id}) to complete...")
    
    start_time = time.time()
    last_status = None
    
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.0f}s] Status: {status}")
            
            if status in ['validating', 'in_progress', 'finalizing']:
                counts = batch.request_counts
                print(f"    Progress: {counts.completed}/{counts.total} completed, {counts.failed} failed")
            
            last_status = status
        
        if status == 'completed':
            elapsed = time.time() - start_time
            print(f"  ✓ Completed in {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
            return {
                'status': 'completed',
                'batch': batch,
                'elapsed': elapsed
            }
        
        elif status == 'failed':
            print(f"  ✗ Batch failed!")
            if batch.errors:
                print(f"    Errors: {batch.errors}")
            return {
                'status': 'failed',
                'batch': batch,
                'errors': batch.errors
            }
        
        elif status in ['expired', 'cancelled']:
            print(f"  ✗ Batch {status}")
            return {
                'status': status,
                'batch': batch
            }
        
        time.sleep(check_interval)


def download_batch_results(client: OpenAI, file_id: str) -> List[Dict]:
    """Download batch results from OpenAI."""
    ensure_openai_network_access()
    file_response = client.files.content(file_id)
    file_content = file_response.text
    
    results = []
    for line in file_content.strip().split('\n'):
        if line:
            results.append(json.loads(line))
    
    return results


def process_batch_results(
    batch_results: List[Dict],
    result_processor_func
) -> tuple[List[Dict], List[Dict]]:
    """
    Process batch results using a custom processor function.
    
    Args:
        batch_results: Raw batch results from OpenAI
        result_processor_func: Function(result) -> (success_data, failure_data or None)
    
    Returns:
        Tuple of (successful_results, failed_results)
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
        
        # Use custom processor
        success_data, failure_data = result_processor_func(custom_id, generated_text)
        
        if success_data:
            successful.append(success_data)
        if failure_data:
            failed.append(failure_data)
    
    return successful, failed


def split_into_batches(
    items: List[tuple],
    estimate_tokens_func,
    max_tokens_per_batch: int = 1_200_000
) -> List[List[tuple]]:
    """
    Split items into batches respecting token limits.
    
    Args:
        items: List of items to split
        estimate_tokens_func: Function(item) -> estimated_tokens
        max_tokens_per_batch: Maximum tokens per batch
    
    Returns:
        List of batches (each batch is a list of items)
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    print(f"\nSplitting into batches (max {max_tokens_per_batch:,} tokens per batch)...")
    
    for item in tqdm(items, desc="Analyzing items"):
        item_tokens = estimate_tokens_func(item)
        
        if current_tokens + item_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            print(f"  ✓ Batch {len(batches)}: {len(current_batch)} items (~{current_tokens:,} tokens)")
            current_batch = []
            current_tokens = 0
        
        current_batch.append(item)
        current_tokens += item_tokens
    
    if current_batch:
        batches.append(current_batch)
        print(f"  ✓ Batch {len(batches)}: {len(current_batch)} items (~{current_tokens:,} tokens)")
    
    return batches


def _default_checkpoint_path(output_dir: Path, base_filename: str) -> Path:
    return output_dir / f"{base_filename}_checkpoint.json"


def _load_generation_checkpoint(checkpoint_path: Path) -> Dict:
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    return {}


def _save_generation_checkpoint(checkpoint_path: Path, state: Dict):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as handle:
        json.dump(state, handle, indent=2)
    temp_path.replace(checkpoint_path)


def _process_completed_batch(
    client: OpenAI,
    result: Dict,
    batch_items: List,
    result_processor_func,
) -> tuple[List[Dict], List[Dict]]:
    if result['status'] != 'completed':
        failed = [{'item': item, 'reason': f"Batch {result['status']}"} for item in batch_items]
        return [], failed

    batch_results = download_batch_results(client, result['batch'].output_file_id)
    successful, failed = process_batch_results(batch_results, result_processor_func)
    return successful, failed


def run_single_generation(
    client: OpenAI,
    batches: List[List[tuple]],
    create_request_func,
    result_processor_func,
    output_dir: Path,
    base_filename: str,
    on_chunk_complete: Optional[Callable[[List[Dict], List[Dict], int], None]] = None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> tuple[List[Dict], List[Dict]]:
    """
    Run OpenAI generation synchronously with one Chat Completions request per item.

    This uses the same request/processor callbacks as Batch API generation, so
    all existing attack scripts inherit OPENAI_INFERENCE_MODE=single without
    per-script flags.
    """
    ensure_openai_network_access()
    all_items = [item for batch in batches for item in batch]
    all_successful = []
    all_failed = []

    checkpoint_file = checkpoint_path or _default_checkpoint_path(output_dir, base_filename)
    checkpoint = _load_generation_checkpoint(checkpoint_file) if resume else {}
    completed_custom_ids = set(checkpoint.get('completed_custom_ids', []))

    pending_items = []
    for item_num, item in enumerate(all_items, 1):
        request = create_request_func(item)
        custom_id = request.get('custom_id', f'request_{item_num}')
        if resume and custom_id in completed_custom_ids:
            continue
        pending_items.append((item_num, item, custom_id))

    print(
        f"OpenAI inference mode: single. Running {len(pending_items)} request(s) "
        f"({len(completed_custom_ids)} already checkpointed) sequentially with Chat Completions."
    )

    for item_num, item, custom_id in tqdm(pending_items, desc="OpenAI single requests"):
        try:
            request = create_request_func(item)
            body = request.get("body", {})
            if not body:
                raise ValueError("OpenAI request is missing body")

            response = client.chat.completions.create(**body)
            generated_text = _extract_chat_completion_text(response)
            if not generated_text:
                failure = {
                    "custom_id": custom_id,
                    "item": item,
                    "reason": "Empty generated text",
                }
                all_failed.append(failure)
                if on_chunk_complete:
                    on_chunk_complete([], [failure], item_num)
                continue

            success_data, failure_data = result_processor_func(custom_id, generated_text)
            chunk_successful = []
            chunk_failed = []
            if success_data:
                if isinstance(success_data, list):
                    chunk_successful.extend(success_data)
                else:
                    chunk_successful.append(success_data)
            if failure_data:
                if isinstance(failure_data, list):
                    chunk_failed.extend(failure_data)
                else:
                    chunk_failed.append(failure_data)

            all_successful.extend(chunk_successful)
            all_failed.extend(chunk_failed)

            if on_chunk_complete:
                on_chunk_complete(chunk_successful, chunk_failed, item_num)

            if chunk_successful:
                completed_custom_ids.add(custom_id)
                checkpoint['completed_custom_ids'] = sorted(completed_custom_ids)
                checkpoint['inference_mode'] = 'single'
                checkpoint['total_items'] = len(all_items)
                _save_generation_checkpoint(checkpoint_file, checkpoint)

        except Exception as exc:
            failure = {
                "custom_id": custom_id,
                "item": item,
                "reason": str(exc),
            }
            all_failed.append(failure)
            if on_chunk_complete:
                on_chunk_complete([], [failure], item_num)

    if resume and len(completed_custom_ids) == len(all_items):
        checkpoint['completed_custom_ids'] = sorted(completed_custom_ids)
        checkpoint['inference_mode'] = 'single'
        checkpoint['total_items'] = len(all_items)
        _save_generation_checkpoint(checkpoint_file, checkpoint)

    print(f"  ✓ Single inference complete: {len(all_successful)} successful, {len(all_failed)} failed")
    return all_successful, all_failed


def run_batch_generation(
    client: OpenAI,
    batches: List[List[tuple]],
    create_request_func,
    result_processor_func,
    output_dir: Path,
    base_filename: str,
    check_interval: int = 120,
    metadata_base: Dict = None,
    on_chunk_complete: Optional[Callable[[List[Dict], List[Dict], int], None]] = None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
) -> tuple[List[Dict], List[Dict]]:
    """
    Run batch generation for all batches sequentially.
    
    Args:
        client: OpenAI client
        batches: List of batches to process
        create_request_func: Function(item) -> batch_request_dict
        result_processor_func: Function(custom_id, generated_text) -> (success_data, failure_data)
        output_dir: Directory for batch files
        base_filename: Base filename for batch files
        check_interval: Seconds between status checks
        metadata_base: Base metadata for all batches
    
    Returns:
        Tuple of (all_successful_results, all_failed_results)
    """
    ensure_openai_network_access()
    inference_mode = get_openai_inference_mode()
    if inference_mode == "single":
        return run_single_generation(
            client,
            batches,
            create_request_func,
            result_processor_func,
            output_dir,
            base_filename,
            on_chunk_complete=on_chunk_complete,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )

    print("OpenAI inference mode: batch. Using OpenAI Batch API.")
    all_successful = []
    all_failed = []

    batch_input_dir = output_dir / "batch_input"
    batch_input_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_path or _default_checkpoint_path(output_dir, base_filename)
    checkpoint = _load_generation_checkpoint(checkpoint_file) if resume else {}
    completed_parts = set(checkpoint.get('completed_parts', []))
    pending = checkpoint.get('pending')

    print(
        f"OpenAI batch execution is sequential: {len(batches)} batch(es) will be "
        f"submitted one at a time ({len(completed_parts)} already checkpointed)."
    )

    for batch_num, batch_items in enumerate(batches, 1):
        if resume and batch_num in completed_parts:
            print(f"  [checkpoint] Skipping completed batch {batch_num}/{len(batches)}")
            continue

        print(f"\n{'='*60}")
        print(f"PROCESSING BATCH {batch_num}/{len(batches)}")
        print(f"{'='*60}")

        batch_id = None
        if (
            resume
            and pending
            and pending.get('part') == batch_num
            and pending.get('batch_id')
        ):
            batch_id = pending['batch_id']
            print(f"  [checkpoint] Resuming in-flight batch {batch_num}: {batch_id}")
        else:
            batch_requests = [create_request_func(item) for item in batch_items]
            batch_file = batch_input_dir / f"{base_filename}_part{batch_num}.jsonl"
            create_batch_input_file(batch_requests, batch_file)

            metadata = {**(metadata_base or {}), "part": str(batch_num), "total_parts": str(len(batches))}
            batch_id = submit_batch_job(client, batch_file, metadata)
            checkpoint['pending'] = {'part': batch_num, 'batch_id': batch_id}
            checkpoint['completed_parts'] = sorted(completed_parts)
            checkpoint['inference_mode'] = 'batch'
            checkpoint['total_parts'] = len(batches)
            _save_generation_checkpoint(checkpoint_file, checkpoint)
            print("  Blocking until this OpenAI batch completes before continuing...")

        result = wait_for_batch_completion(
            client,
            batch_id,
            check_interval,
            f"Batch {batch_num}/{len(batches)}"
        )

        successful, failed = _process_completed_batch(
            client,
            result,
            batch_items,
            result_processor_func,
        )

        flattened_successful = []
        for item in successful:
            if isinstance(item, list):
                flattened_successful.extend(item)
            else:
                flattened_successful.append(item)

        flattened_failed = []
        for item in failed:
            if isinstance(item, list):
                flattened_failed.extend(item)
            else:
                flattened_failed.append(item)

        all_successful.extend(flattened_successful)
        all_failed.extend(flattened_failed)

        if on_chunk_complete and (flattened_successful or flattened_failed):
            on_chunk_complete(flattened_successful, flattened_failed, batch_num)

        if result['status'] == 'completed':
            completed_parts.add(batch_num)
            checkpoint['completed_parts'] = sorted(completed_parts)
            checkpoint['pending'] = None
            checkpoint['inference_mode'] = 'batch'
            checkpoint['total_parts'] = len(batches)
            _save_generation_checkpoint(checkpoint_file, checkpoint)
            print(
                f"  ✓ Batch {batch_num}: {len(flattened_successful)} successful, "
                f"{len(flattened_failed)} failed"
            )
        else:
            checkpoint['pending'] = None
            _save_generation_checkpoint(checkpoint_file, checkpoint)
            print(f"  ✗ Batch {batch_num} failed to complete")

    return all_successful, all_failed


def initialize_openai_client(env_path: Optional[str] = None) -> OpenAI:
    """Initialize OpenAI client with API key from environment (requires internet)."""
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path)

    ensure_openai_network_access()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)
