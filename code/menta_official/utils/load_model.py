import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

from utils.env_config import normalize_cache_dir



def load_generator_model(
    model_name: str,
    use_gpu: bool = False,
    cache_dir: str = None,
    accelerator: Optional[Accelerator] = None
):
    """
    Load transformers model for query generation.
    
    Args:
        model_name: HuggingFace model name or path
        use_gpu: Whether to use GPU (determines dtype)
        cache_dir: Optional cache directory for offline loading
        accelerator: Optional Accelerator instance for multi-GPU setup
    
    Returns:
        tokenizer, model (model will be on accelerator.device if accelerator provided)
    """
    print(f"Loading generation model: {model_name}")

    cache_dir = normalize_cache_dir(cache_dir)
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
                )
                model_loaded = True
            else:
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                                   if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshot_dirs:
                        model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                        print(f"Loading from HF cache (snapshot): {model_path}")
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if use_gpu else torch.float32,
                        )
                        model_loaded = True
        
        if not model_loaded:
            raise FileNotFoundError(f"Model '{model_name}' not found in cache directory: {cache_dir}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_gpu else torch.float32,
        )
    
    # Handle padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            print("Added new pad_token: [PAD]")
    
    tokenizer.padding_side = 'left'
    print(f"Set padding_side to 'left' for decoder-only model")
    print(accelerator)
    # Move model to device if accelerator is provided
    if accelerator is not None:
        model = model.to(accelerator.device)
        model.eval()
        print(f"Model moved to device: {accelerator.device} (process {accelerator.process_index}/{accelerator.num_processes})")
    elif use_gpu:
        model = model.to('cuda')
        model.eval()
        print(f"Model moved to CUDA")
    else:
        model.eval()
    
    return tokenizer, model

def load_embedding_model(
    model_name: str,
    use_gpu: bool = False,
    cache_dir: str = None
) -> SentenceTransformer:
    """Load sentence transformer model for retrieval."""
    print(f"Loading retrieval model: {model_name}")

    cache_dir = normalize_cache_dir(cache_dir)
    model_loaded = False
    if cache_dir:
        model_name_safe = model_name.replace('/', '--')
        model_cache_path = os.path.join(cache_dir, f"models--{model_name_safe}")
        
        if os.path.exists(model_cache_path):
            if os.path.exists(os.path.join(model_cache_path, 'config.json')):
                print(f"Loading from HF cache (direct): {model_cache_path}")
                model = SentenceTransformer(model_cache_path, device='cuda' if use_gpu else 'cpu')
                model_loaded = True
            else:
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                                   if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshot_dirs:
                        model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                        print(f"Loading from HF cache (snapshot): {model_path}")
                        model = SentenceTransformer(model_path, trust_remote_code=True, 
                                                   device='cuda' if use_gpu else 'cpu')
                        model_loaded = True
        
        if not model_loaded:
            raise FileNotFoundError(f"Model '{model_name}' not found in cache directory: {cache_dir}")
    else:
        model = SentenceTransformer(model_name, device='cuda' if use_gpu else 'cpu')
    
    if use_gpu and model.device.type != 'cuda':
        model = model.to('cuda')
    
    return model