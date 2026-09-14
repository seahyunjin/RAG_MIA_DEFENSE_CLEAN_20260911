import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import argparse
import vec2text


# Define your base paths
MODEL_PATH = "~/.cache/huggingface/hub"


def download_model(name: str, task: str = "text-generation"):
    """
    Load a model from Hugging Face or local cache.
    If the model is not found locally, it will be downloaded and saved.
    Args:
        name (str): The model name or path.
        task (str): The task for the model ('embedding' or 'text-generation').
    Returns:
        model: The loaded model.
    """

    print(f"🔍 Loading model '{name}' for task '{task}'...")

    if task == "embedding":
        model_dir = os.path.join(MODEL_PATH, name)
        
        # Special handling for vec2text models
        if name in ["ielabgroup/vec2text_gtr-base-st_inversion", 
                    "ielabgroup/vec2text_corrector_gtr-base-st"]:
            
            # First, ensure the base embedder is downloaded
            embedder_path = "sentence-transformers/gtr-t5-base"
            embedder_dir = os.path.join(MODEL_PATH, embedder_path)
            
            print(f"📥 Ensuring base embedder '{embedder_path}' is available...")
            if not os.path.exists(embedder_dir):
                print(f"📥 Downloading embedder '{embedder_path}'...")
                try:
                    embedder = SentenceTransformer(embedder_path, device='cpu')
                    os.makedirs(embedder_dir, exist_ok=True)
                    embedder.save(embedder_dir)
                    print(f"✅ Saved embedder to {embedder_dir}")
                except Exception as e:
                    print(f"❌ Error downloading embedder: {e}")
                    raise
            else:
                print(f"✅ Embedder already exists at {embedder_dir}")
            
            # Now load the vec2text models
            if not os.path.exists(model_dir):
                print(f"📥 Downloading vec2text model '{name}'...")
                os.makedirs(model_dir, exist_ok=True)
                
                try:
                    # Use the official vec2text loading method
                    if name == "ielabgroup/vec2text_gtr-base-st_inversion":
                        # Load using the corrector which handles both models
                        print("ℹ️  Loading via vec2text.load_pretrained_corrector...")
                        corrector = vec2text.load_pretrained_corrector("gtr-base")
                        print(f"✅ Loaded vec2text corrector for gtr-base")
                        return corrector
                    
                except Exception as e:
                    print(f"❌ Error loading vec2text model: {e}")
                    print(f"ℹ️  Trying alternative loading method...")
                    
                    # Alternative: Download the components separately
                    try:
                        # Set environment variable to force CPU loading
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        
                        if name == "ielabgroup/vec2text_gtr-base-st_inversion":
                            model = vec2text.models.InversionModel.from_pretrained(
                                name,
                                cache_dir="~/.cache/huggingface/hub",
                            )
                        elif name == "ielabgroup/vec2text_corrector_gtr-base-st":
                            model = vec2text.models.CorrectorEncoderModel.from_pretrained(
                                name,
                                cache_dir="~/.cache/huggingface/hub",
                            )
                        
                        model.save_pretrained(model_dir)
                        print(f"✅ Saved model to {model_dir}")
                    except Exception as e2:
                        print(f"❌ Alternative method also failed: {e2}")
                        raise
            else:
                print(f"✅ Found local model at {model_dir}")
        
        else:
            # Regular embedding model loading
            if not os.path.exists(model_dir):
                print(f"📥 Model not found locally. Downloading '{name}' to {model_dir}...")
                model = SentenceTransformer(name, device='cpu')
                os.makedirs(model_dir, exist_ok=True)
                model.save(model_dir)
            else:
                print(f"✅ Found local model at {model_dir}")

    elif task == "text-generation":
        model_dir = os.path.join(MODEL_PATH, name)

        if not os.path.exists(model_dir):
            print(f"📥 Model not found locally. Downloading '{name}' to {model_dir}...")
            
            # Check model config first to determine the correct auto class
            try:
                config = AutoConfig.from_pretrained(name)
                print(f"📋 Model type: {config.model_type}")
                
                tokenizer = AutoTokenizer.from_pretrained(name)
                
                # Try AutoModelForCausalLM first, fall back to AutoModel if it fails
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        name, torch_dtype=torch.float16, device_map="auto"
                    )
                except ValueError as e:
                    print(f"⚠️  Cannot load as causal LM: {e}")
                    print(f"ℹ️  Loading as generic AutoModel instead...")
                    model = AutoModel.from_pretrained(
                        name, torch_dtype=torch.float16, device_map="auto"
                    )
                
                os.makedirs(model_dir, exist_ok=True)
                tokenizer.save_pretrained(model_dir)
                model.save_pretrained(model_dir)
            except Exception as e:
                print(f"❌ Error downloading model: {e}")
                raise
        else:
            print(f"✅ Found local model at {model_dir}")

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument(
        "--task",
        type=str,
        choices=["embedding", "text-generation"],
        help="Task for the model",
    )

    args = parser.parse_args()
    download_model(args.name, args.task)
