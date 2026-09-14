"""
Resolve HuggingFace cache paths from `.env` / environment variables.

Pipeline scripts source `.env` and pass `--cache_dir "${HF_HUB_CACHE_DIR}"`.
Python entry points call `normalize_cache_dir()` so direct `python ...` runs
also honor `HF_HOME` / `HF_HUB_CACHE` when the CLI still has a legacy default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Placeholder defaults in argparse that should be replaced by .env when present.
_LEGACY_CACHE_DEFAULTS = frozenset(
    {
        os.path.normpath(os.path.expanduser("~/.cache/huggingface/hub")),
        os.path.normpath("../hf_cache/hub"),
        os.path.normpath("../hf_cache/hub/"),
    }
)


def _expand_path(value: str) -> str:
    return os.path.normpath(os.path.expanduser(os.path.expandvars(value.strip())))


def load_repo_dotenv(env_path: Optional[str] = None) -> None:
    """Load repo `.env` if present (no-op when python-dotenv missing)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    if env_path is None:
        candidates = [_REPO_ROOT / ".env", Path(".env")]
    else:
        p = Path(env_path)
        candidates = [p if p.is_absolute() else _REPO_ROOT / p]

    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            return


def resolve_hf_hub_cache_dir() -> Optional[str]:
    """
  Return the HuggingFace hub cache directory from the environment.

  Precedence: HF_HUB_CACHE → TRANSFORMERS_CACHE → ${HF_HOME}/hub
  """
    load_repo_dotenv()

    for key in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(key)
        if value:
            return _expand_path(value)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return _expand_path(os.path.join(hf_home, "hub"))

    return None


def normalize_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    """
    Resolve `cache_dir` for offline/local model loading.

    - Expands `~` and env vars.
    - Replaces legacy argparse defaults with paths from `.env` when set.
    """
    load_repo_dotenv()

    if cache_dir:
        expanded = _expand_path(cache_dir)
        if expanded not in _LEGACY_CACHE_DEFAULTS:
            return expanded
    else:
        expanded = None

    resolved = resolve_hf_hub_cache_dir()
    if resolved:
        return resolved

    if expanded is not None:
        return expanded

    return os.path.normpath(os.path.expanduser("~/.cache/huggingface/hub"))


def apply_hf_offline_flags_from_env() -> None:
    """Apply HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE from `.env` when set."""
    load_repo_dotenv()
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_TOKEN"):
        value = os.environ.get(key)
        if value is not None and value != "":
            os.environ[key] = value


def ensure_hf_network_access() -> None:
    """
    Allow HuggingFace Hub downloads (BEIR dataset fetch in process_dataset.py).

    `.env` may set HF_HUB_OFFLINE=1 for attack pipelines; data prep must reach the Hub.
    """
    load_repo_dotenv()
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
