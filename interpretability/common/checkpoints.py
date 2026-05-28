import os
import json

import torch
import torch.nn as nn

from data.loaders import slug_for

# Repo root: interpretability/common/checkpoints.py → root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hardcoded batch sizes for EEGNet (consistent with main.py)
EEGNET_BATCH_SIZES = {
    "Sleep EDF": 128,
    "Physionet Eyes": 128,
    "Physionet MI": 128,
    "Physionet ME": 128,
    "High Gamma": 128,
    "KU ERP": 128,
    "Pavlov memory": 128,
    "KU MI": 128,
}


def unwrap_model(wrapper):
    """Extract the underlying nn.Module from a wrapper."""
    # 1. If the wrapper has a 'model' attribute, drill down
    if hasattr(wrapper, 'model'):
        obj = wrapper.model
        # 2. Check if this object has a nested 'model' attribute that is an nn.Module
        if hasattr(obj, 'model') and isinstance(obj.model, nn.Module):
            return obj.model
        # 3. If not nested, check if the object itself is an nn.Module
        if isinstance(obj, nn.Module):
            return obj

    if isinstance(wrapper, nn.Module):
        return wrapper
    raise ValueError(f"Could not extract nn.Module from {type(wrapper)}")


def get_lrp_batch_size(model_name, benchmark_name, n_chans, n_times, n_outputs, ch_names):
    """Determine batch size for LRP (approx 1/3 of max inference batch size)."""
    if not torch.cuda.is_available():
        return 32

    # Check hardcoded defaults for EEGNet first
    if model_name == "EEGNet":
        bs = EEGNET_BATCH_SIZES.get(benchmark_name, 128)
        lrp_bs = max(1, int(bs // 3))
        print(f"Using hardcoded LRP batch size for EEGNet: {lrp_bs} (derived from {bs})")
        return lrp_bs

    if model_name == "NeuroRVQ":
        print(f"Using hardcoded LRP batch size for NeuroRVQ: 32")
        return 32

    if model_name == "BrainOmni":
        print(f"Using hardcoded LRP batch size for BrainOmni: 32")
        return 32

    if model_name == "LaBraM":
        print(f"Using hardcoded LRP batch size for LaBraM: 100")
        return 100

    gpu_name = torch.cuda.get_device_name(0)
    json_path = os.path.join(_ROOT, "models", "batch_sizes", f"max_batch_sizes_{gpu_name}.json")

    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            pass

    bs = data.get(model_name, {}).get(benchmark_name)

    if bs is None:
        print(f"Batch size not found in {json_path}. Using safe default of 64.")
        bs = 64

    if isinstance(bs, str) or bs is None:
        print(f"Warning: Could not determine batch size (error: {bs}). Defaulting to 32.")
        return 32

    lrp_bs = max(1, int(bs // 3))
    print(f"Using LRP batch size: {lrp_bs} (Max found: {bs})")
    return lrp_bs


def get_ckpt_path(model_name: str, benchmark_name: str, fold: int,
                  large_head: bool = False, train_head_only: bool = False,
                  skip_tokenizer: bool = False) -> str:
    bench_dir = slug_for(benchmark_name)
    finetune_type = "head" if train_head_only else "full"
    if large_head:
        finetune_type += "_large_head"
    if skip_tokenizer:
        finetune_type += "_skip_tokenizer"
    return os.path.join(
        'weights', 'finetuned', model_name, bench_dir,
        f'{finetune_type}_train-clean_fold{fold}_best.pt'
    )


def load_fold_model(wrapper, model_name, benchmark_name, fold,
                    *, large_head=False, train_head_only=False, skip_tokenizer=False):
    """Load a fold checkpoint into ``wrapper`` and return the unwrapped model.

    Returns ``None`` (after printing) when the checkpoint is missing, so the
    caller can ``continue`` to the next fold.
    """
    ckpt_path = get_ckpt_path(
        model_name, benchmark_name, fold,
        large_head=large_head, train_head_only=train_head_only,
        skip_tokenizer=skip_tokenizer,
    )
    if not os.path.exists(ckpt_path):
        print(f'  Checkpoint not found: {ckpt_path}. Skipping.')
        return None
    print(f'  Loading checkpoint from {ckpt_path}...')
    wrapper.load_model(ckpt_path)
    return unwrap_model(wrapper).cuda().eval()
