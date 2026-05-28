"""
Find maximum batch sizes for each model x benchmark combination.
Uses random tensors matching benchmark shapes instead of loading real data.
"""

import gc
import json
from pathlib import Path

import numpy as np
import skorch
import torch

from models.wrappers import get_model



# EEGNet gets slower with larger batch sizes; pinned it here.
_EEGNET_BATCH_SIZES = {
    "Sleep EDF": 128, "Physionet Eyes": 128, "Physionet MI": 128,
    "Physionet ME": 128, "High Gamma": 128, "KU ERP": 128,
    "Pavlov memory": 128, "KU MI": 128,
}

BENCHMARKS = {
    "Physionet Eyes": {
        "n_chans": 64, "n_times": 800, "n_classes": 2, "sfreq": 200,
        "ch_names": [
            'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
            'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7',
            'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
            'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ',
        ],
    },
     "Physionet MI": {
        "n_chans": 64, "n_times": 800, "n_classes": 4, "sfreq": 200,
        "ch_names": [
            'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
            'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7',
            'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
            'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ',
        ],
    },
      "Physionet ME": {
        "n_chans": 64, "n_times": 800, "n_classes": 4, "sfreq": 200,
        "ch_names": [
            'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
            'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7',
            'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
            'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ',
        ],
    },
    "High Gamma": {
        "n_chans": 78, "n_times": 800, "n_classes": 4, "sfreq": 200,
        "ch_names": [
            'FP1', 'FP2', 'FPZ', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1',
            'T7', 'C3', 'CZ', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'PZ', 'P4',
            'P8', 'POZ', 'O1', 'OZ', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
            'FC3', 'FCZ', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPZ', 'CP4', 'P5', 'P1', 'P2',
            'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9',
            'FT10', 'TPP9H', 'TPP10H', 'PO9', 'PO10', 'P9', 'P10', 'AFZ', 'IZ', 'FTT9H',
            'FTT10H', 'TTP7H', 'TPP8H',
        ],
    },
    "KU ERP": {
        "n_chans": 62, "n_times": 200, "n_classes": 2, "sfreq": 200,
        "ch_names": [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3',
            'CZ', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ', 'P4',
            'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3',
            'CPZ', 'CP4', 'P1', 'P2', 'POZ', 'FT9', 'FTT9H', 'TTP7H', 'TP7', 'TPP9H', 'FT10',
            'FTT10H', 'TPP8H', 'TP8', 'TPP10H', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3',
            'PO4',
        ],
    },
    "Pavlov memory": {
        "n_chans": 63, "n_times": 800, "n_classes": 2, "sfreq": 200,
        "ch_names": [
            'FP1', 'FZ', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'PZ',
            'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'CZ', 'C4', 'T8',
            'FT10', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3', 'AFZ', 'F1', 'F5', 'FT7',
            'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6',
            'P2', 'CPZ', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2',
        ],
    },
    "Sleep EDF": {
        "n_chans": 2, "n_times": 6000, "n_classes": 6, "sfreq": 200,
        "ch_names": ['FPZ', 'PZ'],
    },
    "KU MI": {
        "n_chans": 62, "n_times": 800, "n_classes": 2, "sfreq": 200,
        "ch_names": [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3',
            'CZ', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ', 'P4',
            'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3',
            'CPZ', 'CP4', 'P1', 'P2', 'POZ', 'FT9', 'FTT9H', 'TTP7H', 'TP7', 'TPP9H', 'FT10',
            'FTT10H', 'TPP8H', 'TP8', 'TPP10H', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3',
            'PO4',
        ],
    },
}


BATCH_SIZES_TO_TRY = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
VAL_SIZE = 64  # fixed small validation set



def find_max_batch_size(model_name, bench_name, bench_cfg):
    max_bs = 0

    for bs in BATCH_SIZES_TO_TRY:
        torch.cuda.empty_cache()
        gc.collect()

        try:
            n_train = max(bs, 64)
            sfreq = 256 if model_name == "BrainOmni" else 200
            n_times = int(bench_cfg["n_times"] * sfreq / 200)

            X = np.random.randn(n_train + VAL_SIZE, bench_cfg["n_chans"], n_times).astype(np.float32)
            y = np.random.randint(0, bench_cfg["n_classes"], n_train + VAL_SIZE).astype(np.int64)

            train_ds = skorch.dataset.Dataset(X[:n_train], y[:n_train])
            val_ds = skorch.dataset.Dataset(X[n_train:], y[n_train:])

            model = get_model(
                model_name,
                n_chans=bench_cfg["n_chans"],
                ch_names=bench_cfg["ch_names"],
                sfreq=sfreq,
                n_times=n_times,
                n_outputs=bench_cfg["n_classes"],
                sbj_ids=np.arange(n_train),
            )

            model.fit(train_ds, val_ds, batch_size=bs, epochs=1)
            max_bs = bs
            print(f"  batch_size={bs} OK")

            del model, X, y, train_ds, val_ds

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"  batch_size={bs} OOM")
                break
            else:
                print(f"  batch_size={bs} ERROR: {e}")
                return str(e)

    return max_bs



def make_batch_size_resolver(models, benchmarks, run_finder: bool):
    """Return a (model, benchmark) → batch_size lookup, filling cache misses if asked."""
    gpu = torch.cuda.get_device_name(0)
    cache_path = Path("models/batch_sizes") / f"max_batch_sizes_{gpu}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    missing = [(m, b) for m in models if m != "EEGNet"
               for b in benchmarks if b not in cache.get(m, {})]
    if missing and run_finder:
        for m, b in missing:
            print(f"Finding max batch size for {m} on {b}...")
            cache.setdefault(m, {})[b] = find_max_batch_size(m, b, BENCHMARKS[b])
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"Batch size cache updated: {cache_path}")
    elif missing:
        print(f"WARNING: missing batch sizes for {missing} in {cache_path}")

    def resolve(m, b):
        if m == "EEGNet":
            return _EEGNET_BATCH_SIZES[b]
        bs = cache.get(m, {}).get(b)
        if not isinstance(bs, int):
            raise ValueError(f"No valid batch size for {m} x {b} in {cache_path}: {bs!r}")
        return bs
    return resolve
