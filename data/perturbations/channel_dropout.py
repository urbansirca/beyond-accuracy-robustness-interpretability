import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from data.loaders import Benchmark


import mne
import numpy as np
import matplotlib.pyplot as plt



REGION_COLORS = {
    "anterior": "#e41a1c",  # red
    "frontal_left": "#377eb8",  # blue
    "frontal_mid": "#4daf4a",  # green
    "frontal_right": "#984ea3",  # purple
    "frontotemporal_left": "#ff7f00",  # orange
    "frontotemporal_right": "#ffff33",  # yellow
    "central_left": "#a65628",  # brown
    "central_mid": "#f781bf",  # pink
    "central_right": "#999999",  # gray
    "centroparietal_left": "#66c2a5",  # teal
    "centroparietal_mid": "#fc8d62",  # salmon
    "centroparietal_right": "#8da0cb",  # light blue
    "parietal_left": "#e78ac3",  # light pink
    "parietal_mid": "#a6d854",  # lime
    "parietal_right": "#ffd92f",  # gold
    "posterior_left": "#e5c494",  # tan
    "posterior_mid": "#b3b3b3",  # silver
    "posterior_right": "#1b9e77",  # dark teal
    "other": "#000000",  # black
}


def drop_channels_randomly(ch_names, drop_prob, seed=None):
    rng = np.random.default_rng(seed)
    dropped_mask = rng.uniform(size=len(ch_names)) > drop_prob
    return dropped_mask


def create_region_dropout_mask(ch_names, regions_to_drop):
    """Create a mask where channels in specified regions are dropped (False)."""
    dropped_mask = []
    for ch in ch_names:
        region = map_channel_to_region(ch.upper())
        keep = region not in regions_to_drop
        dropped_mask.append(keep)
    return np.array(dropped_mask)


def create_random_dropout_datasets(
    benchmark: Benchmark,
    drop_prob=0.5,
    seed=None,
    save=False,
    plot=False,
    overwrite=False
    ):
    """
    Create random dropout datasets where each channel is independently dropped with a specified probability.
    Dropped channels are removed from the dataset entirely.
    """

    if not overwrite and benchmark.check_augmented_exists(f"dropout_random_p{int(drop_prob*100)}"):
        print(f"Augmented data with random dropout (p={drop_prob}) already exists. Skipping augmentation.")
        return

    X, sbj_id, y, ch_names = benchmark.get_data()
    ch_names_upper = [ch.upper() for ch in ch_names]
    dropped_mask = drop_channels_randomly(ch_names, drop_prob, seed)
    ch_names_kept = [ch for ch, keep in zip(ch_names_upper, dropped_mask) if keep]

    X_dropped = X[:, dropped_mask, :]
    n_dropped = np.sum(~dropped_mask)
    print(f"{benchmark.name} - Random Dropout:")
    print(f"  Drop probability: {drop_prob}")
    print(f"  Dropped {n_dropped}/{len(ch_names)} channels: {[ch for ch, keep in zip(ch_names_upper, dropped_mask) if not keep]}")
    
    fig = plot_dropout(
        ch_names_upper,
        dropped_mask,
        title=f"{benchmark.name} - Random Dropout (p={drop_prob}, {n_dropped}/{len(ch_names)} dropped)",
    )
    if plot:
        plt.show()
    
    metadata = {
            "drop_prob": drop_prob,
            "n_dropped": int(n_dropped),
            "n_total": len(ch_names),
            "dropped_channels": [ch for ch, keep in zip(ch_names_upper, dropped_mask) if not keep],
            "kept_channels": ch_names_kept,
        }
    if save:
        augmentation_name = f"dropout_random_p{int(drop_prob*100)}"
        benchmark.save_augmented_data(X_dropped, augmentation_name, metadata=metadata, figure=fig)
        print(f"  Saved as: {augmentation_name}")
    print("metadata:", metadata)
    

def create_region_dropout_datasets(
    benchmark: Benchmark,
    regions=["Primary", "Control"],
    save=False,
    plot=False,
    overwrite=False
):
    """
    Create dropout datasets for primary, secondary, and control regions.
    Channels in the specified regions are fully removed from the dataset.

    Args:
        batch_size: if specified, process data in batches to reduce memory usage
        benchmark: Name of benchmark
        root: Root directory containing benchmark folders
        subdir: Preprocessing variant subfolder
        save: Whether to save the augmented datasets to disk
        plot: Whether to plot the dropout pattern for each condition

    Returns:
        dict with keys like "primary", "secondary", "control", each containing:
            - X_dropped: data with dropped channels removed
            - ch_names_kept: list of remaining channel names
            - dropped_mask: boolean mask (True=keep, False=dropped)
            - n_dropped: number of channels dropped
            - n_total: total number of channels in benchmark
            - dropped_channels: list of dropped channel names
    """
    config = BENCHMARK_REGION_CONFIG.get(benchmark.name)
    if config is None:
        raise ValueError(f"No region config for benchmark: {benchmark.name}")
        

    X, sbj_id, y, ch_names = benchmark.get_data()
    ch_names_upper = [ch.upper() for ch in ch_names]

    results = {}

    for region in regions:
        
        if not overwrite and benchmark.check_augmented_exists(f"dropout_region_{region}"):
            print(f"Augmented data with region dropout ({region}) already exists. Skipping augmentation for this region.")
            continue
        

        regions_to_drop = config.get(region)
        if regions_to_drop is None:
            continue

        dropped_mask = create_region_dropout_mask(ch_names, regions_to_drop)

        # Remove dropped channels entirely
        ch_names_kept = [ch for ch, keep in zip(ch_names_upper, dropped_mask) if keep]

        X_dropped = X[:, dropped_mask, :]

        # Count dropped channels
        n_dropped = np.sum(~dropped_mask)
        dropped_channels = [
            ch for ch, keep in zip(ch_names_upper, dropped_mask) if not keep
        ]

        results[region] = {
            "benchmark": benchmark.name,
            "region": region,
            "kept_channels": ch_names_kept,
            "dropped_mask": dropped_mask.tolist(),
            "n_dropped": int(n_dropped),
            "n_total": len(ch_names),
            "dropped_channels": dropped_channels,
            "regions": regions_to_drop,
        }

        print(f"{benchmark.name} - {region}:")
        print(f"  Regions: {regions_to_drop}")
        print(f"  Dropped {n_dropped}/{len(ch_names)} channels: {dropped_channels}")
        print(f"  Remaining shape: {X_dropped.shape}")
        
        print("results[region]:")
        print(results[region])

        fig = plot_dropout(
            ch_names_upper,
            dropped_mask,
            title=f"{benchmark.name} - {region} ({n_dropped}/{len(ch_names)} dropped)",
        )
        if plot:
            plt.show()

        if save:
            augmentation_name = f"dropout_region_{region}"
            benchmark.save_augmented_data(
                X_dropped, augmentation_name, metadata=results[region], figure=fig)
            print(f"  Saved as: {augmentation_name}")


    return results, ch_names, benchmark


def plot_dropout(channels, dropped_mask, title):
    channels = [ch.upper() for ch in channels]
    info = mne.create_info(ch_names=channels, sfreq=200, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, match_case=False, on_missing="warn")

    # Color by region
    colors = [REGION_COLORS[map_channel_to_region(ch)] for ch in channels]

    fig = mne.viz.plot_sensors(info, kind="topomap", show_names=True, title=title)

    ax = fig.axes[0]
    # Find the PathCollection with sensor points (has same count as channels)
    sensor_positions = None
    for coll in ax.collections:
        if len(coll.get_offsets()) == len(channels):
            coll.set_color(colors)
            sensor_positions = coll.get_offsets()
            break

    # Add X markers for dropped channels
    if sensor_positions is not None:
        for i, (pos, keep) in enumerate(zip(sensor_positions, dropped_mask)):
            if not keep:
                ax.scatter(
                    pos[0], pos[1], marker="x", s=200, c="red", linewidths=3, zorder=10
                )
                

    return fig


def map_channel_to_region(ch: str) -> str:
    ch = ch.upper()

    if ch in {"FP1", "FPZ", "FP2", "AF3", "AF4", "AF7", "AF8", "AFZ"}:
        return "anterior"

    if ch in {"F7", "F5", "F3", "F9"}:
        return "frontal_left"
    if ch in {"F1", "FZ", "F2", "FC1", "FCZ", "FC2"}:
        return "frontal_mid"
    if ch in {"F4", "F6", "F8", "F10"}:
        return "frontal_right"

    if ch in {"FT7", "FT9", "FC5", "FC3", "FTT9H"}:
        return "frontotemporal_left"
    if ch in {"FC4", "FC6", "FT8", "FT10", "FTT10H"}:
        return "frontotemporal_right"

    if ch in {"T7", "T9", "C5", "C3"}:
        return "central_left"
    if ch in {"C1", "CZ", "C2"}:
        return "central_mid"
    if ch in {"C4", "C6", "T8", "T10"}:
        return "central_right"

    if ch in {"TP7", "TP9", "CP5", "CP3", "TTP7H", "TPP9H"}:
        return "centroparietal_left"
    if ch in {"CP1", "CPZ", "CP2"}:
        return "centroparietal_mid"
    if ch in {"CP4", "CP6", "TP8", "TP10", "TPP8H", "TPP10H"}:
        return "centroparietal_right"

    if ch in {"P7", "P5", "P3", "P9"}:
        return "parietal_left"
    if ch in {"P1", "PZ", "P2"}:
        return "parietal_mid"
    if ch in {"P4", "P6", "P8", "P10"}:
        return "parietal_right"

    if ch in {"PO7", "PO5", "PO9", "CB1"}:
        return "posterior_left"
    if ch in {"PO3", "POZ", "PO4", "O1", "OZ", "O2", "IZ"}:
        return "posterior_mid"
    if ch in {"PO6", "PO8", "PO10", "CB2"}:
        return "posterior_right"

    return "other"


BENCHMARK_REGION_CONFIG = {
    "Physionet Eyes": {
        "primary": [
            "anterior",
            
            "posterior_mid",       
            "posterior_left",
            "posterior_right",
        ],
        "secondary": [
            "parietal_left",          # P7, P5, P3, P9
            "parietal_mid",           # P1, PZ, P2
            "parietal_right",         # P4, P6, P8, P10
            "posterior_left",         # PO7, PO5, PO9
            "posterior_right",        # PO6, PO8, PO10
        ],
        "control": [
            "frontal_left",           # F7, F5, F3
            "frontal_mid",            # F1, FZ, F2, FC1, FCZ, FC2
            "frontal_right",          # F4, F6, F8
            "central_mid",            # C1, CZ, C2
            "central_left",           # T7, T9, C5, C3
            "central_right",          # C4, C6, T8, T10
        ],
    },
    "Physionet MI": {
        "primary": [
            "central_left",
            "central_right",
            "central_mid",
            "centroparietal_left",
            "centroparietal_right",
        ],
        "control": [
            "anterior",
            "posterior_mid",
        ],
    },
    "Physionet ME": {
        "primary": [
            "central_left",
            "central_right",
            "central_mid",
            "centroparietal_left",
            "centroparietal_right",
        ],
        "secondary": [
            "frontotemporal_left",    # FC5, FC3, FT7, FT9
            "frontotemporal_right",   # FC4, FC6, FT8, FT10
        ],
        "control": [
            "posterior_mid",
            "anterior",
        ],
    },
    "KU MI": {
        "primary": [
            "central_left",
            "central_right",
            "central_mid",
            "centroparietal_left",
            "centroparietal_right",
        ],
        "control": [
            "anterior",
            "posterior_mid",
        ],
    },
    "KU ERP": {
        "primary": [
            "posterior_mid"
            
        ],
        "secondary": [
            "parietal_left",
            "parietal_right",
            "frontal_mid",
        ],
        "control": [
            "anterior",
            "posterior_left",
            "posterior_right",
        ],
    },
    "High Gamma": {
        "primary": [
            "central_left",
            "central_right",
            "central_mid",
            "centroparietal_left",
            "centroparietal_right",
        ],
        "secondary": [
            "frontotemporal_left",
            "frontotemporal_right",
        ],
        "control": [
            "posterior_mid",
            "anterior",
        ],
    },
    "Pavlov memory": {
        "primary": [
            "frontal_mid",
            "frontal_left",
            "frontal_right",
        ],
        "secondary": [
            "parietal_mid",
            "centroparietal_mid",
        ],
        "control": [
            "posterior_mid",
            "central_left",
            "central_right",
        ],
    },
    "Sleep EDF": {
        "primary": [
            "anterior", # FPZ
        ],
        "control": [
            "parietal_mid", # PZ
        ],
    },
}
