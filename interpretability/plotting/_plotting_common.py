import os

import matplotlib.pyplot as plt
import mne
import numpy as np


def mne_info(ch_names, sfreq=200.0):
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    try:
        info.set_montage("standard_1005", on_missing="ignore", match_case=False)
    except Exception:
        info.set_montage("standard_1020", on_missing="ignore", match_case=False)
    return info


def valid_channel_indices(info):
    valid = []
    for i, ch in enumerate(info["chs"]):
        loc = ch["loc"][:3]
        if not (np.all(np.isnan(loc)) or np.all(loc == 0)):
            valid.append(i)
    return valid


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
