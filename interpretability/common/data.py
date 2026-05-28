import numpy as np
from sklearn import model_selection

from data.loaders import load_benchmark, zero_pad_channels, get_subdir

from models.LaBraM.utils import get_input_chans
from models.NeuroRVQm.modules import ch_names_global


BENCHMARK_CLASSES = {
    'Physionet MI': {0: 'hands', 1: 'feet', 2: 'left hand', 3: 'right hand'},
    'Physionet ME': {0: 'hands', 1: 'feet', 2: 'left hand', 3: 'right hand'},
    'Physionet Eyes': {0: 'eye_open', 1: 'eye_closed'},
    'Sleep EDF': {0: 'W', 1: 'S1', 2: 'S2', 3: 'S3', 4: 'S4', 5: 'R'},
    'High Gamma': {0: 'no_action', 1: 'left_fist', 2: 'right_fist', 3: 'both_feet'},
    'Pavlov memory': {0: 'control', 1: 'memory'},
    'KU MI': {0: 'left', 1: 'right'},
    'KU ERP': {0: 'nontarget', 1: 'target'},
}


def build_test_set(X, subject_ids, y, fold: int, n_folds: int = 10):
    """Replicate the subject-level KFold split used during training.

    Returns X_test, y_test, subject_ids_test.
    """
    sbj_unique = np.unique(subject_ids)
    kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=99)
    splits = list(kf.split(sbj_unique))
    _, sbj_idx_test = splits[fold]
    mask = np.isin(subject_ids, sbj_unique[sbj_idx_test])
    return X[mask], y[mask], subject_ids[mask]


def resolve_channels(model_name, eval_ch_names):
    """Pick the subset of channels that map back to a topomap for this model.

    Returns a dict with at minimum `plot_ch_names`.  Adapters that use a
    learned channel subset (NeuroRVQ, LaBraM) also expose the mask + (for
    NeuroRVQ) the names; callers pass these through to the
    per-model run functions.
    """
    if model_name in ('REVE', 'CBraMod', 'BrainOmni', 'EEGNet'):
        return {'plot_ch_names': list(eval_ch_names)}

    if model_name == 'NeuroRVQ':
        ch_names_enc = np.array([cn.lower().encode() for cn in eval_ch_names])
        ch_mask = np.isin(ch_names_enc, ch_names_global)
        plot_ch_names = [cn for cn, keep in zip(eval_ch_names, ch_mask) if keep]
        ch_names_masked = ch_names_enc[ch_mask]
        return {
            'plot_ch_names': plot_ch_names,
            'ch_mask': ch_mask,
            'ch_names_masked': ch_names_masked,
        }

    if model_name == 'LaBraM':
        input_chans, ch_mask = get_input_chans(eval_ch_names)
        plot_ch_names = [cn for cn, keep in zip(eval_ch_names, ch_mask) if keep]
        return {
            'plot_ch_names': plot_ch_names,
            'ch_mask': ch_mask,
            'input_chans': input_chans,
        }

    raise ValueError(f"resolve_channels: unknown model '{model_name}'")


def load_data_with_augmentation(model_name, benchmark_name, data_root, augmentation=None):
    """Load benchmark; apply augmentation if requested.

    Channel-reducing augmentations (e.g. channel_dropout) get zero-padded back
    to the clean channel layout so the model sees the trained input shape.

    Returns ``(X, subject_ids, y, ch_names, _)``.  The trailing ``None`` slot
    is retained so callers can keep their 5-tuple unpack — kept for now to
    leave a hook in case per-model eval-time channel routing comes back.
    """
    subdir = get_subdir(model_name)
    clean_bm = load_benchmark(benchmark_name, data_root, subdir=subdir, apply_car=True)
    X, subject_ids, y, ch_names = clean_bm.get_data()
    X = np.array(X, dtype=np.float64)

    if augmentation is not None:
        aug_bm = load_benchmark(benchmark_name, data_root, subdir=subdir,
                                apply_car=True, augmentation=augmentation)
        X_aug, _, _, aug_ch_names = aug_bm.get_data()
        X_aug = np.array(X_aug, dtype=np.float64)
        if len(aug_ch_names) < len(ch_names):
            X = zero_pad_channels(X_aug, aug_ch_names, ch_names)
            print(f'  Zero-padded {len(aug_ch_names)} → {len(ch_names)} channels')
        else:
            X = X_aug

    return X, subject_ids, y, ch_names, None
