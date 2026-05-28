import pdb
import os
import numpy as np
import pandas as pd
from abc import ABC
import json


VARIABLE_CHANNEL_MODELS = {"LaBraM", "NeuroRVQ", "REVE", "BrainOmni"}

def zero_pad_channels(X, src_ch_names, target_ch_names):
    """Zero-pad data to match target channel layout.

    Maps channels from src to their positions in target by name,
    filling missing channels with zeros.

    Args:
        X: array of shape (n_samples, n_src_channels, n_times)
        src_ch_names: channel names present in X
        target_ch_names: full channel list to pad to

    Returns:
        array of shape (n_samples, len(target_ch_names), n_times)
    """
    src_upper = [ch.upper() for ch in src_ch_names]
    target_upper = [ch.upper() for ch in target_ch_names]
    X_padded = np.zeros((X.shape[0], len(target_ch_names), X.shape[2]), dtype=X.dtype)
    for i_target, ch in enumerate(target_upper):
        if ch in src_upper:
            X_padded[:, i_target, :] = X[:, src_upper.index(ch), :]
    return X_padded



def common_average_reference(eeg):
    # Apply common average referencing to signal eeg: (N, C, T)
    return eeg - eeg.mean(axis=-2, keepdims=True)

def get_data_dir(root, main_dir, subdir=None, augmentation=None):
    """
    Returns path to data directory, handling preprocessing subdirs and augmentations.

    Structure: [root]/[main_dir]/[subdir]/[augmented/augmentation_name]
    Examples:
        get_data_dir(root, "PhysionetMI") -> root/PhysionetMI
        get_data_dir(root, "PhysionetMI", "neurogpt_cut") -> root/PhysionetMI/neurogpt_cut
        get_data_dir(root, "PhysionetMI", "neurogpt_cut", "sensor_noise_0db")
            -> root/PhysionetMI/neurogpt_cut/augmented/sensor_noise_0db
    """
    path = os.path.join(root, main_dir)
    if subdir is not None:
        path = os.path.join(path, subdir)
    if augmentation is not None:
        path = os.path.join(path, "augmented", augmentation)
    return path


class AugmentationNotFoundError(Exception):
    """Raised when augmented data doesn't exist."""
    pass

def convert_to_serializable(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class Benchmark(ABC):
    """
    Class for benchmark dataset with expected properties:
        eeg: array of EEG data (samples, channels, time)
        subject_ids: array of subject ID for each data sample (samples,)
        labels: array of target class labels for each data sample (samples,)
        chnames: array of electrode channel names (channels,)
    """
    def __init__(self):
        self.eeg = None
        self.subject_ids = None
        self.labels = None
        self.chnames = None
        self.name = None         # canonical display form (matches YAML / BENCHMARKS key)
        self.slug = None         # snake_case filesystem identifier
        self.eeg_data_name = None
        self.trial_features_name = None
        self.root = None
        self.subdir = None

    def get_data(self):
        return self.eeg, self.subject_ids, self.labels, self.chnames
    

    def sample_balanced_set(self, idx, seed):
        """
        Performs a random sampling of indices to balance classes for each subject
            idx: array of sample indices relative to self.eeg
            seed: random seed for sampling
        Returns:
            filtered indices after random sampling 
        """
        rng = np.random.default_rng(seed)

        subj_all = self.subject_ids[idx]
        y_all = self.labels[idx]

        sampled = []

        for s in np.unique(subj_all):
            mask_s = (subj_all == s)
            idx_s = idx[mask_s]
            y_s = y_all[mask_s]

            labels = np.unique(y_s)

            idx_by_label = [idx_s[y_s == label] for label in labels]

            # minority per subject
            n = min([len(idx_l) for idx_l in idx_by_label])
            if n == 0:
                continue

            take_by_label = [rng.choice(idx_l, size=n, replace=False) for idx_l in idx_by_label]
            sampled.append(np.concatenate(take_by_label))

        sampled_idx = np.concatenate(sampled)
        return sampled_idx
    
    
    def save_augmented_data(self, eeg_augmented, augmentation_name, metadata=None, figure=None):
        """
        Save augmented data to disk in same format as original data.

        Saves to: [root]/[benchmark]/[subdir]/augmented/[augmentation_name]/
        Uses same filenames as original so load_benchmark can load it seamlessly.

        Args:
            eeg_augmented: augmented EEG array (samples, channels, time)
            augmentation_name: e.g. "sensor_noise_0db", "channel_dropout_0.2"
            metadata: optional dictionary with additional metadata to save (e.g. noise parameters), saved as [augmentation_name]_metadata.pd
            figure: optional matplotlib figure to save as [augmentation_name]_figure.png
        """
        save_dir = get_data_dir(self.root, self.slug, self.subdir, augmentation_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save with same filenames as original
        np.save(os.path.join(save_dir, f'{self.eeg_data_name}.npy'), eeg_augmented)

        if metadata is not None:
            # Convert numpy types to native Python types for JSON serialization
            metadata_serializable = convert_to_serializable(metadata)
            # save as json
            with open(os.path.join(save_dir, f'{augmentation_name}_metadata.json'), 'w') as f:
                json.dump(metadata_serializable, f, indent=2)

        # Copy trial features (subject_ids, labels unchanged)
        original_dir = get_data_dir(self.root, self.slug, self.subdir)
        original_tf_path = os.path.join(original_dir, f'{self.trial_features_name}.pd')
        if os.path.exists(original_tf_path):
            trial_features = pd.read_pickle(original_tf_path)

            # For channel dropout augmentations, update channel_names in attrs
            if metadata is not None and 'kept_channels' in metadata:
                kept_channels = metadata['kept_channels']
                # Update attrs with the reduced channel names
                trial_features.attrs['channel_names'] = np.array(kept_channels)
                print(f"  Updated channel_names in trial features: {len(kept_channels)} channels")

            # Save (potentially updated) trial features
            trial_features.to_pickle(os.path.join(save_dir, f'{self.trial_features_name}.pd'))

        if figure is not None:
            figure.savefig(os.path.join(save_dir, f'{augmentation_name}_figure.png'))
        
        print(f"Augmented data saved to: {save_dir}")
        
    def check_augmented_exists(self, augmentation_name):
        """
        Check if augmented data already exists on disk.

        Checks for existence of augmented EEG .npy file in expected path:
        [root]/[benchmark]/[subdir]/augmented/[augmentation_name]/[eeg_data_name].npy

        Args:
            augmentation_name: e.g. "sensor_noise_0db", "channel_dropout_0.2"

        Returns:
            True if augmented data file exists, False otherwise
        """
        aug_dir = get_data_dir(self.root, self.slug, self.subdir, augmentation_name)
        aug_eeg_path = os.path.join(aug_dir, f'{self.eeg_data_name}.npy')
        return os.path.exists(aug_eeg_path)
    
    
    def check_and_save_split(self, fold, train_subjects, test_subjects):
        """
        Save the train/test subject IDs for a given fold to a JSON file if it doesnt yet exist.
        If the file already exists, check that the provided train/test subjects are consistent with the saved ones.

        Saves to: results/splits/benchmark

        Args:
            fold: fold number (e.g. 0, 1, ...)
            train_subjects: list of subject IDs in training set
            test_subjects: list of subject IDs in test set
        """
        splits_dir = os.path.join("data/splits", self.slug)
        os.makedirs(splits_dir, exist_ok=True)
        split_file = os.path.join(splits_dir, f'fold_{fold}_subjects.json')

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                saved_splits = json.load(f)
            saved_train = set(saved_splits['train_subjects'])
            saved_test = set(saved_splits['test_subjects'])
            if set(train_subjects) != saved_train or set(test_subjects) != saved_test:
                raise ValueError(f"Subject splits for fold {fold} are inconsistent with previously saved splits in {split_file}")
            else:
                print(f"Subject splits for fold {fold} are consistent with previously saved splits.")
        else:
            with open(split_file, 'w') as f:
                json.dump({
                    'train_subjects': train_subjects,
                    'test_subjects': test_subjects
                }, f, indent=2)
            print(f"Saved subject splits for fold {fold} to {split_file}")
        
    

class KUERPBenchmark(Benchmark):
    def __init__(self, root, subdir, apply_car, augmentation=None):
        super().__init__()
        self.name = "KU ERP"
        self.slug = "ku_erp"
        print(f"Loading {self.name}...{f' (augmentation={augmentation})' if augmentation else ''}")
        dir = get_data_dir(root, self.slug, subdir, augmentation)
        kuerp_eeg = np.load(os.path.join(dir, 'kuerp_data.npy'), mmap_mode='r')
        kuerp_tf = pd.read_pickle(os.path.join(dir, 'kuerp_trial_features.pd'))

        # Skip preprocessing if loading augmented data (already preprocessed)
        if augmentation is None:
            trial_mask = (kuerp_tf['task'] == 'target') | (kuerp_tf['task'] == 'nontarget')
            kuerp_tf = kuerp_tf[trial_mask]
            kuerp_eeg = kuerp_eeg[trial_mask, :, :]

        if apply_car:
            kuerp_eeg = common_average_reference(kuerp_eeg)

        labels = kuerp_tf['task'].replace({'nontarget': 0, 'target': 1}).to_numpy()
        subject_ids = kuerp_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in kuerp_tf.attrs['channel_names']])

        self.eeg = kuerp_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames
        self.eeg_data_name = "kuerp_data"
        self.trial_features_name = "kuerp_trial_features"
        self.root = root
        self.subdir = subdir

class KUMIBenchmark(Benchmark):
    def __init__(self,root, subdir, apply_car, augmentation=None):
        super().__init__()
        self.name = "KU MI"
        self.slug = "ku_mi"
        print(f"Loading {self.name}...{f' (augmentation={augmentation})' if augmentation else ''}")
        dir = get_data_dir(root, self.slug, subdir, augmentation)
        kumi_eeg = np.load(os.path.join(dir, 'ku_mi_data.npy'), mmap_mode='r')
        kumi_tf = pd.read_pickle(os.path.join(dir, 'ku_mi_trial_features.pd'))
        
        if apply_car:
            kumi_eeg = common_average_reference(kumi_eeg)
            
        labels = kumi_tf['label'].to_numpy().astype(int) # 0, 1
        subject_ids = kumi_tf['subject_id'].to_numpy()
        chnames = np.array([c.upper() for c in kumi_tf.attrs['channel_names']])
        
        if kumi_eeg.shape[0] != len(kumi_tf):
            raise ValueError(f"Mismatch: eeg trials={kumi_eeg.shape[0]} vs trial_features rows={len(kumi_tf)}")
        if kumi_eeg.shape[1] != len(chnames):
            raise ValueError(f"Mismatch: eeg channels={kumi_eeg.shape[1]} vs channel_names={len(chnames)}")

        self.eeg = kumi_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames
        self.eeg_data_name = "ku_mi_data"
        self.trial_features_name = "ku_mi_trial_features"
        self.root = root
        self.subdir = subdir


class PhysionetEyesBenchmark(Benchmark):
    def __init__(self, root, subdir=None, apply_car=False, augmentation=None):
        super().__init__()
        self.name = "Physionet Eyes"
        self.slug = "physionet_eyes"
        print(f"Loading {self.name}...{f' (augmentation={augmentation})' if augmentation else ''}")
        dir = get_data_dir(root, self.slug, subdir, augmentation)
        physioeyes_eeg = np.load(os.path.join(dir, 'mmidb_data.npy'), mmap_mode='r')
        physioeyes_tf = pd.read_pickle(os.path.join(dir, 'mmidb_trial_features.pd'))

        # Skip preprocessing if loading augmented data (already preprocessed)
        if augmentation is None:
            sample_rate = physioeyes_tf['target_fs'][0]

            trial_mask = (physioeyes_tf['type'] == 'eye_open') | (physioeyes_tf['type'] == 'eye_closed')
            physioeyes_tf = physioeyes_tf[trial_mask]

            physioeyes_eeg = physioeyes_eeg[trial_mask, :, :int(4 * sample_rate)]  # cut for max n_patches

        if apply_car:
            physioeyes_eeg = common_average_reference(physioeyes_eeg)

        labels = physioeyes_tf['type'].replace({'eye_closed': 0, 'eye_open': 1}).to_numpy()
        subject_ids = physioeyes_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in physioeyes_tf.attrs['channel_names']])

        self.eeg = physioeyes_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames
        self.eeg_data_name = "mmidb_data"
        self.trial_features_name = "mmidb_trial_features"
        self.root = root
        self.subdir = subdir

    def sample_balanced_set(self, idx, seed):
        print("Classes are already balanced for Physionet Eyes")
        return idx


class PhysionetMIBenchmark(Benchmark):
    def __init__(self, root, subdir=None, apply_car=False, augmentation=None, remove_first_quarter=False):
        super().__init__()
        self.name = "Physionet MI"
        self.slug = "physionet_mi"
        print(f"Loading {self.name}...{f' (augmentation={augmentation})' if augmentation else ''}")
        dir = get_data_dir(root, self.slug, subdir, augmentation)
        physioMI_eeg = np.load(os.path.join(dir, 'mmidb_data.npy'), mmap_mode='r')
        physioMI_tf = pd.read_pickle(os.path.join(dir, 'mmidb_trial_features.pd'))

        # Skip preprocessing if loading augmented data (already preprocessed)
        if augmentation is None:
            sample_rate = physioMI_tf['target_fs'][0]

            trial_mask = (physioMI_tf['type'] == 'imagined')
            physioMI_tf = physioMI_tf[trial_mask]

            physioMI_eeg = physioMI_eeg[trial_mask, :, :int(4 * sample_rate)]  # cut for max n_patches

        if apply_car:
            physioMI_eeg = common_average_reference(physioMI_eeg)

        if remove_first_quarter:
            sample_rate = physioMI_tf['target_fs'].iloc[0]
            print(f"  Removing first quarter: shape before = {physioMI_eeg.shape}")
            start_time = int(1 * sample_rate)  # Skip first 1s out of 4s
            physioMI_eeg = physioMI_eeg[:, :, start_time:]
            print(f"  Removing first quarter: shape after = {physioMI_eeg.shape}")
            print(f"  Removed {start_time} time points ({start_time/sample_rate:.1f}s at {sample_rate}Hz)")

        labels = physioMI_tf['task'].replace({'hands': 0, 'feet': 1, 'left_hand': 2, 'right_hand': 3}).to_numpy()
        subject_ids = physioMI_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in physioMI_tf.attrs['channel_names']])

        self.eeg = physioMI_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames
        self.eeg_data_name = "mmidb_data"
        self.trial_features_name = "mmidb_trial_features"
        self.root = root
        self.subdir = subdir

    def sample_balanced_set(self, idx, seed):
        print("Classes are already balanced for Physionet MI")
        return idx


class PhysionetMEBenchmark(Benchmark):
    def __init__(self, root, subdir=None, apply_car=False, augmentation=None, remove_first_quarter=False):
        super().__init__()
        self.name = "Physionet ME"
        self.slug = "physionet_me"
        print(f"Loading {self.name}...{f' (augmentation={augmentation})' if augmentation else ''}")
        dir = get_data_dir(root, self.slug, subdir, augmentation)
        physioME_eeg = np.load(os.path.join(dir, 'mmidb_data.npy'), mmap_mode='r')
        physioME_tf = pd.read_pickle(os.path.join(dir, 'mmidb_trial_features.pd'))

        # Skip preprocessing if loading augmented data (already preprocessed)
        if augmentation is None:
            sample_rate = physioME_tf['target_fs'][0]

            trial_mask = (physioME_tf['type'] == 'real')
            physioME_tf = physioME_tf[trial_mask]

            physioME_eeg = physioME_eeg[trial_mask, :, :int(4 * sample_rate)]  # cut for max n_patches

        if apply_car:
            physioME_eeg = common_average_reference(physioME_eeg)

        if remove_first_quarter:
            sample_rate = physioME_tf['target_fs'].iloc[0]
            print(f"  Removing first quarter: shape before = {physioME_eeg.shape}")
            start_time = int(1 * sample_rate)  # Skip first 1s out of 4s
            physioME_eeg = physioME_eeg[:, :, start_time:]
            print(f"  Removing first quarter: shape after = {physioME_eeg.shape}")
            print(f"  Removed {start_time} time points ({start_time/sample_rate:.1f}s at {sample_rate}Hz)")

        labels = physioME_tf['task'].replace({'hands': 0, 'feet': 1, 'left_hand': 2, 'right_hand': 3}).to_numpy()
        subject_ids = physioME_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in physioME_tf.attrs['channel_names']])

        self.eeg = physioME_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames
        self.eeg_data_name = "mmidb_data"
        self.trial_features_name = "mmidb_trial_features"
        self.root = root
        self.subdir = subdir

    def sample_balanced_set(self, idx, seed):
        print("Classes are already balanced for Physionet ME")
        return idx

class Pavlov22Benchmark(Benchmark):
    def __init__(self, root, subdir=None, apply_car=False, augmentation=None):
        super().__init__()
        self.name = "Pavlov memory"
        self.slug = "pavlov_memory"
        print(f"Loading {self.name}...{f' (augmentation={augmentation})' if augmentation else ''}")
        dir = get_data_dir(root, self.slug, subdir, augmentation)
        pavlov_eeg = np.load(os.path.join(dir, 'pavlov2022_data.npy'), mmap_mode='r')
        pavlov_tf = pd.read_pickle(os.path.join(dir, 'pavlov2022_trial_features.pd'))

        # Skip preprocessing if loading augmented data (already preprocessed)
        if augmentation is None:
            sample_rate = pavlov_tf['target_fs'][0]
            trial_mask = (pavlov_tf['task'] == 'memory') | (pavlov_tf['task'] == 'control')
            trial_mask = trial_mask & (pavlov_tf['type'] == '13_digits')
            pavlov_tf = pavlov_tf[trial_mask]
            pavlov_eeg = pavlov_eeg[trial_mask, :, int(18. * sample_rate):int(22. * sample_rate)]

        if apply_car:
            pavlov_eeg = common_average_reference(pavlov_eeg)

        labels = pavlov_tf['task'].replace({'control': 0, 'memory': 1}).to_numpy()
        subject_ids = pavlov_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in pavlov_tf.attrs['channel_names']])

        self.eeg = pavlov_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames
        self.eeg_data_name = "pavlov2022_data"
        self.trial_features_name = "pavlov2022_trial_features"
        self.root = root
        self.subdir = subdir

class SleepEDFBenchmark(Benchmark):
    def __init__(self, root, subdir, apply_car, augmentation=None):
        super().__init__()
        self.name = "Sleep EDF"
        self.slug = "sleep_edf"
        print(f"Loading {self.name}...{f' (augmentation={augmentation})' if augmentation else ''}")
        dir = get_data_dir(root, self.slug, subdir, augmentation)
        sleep_eeg = np.load(os.path.join(dir, 'SleepEDF_eeg_trials.npy'), mmap_mode='r')
        sleep_tf = pd.read_pickle(os.path.join(dir, 'SleepEDF_trial_features.pd'))

        if apply_car:
            sleep_eeg = common_average_reference(sleep_eeg)

        labels = sleep_tf['task'].to_numpy()
        subject_ids = sleep_tf['subject_id'].to_numpy()

        chnames = np.array([c.split('-')[0].upper() for c in sleep_tf.attrs['channel_names']])

        self.eeg = sleep_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames
        self.eeg_data_name = "SleepEDF_eeg_trials"
        self.trial_features_name = "SleepEDF_trial_features"
        self.root = root
        self.subdir = subdir

class HighGammaBenchmark(Benchmark):
    def __init__(self, root, subdir, apply_car, augmentation=None):
        super().__init__()
        self.name = "High Gamma"
        self.slug = "high_gamma"
        print(f"Loading {self.name}...{f' (augmentation={augmentation})' if augmentation else ''}")
        dir = get_data_dir(root, self.slug, subdir, augmentation)
        hgd_eeg = np.load(os.path.join(dir, 'highgamma_data.npy'), mmap_mode='r')
        hgd_tf = pd.read_pickle(os.path.join(dir, 'highgamma_trial_features.pd'))

        if augmentation is None:
            # mask out channels
            hgd_chnames = hgd_tf.attrs['channel_names']
            non_EEG = ['EOGh', 'EOGv', 'EMG_RH', 'EMG_LH', 'EMG_RF']
            other_EEG = ['AFF1', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
                'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'I2', 
                'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 
                'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h', 'POO9h',
                'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h'] # i.e. electrodes not in standard 10-20
            hgd_chmask = np.invert(np.isin(hgd_chnames, non_EEG + other_EEG))
            hgd_eeg = hgd_eeg[:, hgd_chmask, :]
            hgd_chnames = hgd_chnames[hgd_chmask]

            sample_rate = hgd_tf['target_fs'][0]
            hgd_eeg = hgd_eeg[:, :, int(2.75 * sample_rate):int(6.75 * sample_rate)] # cut out 4s trial
        else:
            hgd_chnames = hgd_tf.attrs['channel_names']
            non_EEG = ['EOGh', 'EOGv', 'EMG_RH', 'EMG_LH', 'EMG_RF']
            other_EEG = ['AFF1', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
                'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'I2',
                'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h',
                'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h', 'POO9h',
                'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h']
            hgd_chmask = np.invert(np.isin(hgd_chnames, non_EEG + other_EEG))
            hgd_chnames = hgd_chnames[hgd_chmask]


        if apply_car:
            hgd_eeg = common_average_reference(hgd_eeg)

        labels = hgd_tf['task'].replace({'no_action': 0, 'left_fist': 1, 'right_fist': 2, 'both_feet': 3}).to_numpy()
        subject_ids = hgd_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in hgd_chnames])
        
        self.eeg = hgd_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames
        self.eeg_data_name = "highgamma_data"
        self.trial_features_name = "highgamma_trial_features"
        self.root = root
        self.subdir = subdir

    def sample_balanced_set(self, idx, seed):
        print("Classes are already balanced for High Gamma")
        return idx


def load_benchmark(benchmark, root, subdir=None, apply_car=False, augmentation=None, remove_first_quarter=False) -> Benchmark:
    """
    Load a benchmark dataset.

    Args:
        benchmark: Canonical benchmark name ("Physionet MI", "High Gamma", …) — must match a key in BENCHMARKS.
        root: Root directory containing benchmark folders.
        subdir: Preprocessing variant subfolder (e.g. "neurogpt_cut", "01_100Hz").
        apply_car: Apply Common Average Reference preprocessing.
        augmentation: Augmentation variant to load (e.g. "sensor_noise_0db"). None loads the clean data.

    Returns:
        Benchmark object with eeg, subject_ids, labels, chnames.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}. Known: {sorted(BENCHMARKS)}"
        )
    spec = BENCHMARKS[benchmark]

    # Check augmented data exists before instantiating the class (cheaper failure).
    if augmentation is not None:
        data_dir = get_data_dir(root, spec["slug"], subdir, augmentation)
        if not os.path.exists(data_dir):
            raise AugmentationNotFoundError(
                f"Augmented data not found: {data_dir}\n"
                f"Augmentation {augmentation!r} has not been generated for {benchmark!r}."
            )

    cls = spec["class"]
    if benchmark in ("Physionet MI", "Physionet ME"):
        return cls(root, subdir, apply_car, augmentation, remove_first_quarter)
    return cls(root, subdir, apply_car, augmentation)


# Single source of truth for benchmark identifiers.
# - key:        canonical name (matches YAML, plot labels, paper text)
# - "slug":     filesystem-friendly identifier (data dir, splits dir, weights/results subdirs)
# - "class":    the Benchmark subclass that loads it
BENCHMARKS = {
    "Physionet Eyes": {"class": PhysionetEyesBenchmark, "slug": "physionet_eyes"},
    "Physionet MI":   {"class": PhysionetMIBenchmark,   "slug": "physionet_mi"},
    "Physionet ME":   {"class": PhysionetMEBenchmark,   "slug": "physionet_me"},
    "KU MI":          {"class": KUMIBenchmark,          "slug": "ku_mi"},
    "KU ERP":         {"class": KUERPBenchmark,         "slug": "ku_erp"},
    "High Gamma":     {"class": HighGammaBenchmark,     "slug": "high_gamma"},
    "Pavlov memory":  {"class": Pavlov22Benchmark,      "slug": "pavlov_memory"},
    "Sleep EDF":      {"class": SleepEDFBenchmark,      "slug": "sleep_edf"},
}


def slug_for(name: str) -> str:
    """Return the filesystem slug for a canonical benchmark name.

    Tolerates the ' cut' suffix used by cli/_runner.py to differentiate output
    paths for remove_first_quarter runs (Physionet MI/ME variants).
    """
    if name.endswith(" cut"):
        return BENCHMARKS[name[:-4]]["slug"] + "_cut"
    return BENCHMARKS[name]["slug"]



def get_subdir(model_name):
    """
    Returns: str name of subdirectory of preprocessed data for given model
    """
    if model_name in ["EEGNet", "EEGInception", "LaBraM", "CBraMod", "BIOT", "NeuroRVQ", "REVE"]:
        return None
    elif model_name in ["EEGPT"]:
        return "01_100Hz"
    elif model_name in ["BrainOmni"]:
        return "brainomni"
    elif model_name in ["NeuroGPT", "MIRepNet"]:
        return "neurogpt_cut"
    else:
        raise ValueError(f"Unknown model name {model_name!r}")