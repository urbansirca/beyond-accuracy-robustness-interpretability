import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, freqz
from data.loaders import Benchmark
from scipy.fft import irfft



def generate_pink_noise(shape, seed=None):
    """
    Generate pink (1/f) noise.

    Args:
        shape: Output shape, e.g. (n_trials, n_channels, n_samples) or (n_channels, n_samples)
        seed: Random seed for reproducibility

    Returns:
        Pink noise array with the specified shape
    """
    from scipy.fft import rfft, irfft

    white = np.random.default_rng(seed).standard_normal(shape)

    # FFT along last axis (samples)
    spectrum = rfft(white, axis=-1, workers=-1)

    # 1/f scaling (pink noise has power ~ 1/f, so amplitude ~ 1/sqrt(f))
    freqs = np.fft.rfftfreq(shape[-1])
    scale = 1.0 / np.sqrt(freqs + 1e-10)  # +epsilon to avoid division by zero at f=0

    # Broadcasting: scale shape (n_freqs,) broadcasts against last axis automatically
    spectrum *= scale
    pink = irfft(spectrum, n=shape[-1], axis=-1, workers=-1)
    return pink


# Put this near the top of the module (global cache)
_FILTER_MASK_CACHE = {}

def _get_filter_mask(
    n_samples: int,
    sfreq: float,
    lowcut: float,
    highcut: float,
    notch_freqs,
    notch_q: float,
    order: int,
    noise_type: str,
):
    """
    Cached combined magnitude mask for generate_filtered_noise().
    Keyed by parameters that affect the mask.
    """
    nyquist = sfreq / 2.0
    n_freqs = n_samples // 2 + 1

    # Normalise cache key (tuple, hashable)
    notch_key = None if notch_freqs is None else tuple(float(f) for f in notch_freqs)
    key = (n_samples, float(sfreq), float(lowcut), float(highcut), notch_key, float(notch_q) if notch_q is not None else None, int(order), str(noise_type))

    cached = _FILTER_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    # Clamp highcut
    hc = highcut
    if hc >= nyquist:
        hc = nyquist - 1.0
        print(f"Warning: highcut adjusted to {hc} Hz (Nyquist limit)")

    worN = np.linspace(0, np.pi, n_freqs)
    mask = np.ones(n_freqs, dtype=np.float32)

    # Bandpass response, squared to mimic filtfilt magnitude effect
    low = lowcut / nyquist
    high = hc / nyquist
    b_bp, a_bp = butter(order, [low, high], btype="band")
    _, h_bp = freqz(b_bp, a_bp, worN=worN)
    mask *= (np.abs(h_bp) ** 2)

    # Notches
    if notch_freqs is not None:
        for freq in notch_freqs:
            if freq < nyquist:
                b_n, a_n = iirnotch(freq, notch_q, sfreq)
                _, h_n = freqz(b_n, a_n, worN=worN)
                mask *= (np.abs(h_n) ** 2)

    # Pink shaping
    if noise_type == "pink":
        freqs_hz = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
        mask *= (1.0 / np.sqrt(freqs_hz + 1e-10))

    _FILTER_MASK_CACHE[key] = mask
    return mask



def _running_update(n, mean, m2, x):
    """Welford's online algorithm for running mean and variance.
    x is a 1-D array of new observations."""
    for v in x:
        n += 1
        delta = v - mean
        mean += delta / n
        m2 += delta * (v - mean)
    return n, mean, m2


def generate_filtered_noise(
    shape,
    noise_type="white",
    sfreq=200,
    lowcut=0.5,
    highcut=40.0,
    notch_freqs=[50.0, 60.0],
    notch_q=30.0,
    order=4,
    seed=None,
):
    """
    As before, but uses a cached mask so per-batch calls are cheap.
    """

    n_samples = shape[-1]
    n_freqs = n_samples // 2 + 1

    mask = _get_filter_mask(
        n_samples=n_samples,
        sfreq=sfreq,
        lowcut=lowcut,
        highcut=highcut,
        notch_freqs=notch_freqs,
        notch_q=notch_q,
        order=order,
        noise_type=noise_type,
    )

    rng = np.random.default_rng(seed)
    freq_shape = shape[:-1] + (n_freqs,)

    # Complex spectrum
    spectrum = rng.standard_normal(freq_shape) + 1j * rng.standard_normal(freq_shape)

    # Apply mask (broadcast)
    spectrum *= mask

    noise = irfft(spectrum, n=n_samples, axis=-1, workers=-1)

    # Normalise per channel
    noise -= noise.mean(axis=-1, keepdims=True)
    noise /= noise.std(axis=-1, keepdims=True) + 1e-12
    return noise

def _add_sensor_noise_batched(
    benchmark, X, sbj_id, y, ch_names,
    snr_dbs, noise_type, filter_noise, seed, save,
    visualise,
    batch_size, sfreq, lowcut, highcut, notch_freqs, notch_q, order
):
    """
    Batched version of add_sensor_noise_to_dataset for memory efficiency.
    Processes all SNR levels simultaneously so noise is generated only once
    per batch. Uses Welford's online algorithm for power statistics to avoid
    large temporary lists.
    """
    import os
    import json
    import tempfile
    import shutil
    import pandas as pd
    from data.loaders import get_data_dir, convert_to_serializable

    n_trials, n_channels, n_samples = X.shape
    n_batches = int(np.ceil(n_trials / batch_size))

    print(f"  Processing {n_trials} trials in {n_batches} batches of {batch_size}")
    print(f"  SNR levels: {snr_dbs}")

    # Write memmaps to local scratch ($TMPDIR or /tmp) for fast I/O,
    # then move to the final GPFS destination at the end.
    tmp_base = os.environ.get("TMPDIR", None)
    scratch_dir = tempfile.mkdtemp(dir=tmp_base)

    # Create one memmap and running-stats accumulator per SNR
    memmaps = {}
    # Each entry: [n_sp, mean_sp, m2_sp, n_np, mean_np, m2_np, n_nop, mean_nop, m2_nop]
    stats = {}
    first_noisy_trials = {}

    for snr_db in snr_dbs:
        path = os.path.join(scratch_dir, f'{benchmark.eeg_data_name}_snr{snr_db}.npy')
        memmaps[snr_db] = np.lib.format.open_memmap(
            path, mode='w+', dtype=np.float32,
            shape=(n_trials, n_channels, n_samples),
        )
        stats[snr_db] = [0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0]
        first_noisy_trials[snr_db] = None

    rng = np.random.default_rng(seed)

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_trials)

        print(f"    Batch {batch_idx + 1}/{n_batches}: trials {start_idx}-{end_idx}")

        X_batch = np.asarray(X[start_idx:end_idx], dtype=np.float32)
        Xc_batch = X_batch - X_batch.mean(axis=2, keepdims=True)

        # Generate noise ONCE for this batch (shared across all SNRs)
        if filter_noise:
            noise_batch = generate_filtered_noise(
                X_batch.shape, noise_type=noise_type, sfreq=sfreq,
                lowcut=lowcut, highcut=highcut, notch_freqs=notch_freqs,
                notch_q=notch_q, order=order, seed=seed + batch_idx if seed is not None else None,
            )
        else:
            if noise_type == "white":
                noise_batch = rng.standard_normal(X_batch.shape)
            elif noise_type == "pink":
                noise_batch = generate_pink_noise(X_batch.shape, seed=seed + batch_idx)
            else:
                raise ValueError(f"Invalid noise_type: {noise_type}")
            noise_batch -= noise_batch.mean(axis=2, keepdims=True)
            noise_batch /= noise_batch.std(axis=2, keepdims=True) + 1e-12

        # Signal power is the same for all SNRs
        signal_power_batch = np.mean(Xc_batch**2, axis=(1, 2))

        # Scale the same noise for each SNR level
        for snr_db in snr_dbs:
            noise_power_batch = signal_power_batch / (10 ** (snr_db / 10))
            X_noisy_batch = X_batch + np.sqrt(noise_power_batch)[:, None, None] * noise_batch
            noisy_power_batch = np.mean(X_noisy_batch**2, axis=(1, 2))

            # Update running stats (Welford's online algorithm)
            s = stats[snr_db]
            s[0], s[1], s[2] = _running_update(s[0], s[1], s[2], signal_power_batch)
            s[3], s[4], s[5] = _running_update(s[3], s[4], s[5], noise_power_batch)
            s[6], s[7], s[8] = _running_update(s[6], s[7], s[8], noisy_power_batch)

            if batch_idx == 0:
                first_noisy_trials[snr_db] = X_noisy_batch[0].copy()

            # Write directly into memmap
            memmaps[snr_db][start_idx:end_idx] = X_noisy_batch
            del X_noisy_batch, noisy_power_batch

        del X_batch, Xc_batch, noise_batch

    # Save results for each SNR
    for snr_db in snr_dbs:
        augmentation_name = f"{noise_type}_noise_{snr_db}db"

        s = stats[snr_db]
        std_sp = (s[2] / max(s[0] - 1, 1)) ** 0.5
        std_np = (s[5] / max(s[3] - 1, 1)) ** 0.5
        std_nop = (s[8] / max(s[6] - 1, 1)) ** 0.5

        metadata = {
            "signal_type": noise_type,
            "sfreq": sfreq,
            "filter_noise": filter_noise,
            "order": order,
            "notch_freqs": notch_freqs,
            "lowcut": lowcut,
            "highcut": highcut,
            "signal_power": s[1],
            "noise_power": s[4],
            "snr_db": snr_db,
            "mean_signal_power": s[1],
            "mean_noise_power": s[4],
            "mean_noisy_signal_power": s[7],
            "std_signal_power": std_sp,
            "std_noise_power": std_np,
            "std_noisy_signal_power": std_nop,
            "batch_size": batch_size,
            "n_batches": n_batches,
        }

        fig = None
        if visualise:
            visualise_fft_per_channel(first_noisy_trials[snr_db], sfreq=sfreq, name="Noisy Signal")

        # always plot this because we will save the fig
        fig = visualise_trial_before_after_noise(
            X[0], first_noisy_trials[snr_db], benchmark.chnames, trial_idx=0, snr_db=snr_db)

        if save:
            save_dir = get_data_dir(benchmark.root, benchmark.name, benchmark.subdir, augmentation_name)
            os.makedirs(save_dir, exist_ok=True)

            # Close the memmap before moving
            del memmaps[snr_db]
            src_path = os.path.join(scratch_dir, f'{benchmark.eeg_data_name}_snr{snr_db}.npy')
            final_path = os.path.join(save_dir, f'{benchmark.eeg_data_name}.npy')
            print(f"  Moving data from scratch to {save_dir} ...")
            shutil.move(src_path, final_path)

            metadata_serializable = convert_to_serializable(metadata)
            with open(os.path.join(save_dir, f'{augmentation_name}_metadata.json'), 'w') as f:
                json.dump(metadata_serializable, f, indent=2)

            original_dir = get_data_dir(benchmark.root, benchmark.name, benchmark.subdir)
            original_tf_path = os.path.join(original_dir, f'{benchmark.trial_features_name}.pd')
            if os.path.exists(original_tf_path):
                trial_features = pd.read_pickle(original_tf_path)
                trial_features.to_pickle(os.path.join(save_dir, f'{benchmark.trial_features_name}.pd'))

            if fig is not None:
                fig.savefig(os.path.join(save_dir, f'{augmentation_name}_figure.png'))

            print(f"Augmented data saved to: {save_dir}")

        plt.close('all')

    # Clean up any remaining memmaps and scratch directory
    memmaps.clear()
    shutil.rmtree(scratch_dir, ignore_errors=True)

    return None
        

def get_filter_parameters(subdir):
    """
    Get the filter parameters based on the subdir name.
    """
    if subdir is None:
        return {
            "sfreq": 200,
            "lowcut": 0.5,
            "highcut": 45.0,
            "notch_freqs": [50.0, 60.0],
            "notch_q": 30.0,
            "order": 4,
        }
    elif subdir == "neurogpt_cut": # THIS is for NeuroGPT
        return {
            "sfreq": 250,
            "lowcut": 0.05,
            "highcut": 100,
            "notch_freqs": [50.0, 60.0, 100.0],
            "notch_q": 30.0,
            "order": 4,
        }
    elif subdir == "01_100Hz": # This is for EEGPT
        return {
            "sfreq": 256,
            "lowcut": 1,
            "highcut": 100.0,
            "notch_freqs": [50.0, 60.0, 100.0], 
            "notch_q": 30.0,
            "order": 2,  # Lower order for minimal filtering
        }
    elif subdir == "brainomni": # This is for BrainOmni
        return {
            "sfreq": 256,
            "lowcut": 1,
            "highcut": 96.0,
            "notch_freqs": [50.0, 60.0, 100.0],
            "notch_q": 30.0,
            "order": 4,
        }
    else:
        raise ValueError(f"Unknown subdir: {subdir}")


def add_sensor_noise_to_dataset(
    benchmark: Benchmark,
    snr_db=None,
    noise_type="white",
    filter_noise=False,
    seed=None,
    save=True,
    visualise=False,
    batch_size=None,
    overwrite=False
):
    """
    Add independent white sensor noise per channel to each trial in dataset.

    Args:
        benchmark: benchmark object containing the dataset to augment
        snr_db: desired signal-to-noise ratio in dB (scalar or list), lower means more noise
            signal:noise ratios in dB vs linear:
            20 - 100:1 signal dominant
            10 - 10:1 clear signal
            3 - 2:1 signal visible
            0 - 1:1 signal and noise equal --> already aggressive for EEG
            -3 - 1:2 noise stronger
            -10 - 1:10 very noisy
            -20 - 1:100 signal blurred
        seed: random seed for reproducibility
        save: whether to save the augmented dataset to disk
        visualise: whether to visualise the effect of noise on a sample trial
        batch_size: if specified, process data in batches to reduce memory usage

    Returns:
        X_noisy when a single snr_db is given (non-batched), else None
    """

    assert snr_db is not None, "Must specify snr_db to add sensor noise"

    # Normalise to list
    snr_dbs = snr_db if isinstance(snr_db, (list, tuple)) else [snr_db]

    sfreq, lowcut, highcut, notch_freqs, notch_q, order = (
        get_filter_parameters(benchmark.subdir).values()
    )

    # Filter out already-existing augmentations before loading data
    if not overwrite:
        active = []
        for s in snr_dbs:
            if benchmark.check_augmented_exists(f"{noise_type}_noise_{s}db"):
                print(f"Augmented data with {noise_type} noise at {s}dB already exists. Skipping.")
            else:
                active.append(s)
        if not active:
            return None
        snr_dbs = active

    X, sbj_id, y, ch_names = benchmark.get_data()
    n_trials = X.shape[0]

    # Use batch processing if specified
    if batch_size is not None and n_trials > batch_size:
        return _add_sensor_noise_batched(
            benchmark, X, sbj_id, y, ch_names,
            snr_dbs, noise_type, filter_noise, seed, save,
            visualise,
            batch_size, sfreq, lowcut, highcut, notch_freqs, notch_q, order
        )

    # --- Non-batched path: generate noise once, apply for each SNR ---
    X = np.asarray(X, dtype=np.float32)
    Xc = X - X.mean(axis=2, keepdims=True)

    assert noise_type in [
        "white",
        "pink",
    ], "Invalid noise_type, must be 'white' or 'pink'"

    if filter_noise:
        noise = generate_filtered_noise(
            X.shape, noise_type=noise_type, sfreq=sfreq,
            lowcut=lowcut, highcut=highcut, notch_freqs=notch_freqs,
            notch_q=notch_q, order=order, seed=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        if noise_type == "white":
            noise = rng.standard_normal(X.shape)
        elif noise_type == "pink":
            noise = generate_pink_noise(X.shape, seed=seed)
        noise -= noise.mean(axis=2, keepdims=True)
        noise /= noise.std(axis=2, keepdims=True) + 1e-12

    signal_power = np.mean(Xc**2, axis=(1, 2))

    last_X_noisy = None
    for snr_db_val in snr_dbs:
        noise_power = signal_power / (10 ** (snr_db_val / 10))
        X_noisy = X + np.sqrt(noise_power)[:, None, None] * noise

        metadata = {
            "signal_type": noise_type,
            "sfreq": sfreq,
            "filter_noise": filter_noise,
            "order": order,
            "notch_freqs": notch_freqs,
            "lowcut": lowcut,
            "highcut": highcut,
            "signal_power": np.mean(signal_power),
            "noise_power": np.mean(noise_power),
            "snr_db": snr_db_val,
            "mean_signal_power": np.mean(signal_power),
            "mean_noise_power": np.mean(noise_power),
            "mean_noisy_signal_power": np.mean(np.mean(X_noisy**2, axis=(1, 2))),
            "std_signal_power": np.std(signal_power),
            "std_noise_power": np.std(noise_power),
            "std_noisy_signal_power": np.std(np.mean(X_noisy**2, axis=(1, 2))),
        }
        
        
        fig = visualise_trial_before_after_noise(
            X[0], X_noisy[0], benchmark.chnames, trial_idx=0, snr_db=snr_db_val)

        if visualise:
            visualise_fft_per_channel(X[0], sfreq=sfreq, name="Original Signal")
            visualise_fft_per_channel(noise[0], sfreq=sfreq, name="Noise")
            visualise_fft_per_channel(X_noisy[0], sfreq=sfreq, name="Noisy Signal")


        if save:
            augmentation_name = f"{noise_type}_noise_{snr_db_val}db"
            benchmark.save_augmented_data(X_noisy, augmentation_name, metadata=metadata, figure=fig)

        plt.close('all')
        last_X_noisy = X_noisy

    return last_X_noisy if len(snr_dbs) == 1 else None


def visualise_fft_per_channel(X, sfreq, name=""):
    """
    Visualise the FFT of a trial to see how noise affects the frequency spectrum.
    X: (n_channels, n_samples)
    """
    n_channels, n_samples = X.shape
    freqs = np.fft.rfftfreq(n_samples, d=1 / sfreq)
    fft_vals = np.fft.rfft(X, axis=1)  # shape: (n_channels, n_freqs)
    fft_power = np.abs(fft_vals) ** 2  # power spectrum

    plt.figure(figsize=(12, 6))
    for i in range(n_channels):
        plt.plot(freqs, fft_power[i], label=f"Channel {i}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(f"FFT Power Spectrum {name}")
    plt.xlim(0, sfreq / 2)
    plt.show()
    return plt.gcf()  # Return figure for potential saving


def visualise_trial_before_after_noise(trial, noisy_trial, chnames, trial_idx, snr_db):
    # plot original and noisy signals overlaid
    spacing = 200  # adjust this for more/less vertical spacing

    plt.figure(figsize=(14, 10))
    for i in range(trial.shape[0]):
        offset = i * spacing
        plt.plot(
            trial[i] + offset,
            "b",
            linewidth=0.5,
            alpha=0.7,
            label="Original" if i == 0 else None,
        )
        plt.plot(
            noisy_trial[i] + offset,
            "r",
            linewidth=0.5,
            alpha=0.7,
            label="Noisy" if i == 0 else None,
        )

    plt.yticks([i * spacing for i in range(len(chnames))], chnames)
    plt.xlabel("Time (samples)")
    plt.title(f"Trial {trial_idx} | Original (blue) vs Noisy SNR={snr_db}dB (red)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    # plt.show()
    return plt.gcf()  # Return figure for potential saving