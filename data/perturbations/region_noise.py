import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from .channel_dropout import BENCHMARK_REGION_CONFIG, create_region_dropout_mask, plot_dropout
from data.loaders import load_benchmark, AugmentationNotFoundError, Benchmark

def create_region_noise_datasets(
    benchmark: Benchmark,
    snr_levels=None,
    regions=["Primary", "Control"],
    noise_type=None,
    apply_car=False,
    save=True,
    overwrite=False,
):
    """
    Create region noise datasets by replacing dropped-region channels
    with their white-noise-corrupted versions.

    For each benchmark/condition/snr combination:
      - Channels in the target region get their noisy version
      - All other channels remain clean
    """

    # Load clean data once
    cleanX, _, _, chnames = benchmark.get_data()
    
    
    # check if we have the noisy versions of the data for this benchmark and the specified noise levels
    for noise_level in snr_levels:
        noise_aug_name = f"{noise_type}_noise_{noise_level}db"
        if not benchmark.check_augmented_exists(noise_aug_name):
            raise AugmentationNotFoundError(f"Required noise augmentation '{noise_aug_name}' not found for benchmark '{benchmark.name}'. Please create it first using the appropriate augmentation script.")


    dir_name = benchmark.name #TODO: check if needs to be slug
    config = BENCHMARK_REGION_CONFIG.get(dir_name)
    if config is None:
        print(f"No region config for benchmark: {dir_name}. Skipping.")
        return

    for noise_level in snr_levels:
        # Augmentation dir name to load, e.g. "white_noise_5db"
        noise_aug_name = f"{noise_type}_noise_{noise_level}db"

        # Load noisy data (same shape as clean since same subdir)
        noisy_bench = load_benchmark(benchmark.name, root=benchmark.root, subdir=benchmark.subdir, apply_car=apply_car, augmentation=noise_aug_name)
            
        noisyX, _, _, _ = noisy_bench.get_data()
        
        assert cleanX.shape == noisyX.shape, (
            f"Shape mismatch: clean {cleanX.shape} vs noisy {noisyX.shape}. "
            f"Both must come from the same preprocessing."
        )

        for region in regions:
            if region not in config:
                continue

            aug_name = f"region_noise_{noise_type}_{noise_level}db_{region}"

            if not overwrite and benchmark.check_augmented_exists(aug_name):
                print(f"  {aug_name} already exists. Skipping.")
                continue

            regions_to_drop = config[region]
            drop_mask = create_region_dropout_mask(chnames, regions_to_drop)

            compositeX = cleanX.copy()
            compositeX[:, ~drop_mask, :] = noisyX[:, ~drop_mask, :]

            n_replaced = int(np.sum(~drop_mask))
            replaced_channels = [ch for ch, keep in zip(chnames, drop_mask) if not keep]

            print(f"{dir_name} - {aug_name}:")
            print(f"  Regions: {regions_to_drop}")
            print(f"  Replaced {n_replaced}/{len(chnames)} channels with {noise_aug_name}: {replaced_channels}")

            metadata = {
                "benchmark": dir_name,
                "region": region,
                "noise_augmentation": noise_aug_name,
                "noise_type": noise_type,
                "noise_level_db": noise_level,
                "regions": regions_to_drop,
                "n_replaced": n_replaced,
                "n_total": len(chnames),
                "replaced_channels": replaced_channels,
                "kept_channels": chnames, # all are kept in the sense that we don't drop any, just replace some with noisy versions
            }
            
            fig = plot_dropout(chnames, drop_mask, title=f"{dir_name} {region} Region Noise ({noise_aug_name})")

            if save:
                benchmark.save_augmented_data(compositeX, aug_name, metadata=metadata, figure=fig)
                print(f"  Saved as: {aug_name}")

