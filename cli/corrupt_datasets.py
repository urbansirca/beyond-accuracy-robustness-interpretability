"""Generate perturbed EEG datasets on disk. Defined by YAML config.
python -m cli.corrupt_datasets --config configs/corrupt/default.yaml
"""
import argparse
import gc

from cli._config import load_config
from data.loaders import load_benchmark
from data.perturbations.channel_dropout import (
    create_random_dropout_datasets,
    create_region_dropout_datasets,
)
from data.perturbations.region_noise import create_region_noise_datasets
from data.perturbations.sensor_noise import add_sensor_noise_to_dataset


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate perturbed EEG datasets on disk.")
    p.add_argument("--config", required=True,
                   help="Path to a YAML config (configs/corrupt/*.yaml).")
    p.add_argument("--benchmarks", nargs="+",
                   help="Override config's benchmarks list.")
    p.add_argument("--subdirs", nargs="+",
                   help="Override config's subdirs list. Pass the literal 'null' for the base subdir.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.benchmarks:
        cfg["benchmarks"] = args.benchmarks
    if args.subdirs:
        cfg["subdirs"] = [None if s.lower() == "null" else s for s in args.subdirs]

    process_all_augmentations(cfg)


def process_all_augmentations(cfg: dict) -> None:
    """Iterate (benchmark × subdir) and apply every enabled augmentation block."""
    enabled = {k: cfg[k] for k in ("sensor_noise", "random_dropout", "region_dropout", "region_noise")
               if cfg.get(k)}

    print("\n" + "=" * 60)
    print("DATASET CORRUPTION PIPELINE")
    print("=" * 60)
    print(f"Benchmarks: {cfg['benchmarks']}")
    print(f"Subdirs:    {cfg['subdirs']}")
    print(f"Augmentations: {list(enabled) or '(none)'}")
    print("=" * 60)

    for benchmark_name in cfg["benchmarks"]:
        for subdir in cfg["subdirs"]:
            label = subdir or "base"
            print(f"\n{'#' * 60}\nLOADING: {benchmark_name} | {label}\n{'#' * 60}")
            try:
                benchmark = load_benchmark(
                    benchmark_name, cfg["data_root"], subdir=subdir, apply_car=cfg["apply_car"]
                )
                X, _, _, _ = benchmark.get_data()
                print(f"✓ Loaded: shape {X.shape}")
            except Exception as e:
                print(f"✗ Error loading {benchmark_name} ({label}): {e}")
                gc.collect()
                continue

            if "sensor_noise" in enabled:
                _run_sensor_noise(benchmark, cfg, enabled["sensor_noise"])
            if "region_dropout" in enabled:
                _run_region_dropout(benchmark, cfg, enabled["region_dropout"])
            if "random_dropout" in enabled:
                _run_random_dropout(benchmark, cfg, enabled["random_dropout"])
            if "region_noise" in enabled:
                _run_region_noise(benchmark, cfg, enabled["region_noise"])

            del benchmark, X
            gc.collect()

    print(f"\n{'=' * 60}\nDONE\n{'=' * 60}")


def _run_sensor_noise(benchmark, cfg, params):
    for noise_type in params["noise_types"]:
        print(f"\n— sensor noise [{noise_type}] SNRs={params['snr_db']}")
        try:
            add_sensor_noise_to_dataset(
                benchmark=benchmark,
                snr_db=params["snr_db"],
                noise_type=noise_type,
                filter_noise=cfg["filter_noise"],
                seed=cfg["seed"],
                save=cfg["save"],
                visualise=cfg["visualise"],
                batch_size=cfg["batch_size"],
                overwrite=params.get("overwrite", False),
            )
        except Exception as e:
            print(f"  ✗ Error: {e}")
        gc.collect()


def _run_region_dropout(benchmark, cfg, params):
    print(f"region dropout regions={params['regions']}")
    
    create_region_dropout_datasets(
        benchmark=benchmark,
        regions=params["regions"],
        save=cfg["save"],
        plot=cfg["visualise"],
        overwrite=params.get("overwrite", False)
    )
    gc.collect()


def _run_random_dropout(benchmark, cfg, params):
    for drop_prob in params["drop_rates"]:
        print(f"random dropout p={drop_prob}")
        create_random_dropout_datasets(
            benchmark=benchmark,
            drop_prob=drop_prob,
            seed=cfg["seed"],
            save=cfg["save"],
            plot=cfg["visualise"],
            overwrite=params.get("overwrite", False),
        )
        gc.collect()


def _run_region_noise(benchmark, cfg, params):
    print(f"\n— region noise [{params['noise_type']}, regions={params['regions']}] SNRs={params['snr_db']}")
    try:
        create_region_noise_datasets(
            benchmark=benchmark,
            snr_levels=params["snr_db"],
            noise_type=params["noise_type"],
            regions=params["regions"],
            apply_car=cfg["apply_car"],
            save=cfg["save"],
            overwrite=params.get("overwrite", False),
        )
    except Exception as e:
        print(f"  ✗ Error: {e}")
    gc.collect()


if __name__ == "__main__":
    main()
