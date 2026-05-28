# EEG Robustness and Interpretability Benchmarking

Code for the paper [*Beyond Accuracy: Robustness, Interpretability and
Expressiveness of EEG Foundation Models*](https://doi.org/10.48550/arXiv.2605.17562)
(Širca, Alimardani, Zafeiriou, Barmpas — arXiv:2605.17562, 2026).

This repository **extends** the [*codebase*](https://github.com/dykestra/EEG-Benchmarking) and benchmarking protocol from the
IEEE MLSP 2025 paper [*Assessing the Capabilities of Large Brainwave Foundation
Models*](https://ieeexplore.ieee.org/abstract/document/11204282) (Lee et al.), which
introduced the subject-independent cross-validation protocol and the initial
five-dataset suite. We add more datasets (KU MI, PhysioNet MI/ME), more
foundation models, the robustness suite, and the interpretability analyses listed above.


## Abstract

EEG foundation models (EEG-FMs) have been evaluated predominantly on clean, in-distribution accuracy, leaving their robustness, interpretability and representational quality largely unexamined. This study addresses these gaps by benchmarking six EEG-FMs against a baseline deep learning model across eight datasets. Beyond clean accuracy, we conduct three layers of analysis: (i) Robustness: we apply test-time perturbations including additive noise, random and region-based channel dropout and region-specific noise injection. Our analyses show that no single model dominates all failure modes. The most noise-robust model is among the most fragile under channel dropout and much of the dropout fragility disappears when channels are removed rather than zero-padded. (ii) Interpretability: we present the first application of Attention-Aware Layer-Wise Relevance Propagation (AttnLRP) to EEG-FMs and show that models broadly concentrate relevance on task-appropriate brain regions consistent with known neurophysiology. However, attribution maps remain spatially stable under perturbation while predictions degrade, suggesting that the models attend to the correct brain regions but decode corrupted content. (iii) Expressiveness: With block-wise probing we show that late blocks are repurposed during fine-tuning, while early blocks already hold task-related information. Furthermore, we demonstrate that the poor head-only performance previously attributed to low-quality pre-trained representations is largely explained by pooling and that EEG-FMs possess sufficient representational capacity when their token-level embeddings are preserved. Together, these findings provide the first systematic assessment of robustness, interpretability and expressiveness for EEG-FMs and highlight critical considerations for their development.

### Clean accuracy across all eight benchmarks

<img src="readme-plots/sec1_clean_bars_combined.png" alt="Clean balanced accuracy per (model, benchmark)" width="700">

### Robustness summary across perturbations

<img src="readme-plots/sec2_robustness_summary_full_realdrop.png" alt="Robustness summary — full fine-tuning, real channel dropout" width="700">

### Class-averaged attribution maps

<img src="readme-plots/sec3_lrp_grid_class_avg_all.png" alt="Class-averaged attribution topographies — all models × benchmarks" width="700">

AttnLRP is used for EEGNet, LaBraM, CBraMod and REVE. NeuroRVQ and BrainOmni
fall back to Input × Gradient (IxG).

### Block-wise mean-pooled probing (pre-trained vs fine-tuned)

<img src="readme-plots/sec4_probing_grid_mean_pt_vs_ft.png" alt="Block-wise mean-pooled probing" width="700">

## Models

Benchmarking is currently supported for the following models:

- [**EEGNet**](https://braindecode.org/dev/generated/braindecode.models.EEGNet.html)
- [**LaBraM**](https://github.com/935963004/LaBraM)
- [**CBraMod**](https://github.com/wjq-learning/CBraMod)
- [**BIOT**](https://github.com/ycq091044/BIOT)
- [**REVE**](https://brain-bzh.github.io/reve/)
- [**BrainOmni**](https://github.com/OpenTSLab/BrainOmni)
- [**NeuroRVQ**](https://neurorvq.github.io)


### BrainOmni setup

BrainOmni's source is not vendored in this repo. There are two ways to make it
available:

1. **Use a separate checkout.** Clone the upstream repo anywhere, then point
   `brainomni_repo` in [configs/_base.yaml](configs/_base.yaml) at the
   checkout (absolute path, or relative to the project root):

   ```commandline
   git clone https://github.com/OpenTSLab/BrainOmni /path/to/BrainOmni
   ```

   ```yaml
   # configs/_base.yaml
   brainomni_repo: /path/to/BrainOmni
   ```

### Pretrained weights

Each foundation-model wrapper looks for its checkpoint at a default path under
`weights/pretrained/`. You can override per run by passing `--ckpt-path` (or
setting `ckpt_path` in the config); when unset, the defaults below are used.
EEGNet is trained from scratch and needs no download.

| Model | Default path | Where to get the checkpoint |
|-|-|-|
| LaBraM | `weights/pretrained/labram-base.pth` | Upstream [LaBraM repo](https://github.com/935963004/LaBraM) |
| CBraMod | `weights/pretrained/cbramod-base.pth` | Upstream [CBraMod repo](https://github.com/wjq-learning/CBraMod)|
| BIOT | `weights/pretrained/biot-base.ckpt` | Upstream [BIOT repo](https://github.com/ycq091044/BIOT) |
| BrainOmni | `weights/pretrained/BrainOmni/` (directory containing `BrainOmni.pt` and `model_cfg.json`) | HuggingFace `OpenTSLab/BrainOmni` |
| REVE | auto-downloaded from HuggingFace (`brain-bzh/reve-base` + `brain-bzh/reve-positions`) — no manual step | Loaded by [models/REVE/modules.py](models/REVE/modules.py) |
| NeuroRVQ | `weights/pretrained/neurorvq-base.pt` | HuggingFace [`ntinosbarmpas/NeuroRVQ`](https://huggingface.co/ntinosbarmpas/NeuroRVQ) |

The BrainOmni checkpoint can be fetched directly:

```commandline
huggingface-cli download OpenTSLab/BrainOmni --local-dir weights/pretrained/BrainOmni
```

For LaBraM / CBraMod / BIOT / NeuroRVQ, download the released checkpoint from the
upstream repo linked above and rename it to match the default path in the
table. REVE pulls its weights automatically the first time it's instantiated (cached
under `~/.cache/huggingface/`).

## Environment Setup
[![python](https://img.shields.io/badge/Python-3.11.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

Create the conda environment from [env.yml](env.yml):

```commandline
conda env create -f env.yml
conda activate benchmark-env
```


## Data Pre-processing

### Download Data

The following EEG datasets are selected for benchmarking.

| Dataset | Paradigm | Classes | Tasks |
|-|-|-|-|
| [High Gamma](https://github.com/robintibor/high-gamma-dataset) | Executed Movement | 4 | `no_action`, `left_fist`, `right_fist`, `both_feet` |
| [KU MI](http://gigadb.org/dataset/100542) (OpenBMI) | Motor Imagery | 2 | `left_hand`, `right_hand` |
| [KU ERP](http://gigadb.org/dataset/100542) (OpenBMI) | ERP | 2 | `target`, `nontarget` |
| [Pavlov 2022](https://openneuro.org/datasets/ds003838/versions/1.0.2) | Working Memory | 2 | `memory`, `control` (13-digit trials) |
| [Sleep-EDF](https://www.physionet.org/content/sleep-edfx/1.0.0/) | Sleep Stage | 6 | `Sleep stage W/1/2/3/4/R` |
| [PhysioNet Eyes](https://physionet.org/content/eegmmidb/1.0.0/) | Eyes Open-Closed | 2 | `eye_open`, `eye_closed` |
| [PhysioNet MI](https://physionet.org/content/eegmmidb/1.0.0/) | Motor Imagery | 4 | `left_hand`, `right_hand`, `hands`, `feet` (imagined) |
| [PhysioNet ME](https://physionet.org/content/eegmmidb/1.0.0/) | Motor Execution | 4 | `left_hand`, `right_hand`, `hands`, `feet` (executed) |

### Pre-process EEG Signal Data
To reproduce the results in the paper, the raw EEG signals for each dataset
should be:

- **resampled to 200 Hz** for all models, except **256 Hz for BrainOmni**
  (the wrapper picks the rate via [experiments/config.py:77-81](experiments/config.py#L77-L81);
  the saved `target_fs` metadata field must match).
- **bandpass filtered at 0.5–45 Hz** (0.1–96 Hz for BrainOmni)
- cut into trials:
  - **High Gamma:** 0s-4s after each cue
  - **KU MI:** 0s-4s after each cue (motor imagery period)
  - **KU ERP:** 0.2s before - 0.8s after each cue
  - **Pavlov 2022:** 14s-18s after each 13-digits trial cue (corresponding to the peak in pupil size reported in the dataset publication)
  - **Sleep-EDF:** 30s epochs from the original continuous recording, and discard any data from the awake condition except for the 30 minutes before and after sleep
  - **PhysioNet Eyes:** 4s epochs from the original continuous recording
  - **PhysioNet MI:** 4s epochs aligned to imagined movement cues
  - **PhysioNet ME:** 4s epochs aligned to executed movement cues
- saved in numpy format as an array with shape (N_trials, Channels, Time)

### Save Metadata
For compatability with our data loading functions, metadata about each dataset
should be saved as a pandas DataFrame where each row corresponds to a single trial:
- each row should contain the subject ID
- each row should give the 'task' or 'type' of the trial
- each row should include `target_fs` (200 or 256, see above)
- the attributes of the file should include the list of channel names

### Subject-Independent Cross-Validation Splits
The 10-fold subject-independent splits used in the paper are checked into
[data/splits/](data/splits/), one directory per benchmark slug (`ku_mi/`,
`ku_erp/`, `physionet_eyes/`, `physionet_mi/`, `physionet_me/`, `high_gamma/`,
`pavlov_memory/`, `sleep_edf/`). Each fold is a JSON file
`fold_<i>_subjects.json` of the form:

```json
{
  "train_subjects": ["s1", "s10", ...],
  "test_subjects":  ["s3", "s7",  ...]
}
```

The loaders compare the in-memory split against the on-disk file on every
run and abort if they disagree, so historical results stay reproducible.

## Run Benchmarking

Once data has been pre-processed to the expected format and saved under
`data_root` (set in [configs/_base.yaml](configs/_base.yaml)), use the
`cli.*` entry points described below.


## Configs

All configs use a shallow inheritance chain via the `extends:` key. A child config
overrides only what differs from [configs/_base.yaml](configs/_base.yaml), which
fixes `data_root`, `n_folds`, the default `models`/`benchmarks` lists, and the
finetune sweep axes (`finetune_modes`, `large_head`, `exit_block`).

Every CLI takes `--config <path>` and accepts override flags
(`--models`, `--benchmarks`, `--output-dir`, `--overwrite`, …) — see
`python -m cli.<name> --help`.

## Running

Train baseline checkpoints (cross-validated, both head-only and full finetuning):

```commandline
python -m cli.train --config configs/train/baseline.yaml
python -m cli.train --config configs/train/baseline.yaml --models REVE --benchmarks "KU MI"
```

Evaluate existing checkpoints — clean or under perturbation:

```commandline
python -m cli.evaluate --config configs/eval/baseline.yaml
python -m cli.evaluate --config configs/eval/pink_noise.yaml --finetune head_only
```

Generate corrupted datasets on disk (consumed by `cli.evaluate` via
`augmentations:`):

```commandline
python -m cli.corrupt_datasets --config configs/corrupt/default.yaml
```

Linear-probe transformer blocks:

```commandline
python -m cli.probe --config configs/probing/mean.yaml
python -m cli.probe --config configs/probing/concat.yaml --models CBraMod
```

Run attribution analyses (LRP / IxG / GradCAM / attention):

```commandline
python -m cli.interpret lrp --models REVE --benchmarks "Physionet Eyes" --fold 0
python -m cli.interpret ixg --augmentations pink_noise_0db --fold -1
python -m cli.interpret gradcam --models LaBraM --target_layer 5
```

`--fold -1` (the default) iterates all 10 folds.
## Result Aggregation

Per-fold CSVs are written under `results/<experiment>/`. The `scripts/` helpers
collect and summarize them:

```commandline
python scripts/collect_csvs.py        # merge fold CSVs into one table
python scripts/print_result_tables.py # summary tables
python scripts/print_param_counts.py  # trainable params per (model, head)
```


## Repository Layout

```
cli/                       # Entry points
  train.py                 #   fit checkpoints (per fold / model / benchmark)
  evaluate.py              #   eval checkpoints, clean or under augmentations
  probe.py                 #   linear-probe transformer blocks
  interpret.py             #   LRP / IxG / GradCAM / attention
  corrupt_datasets.py      #   write perturbed datasets to disk

configs/                   # YAML configs (child configs extend _base.yaml)
  _base.yaml               #   shared defaults: data_root models, benchmarks, sweep axes...
  train/                   #   baseline, head_experiments, block_exit
  eval/                    #   baseline, white_noise, region_noise, channel_dropout_*...
  probing/                 #   mean-pool / concat-pool of intermediate blocks
  interpret/               #   lrp, ixg, gradcam, attention
  corrupt/                 #   on-disk dataset corruption recipes

data/
  loaders.py               # benchmark loaders + split verification
  splits/                  # fold_<i>_subjects.json
  perturbations/           # sensor_noise, channel_dropout, region_noise generators

models/
  wrappers.py              # FinetuningWrapper subclass per model
  EEGNet/ LaBraM/ CBraMod/ BIOT/ REVE/ BrainOmni/ NeuroRVQm/

interpretability/
  lrp/ ixg/ gradcam/ attention/   # attribution methods
  probing/                        # block-wise probing utilities
  plotting/                       # topomaps, grids
  common/                         # shared helpers

utils/
  collect_csvs.py          # merge per-fold CSVs into one table
  print_result_tables.py   # summary tables
  print_param_counts.py    # trainable params per (model, head)
  plot_results.py          # plotting helpers

weights/                   # pretrained/ (downloaded), finetuned/ (produced by cli.train)
results/                   # per-experiment CSVs (one per fold) and attribution outputs
```


## Citation

If you use this codebase, please cite our paper:

```bibtex
@article{sirca2026beyond,
  title        = {Beyond Accuracy: Robustness, Interpretability and Expressiveness of {EEG} Foundation Models},
  author       = {{\v{S}}irca, Urban and Alimardani, Maryam and Zafeiriou, Stefanos and Barmpas, Konstantinos},
  journal      = {arXiv preprint arXiv:2605.17562},
  year         = {2026},
  doi          = {10.48550/arXiv.2605.17562},
  url          = {https://doi.org/10.48550/arXiv.2605.17562}
}
```

