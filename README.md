# STFC-DANet

This repository provides the official implementation of **STFC-DANet** for binary classification of autism spectrum disorder (ASD) using resting-state functional magnetic resonance imaging (rs-fMRI) data from ABIDE. The model uses a `CNN-Transformer` backbone and integrates three additional modules:

```text
Backbone:     CNN-Transformer (ROI time series)
Innovation 1: Spectral Branch (low-frequency BOLD spectrum)
Innovation 2: FC Branch (static functional connectivity)
Innovation 3: DANN (cross-site domain adversarial learning)
```

## Highlights

1. **STFC-DANet model.** The model combines temporal dynamics, low-frequency spectral information, static functional connectivity, and domain adversarial learning to jointly model ROI time series, spectral BOLD fluctuations, functional connectivity, and cross-site domain shifts.

2. **Cross-site generalization evaluation.** Standard five-fold cross-validation and Leave-One-Site-Out (LOSO) protocols are provided to evaluate generalization under distribution shifts among acquisition sites.

## Setup

The following commands assume that the current working directory is the repository root:

```bash
pip install -r requirements.txt
```

## Data

Due to their large size, the raw and preprocessed data files are **not included in this repository** and must be prepared locally:

```text
ABIDE_CC400_NotCC_NotQC_shufflerandomSeed42.npz                       # Functional connectivity data provided by the authors
ABIDE_CC400_NOTQC_TPE_withSITE_shuffle_randomSeed1234_filtGlobal.npz # Time-series data provided by the authors
```

Place both `.npz` files in the repository root, then run the preprocessing script to generate the local training cache:

```bash
python scripts/preprocess_author_npz.py \
  --connectivity-npz ABIDE_CC400_NotCC_NotQC_shufflerandomSeed42.npz \
  --timeseries-npz ABIDE_CC400_NOTQC_TPE_withSITE_shuffle_randomSeed1234_filtGlobal.npz \
  --output-dir data/abide_cc400_author_npz
```

The preprocessing script automatically creates `data/abide_cc400_author_npz/` and generates the following files:

```text
X.npy                   # ROI time series
fc.npy                  # Flattened upper-triangular functional connectivity features
y.npy                   # ASD/control labels
domain.npy              # Site domain labels
metadata.csv            # Metadata
site_mapping.json       # Site mapping
preprocess_summary.json # Preprocessing summary
```

These files are local data caches. They are excluded by `.gitignore` and are not distributed with the repository.

The generated raw cache uses the source convention `0=ASD, 1=HC`. Training retains this cache encoding, while reported classification metrics and exported predictions use the ASD-positive convention `1=ASD, 0=HC`. Thus, sensitivity, F1, AUC, and ASD probabilities consistently treat ASD as the positive class.

## Fixed Benchmark Splits

Generate the fixed five-fold train/validation/test partitions once after preprocessing:

```bash
python scripts/export_benchmark_splits.py \
  --data-dir data/abide_cc400_author_npz \
  --output-dir data/benchmark_splits \
  --split-seed 42 \
  --folds 5
```

This creates `data/benchmark_splits/abide_cc400_5fold_seed42.npz` and a manifest containing the split settings and subject-order checksum. Pass this split file to every cross-validation run to reuse exactly the same train, validation, and test subjects.

## Network Overview

### Backbone: CNN-Transformer

The backbone uses only ROI time series:

```text
ROI time series [B, T, ROI]
→ Multi-scale temporal CNN stem
→ Residual temporal CNN blocks
→ Transformer encoder
→ CLS / mean / max pooling
→ Temporal feature
→ Classifier
```

This branch learns local dynamic patterns and long-range temporal dependencies in BOLD time series.

### Innovation 1: Spectral Branch

The Spectral Branch extracts low-frequency spectral information from the same ROI time series:

```text
ROI time series
→ rFFT along the temporal dimension
→ Remove the DC component
→ Retain the first 32 low-frequency bins
→ Apply log1p to the magnitude
→ Per-ROI spectral projection
→ Mean / max pooling across ROIs
→ Spectral feature
```

**Motivation:** BOLD signals in rs-fMRI primarily reflect low-frequency spontaneous neural activity. ASD-related abnormalities may appear not only in temporal dynamics but also in the intensity of low-frequency oscillations and the distribution of spectral energy across ROIs. The Spectral Branch transforms ROI time series into the frequency domain and directly models low-frequency BOLD fluctuations, providing information complementary to the temporal branch. The default setting is:

```text
spectral_bins = 32
```

### Innovation 2: FC Branch

This branch uses static functional connectivity features:

```text
Upper triangle of FC matrix [B, 76636]
→ Linear(76636, 512)
→ LayerNorm + GELU + Dropout
→ Linear(512, 128)
→ LayerNorm + GELU + Dropout
→ Linear(128, 128)
→ FC feature
```

**Motivation:** ASD is believed to be associated with abnormalities in functional brain network connectivity. ROI time series represent dynamic activity trajectories, whereas functional connectivity matrices characterize long-term synchronization among ROIs. The FC Branch uses pairwise functional connectivity as an independent input to complement global connectivity patterns that are difficult for the temporal branch to capture directly. `FC-MLP` is used by default; `wide_mlp`, `gated_mlp`, and `transformer` are also available for FC branch architecture comparisons.

### Innovation 3: DANN

DANN is used to reduce site-related bias in the multi-site ABIDE dataset:

```text
Fused feature
→ Label classifier
→ Gradient Reversal Layer
→ Domain classifier
```

The default value of `dann_lambda` is:

```text
dann_lambda = 0.1
```

**Motivation:** ABIDE is a typical multi-site dataset. Sites differ in scanners, acquisition protocols, participant composition, and preprocessing quality. Through a Gradient Reversal Layer, DANN reduces site discriminability in the shared features while preserving their ASD classification ability, thereby encouraging the model to learn site-invariant representations.

### Final Model

```text
CNN-Transformer Temporal Branch
+ Spectral Branch
+ FC-MLP Branch
+ DANN
```

After fusion, the three feature streams are passed to the label classifier and the domain adversarial head.

## Training

### Final Model (Five-Fold Cross-Validation)

```bash
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 1 --spectral-bins 32 --output-dir stfc_danet
```

When `--output-dir` is a relative name, results are saved to `outputs/<name>`. Absolute paths and paths beginning with `outputs/` are used as provided.

### Ablation Studies

```bash
# Baseline: CNN-Transformer only
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 0 --use-fc-branch 0 --use-dann 0 --output-dir abl_baseline

# Add the Spectral Branch
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 1 --use-fc-branch 0 --use-dann 0 --output-dir abl_spectral

# Add the FC Branch
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 0 --use-fc-branch 1 --use-dann 0 --output-dir abl_fc

# Add DANN
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 0 --use-fc-branch 0 --use-dann 1 --output-dir abl_dann

# Add the Spectral and FC Branches
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 1 --use-fc-branch 1 --use-dann 0 --output-dir abl_spectral_fc

# Add the Spectral Branch and DANN
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 1 --use-fc-branch 0 --use-dann 1 --output-dir abl_spectral_dann

# Add the FC Branch and DANN
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 0 --use-fc-branch 1 --use-dann 1 --output-dir abl_fc_dann

# Full model: Spectral Branch + FC Branch + DANN
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 1 --use-fc-branch 1 --use-dann 1 --output-dir abl_all
```

Optional FC branch architecture comparisons:

```bash
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 0 --fc-branch-type wide_mlp --output-dir abl_fc_wide_mlp
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 0 --fc-branch-type gated_mlp --output-dir abl_fc_gated_mlp
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 0 --fc-branch-type transformer --output-dir abl_fc_transformer
```

Optional spectral-bin comparisons:

```bash
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 1 --spectral-bins 16 --output-dir abl_all_spectral_bins16
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 1 --spectral-bins 24 --output-dir abl_all_spectral_bins24
python scripts/train_cnn_transformer_cv.py --split-file data/benchmark_splits/abide_cc400_5fold_seed42.npz --use-spectral-branch 1 --spectral-bins 48 --output-dir abl_all_spectral_bins48
```

### Leave-One-Site-Out (LOSO)

LOSO evaluates the model's ability to generalize across acquisition sites. The CC400 cache contains 20 sites. In each iteration, one site is held out as the test set, while the remaining 19 sites are used for training and validation.

```bash
python scripts/train_cnn_transformer_loso.py --output-dir loso_stfc_danet --device cuda
```

If CUDA is unavailable, run the model on the CPU:

```bash
python scripts/train_cnn_transformer_loso.py --output-dir loso_stfc_danet --device cpu --no-amp
```

## Outputs

Each experiment directory contains the following files:

```text
fold_metrics.csv
summary_metrics.csv
summary_metrics.json
summary_metrics_raw.csv
summary_metrics_raw.json
fold_*/history.csv
fold_*/metrics.json
fold_*/model.pt
fold_*/best_model.pt
```

The following metrics are printed to the console at the end:

```text
Accuracy
Sensitivity
Specificity
F1
AUC
```
