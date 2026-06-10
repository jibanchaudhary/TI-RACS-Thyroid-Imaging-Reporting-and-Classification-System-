# ThyFormer

A hybrid CNN-Transformer for thyroid nodule TI-RADS classification from ultrasound images.

ThyFormer combines a novel despeckling CNN stem, echogenicity-aware channel attention, a Swin Transformer encoder, and a dual-head architecture (classification + segmentation) with a MedSAM-guided boundary loss.


## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [How to Run Training](#how-to-run-training)
- [How to Run Evaluation](#how-to-run-evaluation)
- [How to Run Inference](#how-to-run-inference)
- [How to Run Ablation Study](#how-to-run-ablation-study)
- [How Modules Connect](#how-modules-connect)
- [Configuration Reference](#configuration-reference)
- [Model Architecture Details](#model-architecture-details)
- [Loss Function Details](#loss-function-details)
- [Metrics and Evaluation](#metrics-and-evaluation)
- [Explainability](#explainability)
- [Troubleshooting](#troubleshooting)


## Architecture Overview

```
Input Image [B,3,224,224]
        │
        ▼
┌─────────────────────────┐
│  Stage 1: Despeckling    │  ★ Novel
│  CNN Stem                │  DWConv → GroupNorm → GELU → PatchEmbed
│  → tokens [B,3136,96]   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Stage 2: Echogenicity   │  ★ Novel
│  Channel Attention (ECA) │  3 echo bins: hypo/iso/hyperechoic
│  → tokens [B,3136,96]   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Stage 3: Swin-T         │  timm backbone (features_only)
│  Transformer Encoder     │  Outputs: F1[96] F2[192] F3[384] F4[768]
└──────┬──────────┬───────┘
       ▼          ▼
┌────────────┐ ┌──────────────┐
│ Stage 4a:  │ │ Stage 4b:    │
│ Cls Head   │ │ FPN Decoder  │
│ → [B,4]    │ │ → [B,1,H,W]  │
│ (T1–T4)    │ │ (nodule mask) │
└────────────┘ └──────────────┘
```


## Project Structure

```
thyformer/
├── configs/
│   ├── __init__.py
│   └── config.py               # ThyFormerConfig — all hyperparameters
│
├── data/
│   ├── __init__.py
│   ├── make_splits.py          # Step 1: annotations → train/val/test CSVs
│   ├── precompute_medsam.py    # Step 2: images → boundary .npy files
│   └── dataset.py              # Dataset class, augmentation, DataLoader factory
│
├── models/
│   ├── __init__.py
│   └── thyformer.py            # Full model: Stem → ECA → Swin → Cls + FPN
│
├── losses/
│   ├── __init__.py
│   └── composite_loss.py       # L = α·CE + β·Dice + γ(t)·Boundary
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py              # AUC, F1, sensitivity, specificity, DeLong, kappa
│   ├── logging_utils.py        # CSV logger + optional W&B
│   └── explainability.py       # GradCAM + Attention Rollout
│
├── scripts/
│   ├── __init__.py
│   ├── train.py                # Training loop — run directly
│   ├── evaluate.py             # Full evaluation
│   ├── inference.py            # Predict on new images
│   └── ablation.py             # 4-variant ablation study
│
├── data/
│   ├── vindr_thyroid/           # ← Put your raw data here
│   │   ├── annotations.csv
│   │   ├── images/
│   │   └── masks/               # (optional)
│   ├── splits/                  # ← Created by Step 1
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── medsam_boundaries/       # ← Created by Step 2
│       └── {stem}.npy
│
├── checkpoints/                 # ← Created during training
├── logs/                        # ← Training logs
├── outputs/                     # ← Evaluation outputs
└── requirements.txt
```


## Installation

```
pip install torch torchvision timm albumentations opencv-python-headless pandas numpy scikit-learn scipy matplotlib
```

Optional:
```
pip install wandb              # for W&B logging
pip install segment-anything   # for MedSAM boundary precomputation
```


## Data Preparation

Before training, you need to run two preprocessing steps.

### Step 1: Create Train/Val/Test Splits

This reads your raw annotations CSV and creates stratified 70/15/15 splits.

```
python -m data.make_splits \
    --annotations data/vindr_thyroid/annotations.csv \
    --images_dir  data/vindr_thyroid/images \
    --out_dir     data/splits \
    --seed        42
```

This creates `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`.

Label mapping: TR1/T1/1 → 0, TR2/T2/2 → 1, TR3/T3/3 → 2, TR4/T4/TR5/T5/4/5 → 3.


### Step 2: Precompute MedSAM Boundary Maps

Run this once. It generates `.npy` boundary files used by the boundary loss during training.

Usage:
    # Just provide the dataset directory — masks are auto-generated from JSON annotations
    python data/precompute_medsam.py \
        --data_dir  data/vindr_thyroid \
        --out_dir   data/medsam_boundaries \
        --device    cuda

    # Or provide separate dirs if masks already exist
    python data/precompute_medsam.py \
        --images_dir  data/vindr_thyroid/images \
        --masks_dir   data/vindr_thyroid/masks \
        --out_dir     data/medsam_boundaries \
        --device      cuda

Requires (optional):
    MedSAM model weights downloaded from:
    https://github.com/bowang-lab/MedSAM
    Place at: checkpoints/medsam_vit_b.pth
"""

## How to Run Training

From the project root directory:

```
python -m scripts.train
```

That's it. It uses default settings from `configs/config.py`.

To override settings from the command line:

```
python -m scripts.train \
    --data_root   data/vindr_thyroid \
    --train_csv   data/splits/train.csv \
    --val_csv     data/splits/val.csv \
    --test_csv    data/splits/test.csv \
    --medsam_dir  data/medsam_boundaries \
    --epochs      50 \
    --batch_size  16 \
    --lr_backbone 1e-4 \
    --lr_head     1e-3
```

All available training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `data/vindr_thyroid` | Root directory containing images |
| `--train_csv` | `data/splits/train.csv` | Path to training split CSV |
| `--val_csv` | `data/splits/val.csv` | Path to validation split CSV |
| `--test_csv` | `data/splits/test.csv` | Path to test split CSV |
| `--medsam_dir` | `data/medsam_boundaries` | Path to boundary .npy files |
| `--epochs` | `50` | Maximum training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr_backbone` | `1e-4` | Swin backbone learning rate |
| `--lr_head` | `1e-3` | Classification/segmentation head learning rate |
| `--no_fp16` | `False` | Disable mixed precision (use if GPU doesn't support FP16) |
| `--wandb` | `False` | Enable Weights & Biases logging |
| `--experiment` | `thyformer_v1` | Experiment name for logging |
| `--checkpoint_dir` | `checkpoints` | Where to save model checkpoints |

What happens during training:
- AdamW optimizer with differential LR (backbone vs head).
- 5-epoch linear warmup, then cosine annealing to 1e-6.
- FP16 mixed precision with gradient clipping (max norm 1.0).
- Weighted random sampling to handle class imbalance.
- Mixup augmentation (alpha=0.2, probability=0.3).
- Early stopping on `val_auc` with patience=10.
- Saves top-3 checkpoints by AUC to `checkpoints/`.
- After training completes, automatically loads the best checkpoint and evaluates on the test set.

Output:
- Checkpoints: `checkpoints/ep{epoch}_auc{score}.pt`
- Metrics log: `logs/metrics.csv`
- Console output with per-epoch loss, AUC, F1.


## How to Run Evaluation

```
python -m scripts.evaluate --checkpoint checkpoints/best.pt
```

With all options:

```
python -m scripts.evaluate \
    --checkpoint checkpoints/best.pt \
    --baseline   checkpoints/efficientnet_best.pt \
    --rad_csv    data/radiologist_grades.csv \
    --n_gradcam  50
```

| Flag | Description |
|------|-------------|
| `--checkpoint` | (Required) Path to trained ThyFormer checkpoint |
| `--baseline` | Baseline checkpoint for DeLong's statistical comparison |
| `--rad_csv` | CSV with radiologist grades for Cohen's kappa (columns: `stem`, `radiologist_label`) |
| `--n_gradcam` | Number of GradCAM heatmaps to generate (default: 50) |

Output: test metrics table, GradCAM images in `outputs/gradcam/`, DeLong p-value, Cohen's kappa.


## How to Run Inference

Single image:
```
python -m scripts.inference \
    --checkpoint checkpoints/best.pt \
    --image path/to/ultrasound.png
```

Folder of images:
```
python -m scripts.inference \
    --checkpoint checkpoints/best.pt \
    --input_dir  data/new_cases/ \
    --output_dir outputs/predictions/ \
    --gradcam \
    --save_masks
```

CSV batch:
```
python -m scripts.inference \
    --checkpoint checkpoints/best.pt \
    --csv        data/splits/test.csv \
    --output_dir outputs/predictions/
```

Output per image: predicted TI-RADS class (T1–T4) with confidence scores, optional segmentation mask, optional GradCAM overlay, and a summary CSV report.


## How to Run Ablation Study

```
python -m scripts.ablation --config_preset base
```

Trains 4 variants to measure each novel component's contribution:

| Variant | What Changes |
|---------|-------------|
| Full ThyFormer | All modules active (baseline) |
| w/o Despeckling stem | Replaced with standard Conv2d patch embedding |
| w/o ECA module | Replaced with standard Squeeze-and-Excitation |
| w/o Boundary loss | γ=0, only CE + Dice loss |

Runs at half epochs with patience=5 for speed. Output: comparison table + ΔAUC vs full model.


## How Modules Connect

```
annotations.csv ──► make_splits.py ──► train.csv, val.csv, test.csv
                                                │
images/ + masks/ ──► precompute_medsam.py ──► boundary .npy files
                                                │
                                                ▼
                                        dataset.py
                                    (loads CSVs + .npy,
                                     augmentation, DataLoaders)
                                                │
                    config.py ──────────────────►│
                    thyformer.py (model) ───────►│
                    composite_loss.py (loss) ───►│
                                                ▼
                                          train.py
                                    (uses metrics.py + logging_utils.py)
                                                │
                                      saves checkpoints/
                                                │
                    ┌───────────────┬────────────┼──────────────┐
                    ▼               ▼            ▼              ▼
              evaluate.py     inference.py  explainability.py  ablation.py
```

Import dependency table:

| File | Imports From |
|------|-------------|
| `scripts/train.py` | config, thyformer, composite_loss, metrics, logging_utils |
| `data/dataset.py` | config (DataConfig, AugmentationConfig) |
| `models/thyformer.py` | config (ModelConfig), timm |
| `losses/composite_loss.py` | config (LossConfig) |
| `scripts/evaluate.py` | config, dataset, thyformer, composite_loss, train, explainability, metrics |
| `scripts/inference.py` | config, thyformer |
| `scripts/ablation.py` | config, dataset, thyformer, composite_loss, train, metrics |
| `utils/explainability.py` | thyformer (model class) |


## Configuration Reference

All settings are in `configs/config.py`. Key values:

**ModelConfig** — `swin_tiny_patch4_window7_224`, pretrained=True, drop_path=0.1, ECA with 3 echo bins, FPN with 128 channels.

**TrainingConfig** — 50 epochs, batch 16, lr_backbone=1e-4, lr_head=1e-3, weight_decay=1e-4, 5-epoch warmup, cosine scheduler, FP16 on, early stop patience=10 on val_auc.

**LossConfig** — α=1.0 (CE), β=0.5 (Dice), γ=0.3 (Boundary), boundary warmup over 10 epochs, label smoothing=0.1, class weighting enabled.

**AugmentationConfig** — CLAHE (clip=2.0), horizontal flip (p=0.5), rotation ±15°, elastic deformation (p=0.3), brightness/contrast ±0.15, speckle noise (var=0.05, p=0.4), Mixup (alpha=0.2, p=0.3).

To change settings, either edit `configs/config.py` directly, or use CLI flags when running `scripts/train.py`.


## Model Architecture Details

**Stage 1 — DespecklingCNNStem (Novel):** Depthwise-separable convolution initialised with Gaussian kernel weights for speckle suppression, followed by 4×4 patch embedding. Replaces the standard linear projection in vanilla ViTs with an inductive bias for noisy ultrasound inputs.

**Stage 2 — EchogenicityChannelAttention (Novel):** Channel attention whose hidden layer projects to 3 echogenicity bins (hypoechoic, isoechoic, hyperechoic), directly mapping to ACR TI-RADS feature language. Unlike standard SE blocks, the intermediate representation is clinically interpretable.

**Stage 3 — Swin Transformer Encoder:** Swin-T backbone from timm with shifted-window attention. Tokens from Stages 1–2 bypass the built-in patch embedding and feed directly into Swin stages. Outputs hierarchical features F1(96), F2(192), F3(384), F4(768).

**Stage 4a — ClassificationHead:** GAP → LayerNorm → Dropout(0.2) → FC(4) for T1–T4 classification.

**Stage 4b — FPNDecoder:** Feature Pyramid Network fusing F1–F4 via top-down pathway for binary nodule segmentation at 224×224.


## Loss Function Details

```
L_total = α · L_ce + β · L_dice + γ(t) · L_boundary
```

- **L_ce (α=1.0):** Soft cross-entropy supporting hard labels and Mixup one-hot targets, with label smoothing (0.1) and optional class weighting.
- **L_dice (β=0.5):** Standard Dice loss for binary segmentation.
- **L_boundary (γ=0.3 max, Novel):** Compares morphological boundary of predicted mask against MedSAM-precomputed boundary maps via weighted BCE. Boundary pixels get 5× higher weight, forcing the model to attend to nodule margins — the critical region for T2↔T3 confusion.
- **γ(t) warmup:** Boundary weight ramps linearly from 0 to 0.3 over the first 10 epochs, allowing stable early convergence before boundary precision kicks in.


## Metrics and Evaluation

`utils/metrics.py` computes: macro AUC (primary, one-vs-rest), per-class AUC (T1–T4), macro and per-class F1, sensitivity, specificity, accuracy, DeLong's test (bootstrap, 1000 iterations) for comparing two models, and Cohen's quadratic kappa for ordinal agreement with radiologist grades.


## Explainability

**GradCAM** hooks on the last Swin stage's norm layer, producing a 224×224 heatmap showing which regions most influence classification.

**Attention Rollout** (Abnar & Zuidema, ACL 2020) multiplicatively accumulates attention maps across all Swin layers for a global view of information flow.

Both produce overlay visualisations saved as side-by-side plots (original / heatmap / overlay) with prediction labels.


## Troubleshooting

**"Image not found" errors:** Check that `data_root` in `configs/config.py` points to your image directory, and that paths in CSVs are relative to it.

**Empty boundary maps (all zeros):** Neither GT masks nor MedSAM were available in Step 2. Training works but boundary loss contributes nothing. Provide masks or install segment-anything with MedSAM weights.

**CUDA out of memory:** Reduce `batch_size` (try 8 or 4), or ensure `fp16 = True`.

**Poor T2 vs T3 discrimination:** Ensure boundary maps exist. Consider increasing `gamma` in LossConfig.

**timm model not found:** Run `pip install timm --upgrade` (needs ≥0.9).

**ModuleNotFoundError for configs/data/models:** Make sure you're running from the project root directory, not from inside `scripts/`.
