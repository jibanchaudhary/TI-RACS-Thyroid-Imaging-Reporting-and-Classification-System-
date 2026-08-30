# TI-RACS: TI-RADS Thyroid Nodule Classification

**Automated risk stratification of thyroid nodules from ultrasound imaging.**

TI-RACS classifies thyroid ultrasound images into the five ACR TI-RADS categories (TR1–TR5) using an ensemble of four modern deep learning backbones — ConvNeXt-T, EfficientNetV2-S, Swin-T, and ViT-B/16 — combined by soft voting, with Grad-CAM support for visual explainability.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-1.0-lightgrey)

---

## Demo

https://github.com/user-attachments/assets/150b5544-d6e6-470e-9769-28fedc1d5b0a

---

## Overview

The ACR Thyroid Imaging Reporting and Data System (TI-RADS) stratifies thyroid nodules by malignancy risk:

| Category | Risk level |
|----------|------------|
| TR1 | Benign |
| TR2 | Not suspicious |
| TR3 | Mildly suspicious |
| TR4 | Moderately suspicious |
| TR5 | Highly suspicious |

TI-RACS trains four backbone architectures on annotated ultrasound images and aggregates their predictions:

| Backbone | Family |
|----------|--------|
| ConvNeXt-T | Modernized ConvNet |
| EfficientNetV2-S | Compound-scaled ConvNet |
| Swin-T | Hierarchical vision transformer |
| ViT-B/16 | Vision transformer |

**Key features**

- End-to-end pipeline: dataset construction → training → cross-validated evaluation → inference
- Soft-voting ensemble across all four backbones
- 5-fold cross-validation with ablation tables and confusion matrices
- Grad-CAM heatmaps for model interpretability
- Checkpointing with early stopping

---

## Project Structure

```
├── data_pipeline/
│   ├── create_dataset.py      # Builds train/val/test splits from raw data
│   └── dataset_conversion.py  # Parses XML annotations, converts SVG masks
├── models/
│   └── models.py              # All four backbone models + ensemble wrapper
├── train/
│   └── trainer.py             # Training loop, checkpointing, early stopping
├── scripts/
│   ├── evaluate.py            # 5-fold cross-validation, ablation table
│   └── inference.py           # Predict a single image or the full test set
└── artifacts/ckpts/           # Saved checkpoints
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended for training)

### Installation

```bash
git clone <repository-url>
cd TI-RACS
pip install -r requirements.txt
```

### Data Layout

Place your data under `data/` as follows:

```
data/
├── images/       # Ultrasound PNG/JPEG files (1.png, 2.png, …)
└── annotations/  # One XML annotation file per case (1.xml, 2.xml, …)
```

---

## Usage

### 1. Prepare the dataset

```bash
python data_pipeline/create_dataset.py --data_dir data
```

### 2. Train

Train all four backbones:

```bash
python train/trainer.py --backbone all --data_dir data --output_dir artifacts/output
```

Or a single backbone:

```bash
python train/trainer.py --backbone convnext --data_dir data --output_dir artifacts/output
```

Checkpoints are saved to `artifacts/ckpts/<backbone>/best.pt`.

### 3. Evaluate (5-fold cross-validation)

```bash
python scripts/evaluate.py --data_dir data --output_dir artifacts/output --n_folds 5
```

Produces `ablation_table.csv`, `ablation_bar.png`, and per-model confusion matrix PNGs.

### 4. Inference

Single image with a single backbone (optionally with a Grad-CAM overlay):

```bash
python scripts/inference.py \
    --mode single \
    --image path/to/image.png \
    --backbone convnext \
    --checkpoint artifacts/ckpts/convnext/best.pt \
    --gradcam
```

Soft-voting ensemble across all four backbones:

```bash
python scripts/inference.py \
    --mode ensemble \
    --image path/to/image.png \
    --checkpoints artifacts/ckpts/convnext/best.pt \
                  artifacts/ckpts/efficientnet/best.pt \
                  artifacts/ckpts/swin/best.pt \
                  artifacts/ckpts/vit/best.pt
```

---

## Disclaimer

This project is intended for research and educational purposes only. It is **not** a medical device and must not be used for clinical diagnosis or treatment decisions.
