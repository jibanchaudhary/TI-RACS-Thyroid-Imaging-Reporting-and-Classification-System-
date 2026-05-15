# TI-RADS Thyroid Nodule Classification

Classifies thyroid ultrasound images into 5 TI-RADS categories (TR-1 to TR-5) using an ensemble of four deep learning models: ConvNeXt-T, EfficientNetV2-S, Swin-T, and ViT-B/16.

---

## Project Structure

```
├── data_pipeline/
│   ├── create_dataset.py       ← builds train/val/test splits from raw data
│   └── dataset_conversion.py  ← parses XML annotations, converts SVG masks
├── models/
│   └── models.py              ← all four backbone models + ensemble wrapper
├── train/
│   └── trainer.py             ← training loop, checkpointing, early stopping
├── scripts/
│   ├── evaluate.py            ← 5-fold cross-validation, ablation table
│   └── inference.py           ← predict a single image or full test set
└── artifacts/ckpts                ← saved checkpoints go here
```

---

## Data Layout

```
data/
├── images/       ← ultrasound PNG/JPEG files (1.png, 2.png …)
└── annotations/  ← one XML file per case     (1.xml, 2.xml …)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## 1. Prepare the dataset

```bash
python data_pipeline/create_dataset.py --data_dir data
```

---

## 2. Train

```bash
# All four models
python train/trainer.py --backbone all --data_dir data --output_dir artifacts/output

# Single model
python train/trainer.py --backbone convnext --data_dir data --output_dir artifacts/output
```

Checkpoints are saved to `artifacts/ckpts/<backbone>/best.pt`.

---

## 3. Evaluate (5-fold cross-validation)

```bash
python scripts/evaluate.py --data_dir data --output_dir artifacts/output --n_folds 5
```

Produces `ablation_table.csv`, `ablation_bar.png`, and confusion matrix PNGs.

---

## 4. Inference

```bash
# Single image
python scripts/inference.py \
    --mode single \
    --image path/to/image.png \
    --backbone convnext \
    --checkpoint artifacts/ckpts/convnext/best.pt \
    --gradcam

# Ensemble (all four models, soft-voting)
python scripts/inference.py \
    --mode ensemble \
    --image path/to/image.png \
    --checkpoints artifacts/ckpts/convnext/best.pt \
                  artifacts/ckpts/efficientnet/best.pt \
                  artifacts/ckpts/swin/best.pt \
                  artifacts/ckpts/vit/best.pt
