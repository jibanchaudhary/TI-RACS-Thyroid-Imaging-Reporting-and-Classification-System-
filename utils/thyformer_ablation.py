"""
ThyFormer — Ablation Study Runner

Trains 4 variants by disabling each novel module:
  1. Full ThyFormer        (all modules ON)
  2. w/o Despeckling stem  (replace with standard Conv2d stem)
  3. w/o ECA module        (replace with standard SE attention)
  4. w/o Boundary loss     (set gamma=0, CE+Dice only)

Usage:
    python scripts/ablation.py --config_preset base
"""
import argparse
import copy

import torch

from configs.config import get_config
from data.dataset import build_dataloaders
from models.thyformer import ThyFormer, build_model
from losses.composite_loss import build_loss
from scripts.train import train
from utils.metrics import print_comparison_table


# ─────────────────────────────────────────────────────────────────
# Variant 2 — vanilla stem (replaces DespecklingCNNStem)
# ─────────────────────────────────────────────────────────────────

import torch.nn as nn


class VanillaStem(nn.Module):
    """Standard 4×4 patch embedding — no speckle-aware processing."""

    def __init__(self, in_ch=3, out_ch=96, patch=4):
        super().__init__()
        self.pe = nn.Conv2d(in_ch, out_ch, patch, stride=patch, bias=False)
        self.ln = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.pe(x)
        B, C, H, W = x.shape
        return self.ln(x.flatten(2).transpose(1, 2)), H, W


# ─────────────────────────────────────────────────────────────────
# Variant 3 — standard SE attention (replaces ECA)
# ─────────────────────────────────────────────────────────────────


class StandardSE(nn.Module):
    """Classic Squeeze-and-Excitation (no echo-bin structure)."""

    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(inplace=True), nn.Linear(mid, channels), nn.Sigmoid()
        )

    def forward(self, x):
        sq = self.pool(x.transpose(1, 2)).squeeze(-1)
        w = self.fc(sq)
        echo_dummy = torch.zeros(x.size(0), 3, device=x.device)
        return x * w.unsqueeze(1), echo_dummy


# ─────────────────────────────────────────────────────────────────
# Build ablation variants
# ─────────────────────────────────────────────────────────────────


def build_variant(name: str, cfg) -> ThyFormer:
    model = build_model(cfg.model)

    if name == "no_stem":
        model.stem = VanillaStem(out_ch=96)

    elif name == "no_eca":
        model.eca = StandardSE(channels=96, reduction=cfg.model.eca_reduction_ratio)

    # "no_boundary" handled via loss config — no model change needed
    return model


def build_loss_variant(name: str, cfg):
    loss_cfg = cfg.loss
    if name == "no_boundary":
        loss_cfg = copy.copy(loss_cfg)
        loss_cfg.gamma = 0.0
        loss_cfg.boundary_warmup_epochs = 0
    return build_loss(loss_cfg)


# ─────────────────────────────────────────────────────────────────
# Main ablation runner
# ─────────────────────────────────────────────────────────────────

VARIANTS = [
    ("full", "Full ThyFormer (all modules)"),
    ("no_stem", "w/o Despeckling stem"),
    ("no_eca", "w/o ECA module"),
    ("no_boundary", "w/o Boundary loss"),
]


def run_ablation(cfg):
    loaders = build_dataloaders(
        cfg.data, cfg.augmentation, cfg.training.batch_size, cfg.training.num_workers
    )

    all_results, all_names = [], []

    for variant_id, variant_label in VARIANTS:
        print(f"\n{'='*55}")
        print(f"  VARIANT: {variant_label}")
        print(f"{'='*55}")

        # Shorten training for ablation (half epochs)
        abl_cfg = copy.deepcopy(cfg)
        abl_cfg.training.epochs = max(cfg.training.epochs // 2, 10)
        abl_cfg.training.early_stopping_patience = 5
        abl_cfg.training.experiment_name = f"ablation_{variant_id}"
        abl_cfg.training.checkpoint_dir = f"checkpoints/ablation_{variant_id}"

        model = build_variant(variant_id, abl_cfg)
        loss_fn = build_loss_variant(variant_id, abl_cfg)
        results = train(model, loaders, loss_fn, abl_cfg)
        all_results.append(results)
        all_names.append(variant_label)

    # Print comparison
    print_comparison_table(
        all_names, all_results, keys=["auc", "f1", "sensitivity", "specificity", "accuracy"]
    )

    # Delta vs full model
    base_auc = all_results[0].get("test_auc", 0.0)
    print("\nΔ AUC vs full ThyFormer:")
    for name, res in zip(all_names[1:], all_results[1:]):
        delta = res.get("test_auc", 0.0) - base_auc
        print(f"  {name:<35s}  {delta:+.4f}")

    return dict(zip(all_names, all_results))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_preset", default="base", choices=["base"], help="Config preset to use")
    args = p.parse_args()

    cfg = get_config()
    run_ablation(cfg)
