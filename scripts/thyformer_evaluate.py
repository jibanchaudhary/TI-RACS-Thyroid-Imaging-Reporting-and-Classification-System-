"""
ThyFormer — Evaluation Runner

Loads a trained checkpoint and computes:
  • Full test metrics (AUC, F1, sensitivity, specificity)
  • GradCAM heatmaps for n samples
  • DeLong's test vs a baseline checkpoint
  • Cohen's kappa (clinical agreement) from a radiologist CSV

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --baseline   checkpoints/efficientnet_best.pt \
        --rad_csv    data/radiologist_grades.csv \
        --n_gradcam  50
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from configs.thyformer_config import get_config
from data_pipeline.thyformer_create_dataset import build_dataloaders
from utils.thyformer_loss import build_loss
from models.thyformer_models import build_model
from train.thyformer_train import evaluate
from utils.thyformer_explainability import run_gradcam_batch
from utils.thyformer_metrics import compute_kappa, delong_test, print_results_table


def load_checkpoint(path: str, device: str = "cuda") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"Loaded: {path}  (epoch {ckpt.get('epoch','?')})")
    return ckpt


def run_evaluation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()

    # ── Load model ────────────────────────────────────────────────
    model = build_model(cfg.model)
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    # ── Data ──────────────────────────────────────────────────────
    loaders = build_dataloaders(
        cfg.data, cfg.augmentation, cfg.training.batch_size, cfg.training.num_workers
    )

    loss_fn = build_loss(cfg.loss)

    # ── Test metrics ──────────────────────────────────────────────
    test_m = evaluate(model, loaders["test"], loss_fn, 0, "test", cfg)
    print_results_table(test_m, "ThyFormer — Test Results")

    # ── GradCAM ───────────────────────────────────────────────────
    if args.n_gradcam > 0:
        print(f"\nGenerating {args.n_gradcam} GradCAM figures …")
        run_gradcam_batch(
            model,
            loaders["test"],
            out_dir=cfg.evaluation.gradcam_output_dir,
            n=args.n_gradcam,
            device=device,
        )

    # ── DeLong test vs baseline ───────────────────────────────────
    if args.baseline:
        print("\nComputing DeLong test vs baseline …")

        # Collect ThyFormer predictions
        thy_probs, thy_labels = _collect_probs(model, loaders["test"], device)

        # Load baseline model (assume same architecture for simplicity)
        # In practice: load the specific baseline model class here
        base_model = build_model(cfg.model)
        base_ckpt = load_checkpoint(args.baseline, device)
        base_model.load_state_dict(base_ckpt["model"])
        base_model = base_model.to(device)
        base_probs, _ = _collect_probs(base_model, loaders["test"], device)

        auc_a, auc_b, z, p = delong_test(thy_labels, thy_probs, base_probs)
        print("\nDeLong's test:")
        print(f"  ThyFormer AUC : {auc_a:.4f}")
        print(f"  Baseline AUC  : {auc_b:.4f}")
        print(f"  z-statistic   : {z:.3f}")
        print(f"  p-value       : {p:.4f}  ({'significant' if p<0.05 else 'n.s.'} at α=0.05)")

    # ── Cohen's kappa ─────────────────────────────────────────────
    if args.rad_csv:
        print("\nComputing Cohen's kappa (clinical agreement) …")
        rad_df = pd.read_csv(args.rad_csv)
        # Expected columns: stem, radiologist_label (0..3)
        pred_map = _build_pred_map(model, loaders["test"], device)

        stems = rad_df["stem"].values
        rad_l = rad_df["radiologist_label"].values
        mod_l = np.array([pred_map.get(s, -1) for s in stems])
        valid = mod_l >= 0
        if valid.sum() < len(stems):
            print(f"  WARNING: {(~valid).sum()} stems not found in predictions — skipped")

        kappa_m = compute_kappa(mod_l[valid], rad_l[valid])
        print_results_table(kappa_m, "Clinical Agreement (Cohen's κ)")
        k = kappa_m.get("kappa", 0.0)
        interp = (
            "Almost perfect"
            if k > 0.80
            else "Substantial"
            if k > 0.60
            else "Moderate"
            if k > 0.40
            else "Fair"
            if k > 0.20
            else "Slight"
        )
        print(f"  Interpretation: {interp}")


@torch.no_grad()
def _collect_probs(model, loader, device) -> tuple:
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        out = model(imgs)
        probs = F.softmax(out["cls_logits"], 1).cpu().numpy()
        lbs = batch["label"]
        if lbs.dim() == 2:
            lbs = lbs.argmax(1)
        all_probs.append(probs)
        all_labels.append(lbs.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


@torch.no_grad()
def _build_pred_map(model, loader, device) -> dict:
    model.eval()
    pred_map = {}
    for batch in loader:
        imgs = batch["image"].to(device)
        stems = batch["stem"]
        out = model(imgs)
        preds = out["cls_logits"].argmax(1).cpu().numpy()
        for stem, pred in zip(stems, preds):
            pred_map[stem] = int(pred)
    return pred_map


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--baseline", default=None, help="Baseline checkpoint for DeLong test")
    p.add_argument("--rad_csv", default=None, help="CSV with radiologist grades for kappa")
    p.add_argument("--n_gradcam", type=int, default=50)
    args = p.parse_args()
    run_evaluation(args)
