"""
evaluate.py
-----------
5-Fold cross-validation runner and ablation study generator.

This is the evaluation script you run to produce the numbers for your
research paper tables.

Outputs (all saved to output_dir/)
-------
  cv_results.csv       — per-fold metrics for each backbone
  ablation_table.csv   — side-by-side comparison of all models
  roc_curves.png       — OvR ROC curves for each backbone
  confusion_matrices/  — one PNG per backbone × fold

Usage
-----
  python evaluate.py --data_dir data --output_dir paper_results --epochs 30
"""

import csv
import os
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import (
    DataLoader,
    Subset,
)

from data_pipeline.Binary_classification_create_dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    BinaryThyroidDataset,
)
from models.models import build_model
from scripts.inference import evaluate_test_set
from train.trainer import (
    LabelSmoothingCE,
    WarmupCosineScheduler,
)

matplotlib.use("Agg")


def run_kfold(
    backbone: str,
    data_dir: str,
    output_dir: str,
    epochs: int = 20,
    n_folds: int = 3,
    batch_size: int = 16,
    lr: float = 3e-4,
    seed: int = 42,
) -> list[dict]:
    """
    Returns list of n_folds dicts, each with accuracy, f1, auc.
    Also saves fold-level confusion matrix PNGs.
    """

    ckpt_dir = os.path.join(output_dir, "resume_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load full dataset (no split — kfold handles it)
    full_ds = BinaryThyroidDataset(
        data_dir, "train", backbone, train_ratio=1.0, val_ratio=0.0, seed=seed
    )
    all_labels = [full_ds.cases[i]["label"] for i in range(len(full_ds))]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(full_ds)), all_labels), start=1
    ):
        ckpt_path = os.path.join(ckpt_dir, f"{backbone}_fold{fold}.pt")
        done_path = os.path.join(ckpt_dir, f"{backbone}_fold{fold}_done.json")
        if os.path.exists(done_path):
            with open(done_path) as f:
                results.append(json.load(f))
            print(f"Skipping fold {fold}")
            continue

        print(f"\n{'='*55}")
        print(f" {backbone.upper()}  |  Fold {fold}/{n_folds}")
        print(f"{'='*55}")

        train_sub = Subset(full_ds, train_idx)
        val_sub = Subset(full_ds, val_idx)

        # Class-weighted sampler for this fold
        fold_labels = [all_labels[i] for i in train_idx]
        from collections import Counter

        counts = Counter(fold_labels)
        weights = [1.0 / counts[fold_labels[i]] for i in range(len(fold_labels))]
        from torch.utils.data import WeightedRandomSampler

        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(
            train_sub,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_sub, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        model = build_model(backbone, freeze_epochs=5)
        model.to(device)

        criterion = LabelSmoothingCE()
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": lr * 0.1},
                {"params": model.head.parameters(), "lr": lr},
            ],
            weight_decay=1e-4,
        )
        scheduler = WarmupCosineScheduler(optimizer, 3, epochs)

        best_val_loss = float("inf")
        best_state = None

        resume_epoch = 1
        if os.path.exists(ckpt_path):
            print(f"Resuming_{backbone}_fold:{fold}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_val_loss = ckpt["best_val_loss"]
            best_state = ckpt["best_state"]
            resume_epoch = ckpt["epoch"] + 1
            print(
                f"Loaded checkpoint: "
                f"epoch {ckpt['epoch']} "
                f"→ resuming at "
                f"epoch {resume_epoch}"
            )

        for epoch in range(resume_epoch, epochs + 1):
            model.train()
            model.on_epoch_start(epoch)
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Quick val check
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    val_loss += criterion(model(images), labels).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                import copy

                best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_state": best_state,
                },
                ckpt_path,
            )
        # Evaluate best state
        model.load_state_dict(best_state)
        fold_metrics = evaluate_test_set(model, val_loader, device)

        print(
            f"\nFold {fold} | acc {fold_metrics['accuracy']:.4f} | "
            f"f1 {fold_metrics['macro_f1']:.4f} | "
            f"auc {fold_metrics['macro_auc']:.4f}"
        )

        result = {
            "fold": fold,
            "accuracy": fold_metrics["accuracy"],
            "macro_f1": fold_metrics["macro_f1"],
            "macro_auc": fold_metrics["macro_auc"],
        }
        results.append(result)
        with open(done_path, "w") as f:
            json.dump(result, f)

        # Save confusion matrix
        _save_confusion_matrix(
            fold_metrics["confusion_matrix"],
            backbone,
            fold,
            os.path.join(output_dir, "confusion_matrices", f"{backbone}_fold{fold}.png"),
        )
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    return results


def run_ablation(
    data_dir: str,
    output_dir: str,
    epochs: int = 30,
    n_folds: int = 5,
):
    os.makedirs(output_dir, exist_ok=True)
    backbones = ["convnext", "efficientnet", "swin", "vit"]
    all_results = {}

    for bb in backbones:
        fold_results = run_kfold(bb, data_dir, output_dir, epochs=epochs, n_folds=n_folds)
        all_results[bb] = fold_results

        # Save per-fold CSV
        fold_csv = os.path.join(output_dir, f"cv_{bb}.csv")
        with open(fold_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["fold", "accuracy", "macro_f1", "macro_auc"])
            writer.writeheader()
            writer.writerows(fold_results)

    # Build ablation table
    ablation_rows = []
    print(f"\n{'='*65}")
    print(f"{'Model':<18} {'Accuracy':>10} {'Macro F1':>10} {'AUC':>10}")
    print(f"{'-'*65}")

    for bb in backbones:
        folds = all_results[bb]
        accs = [r["accuracy"] for r in folds]
        f1s = [r["macro_f1"] for r in folds]
        aucs = [r["macro_auc"] for r in folds]

        row = {
            "model": bb,
            "acc_mean": round(np.mean(accs), 4),
            "acc_std": round(np.std(accs), 4),
            "f1_mean": round(np.mean(f1s), 4),
            "f1_std": round(np.std(f1s), 4),
            "auc_mean": round(np.mean(aucs), 4),
            "auc_std": round(np.std(aucs), 4),
        }
        ablation_rows.append(row)
        print(
            f"{bb:<18} "
            f"{row['acc_mean']:.4f}±{row['acc_std']:.4f}  "
            f"{row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
            f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}"
        )

    print(f"{'='*65}")

    abl_csv = os.path.join(output_dir, "ablation_table.csv")
    with open(abl_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ablation_rows[0].keys())
        writer.writeheader()
        writer.writerows(ablation_rows)

    print(f"\nAblation table saved → {abl_csv}")
    return all_results


def _save_confusion_matrix(cm, backbone, fold, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im)
    ax.set(
        xticks=range(NUM_CLASSES),
        yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{backbone} — Fold {fold}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation_bar(ablation_csv: str, output_path: str):
    """Bar chart comparing all four models across three metrics."""
    rows = []
    with open(ablation_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    models = [r["model"] for r in rows]
    metrics = ["acc_mean", "f1_mean", "auc_mean"]
    labels = ["Accuracy", "Macro F1", "AUC"]
    colors = ["#378ADD", "#1D9E75", "#D85A30"]

    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [float(r[metric]) for r in rows]
        errs = [float(r[metric.replace("mean", "std")]) for r in rows]
        ax.bar(x + i * w, vals, w, label=label, color=color, yerr=errs, capsize=4, alpha=0.85)

    ax.set_xticks(x + w)
    ax.set_xticklabels(models)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: backbone comparison (5-fold CV, mean ± std)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Ablation bar chart saved → {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TI-RADS ablation evaluation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="paper_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--backbone", type=str, default="all", help="Run only one backbone, or 'all'"
    )
    args = parser.parse_args()

    if args.backbone == "all":
        results = run_ablation(args.data_dir, args.output_dir, args.epochs, args.n_folds)
        plot_ablation_bar(
            os.path.join(args.output_dir, "ablation_table.csv"),
            os.path.join(args.output_dir, "ablation_bar.png"),
        )
    else:
        fold_results = run_kfold(
            args.backbone,
            args.data_dir,
            args.output_dir,
            args.epochs,
            args.n_folds,
        )
        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["macro_f1"] for r in fold_results]
        aucs = [r["macro_auc"] for r in fold_results]
        print(
            f"\n{args.backbone} | "
            f"Acc {np.mean(accs):.4f}±{np.std(accs):.4f} | "
            f"F1 {np.mean(f1s):.4f}±{np.std(f1s):.4f} | "
            f"AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f}"
        )
