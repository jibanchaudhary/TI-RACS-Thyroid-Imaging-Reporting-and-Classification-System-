"""
ThyFormer — Training Curve Plotter

Reads the per-epoch metrics CSV written by utils.thyformer_logging.MetricLogger
and renders a panel of training curves:

  • Loss              (train vs val)
  • Accuracy          (train vs val)
  • Macro AUC         (train vs val)
  • Macro F1          (train vs val)
  • Sensitivity       (train vs val)
  • Specificity       (train vs val)
  • Per-class AUC     (val: t1..t4)
  • Per-class F1      (val: t1..t4)

Usage:
    python -m scripts.thyformer_plot_metrics
    python -m scripts.thyformer_plot_metrics --csv logs/metrics.csv --out artifacts/curves.png
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / no display needed

CLASS_NAMES = ["t1", "t2", "t3", "t4"]


def load_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Coerce everything to numeric; "inf"/"nan"/"" become NaN so they leave gaps.
    df = df.apply(pd.to_numeric, errors="coerce")
    # Drop the trailing final-test row (no epoch) and any all-empty rows.
    if "epoch" in df.columns:
        df = df[df["epoch"].notna()].copy()
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


def _paired_panel(ax, df, metric, title):
    """Plot train vs val curves for a single scalar metric."""
    x = df["epoch"] if "epoch" in df else np.arange(len(df))
    plotted = False
    for split, color in (("train", "tab:blue"), ("val", "tab:orange")):
        col = f"{split}_{metric}"
        if col in df.columns and df[col].notna().any():
            ax.plot(x, df[col], marker="o", ms=3, color=color, label=split)
            plotted = True
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)


def _perclass_panel(ax, df, metric, split, title):
    """Plot per-class (t1..t4) curves for one split."""
    x = df["epoch"] if "epoch" in df else np.arange(len(df))
    for c in CLASS_NAMES:
        col = f"{split}_{metric}_{c}"
        if col in df.columns and df[col].notna().any():
            ax.plot(x, df[col], marker="o", ms=3, label=c.upper())
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_metrics(csv_path: str, out_path: str) -> str:
    df = load_metrics(csv_path)
    if df.empty:
        raise ValueError(f"No usable rows in {csv_path}")

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("ThyFormer Training Metrics", fontsize=16, fontweight="bold")

    _paired_panel(axes[0, 0], df, "loss", "Loss")
    _paired_panel(axes[0, 1], df, "accuracy", "Accuracy")
    _paired_panel(axes[0, 2], df, "auc", "Macro AUC")
    _paired_panel(axes[1, 0], df, "f1", "Macro F1")
    _paired_panel(axes[1, 1], df, "sensitivity", "Sensitivity (macro)")
    _paired_panel(axes[1, 2], df, "specificity", "Specificity (macro)")
    _perclass_panel(axes[2, 0], df, "auc", "val", "Per-class AUC (val)")
    _perclass_panel(axes[2, 1], df, "f1", "val", "Per-class F1 (val)")
    _perclass_panel(axes[2, 2], df, "auc", "train", "Per-class AUC (train)")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot ThyFormer training curves")
    ap.add_argument("--csv", default="logs/metrics.csv", help="Path to metrics.csv")
    ap.add_argument(
        "--out", default="artifacts/thyformer_v2_720/training_curves.png", help="Output PNG path"
    )
    args = ap.parse_args()
    out = plot_metrics(args.csv, args.out)
    print(f"Saved training curves → {out}")


if __name__ == "__main__":
    main()
