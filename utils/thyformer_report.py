import dataclasses
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)

import matplotlib

matplotlib.use("Agg")  # headless / no display needed

# Validated categorical palette (slots 1-4) — CVD-safe in this order; the
# aqua/yellow slots are low-contrast on white, so every figure also carries
# direct value labels and a CSV table twin.
CLASS_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#d11717"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Sequential blue ramp (light→dark) for the confusion-matrix heatmap.
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "thy_blues",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _save_fig(fig, path: Path) -> Path:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_json(obj: dict, path: Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return Path(path)


def save_metrics(out_dir: Path, metrics: Dict[str, float], stem: str = "metrics") -> None:
    """Scalar metrics → <stem>.json + <stem>.csv. `stem` lets a run write more
    than one metric set (e.g. frame-level "metrics" and "metrics_clip_level")."""
    save_json(metrics, out_dir / f"{stem}.json")
    pd.DataFrame([metrics]).to_csv(out_dir / f"{stem}.csv", index=False)


def save_metric_cis(out_dir: Path, cis: Dict[str, Dict[str, float]]) -> None:
    """Bootstrap CIs → metrics_ci.json + a tidy metrics_ci.csv (one row per metric)."""
    save_json(cis, out_dir / "metrics_ci.json")
    rows = [
        {"metric": k, "value": ci["value"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
        for k, ci in cis.items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "metrics_ci.csv", index=False)


def save_reliability_diagram(out_dir: Path, bins: List[Dict], calib: Dict[str, float]) -> None:
    """
    Reliability diagram (top-label calibration) + confidence histogram, with a
    CSV twin of the bin table. `bins`/`calib` come from
    thyformer_metrics.compute_calibration.
    """
    df = pd.DataFrame(bins)
    df.to_csv(out_dir / "calibration_bins.csv", index=False)

    centers = (df["bin_lower"] + df["bin_upper"]) / 2
    width = (df["bin_upper"] - df["bin_lower"]).iloc[0]
    occupied = df["count"] > 0

    fig, (ax, ax_h) = plt.subplots(
        2,
        1,
        figsize=(6.0, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # ── Top: observed accuracy per confidence bin vs the diagonal ──
    ax.bar(
        centers[occupied],
        df.loc[occupied, "avg_accuracy"],
        width * 0.9,
        color=CLASS_COLORS[0],
        label="Observed accuracy",
    )
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=2, linestyle="--", label="Perfect calibration")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram (top-label calibration)", fontsize=12)
    _style_axis(ax)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor=GRID)
    ax.text(
        0.97,
        0.05,
        f"ECE = {calib['ece']:.4f}\nMCE = {calib['mce']:.4f}\nBrier = {calib['brier']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=INK_2,
        bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.4"),
    )

    # ── Bottom: how many samples land in each confidence bin ──
    ax_h.bar(centers[occupied], df.loc[occupied, "count"], width * 0.9, color=BASELINE)
    ax_h.set_xlim(-0.02, 1.02)
    ax_h.set_xlabel("Predicted confidence (max softmax)")
    ax_h.set_ylabel("Samples")
    _style_axis(ax_h)
    _save_fig(fig, out_dir / "reliability_diagram.png")


def save_predictions_csv(
    out_dir: Path,
    stems: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> None:
    df = pd.DataFrame(
        {
            "stem": stems,
            "true_label": [class_names[i] for i in y_true],
            "pred_label": [class_names[i] for i in y_pred],
            "correct": (y_true == y_pred).astype(int),
        }
    )
    for c, name in enumerate(class_names):
        df[f"prob_{name}"] = probs[:, c]
    df.to_csv(out_dir / "predictions.csv", index=False)


def save_classification_report(
    out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> None:
    txt = classification_report(
        y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0
    )
    (out_dir / "classification_report.txt").write_text(txt)


def save_confusion_matrix(
    out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> None:
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / "confusion_matrix.csv", index_label="true\\pred"
    )
    pd.DataFrame(np.round(cm_norm, 4), index=class_names, columns=class_names).to_csv(
        out_dir / "confusion_matrix_normalized.csv", index_label="true\\pred"
    )

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm_norm, cmap=SEQ_CMAP, vmin=0.0, vmax=1.0)
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.55 else INK
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )
    ax.set_xticks(range(n), class_names)
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (count, row %)", fontsize=12)
    ax.tick_params(colors=INK_2, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized recall", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=INK_2, labelsize=8)
    cbar.outline.set_visible(False)
    _save_fig(fig, out_dir / "confusion_matrix.png")


def save_roc_curves(
    out_dir: Path, y_true: np.ndarray, probs: np.ndarray, class_names: List[str]
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    rows = []
    grid = np.linspace(0.0, 1.0, 200)
    mean_tpr = np.zeros_like(grid)
    n_valid = 0

    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        if y_bin.min() == y_bin.max():  # class absent from the test split
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        auc_c = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[c], linewidth=2, label=f"{name} (AUC = {auc_c:.3f})")
        rows.extend({"class": name, "fpr": f, "tpr": t} for f, t in zip(fpr, tpr))
        mean_tpr += np.interp(grid, fpr, tpr)
        n_valid += 1

    if n_valid:
        mean_tpr /= n_valid
        mean_tpr[0] = 0.0
        ax.plot(
            grid,
            mean_tpr,
            color=INK_2,
            linewidth=2,
            linestyle="--",
            label=f"Macro average (AUC = {auc(grid, mean_tpr):.3f})",
        )
        rows.extend({"class": "macro", "fpr": f, "tpr": t} for f, t in zip(grid, mean_tpr))

    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1, linestyle=":")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curves (one-vs-rest)", fontsize=12)
    _style_axis(ax)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9, edgecolor=GRID)
    _save_fig(fig, out_dir / "roc_curves.png")
    pd.DataFrame(rows).to_csv(out_dir / "roc_curves.csv", index=False)


def save_pr_curves(
    out_dir: Path, y_true: np.ndarray, probs: np.ndarray, class_names: List[str]
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    rows = []
    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        if y_bin.min() == y_bin.max():
            continue
        prec, rec, _ = precision_recall_curve(y_bin, probs[:, c])
        ap = average_precision_score(y_bin, probs[:, c])
        ax.plot(rec, prec, color=CLASS_COLORS[c], linewidth=2, label=f"{name} (AP = {ap:.3f})")
        rows.extend({"class": name, "recall": r, "precision": p} for r, p in zip(rec, prec))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (one-vs-rest)", fontsize=12)
    _style_axis(ax)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9, edgecolor=GRID)
    _save_fig(fig, out_dir / "pr_curves.png")
    pd.DataFrame(rows).to_csv(out_dir / "pr_curves.csv", index=False)


def save_per_class_metrics(
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """Per-class AUC/AP/F1/sensitivity/specificity/precision/NPV table + grouped bars."""
    n = len(class_names)
    f1s = f1_score(y_true, y_pred, labels=range(n), average=None, zero_division=0)
    precs = precision_score(y_true, y_pred, labels=range(n), average=None, zero_division=0)
    records = []
    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        tn = int(((y_pred != c) & (y_true != c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        if y_bin.min() == y_bin.max():  # class absent from this split -> undefined
            auc_c = ap_c = float("nan")
        else:
            auc_c = float(roc_auc_score(y_bin, probs[:, c]))
            ap_c = float(average_precision_score(y_bin, probs[:, c]))
        records.append(
            {
                "class": name,
                "support": int(y_bin.sum()),
                "auc": auc_c,
                "ap": ap_c,
                "f1": float(f1s[c]),
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "precision": float(precs[c]),
                "npv": tn / max(tn + fn, 1),
            }
        )
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "per_class_metrics.csv", index=False)

    metric_cols = ["auc", "f1", "sensitivity", "specificity", "precision"]
    metric_labels = ["AUC", "F1", "Sensitivity", "Specificity", "Precision"]
    x = np.arange(len(metric_cols))
    width = 0.19

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c, name in enumerate(class_names):
        vals = df.loc[c, metric_cols].astype(float).values
        offset = (c - (n - 1) / 2) * (width + 0.015)
        bars = ax.bar(x + offset, vals, width, color=CLASS_COLORS[c], label=name)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.015,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=INK_2,
                )
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Score")
    ax.set_title("Per-class Test Metrics", fontsize=12)
    _style_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(
        fontsize=9,
        ncols=len(class_names),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
    )
    _save_fig(fig, out_dir / "per_class_metrics.png")
    return df


def _macro_roc(y_true: np.ndarray, probs: np.ndarray, n_classes: int):
    """Macro-average ROC over a common FPR grid. Returns (grid, mean_tpr, macro_auc)."""
    grid = np.linspace(0.0, 1.0, 200)
    mean_tpr = np.zeros_like(grid)
    n_valid = 0
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.min() == y_bin.max():  # class absent from the test split
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        mean_tpr += np.interp(grid, fpr, tpr)
        n_valid += 1
    if n_valid:
        mean_tpr /= n_valid
        mean_tpr[0] = 0.0
        return grid, mean_tpr, float(auc(grid, mean_tpr))
    return grid, mean_tpr, 0.0


def save_comparison(
    out_dir: Path,
    name_a: str,
    metrics_a: Dict[str, float],
    probs_a: np.ndarray,
    name_b: str,
    metrics_b: Dict[str, float],
    probs_b: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    delong_results: Dict,
) -> None:
    """
    Head-to-head comparison artifacts for two independently evaluated models:

      • auc_comparison.json      significance of the macro-AUC difference
      • comparison_metrics.csv   side-by-side scalar metrics
      • comparison_metrics.png   grouped bars (AUC/F1/sens/spec/accuracy)
      • comparison_roc.png       macro-average ROC overlay for both models

    `delong_results` keeps its parameter name for callers, but the statistic is a
    paired cluster bootstrap over clips — not DeLong's test — so it is written to
    auc_comparison.json. Older runs have a delong_test.json holding the invalid
    frame-level numbers; the filenames differ deliberately so the two cannot be
    confused.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(delong_results, out_dir / "auc_comparison.json")

    n_classes = len(class_names)
    models = [(name_a, metrics_a, probs_a), (name_b, metrics_b, probs_b)]
    # ThyFormer (blue) vs baseline (amber) — a CVD-safe 2-series pair.
    colors = [CLASS_COLORS[0], CLASS_COLORS[2]]

    # ── Side-by-side metrics table ────────────────────────────────
    keys = [
        "test_auc",
        "test_ap",
        "test_f1",
        "test_sensitivity",
        "test_specificity",
        "test_ppv",
        "test_npv",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_mcc",
        "test_within1_accuracy",
        "test_grade_mae",
        "test_ece",
        "test_brier",
        "test_loss",
    ]
    rows = [
        {"model": nm, **{k: float(m.get(k, float("nan"))) for k in keys}} for nm, m, _ in models
    ]
    pd.DataFrame(rows).to_csv(out_dir / "comparison_metrics.csv", index=False)

    # ── Grouped metric bars (0–1 metrics only; loss stays in the CSV) ─
    bar_keys = ["test_auc", "test_f1", "test_sensitivity", "test_specificity", "test_accuracy"]
    bar_labels = ["AUC", "F1", "Sensitivity", "Specificity", "Accuracy"]
    x = np.arange(len(bar_keys))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, (nm, m, _) in enumerate(models):
        vals = [float(m.get(k, 0.0)) for k in bar_keys]
        offset = (i - 0.5) * (width + 0.02)
        bars = ax.bar(x + offset, vals, width, color=colors[i], label=nm)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.015,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK_2,
            )
    ax.set_xticks(x, bar_labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Metrics", fontsize=12)
    _style_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(
        fontsize=9,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
    )
    _save_fig(fig, out_dir / "comparison_metrics.png")

    # ── Macro-average ROC overlay ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    for i, (nm, _, pr) in enumerate(models):
        grid, mean_tpr, macro_auc = _macro_roc(labels, pr, n_classes)
        ax.plot(
            grid,
            mean_tpr,
            color=colors[i],
            linewidth=2,
            label=f"{nm} (macro AUC = {macro_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1, linestyle=":")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Macro-average ROC — Model Comparison", fontsize=12)
    _style_axis(ax)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9, edgecolor=GRID)
    _save_fig(fig, out_dir / "comparison_roc.png")


def save_run_metadata(
    out_dir: Path,
    cfg,
    args,
    ckpt,
    model: torch.nn.Module,
    n_test: int,
    class_counts: Dict[str, int],
) -> None:
    """Full experiment record: config, CLI args, checkpoint, environment, model size."""
    ckpt_meta = {"path": args.checkpoint}
    if isinstance(ckpt, dict):
        ckpt_meta.update({k: v for k, v in ckpt.items() if isinstance(v, (int, float, str, bool))})
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "checkpoint": ckpt_meta,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "model": {"num_parameters": n_params, "num_trainable_parameters": n_trainable},
        "test_set": {"num_samples": n_test, "class_distribution": class_counts},
        "config": dataclasses.asdict(cfg),
    }
    save_json(meta, out_dir / "run_config.json")
