"""
ThyFormer — Evaluation Metrics

• Macro-averaged AUC (primary) + per-class AUC
• Macro AP (average precision) + per-class AP
• Macro F1, sensitivity, specificity, PPV, NPV, accuracy, balanced accuracy, MCC
• Ordinal grade metrics: within-one-grade accuracy, grade MAE (TI-RADS is ordinal)
• Calibration: ECE, MCE, multiclass Brier score (+ reliability-diagram bins)
• Bootstrap 95% confidence intervals for every scalar metric
• DeLong's test (bootstrap) for significance comparison
• Cohen's kappa for clinical radiologist agreement
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from tqdm import tqdm


def compute_calibration(
    y: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Top-label calibration: bin samples by max-softmax confidence, compare each
    bin's mean confidence to its empirical accuracy.

    Returns ({ece, mce, brier}, bins) where bins is a per-bin table
    (bin_lower/bin_upper/count/avg_confidence/avg_accuracy) for the
    reliability diagram and its CSV twin.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    n = len(y)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # right-inclusive last bin so conf == 1.0 is counted
        mask = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo)
        count = int(mask.sum())
        avg_conf = float(conf[mask].mean()) if count else 0.0
        avg_acc = float(correct[mask].mean()) if count else 0.0
        if count:
            gap = abs(avg_acc - avg_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        rows.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": count,
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
            }
        )

    # Multiclass Brier score: mean squared distance from the one-hot target.
    onehot = np.eye(probs.shape[1])[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return {"ece": float(ece), "mce": float(mce), "brier": brier}, rows


def _scalar_metrics(y: np.ndarray, probs: np.ndarray, num_classes: int = 5) -> Dict[str, float]:
    """
    All scalar metrics (unprefixed) from labels + probabilities. Shared by
    compute_metrics (point estimates) and bootstrap_cis (resampled estimates)
    so both always agree on definitions.
    """
    preds = probs.argmax(axis=1)
    m = {}
    cn = [f"t{i + 1}" for i in range(num_classes)]

    # Macro AUC
    try:
        m["auc"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        m["auc"] = 0.0

    # Per-class AUC
    for c in range(num_classes):
        try:
            m[f"auc_{cn[c]}"] = float(roc_auc_score((y == c).astype(int), probs[:, c]))
        except ValueError:
            m[f"auc_{cn[c]}"] = 0.0

    # Average precision (per class + macro) — the imbalance-appropriate summary
    aps = []
    for c in range(num_classes):
        y_bin = (y == c).astype(int)
        if y_bin.min() == y_bin.max():  # class absent
            m[f"ap_{cn[c]}"] = 0.0
            continue
        ap_c = float(average_precision_score(y_bin, probs[:, c]))
        m[f"ap_{cn[c]}"] = ap_c
        aps.append(ap_c)
    m["ap"] = float(np.mean(aps)) if aps else 0.0

    # F1
    m["f1"] = float(f1_score(y, preds, average="macro", zero_division=0))
    f1s = f1_score(y, preds, labels=range(num_classes), average=None, zero_division=0)
    for c, name in enumerate(cn):
        m[f"f1_{name}"] = float(f1s[c]) if c < len(f1s) else 0.0

    # Sensitivity / specificity / PPV / NPV (macro avg over one-vs-rest)
    sens, spec, npvs = [], [], []
    for c in range(num_classes):
        tp = int(((preds == c) & (y == c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        tn = int(((preds != c) & (y != c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        sens.append(tp / max(tp + fn, 1))
        spec.append(tn / max(tn + fp, 1))
        npvs.append(tn / max(tn + fn, 1))
    m["sensitivity"] = float(np.mean(sens))
    m["specificity"] = float(np.mean(spec))
    m["ppv"] = float(precision_score(y, preds, average="macro", zero_division=0))
    m["npv"] = float(np.mean(npvs))

    # Accuracy family
    m["accuracy"] = float(accuracy_score(y, preds))
    m["balanced_accuracy"] = float(balanced_accuracy_score(y, preds))
    m["mcc"] = float(matthews_corrcoef(y, preds))

    # Ordinal grade metrics — TI-RADS grades are ordered, so T4→T3 is a far
    # smaller error than T4→T1; plain accuracy/F1 treat them identically.
    grade_err = np.abs(preds.astype(int) - y.astype(int))
    m["within1_accuracy"] = float((grade_err <= 1).mean())
    m["grade_mae"] = float(grade_err.mean())

    # Calibration scalars (bin table is regenerated separately for the figure)
    calib, _ = compute_calibration(y, probs)
    m.update(calib)
    return m


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor, prefix: str = "val", num_classes: int = 5
) -> Dict[str, float]:
    # Cast to float32 first: under FP16 autocast cls_logits are float16, and a
    # float16 softmax row sums to ~1.0003, which trips sklearn's multi_class="ovr"
    # "scores must be probabilities" check (tol ~1e-5) and silently zeroed the AUC.
    logits = logits.float()
    probs = F.softmax(logits, dim=-1).numpy()
    y = labels.numpy()
    return {f"{prefix}_{k}": v for k, v in _scalar_metrics(y, probs, num_classes).items()}


def bootstrap_cis(
    y: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    num_classes: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Percentile-bootstrap 95% CIs for every scalar metric.

    Resamples the test set with replacement n_bootstrap times, recomputes the
    full metric set on each resample, and reports the (alpha/2, 1-alpha/2)
    percentiles. Resamples that drop a class entirely are skipped (they would
    zero the OvR AUC and poison the percentiles).

    Returns {metric: {"value": point_estimate, "ci_low": lo, "ci_high": hi}}.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    present = np.unique(y)
    point = _scalar_metrics(y, probs, num_classes)

    samples: Dict[str, List[float]] = {k: [] for k in point}
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap CIs", unit="it", leave=False):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < len(present):
            continue
        for k, v in _scalar_metrics(y[idx], probs[idx], num_classes).items():
            samples[k].append(v)

    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    out = {}
    for k, vals in samples.items():
        if vals:
            lo, hi = np.percentile(vals, [lo_q, hi_q])
        else:
            lo = hi = point[k]
        out[k] = {"value": point[k], "ci_low": float(lo), "ci_high": float(hi)}
    return out


def delong_test(
    labels: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, n_bootstrap: int = 1000
) -> Tuple[float, float, float, float]:
    """
    Bootstrap DeLong test comparing macro AUC of two models.
    Returns (auc_a, auc_b, z, p_value).
    """
    rng = np.random.default_rng(42)
    n = len(labels)
    auc_a = roc_auc_score(labels, probs_a, multi_class="ovr", average="macro")
    auc_b = roc_auc_score(labels, probs_b, multi_class="ovr", average="macro")

    diffs = []
    for _ in tqdm(range(n_bootstrap), desc="DeLong bootstrap", unit="it", leave=False):
        idx = rng.integers(0, n, n)
        try:
            da = roc_auc_score(labels[idx], probs_a[idx], multi_class="ovr", average="macro")
            db = roc_auc_score(labels[idx], probs_b[idx], multi_class="ovr", average="macro")
            diffs.append(da - db)
        except ValueError:
            pass

    se = np.std(diffs)
    z = (auc_a - auc_b) / max(se, 1e-9)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(auc_a), float(auc_b), float(z), float(p)


def compute_kappa(
    pred: np.ndarray, rad: np.ndarray, weights: str = "quadratic"
) -> Dict[str, float]:
    """
    Weighted Cohen's kappa (quadratic = standard for ordinal TI-RADS).
    Interpretation: >0.80 almost perfect, 0.61–0.80 substantial.
    """
    kappa = cohen_kappa_score(pred, rad, weights=weights)
    acc = accuracy_score(pred, rad)
    per_class = {}
    for c in range(4):
        mask = rad == c
        if mask.sum() > 0:
            per_class[f"agreement_t{c+1}"] = float((pred[mask] == c).mean())
    return {"kappa": float(kappa), "accuracy": float(acc), **per_class}


def print_results_table(metrics: Dict[str, float], title: str = "Results"):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:<35s} {v:.4f}")
    print("=" * 55)


def print_ci_table(cis: Dict[str, Dict[str, float]], title: str = "95% Confidence Intervals"):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")
    for k, ci in sorted(cis.items()):
        print(f"  {k:<25s} {ci['value']:.4f}  ({ci['ci_low']:.4f}–{ci['ci_high']:.4f})")
    print("=" * 55)


def print_comparison_table(
    names: List[str], all_metrics: List[Dict], keys: Optional[List[str]] = None
):
    if keys is None:
        keys = ["auc", "f1", "sensitivity", "specificity", "accuracy"]
    prefix = "val" if "val_auc" in all_metrics[0] else "test"
    full_keys = [f"{prefix}_{k}" for k in keys]
    hdr = f"{'Model':<20}" + "".join(f"{k:>14}" for k in keys)
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for name, m in zip(names, all_metrics):
        row = f"{name:<20}" + "".join(f"{m.get(k,0.0):>14.4f}" for k in full_keys)
        print(row)
    print("=" * len(hdr))
