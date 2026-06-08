"""
ThyFormer — Evaluation Metrics

• Macro-averaged AUC (primary)
• Per-class AUC
• Macro F1, sensitivity, specificity, accuracy
• DeLong's test (bootstrap) for significance comparison
• Cohen's kappa for clinical radiologist agreement
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor, prefix: str = "val", num_classes: int = 4
) -> Dict[str, float]:
    probs = F.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    y = labels.numpy()
    m = {}
    cn = ["t1", "t2", "t3", "t4"]

    # Macro AUC
    try:
        m[f"{prefix}_auc"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        m[f"{prefix}_auc"] = 0.0

    # Per-class AUC
    for c in range(num_classes):
        try:
            m[f"{prefix}_auc_{cn[c]}"] = float(roc_auc_score((y == c).astype(int), probs[:, c]))
        except ValueError:
            m[f"{prefix}_auc_{cn[c]}"] = 0.0

    # F1
    m[f"{prefix}_f1"] = float(f1_score(y, preds, average="macro", zero_division=0))
    for c, name in enumerate(cn):
        f1s = f1_score(y, preds, average=None, zero_division=0)
        m[f"{prefix}_f1_{name}"] = float(f1s[c]) if c < len(f1s) else 0.0

    # Sensitivity / specificity (macro avg)
    sens, spec = [], []
    for c in range(num_classes):
        tp = int(((preds == c) & (y == c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        tn = int(((preds != c) & (y != c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        sens.append(tp / max(tp + fn, 1))
        spec.append(tn / max(tn + fp, 1))
    m[f"{prefix}_sensitivity"] = float(np.mean(sens))
    m[f"{prefix}_specificity"] = float(np.mean(spec))

    # Accuracy
    m[f"{prefix}_accuracy"] = float(accuracy_score(y, preds))
    return m


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
    for _ in range(n_bootstrap):
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
