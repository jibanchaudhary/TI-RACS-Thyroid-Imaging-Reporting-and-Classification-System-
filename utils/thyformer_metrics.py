"""
ThyFormer — Evaluation Metrics

• Macro-averaged AUC (primary) + per-class AUC
• Macro AP (average precision) + per-class AP
• Macro F1, sensitivity, specificity, PPV, NPV, accuracy, balanced accuracy, MCC
• Ordinal grade metrics: within-one-grade accuracy, grade MAE (TI-RADS is ordinal)
• Calibration: ECE, MCE, multiclass Brier score (+ reliability-diagram bins)
• Bootstrap 95% confidence intervals for every scalar metric
• Paired cluster bootstrap (over clips) for model-vs-model significance
• Cohen's kappa for clinical radiologist agreement
"""
import warnings
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


def _ovr_auc(y: np.ndarray, probs: np.ndarray, num_classes: int) -> Tuple[List[float], float]:
    """
    One-vs-rest AUC per class plus the macro average, robust to classes that are
    missing from ``y``. A class with no positives (absent from this split) or no
    negatives has an undefined OvR AUC, so it is reported as NaN and excluded
    from the macro mean — instead of letting sklearn emit UndefinedMetricWarning
    or raise on a probs-column/label-count mismatch. This lets evaluation run on
    any subset of the classes (all 5, or just 3, etc.).

    The macro value here equals sklearn's roc_auc_score(multi_class="ovr",
    average="macro") whenever every class is present, so numbers are unchanged
    on a full test set.
    """
    per_class: List[float] = []
    valid: List[float] = []
    for c in range(num_classes):
        y_bin = (y == c).astype(int)
        if y_bin.min() == y_bin.max():  # class has no positives (or no negatives)
            per_class.append(float("nan"))
            continue
        auc_c = float(roc_auc_score(y_bin, probs[:, c]))
        per_class.append(auc_c)
        valid.append(auc_c)
    macro = float(np.mean(valid)) if valid else 0.0
    return per_class, macro


def _scalar_metrics(y: np.ndarray, probs: np.ndarray, num_classes: int = 5) -> Dict[str, float]:
    """
    All scalar metrics (unprefixed) from labels + probabilities. Shared by
    compute_metrics (point estimates) and bootstrap_cis (resampled estimates)
    so both always agree on definitions.
    """
    preds = probs.argmax(axis=1)
    m = {}
    cn = [f"t{i + 1}" for i in range(num_classes)]

    # AUC — one-vs-rest, macro-averaged over the classes actually present in this
    # split (absent classes -> NaN, excluded from the macro). See _ovr_auc.
    per_class_auc, m["auc"] = _ovr_auc(y, probs, num_classes)
    for c in range(num_classes):
        m[f"auc_{cn[c]}"] = per_class_auc[c]

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

    # Accuracy family. balanced_accuracy warns "y_pred contains classes not in
    # y_true" when the model predicts a class that is absent from this split —
    # expected when evaluating on a subset of the classes; the value is still
    # valid (recall averaged over the classes present in y_true).
    m["accuracy"] = float(accuracy_score(y, preds))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
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


def clip_ids(stems: List[str]) -> np.ndarray:
    """
    Clip (video) id for each frame stem: "138_69" -> "138".

    The Stanford frames are video frames — every clip contributes many
    near-duplicate frames that share one label. The clip, not the frame, is the
    independent sampling unit, so every bootstrap in this module resamples these
    ids rather than rows. See ``_resampler``.
    """
    return np.array([str(s).split("_")[0] for s in stems])


def _resampler(groups: Optional[np.ndarray], n: int):
    """
    Build a bootstrap index generator.

    With ``groups`` (one clip id per sample) this is a *cluster* bootstrap: whole
    clips are drawn with replacement and every frame of a drawn clip comes along.
    That is the statistically correct unit here — frames within a clip are
    near-duplicates of one nodule, so resampling them independently treats
    ~29 real observations as ~3066 and understates the standard error by roughly
    sqrt(frames/clips) (~10x on this test set), which is what produced
    "p = 0.0000" for differences that are not resolvable at this sample size.

    Without ``groups`` it degrades to the old i.i.d. row bootstrap.

    Returns (draw_fn, n_units) where draw_fn(rng) -> integer index array.
    """
    if groups is None:
        return (lambda rng: rng.integers(0, n, n)), n

    uniq = np.unique(groups)
    idx_by_clip = [np.where(groups == c)[0] for c in uniq]

    def draw(rng):
        pick = rng.integers(0, len(uniq), len(uniq))
        return np.concatenate([idx_by_clip[i] for i in pick])

    return draw, len(uniq)


def bootstrap_cis(
    y: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    num_classes: int = 5,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Percentile-bootstrap 95% CIs for every scalar metric.

    Resamples the test set with replacement n_bootstrap times, recomputes the
    full metric set on each resample, and reports the (alpha/2, 1-alpha/2)
    percentiles. Resamples that drop a class entirely are skipped (they would
    zero the OvR AUC and poison the percentiles).

    ``groups`` (per-sample clip ids from ``clip_ids``) switches this to a cluster
    bootstrap over clips — required for honest interval widths on frame-level
    video data. Omitting it reproduces the old, over-narrow frame-level CIs.

    Returns {metric: {"value": point_estimate, "ci_low": lo, "ci_high": hi}}.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    present = np.unique(y)
    point = _scalar_metrics(y, probs, num_classes)
    draw, _ = _resampler(groups, n)

    samples: Dict[str, List[float]] = {k: [] for k in point}
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap CIs", unit="it", leave=False):
        idx = draw(rng)
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


def paired_auc_test(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstrap: int = 1000,
    groups: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap test on the macro-AUC difference between two models
    evaluated on the same samples.

    This is NOT DeLong's test (it was previously mislabelled as such). DeLong's
    test uses the analytic covariance of the Mann-Whitney statistic and assumes
    independent observations; this is a paired *cluster* bootstrap, which is the
    appropriate tool here because the observations are video frames nested in
    clips. Pass ``groups`` (per-frame clip ids from ``clip_ids``) so whole clips
    are resampled — see ``_resampler`` for why the frame-level version is invalid.

    Both models must be scored on the same samples in the same order, so each
    resample re-scores both models on one index set (paired), which cancels the
    shared sampling noise.

    Returns a dict with both AUCs, the difference, a percentile CI on the
    difference, and the two-sided bootstrap p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    nc_a, nc_b = probs_a.shape[1], probs_b.shape[1]
    _, auc_a = _ovr_auc(labels, probs_a, nc_a)
    _, auc_b = _ovr_auc(labels, probs_b, nc_b)
    draw, n_units = _resampler(groups, n)

    diffs = []
    for _ in tqdm(range(n_bootstrap), desc="Paired AUC bootstrap", unit="it", leave=False):
        idx = draw(rng)
        yb = labels[idx]
        if len(np.unique(yb)) < 2:  # degenerate resample — no AUC is defined
            continue
        _, da = _ovr_auc(yb, probs_a[idx], nc_a)
        _, db = _ovr_auc(yb, probs_b[idx], nc_b)
        diffs.append(da - db)

    diffs = np.asarray(diffs, dtype=float)
    obs = auc_a - auc_b
    if diffs.size == 0:
        return {
            "auc_a": float(auc_a),
            "auc_b": float(auc_b),
            "auc_difference": float(obs),
            "diff_ci_low": float("nan"),
            "diff_ci_high": float("nan"),
            "z_statistic": float("nan"),
            "p_value": float("nan"),
            "n_bootstrap_used": 0,
            "n_units": int(n_units),
            "n_samples": int(n),
            "unit": "clip" if groups is not None else "frame",
            "method": "paired cluster bootstrap" if groups is not None else "paired bootstrap",
        }

    se = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    z = obs / max(se, 1e-9)
    # Two-sided percentile p: how often the resampled difference falls on the
    # other side of zero. Floor at 1/n_bootstrap — a bootstrap can never
    # legitimately report p = 0.
    tail = min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    p = max(2.0 * tail, 1.0 / diffs.size)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_difference": float(obs),
        "diff_ci_low": float(lo),
        "diff_ci_high": float(hi),
        "diff_se": se,
        "z_statistic": float(z),
        "p_value": float(p),
        "p_value_normal": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
        "n_bootstrap_used": int(diffs.size),
        "n_units": int(n_units),
        "n_samples": int(n),
        "unit": "clip" if groups is not None else "frame",
        "method": "paired cluster bootstrap" if groups is not None else "paired bootstrap",
    }


def aggregate_by_group(
    y: np.ndarray, probs: np.ndarray, groups: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse frame-level predictions to one row per clip by averaging the
    probability vectors (labels are constant within a clip, so ``first`` is
    exact). Returns (y_clip, probs_clip, clip_ids) ordered by clip id.

    Clip-level metrics answer the clinically meaningful question — "how many
    nodules did it grade correctly" — instead of weighting each nodule by how
    many frames its video happens to contain (25 to 293 here).
    """
    uniq = np.unique(groups)
    yc = np.empty(len(uniq), dtype=y.dtype)
    pc = np.empty((len(uniq), probs.shape[1]), dtype=float)
    for i, c in enumerate(uniq):
        m = groups == c
        yc[i] = y[m][0]
        pc[i] = probs[m].mean(axis=0)
    return yc, pc, uniq


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
