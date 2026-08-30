"""
ThyFormer — Evaluation Runner

Evaluates a trained ThyFormer checkpoint (--checkpoint), a task-trained
baseline (--baseline), or both. Each model is swept over the test set fully
independently and gets its own complete, paper-ready artifact folder; when
both are evaluated, they are also compared head-to-head:

    <run>/thyformer/           ThyFormer's full evaluation
    <run>/<baseline_arch>/     the baseline's full evaluation (same artifacts)
    <run>/comparison/          paired AUC test + comparison figures (both models only)

Baseline dispatch — evaluate_one_model routes each --baseline name to its
dedicated evaluator, all of which produce the same artifacts as ThyFormer's:

    efficientnet*  → evaluate_efficientnet   (EfficientNetV2-S, native 300px)
    vit*           → evaluate_vit            (ViT-B/16,         native 224px)
    swin*          → evaluate_swin           (Swin-Tiny,        native 224px)
    convnext*      → evaluate_convnext       (ConvNeXt-Tiny,    native 224px)
    anything else  → generic path            (any timm name, or "thyformer" to
                                              compare two ThyFormer checkpoints)

The four family evaluators load the train/trainer.py BackboneModel checkpoints
(e.g. artifacts/multiple_model_stanford_output/<family>/best.pt) and run them
exactly as scripts/inference.py does: the model is rebuilt with
models.models.BackboneModel, inputs are resized to the backbone's native
training resolution (BACKBONE_SIZE: efficientnet 300px, the rest 224px), and
the 5-class TR-1..TR-5 probabilities are folded to this pipeline's 4 classes
(TR-4 + TR-5 → T4). The outputs then feed the same metrics/artifact pipeline
as ThyFormer, unchanged.

Per-model artifacts (in each model's folder):

    metrics.json/.csv            scalar test metrics (incl. calibration + ordinal)
    metrics_clip_level.json/.csv same metrics with one row per clip (nodule)
    metrics_ci.json/.csv         cluster-bootstrap 95% CI for every scalar metric
    predictions.csv              per-sample labels + probabilities
    classification_report.txt    precision/recall/F1/support
    per_class_metrics.csv/.png   AUC/AP/F1/sens/spec/precision/NPV per class
    confusion_matrix*.csv/.png   counts + row-normalized heatmap
    roc_curves.csv/.png          one-vs-rest ROC + macro average
    pr_curves.csv/.png           precision-recall per class
    reliability_diagram.png      calibration (ECE/MCE/Brier) + calibration_bins.csv
    gradcam/                     GradCAM heatmaps (ThyFormer only)

Comparison artifacts (only when both models are evaluated):

    auc_comparison.json          significance of the macro-AUC difference
                                 (paired CLUSTER bootstrap over clips — not DeLong's test)
    comparison_metrics.csv/.png  side-by-side metrics + grouped bars
    comparison_roc.png           macro-average ROC overlay

Run-level artifacts (at the run root):

    clinical_agreement.json      Cohen's kappa vs radiologist CSV (if --rad_csv)
    run_config.json              full config, args, environment, model size
    README.md                    headline results + file manifest

Usage:
    # ThyFormer only
    python -m scripts.thyformer_evaluate \
        --checkpoint artifacts/v2_thyformer_v2_720/checkpoints/ep009_auc0.9999.pt

    # Baseline only — checkpoint auto-resolved from
    # artifacts/multiple_model_stanford_output/<family>/best.pt, or explicit:
    python -m scripts.thyformer_evaluate --baseline efficientnet
    python -m scripts.thyformer_evaluate --baseline swin --baseline_ckpt runs/swin/best.pt

    # ThyFormer vs baseline: both evaluated independently, then compared
    python -m scripts.thyformer_evaluate \
        --checkpoint artifacts/v2_thyformer_v2_720/checkpoints/ep009_auc0.9999.pt \
        --baseline   efficientnet

    # Legacy form (still supported): checkpoint path as --baseline
    python -m scripts.thyformer_evaluate \
        --checkpoint artifacts/v2_thyformer_v2_720/checkpoints/ep010_auc0.9999.pt \
        --baseline artifacts/multiple_model_stanford_output/efficientnet/best.pt \
        --baseline_arch efficientnet
"""
import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from configs.thyformer_config import get_config
from data_pipeline.thyformer_create_dataset import build_dataloaders
from models.thyformer_models import (
    build_model,
    build_baseline_model,
    model_cfg_for_checkpoint,
    TimmClassifier,
)
from train.thyformer_train import amp_settings
from utils.thyformer_explainability import run_gradcam_batch
from utils.thyformer_metrics import (
    aggregate_by_group,
    bootstrap_cis,
    clip_ids,
    compute_calibration,
    compute_kappa,
    compute_metrics,
    paired_auc_test,
    print_ci_table,
    print_comparison_table,
    print_results_table,
    _scalar_metrics,
)
from utils import thyformer_report as report


def load_checkpoint(path: str, device: str = "cuda") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(f"Loaded: {path}  (epoch {epoch})")
    return ckpt


def _state_dict_from(ckpt) -> dict:
    """
    Pull model weights out of a checkpoint that may be a full training payload
    ({'model': ...}) or a bare state_dict, and strip any DataParallel /
    torch.compile key prefixes so weights from differently-wrapped models load
    cleanly.
    """
    sd = ckpt
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(ckpt.get(key), dict):
                sd = ckpt[key]
                break

    cleaned = {}
    for k, v in sd.items():
        for pref in ("module.", "_orig_mod."):
            if k.startswith(pref):
                k = k[len(pref) :]
        cleaned[k] = v
    return cleaned


# Baseline families and their default timm backbones — the same four
# architectures benchmarked by scripts/evaluate.py (ConvNeXt-Tiny,
# EfficientNetV2-S, Swin-Tiny, ViT-B/16). Exact timm names like
# "efficientnet_b0" pass through unchanged; a bare family name resolves here.
BASELINE_TIMM_NAMES = {
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
    "efficientnet": "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "vit": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
}

# Checkpoint search locations for --baseline, in priority order: the
# train/trainer.py Stanford runs, thyformer_train_baseline.py output, then the
# legacy scripts/evaluate.py run.
DEFAULT_BASELINE_DIRS = [
    Path("artifacts/multiple_model_stanford_output"),
    Path("artifacts/v2_thyformer_v2_720/baselines"),
    Path("artifacts/multiple_test_v1/ckpts"),
]


def _baseline_family(name: str) -> str | None:
    """Map an architecture name to its family, e.g. "efficientnet_b0" → "efficientnet"."""
    n = name.lower()
    if "thyformer" in n or n == "same":
        return None
    for family in BASELINE_TIMM_NAMES:
        if family in n:
            return family
    return None


class LegacyBackboneAdapter(torch.nn.Module):
    """
    Wraps an old-pipeline models.models.BackboneModel (5-class TR-1..TR-5,
    trained on 224/300px nodule crops via scripts/evaluate.py) so it plugs into
    this 4-class 720px evaluation:

      * inputs are resized to the backbone's native training resolution
        (both pipelines normalize with the same ImageNet mean/std);
      * the 5 output probabilities are folded to the 4 thyformer classes —
        TR-1..TR-3 map 1:1 and p(T4) = p(TR-4) + p(TR-5), mirroring
        thyformer_create_dataset.TIRADS_MAP which sends TR-4 and TR-5 to
        class 3. Returned `cls_logits` are log-probabilities, so downstream
        softmax recovers exactly these merged probabilities.
    """

    def __init__(self, model, img_size: int, num_classes_out: int):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.num_classes_out = num_classes_out

    def forward(self, x):
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        logits = self.model(x)
        probs = logits.float().softmax(dim=-1)
        keep = self.num_classes_out - 1
        probs = torch.cat([probs[:, :keep], probs[:, keep:].sum(dim=-1, keepdim=True)], dim=-1)
        return {"cls_logits": probs.clamp_min(1e-12).log()}


def _load_legacy_baseline(ckpt, state, cfg):
    """Rebuild a scripts/evaluate.py BackboneModel from its checkpoint and wrap
    it for this pipeline. The checkpoint stores its own backbone name."""
    from data_pipeline.create_dataset import BACKBONE_SIZE
    from models.models import BackboneModel

    backbone = ckpt.get("backbone") if isinstance(ckpt, dict) else None
    if backbone not in BACKBONE_SIZE:
        raise SystemExit(
            f"Legacy checkpoint has unknown backbone {backbone!r} "
            f"(expected one of {list(BACKBONE_SIZE)})."
        )
    n_out = state["head.4.weight"].shape[0]
    if n_out < cfg.data.num_classes:
        raise SystemExit(
            f"Legacy checkpoint has {n_out} classes — cannot map onto "
            f"{cfg.data.num_classes} thyformer classes."
        )
    model = BackboneModel(backbone, num_classes=n_out, pretrained=False, freeze_epochs=0)
    model.load_state_dict(state)
    print(
        f"Legacy baseline [{backbone}]  {n_out}-class head folded to "
        f"{cfg.data.num_classes} classes, inputs resized to {BACKBONE_SIZE[backbone]}px"
    )
    return LegacyBackboneAdapter(model, BACKBONE_SIZE[backbone], cfg.data.num_classes)


def load_baseline(arch: str, path: str, cfg, device: str):
    """Build the requested baseline architecture and load its task-trained weights."""
    ckpt = load_checkpoint(path, device)
    state = _state_dict_from(ckpt)

    if any(k.startswith("backbone.") for k in state) and "head.4.weight" in state:
        return _load_legacy_baseline(ckpt, state, cfg).to(device), ckpt

    model = build_baseline_model(arch, cfg.model, num_classes=cfg.data.num_classes)
    try:
        if isinstance(model, TimmClassifier):
            model.load_compatible(state)
        else:
            model.load_state_dict(state)
    except RuntimeError as e:
        hint = (
            "it looks like torchvision-format weights (this pipeline builds timm "
            "models — torchvision checkpoints are incompatible)"
            if any(k.startswith(("features.", "classifier.")) for k in state)
            else "its keys don't match this architecture"
        )
        raise SystemExit(
            f"Checkpoint {path} does not fit baseline '{arch}': {hint}. "
            f"Baselines must be trained on this task with "
            f"scripts/thyformer_train_baseline.py --arch {arch}, which saves a "
            f"compatible best.pt.\nOriginal error: {e}"
        ) from e
    return model.to(device), ckpt


@torch.no_grad()
def collect_outputs(model, loader, cfg, device, desc):
    """
    One pass over a loader → per-sample logits, labels, stems. Model-agnostic:
    works for ThyFormer and any TimmClassifier baseline (both emit `cls_logits`).
    """
    model.eval()
    amp_enabled, amp_dtype = amp_settings(cfg)
    all_logits, all_labels, all_stems = [], [], []

    for batch in tqdm(loader, desc=desc, unit="batch"):
        imgs = batch["image"].to(device, non_blocking=True)
        lbs = batch["label"]
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            preds = model(imgs)
        all_logits.append(preds["cls_logits"].float().cpu())
        hard = lbs.argmax(1) if lbs.dim() == 2 else lbs
        all_labels.append(hard.cpu())
        all_stems.extend(batch["stem"])

    return torch.cat(all_logits), torch.cat(all_labels), all_stems


def _evaluate_model_core(
    model,
    model_name,
    loader,
    cfg,
    device,
    out_dir,
    class_names,
    n_gradcam=0,
    n_bootstrap=1000,
):
    """
    Full, independent evaluation of a single already-built model. Writes the
    complete paper-ready artifact set into out_dir and returns everything the
    caller needs for a downstream comparison.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Evaluating: {model_name}  →  {out_dir}\n{'='*60}")

    logits, labels_t, stems = collect_outputs(
        model, loader, cfg, device, desc=f"Evaluating {model_name}"
    )
    # Cross-entropy on the class head — comparable across ThyFormer and baselines
    # (baselines have no seg/boundary heads, so we don't use the composite loss).
    test_loss = float(F.cross_entropy(logits, labels_t).item())
    metrics = compute_metrics(logits, labels_t, prefix="test")
    metrics["test_loss"] = test_loss
    print_results_table(metrics, f"{model_name} — Test Results")

    probs = F.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    labels = labels_t.numpy()

    print("Saving metrics, predictions, and figures …")
    report.save_metrics(out_dir, metrics)
    report.save_predictions_csv(out_dir, stems, labels, preds, probs, class_names)
    report.save_classification_report(out_dir, labels, preds, class_names)
    report.save_confusion_matrix(out_dir, labels, preds, class_names)
    report.save_roc_curves(out_dir, labels, probs, class_names)
    report.save_pr_curves(out_dir, labels, probs, class_names)
    report.save_per_class_metrics(out_dir, labels, preds, probs, class_names)

    # Calibration: reliability diagram + bin table (ECE/MCE/Brier scalars are
    # already in `metrics` via compute_metrics).
    calib, calib_bins = compute_calibration(labels, probs)
    report.save_reliability_diagram(out_dir, calib_bins, calib)

    # Bootstrap 95% CIs for every scalar metric (percentile method). Frames are
    # video frames nested in clips, so resampling is done over CLIPS — a frame
    # bootstrap treats ~29 nodules as ~3066 observations and reports intervals
    # roughly 10x too narrow.
    groups = clip_ids(stems)
    n_clips = len(np.unique(groups))
    cis = None
    if n_bootstrap > 0:
        print(
            f"Bootstrapping 95% CIs ({n_bootstrap} resamples, clustered over " f"{n_clips} clips) …"
        )
        cis = bootstrap_cis(labels, probs, n_bootstrap=n_bootstrap, groups=groups)
        report.save_metric_cis(out_dir, cis)
        print_ci_table(
            {k: v for k, v in cis.items() if "_t" not in k},  # headline metrics only
            f"{model_name} — 95% CI (cluster bootstrap over {n_clips} clips)",
        )

    # Clip-level view: one row per nodule (mean probability over its frames), so
    # a 293-frame clip does not outweigh a 25-frame one.
    y_clip, probs_clip, _ = aggregate_by_group(labels, probs, groups)
    clip_metrics = {f"test_{k}": v for k, v in _scalar_metrics(y_clip, probs_clip).items()}
    report.save_metrics(out_dir, clip_metrics, stem="metrics_clip_level")
    print_results_table(
        {k: v for k, v in clip_metrics.items() if "_t" not in k},
        f"{model_name} — clip-level test results (n={n_clips} nodules)",
    )

    if n_gradcam > 0:
        print(f"Generating {n_gradcam} GradCAM figures …")
        run_gradcam_batch(
            model, loader, out_dir=str(out_dir / "gradcam"), n=n_gradcam, device=device
        )

    return {
        "name": model_name,
        "metrics": metrics,
        "clip_metrics": clip_metrics,
        "probs": probs,
        "preds": preds,
        "labels": labels,
        "stems": stems,
        "groups": groups,
        "n_clips": n_clips,
        "cis": cis,
    }


def _evaluate_baseline_arch(
    family,
    arch,
    checkpoint,
    loader,
    cfg,
    device,
    out_dir,
    class_names,
    n_bootstrap=1000,
):
    """
    Shared path behind the per-family evaluators. The checkpoint is a
    train/trainer.py BackboneModel trained at its own native resolution, so the
    model is rebuilt and run exactly as scripts/inference.py does it:
    BackboneModel(family, pretrained=False) + the checkpoint's
    model_state_dict, inputs resized to BACKBONE_SIZE[family], and the 5-class
    TR-1..TR-5 probabilities folded to the 4 thyformer classes. The outputs
    then get the exact same full evaluation ThyFormer gets (metrics, CIs,
    curves, calibration, predictions).
    """
    from data_pipeline.create_dataset import BACKBONE_SIZE
    from models.models import BackboneModel

    ckpt = load_checkpoint(str(checkpoint), device)
    state = _state_dict_from(ckpt)
    model = BackboneModel(family, pretrained=False)  # 5-class head, as in inference.py
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise SystemExit(
            f"Checkpoint {checkpoint} does not fit the '{family}' BackboneModel "
            f"from models/models.py — expected a train/trainer.py checkpoint "
            f"(e.g. artifacts/multiple_model_stanford_output/{family}/best.pt)."
            f"\nOriginal error: {e}"
        ) from e

    img_size = BACKBONE_SIZE[family]
    print(
        f"[{arch}] BackboneModel loaded as in scripts/inference.py — inputs "
        f"resized to native {img_size}px, TR-4/TR-5 folded into {class_names[-1]}"
    )
    model = LegacyBackboneAdapter(model, img_size, cfg.data.num_classes).to(device)

    out = _evaluate_model_core(
        model,
        arch,
        loader,
        cfg,
        device,
        out_dir,
        class_names,
        n_gradcam=0,
        n_bootstrap=n_bootstrap,
    )
    out.update(model=model, ckpt=ckpt, checkpoint_path=str(checkpoint))
    return out


def evaluate_efficientnet(
    arch,
    checkpoint,
    loader,
    cfg,
    device,
    out_dir,
    class_names,
    n_bootstrap=1000,
):
    """EfficientNet baseline: the train/trainer.py EfficientNetV2-S checkpoint,
    run at its native 300px as scripts/inference.py does."""
    return _evaluate_baseline_arch(
        "efficientnet",
        arch,
        checkpoint,
        loader,
        cfg,
        device,
        out_dir,
        class_names,
        n_bootstrap,
    )


def evaluate_vit(
    arch,
    checkpoint,
    loader,
    cfg,
    device,
    out_dir,
    class_names,
    n_bootstrap=1000,
):
    """ViT baseline: the train/trainer.py ViT-B/16 checkpoint, run at its
    native 224px as scripts/inference.py does."""
    return _evaluate_baseline_arch(
        "vit",
        arch,
        checkpoint,
        loader,
        cfg,
        device,
        out_dir,
        class_names,
        n_bootstrap,
    )


def evaluate_swin(
    arch,
    checkpoint,
    loader,
    cfg,
    device,
    out_dir,
    class_names,
    n_bootstrap=1000,
):
    """Swin baseline: the train/trainer.py Swin-Tiny checkpoint, run at its
    native 224px as scripts/inference.py does."""
    return _evaluate_baseline_arch(
        "swin",
        arch,
        checkpoint,
        loader,
        cfg,
        device,
        out_dir,
        class_names,
        n_bootstrap,
    )


def evaluate_convnext(
    arch,
    checkpoint,
    loader,
    cfg,
    device,
    out_dir,
    class_names,
    n_bootstrap=1000,
):
    """ConvNeXt baseline: the train/trainer.py ConvNeXt-Tiny checkpoint, run at
    its native 224px as scripts/inference.py does."""
    return _evaluate_baseline_arch(
        "convnext",
        arch,
        checkpoint,
        loader,
        cfg,
        device,
        out_dir,
        class_names,
        n_bootstrap,
    )


BASELINE_EVALUATORS = {
    "efficientnet": evaluate_efficientnet,
    "vit": evaluate_vit,
    "swin": evaluate_swin,
    "convnext": evaluate_convnext,
}


def evaluate_one_model(
    model,
    model_name,
    loader,
    cfg,
    device,
    out_dir,
    class_names,
    n_gradcam=0,
    n_bootstrap=1000,
    checkpoint=None,
):
    """
    Evaluate one model and write its full artifact set into out_dir.

    Two entry modes:
      * `model` is a built module (the ThyFormer path) → evaluated directly,
        exactly as before.
      * `model` is None and `checkpoint` is a path → `model_name` picks the
        evaluator: efficientnet / vit / swin / convnext names dispatch to
        evaluate_efficientnet / evaluate_vit / evaluate_swin /
        evaluate_convnext; anything else (another timm arch or "thyformer")
        is built generically and evaluated the same way.
    """
    if model is not None:
        return _evaluate_model_core(
            model,
            model_name,
            loader,
            cfg,
            device,
            out_dir,
            class_names,
            n_gradcam=n_gradcam,
            n_bootstrap=n_bootstrap,
        )

    if checkpoint is None:
        raise ValueError(
            f"evaluate_one_model needs either a built model or a checkpoint "
            f"path for {model_name!r}"
        )
    family = _baseline_family(model_name)
    if family is not None:
        return BASELINE_EVALUATORS[family](
            model_name,
            checkpoint,
            loader,
            cfg,
            device,
            out_dir,
            class_names,
            n_bootstrap=n_bootstrap,
        )
    built, ckpt = load_baseline(model_name, checkpoint, cfg, device)
    out = _evaluate_model_core(
        built,
        model_name,
        loader,
        cfg,
        device,
        out_dir,
        class_names,
        n_gradcam=0,
        n_bootstrap=n_bootstrap,
    )
    out.update(model=built, ckpt=ckpt, checkpoint_path=str(checkpoint))
    return out


def _resolve_baseline_checkpoint(base_name: str, explicit: str | None) -> Path:
    """
    Locate the baseline's checkpoint: --baseline_ckpt when given, otherwise the
    default folder scripts/thyformer_train_baseline.py saves into — tried under
    the name as passed, then under the resolved family default.
    """
    if explicit:
        return Path(explicit)
    family = _baseline_family(base_name)
    names = [base_name]
    if family:
        if base_name.lower() == family:
            names.append(BASELINE_TIMM_NAMES[family])
        else:
            # legacy scripts/evaluate.py runs store checkpoints under the family name
            names.append(family)
    candidates = [
        root / name / stem
        for name in names
        for root in DEFAULT_BASELINE_DIRS
        for stem in ("best.pt", "best.pth")
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(
        f"No checkpoint found for baseline '{base_name}' (tried: {tried}). "
        f"Train one with scripts/thyformer_train_baseline.py or pass --baseline_ckpt."
    )


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def run_evaluation(args):
    if not args.checkpoint and not args.baseline:
        raise SystemExit("Nothing to evaluate: pass --checkpoint, --baseline, or both.")

    # Resolve the baseline up front so a missing checkpoint fails fast,
    # before any data loading.
    # Legacy form: --baseline <checkpoint path> --baseline_arch <arch>.
    # New form: --baseline <arch>, checkpoint from --baseline_ckpt or the
    # default training output folder.
    base_name = base_ckpt = None
    if args.baseline:
        if Path(args.baseline).is_file():
            base_name, base_ckpt = args.baseline_arch, Path(args.baseline)
        else:
            base_name = args.baseline
            base_ckpt = _resolve_baseline_checkpoint(base_name, args.baseline_ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    class_names = cfg.data.class_names
    if args.out_dir:
        run_dir = Path(args.out_dir)
    else:
        run_dir = Path(
            cfg.evaluation.output_dir
        ) / f"{base_name}_{datetime.now().strftime('run_%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation run → {run_dir}  (device: {device})")

    # ── Data (shared, shuffle=False → identical sample order for both models) ──
    loaders = build_dataloaders(
        cfg.data, cfg.augmentation, cfg.training.batch_size, cfg.training.num_workers
    )
    # ── ThyFormer: independent full evaluation ────────────────────
    thy_out = thy_model = thy_ckpt = None
    if args.checkpoint:
        thy_ckpt = load_checkpoint(args.checkpoint, device)
        # Rebuild the architecture the checkpoint was trained under. eca_gate in
        # particular changes the ECA block's function without changing any
        # parameter shape, so loading a legacy checkpoint into today's default
        # would load cleanly and silently score a different model.
        thy_model = build_model(model_cfg_for_checkpoint(thy_ckpt, cfg.model))
        thy_model.load_state_dict(_state_dict_from(thy_ckpt))
        thy_model = thy_model.to(device)
        thy_out = evaluate_one_model(
            thy_model,
            "ThyFormer",
            loaders["test"],
            cfg,
            device,
            run_dir / "thyformer",
            class_names,
            n_gradcam=args.n_gradcam,
            n_bootstrap=args.n_bootstrap,
        )

    # ── Baseline: independent full evaluation, then comparison ────
    base_out = None
    delong_results = None
    if base_name:
        base_out = evaluate_one_model(
            None,
            base_name,
            loaders["test"],
            cfg,
            device,
            run_dir / _safe_name(base_name),
            class_names,
            n_gradcam=0,
            n_bootstrap=args.n_bootstrap,
            checkpoint=base_ckpt,
        )

    if thy_out and base_out:
        base_name = base_out["name"]
        if not np.array_equal(thy_out["labels"], base_out["labels"]):
            print(
                "WARNING: label order differs between models — the paired test "
                "assumes both were scored on the same samples in the same order."
            )

        print(f"\n{'='*60}\n  Comparison: ThyFormer vs {base_name}\n{'='*60}")
        groups = thy_out["groups"]
        n_clips = thy_out["n_clips"]
        print(
            f"Paired cluster bootstrap over {n_clips} clips "
            f"({len(groups)} frames) — frames within a clip are not independent."
        )
        test = paired_auc_test(
            thy_out["labels"],
            thy_out["probs"],
            base_out["probs"],
            n_bootstrap=args.n_bootstrap or 1000,
            groups=groups,
        )
        p_val = test["p_value"]
        auc_a, auc_b, z = test["auc_a"], test["auc_b"], test["z_statistic"]
        delong_results = {
            "model_a": "ThyFormer",
            "model_b": base_name,
            "thyformer_auc": auc_a,
            "baseline_auc": auc_b,
            "auc_difference": test["auc_difference"],
            "diff_ci_low": test["diff_ci_low"],
            "diff_ci_high": test["diff_ci_high"],
            "z_statistic": z,
            "p_value": p_val,
            "significant_at_0.05": bool(p_val < 0.05),
            # Provenance for the statistic itself: this is a paired cluster
            # bootstrap over clips, NOT DeLong's test, and the number of
            # independent units is n_units — not n_samples.
            "method": test["method"],
            "resampling_unit": test["unit"],
            "n_units": test["n_units"],
            "n_samples": test["n_samples"],
            "n_bootstrap_used": test["n_bootstrap_used"],
            "thyformer_auc_clip_level": thy_out["clip_metrics"].get("test_auc"),
            "baseline_auc_clip_level": base_out["clip_metrics"].get("test_auc"),
            "thyformer_checkpoint": args.checkpoint,
            "baseline_checkpoint": base_out["checkpoint_path"],
            "baseline_arch": base_name,
        }
        report.save_comparison(
            run_dir / "comparison",
            "ThyFormer",
            thy_out["metrics"],
            thy_out["probs"],
            base_name,
            base_out["metrics"],
            base_out["probs"],
            thy_out["labels"],
            class_names,
            delong_results,
        )
        print_comparison_table(["ThyFormer", base_name], [thy_out["metrics"], base_out["metrics"]])
        print(f"\nPaired cluster bootstrap on macro AUC (unit: clip, n={test['n_units']}):")
        print(
            f"  ThyFormer AUC : {auc_a:.4f}   (clip-level {thy_out['clip_metrics']['test_auc']:.4f})"
        )
        print(
            f"  {base_name} AUC : {auc_b:.4f}   (clip-level {base_out['clip_metrics']['test_auc']:.4f})"
        )
        print(
            f"  Δ AUC         : {test['auc_difference']:+.4f}  "
            f"95% CI [{test['diff_ci_low']:+.4f}, {test['diff_ci_high']:+.4f}]"
        )
        print(
            f"  p-value       : {p_val:.4f}  ({'significant' if p_val<0.05 else 'n.s.'} at α=0.05)"
        )
        if test["n_units"] < 50:
            print(
                f"  ⚠ only {test['n_units']} independent clips — this design cannot "
                f"resolve small AUC differences; treat a non-significant result as "
                f"'underpowered', not 'equivalent'."
            )

    # The model whose predictions feed the run-level artifacts (kappa, metadata):
    # ThyFormer when evaluated, otherwise the baseline.
    primary = thy_out or base_out

    # ── Cohen's kappa (model vs radiologist grades) ───────────────
    kappa_results = None
    if args.rad_csv:
        print(f"\nComputing Cohen's kappa ({primary['name']} vs radiologist) …")
        rad_df = pd.read_csv(args.rad_csv)
        # Expected columns: stem, radiologist_label (0..3)
        pred_map = {stem: int(pred) for stem, pred in zip(primary["stems"], primary["preds"])}

        rad_stems = rad_df["stem"].values
        rad_l = rad_df["radiologist_label"].values
        mod_l = np.array([pred_map.get(s, -1) for s in rad_stems])
        valid = mod_l >= 0
        if valid.sum() < len(rad_stems):
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
        kappa_results = {
            **kappa_m,
            "model": primary["name"],
            "interpretation": interp,
            "n_matched": int(valid.sum()),
            "rad_csv": args.rad_csv,
        }
        report.save_json(kappa_results, run_dir / "clinical_agreement.json")

    # ── Run record + README ───────────────────────────────────────
    labels = primary["labels"]
    class_counts = {class_names[c]: int((labels == c).sum()) for c in range(cfg.data.num_classes)}
    meta_model = thy_model if thy_model is not None else base_out["model"]
    meta_ckpt = thy_ckpt if thy_ckpt is not None else base_out["ckpt"]
    report.save_run_metadata(run_dir, cfg, args, meta_ckpt, meta_model, len(labels), class_counts)
    _write_readme(run_dir, thy_out, base_out, delong_results, kappa_results, args)
    print(f"\nAll evaluation artifacts saved to: {run_dir}")


_README_HEADLINE_KEYS = (
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
)


def _metric_rows(metrics, cis) -> List[str]:
    """Markdown table rows: metric | value (95% CI lo–hi when available)."""
    rows = []
    for key in _README_HEADLINE_KEYS:
        if key not in metrics:
            continue
        ci = (cis or {}).get(key.removeprefix("test_"))
        val = f"{metrics[key]:.4f}"
        if ci:
            val += f" ({ci['ci_low']:.4f}–{ci['ci_high']:.4f})"
        rows.append(f"| {key} | {val} |")
    return rows


def _write_readme(run_dir, thy_out, base_out, delong, kappa, args) -> None:
    lines = ["# ThyFormer Evaluation Run", ""]
    if args.checkpoint:
        lines.append(f"- ThyFormer checkpoint: `{args.checkpoint}`")
    lines += [f"- Generated: {datetime.now().isoformat(timespec='seconds')}"]

    if thy_out:
        lines += [
            "",
            "## ThyFormer — headline test metrics",
            "",
            "| Metric | Value (95% CI) |",
            "|---|---|",
        ]
        lines += _metric_rows(thy_out["metrics"], thy_out.get("cis"))

    if base_out:
        lines += [
            "",
            f"## {base_out['name']} baseline — headline test metrics",
            "",
            "| Metric | Value (95% CI) |",
            "|---|---|",
        ]
        lines += _metric_rows(base_out["metrics"], base_out.get("cis"))

    if delong:
        verdict = "significant" if delong["significant_at_0.05"] else "not significant"
        lines += [
            "",
            "## Comparison — paired cluster bootstrap (macro AUC)",
            "",
            f"- Baseline: `{delong['baseline_checkpoint']}` ({delong['baseline_arch']})",
            f"- ThyFormer AUC **{delong['thyformer_auc']:.4f}** vs "
            f"{delong['baseline_arch']} **{delong['baseline_auc']:.4f}** "
            f"(Δ = {delong['auc_difference']:+.4f}, 95% CI "
            f"{delong['diff_ci_low']:+.4f} to {delong['diff_ci_high']:+.4f})",
            f"- p = {delong['p_value']:.4f} → difference is **{verdict}** at α = 0.05",
            "",
            f"> Resampling unit: **{delong['resampling_unit']}** — "
            f"{delong['n_units']} independent clips behind {delong['n_samples']} frames. "
            "Frames from one clip are near-duplicate views of the same nodule, so they "
            "are resampled together. This is a paired cluster bootstrap, **not DeLong's "
            "test**; a frame-level bootstrap understates the standard error by roughly "
            "sqrt(frames/clips) and reports spuriously tiny p-values.",
        ]
        if delong.get("thyformer_auc_clip_level") is not None:
            lines.append(
                f"> Clip-level macro AUC (one row per nodule, mean probability): "
                f"ThyFormer {delong['thyformer_auc_clip_level']:.4f} vs "
                f"{delong['baseline_arch']} {delong['baseline_auc_clip_level']:.4f}."
            )
        if delong["n_units"] < 50:
            lines.append(
                f"> **Underpowered:** {delong['n_units']} clips cannot resolve small AUC "
                "differences. A non-significant result here means *not measurable*, not "
                "*equivalent*."
            )
    if kappa:
        lines += [
            "",
            "## Clinical agreement",
            "",
            f"- Cohen's κ = {kappa['kappa']:.4f} ({kappa['interpretation']}), n = {kappa['n_matched']}",
        ]

    lines += [
        "",
        "## Folder layout",
        "",
    ]
    if thy_out:
        lines.append(
            "- `thyformer/` — ThyFormer's full evaluation (metrics + bootstrap CIs, "
            "predictions, classification report, confusion matrix, ROC/PR curves, "
            "reliability diagram, per-class bars, GradCAM)"
        )
    if base_out:
        lines.append(
            f"- `{_safe_name(base_out['name'])}/` — the {base_out['name']} baseline's "
            "full evaluation (same artifacts, no GradCAM)"
        )
    if delong:
        lines.append(
            "- `comparison/` — `auc_comparison.json`, `comparison_metrics.csv`/`.png`, `comparison_roc.png`"
        )
    if kappa:
        lines.append("- `clinical_agreement.json` — Cohen's κ vs radiologist grades")
    lines += [
        "- `run_config.json` — full config, CLI args, environment, model size",
        "",
    ]
    (run_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate ThyFormer and/or a baseline model")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="ThyFormer checkpoint to evaluate (omit for a baseline-only run)",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output folder for this run (default: timestamped subfolder of "
        "cfg.evaluation.output_dir)",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Baseline to evaluate independently: an architecture name — a family "
        'alias ("efficientnet", "vit", "swin", "convnext"), any timm model name '
        '(e.g. efficientnet_b0, resnet50), or "thyformer" — or, legacy form, a '
        "checkpoint path combined with --baseline_arch",
    )
    p.add_argument(
        "--baseline_ckpt",
        default=None,
        help="Baseline checkpoint (default: <arch>/best.pt searched under "
        "artifacts/multiple_model_stanford_output, then "
        "artifacts/v2_thyformer_v2_720/baselines, then "
        "artifacts/multiple_test_v1/ckpts)",
    )
    p.add_argument(
        "--baseline_arch",
        default="thyformer",
        help="Legacy: baseline architecture when --baseline is a checkpoint path",
    )
    p.add_argument("--rad_csv", default=None, help="CSV with radiologist grades for kappa")
    p.add_argument(
        "--n_gradcam", type=int, default=50, help="GradCAM figures for ThyFormer (0 to skip)"
    )
    p.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for 95%% CIs on all metrics (0 to skip)",
    )
    args = p.parse_args()
    run_evaluation(args)
