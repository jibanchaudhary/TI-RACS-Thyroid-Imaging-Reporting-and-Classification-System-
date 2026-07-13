# """
# trainer.py
# ----------
# Training engine for individual BackboneModels.

# Features
# --------
#   - Label-smoothed cross-entropy loss
#   - Cosine annealing LR scheduler with linear warm-up
#   - WeightedRandomSampler to handle class imbalance
#   - Gradient clipping
#   - Early stopping on val loss
#   - Per-epoch checkpoint saving (best + last)
#   - Resume interrupted training (--resume auto | --resume <ckpt path>)
#   - CSV metrics log
#   - Saved metric plots (training curves, confusion matrix, ROC/PR curves,
#     per-class precision/recall/F1) under <output_dir>/<backbone>/plots/
#   - Grad-CAM hook registration (used by inference.py)
#   - Supports mixed precision (torch.amp) when CUDA available

# Usage
# -----
#     from trainer import Trainer
#     trainer = Trainer(backbone="convnext", data_dir="data", output_dir="runs")
#     trainer.fit(epochs=30)
# """

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

import numpy as np
import matplotlib


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

from data_pipeline.create_dataset import ThyroidDataset, CLASS_NAMES, NUM_CLASSES
from models.models import build_model
import argparse

matplotlib.use("Agg")

# Plot styling: CVD-safe categorical series + one-hue sequential ramp for heatmaps
PLOT_SERIES = [
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#e87ba4",
    "#eb6834",
]
PLOT_SEQ_RAMP = [
    "#fcfcfb",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#1c5cab",
    "#0d366b",
]
PLOT_SURFACE = "#fcfcfb"
PLOT_INK = "#0b0b0b"
PLOT_INK_2 = "#52514e"
PLOT_MUTED = "#898781"
PLOT_GRID = "#e1e0d9"
PLOT_AXIS = "#c3c2b7"


def _style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PLOT_AXIS)
    ax.tick_params(colors=PLOT_MUTED, labelsize=9)
    ax.grid(True, color=PLOT_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(PLOT_SURFACE)


def compute_epoch_metrics(all_labels, all_preds, all_probs) -> dict:
    """
    Full per-epoch metric set from collected predictions.
    Returns acc, macro f1/auc/sensitivity/specificity, and per-class auc/f1
    (NaN for classes absent from this epoch's labels).
    """
    labels = np.asarray(all_labels)
    preds = np.asarray(all_preds)
    probs = np.asarray(all_probs)

    m = {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "sens": recall_score(labels, preds, average="macro", zero_division=0),
        "per_f1": f1_score(
            labels, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
        ).tolist(),
    }

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    total = cm.sum()
    support = cm.sum(axis=1)
    spec = np.full(NUM_CLASSES, np.nan)
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        tn = total - cm[i, :].sum() - fp
        if tn + fp > 0:
            spec[i] = tn / (tn + fp)
    present = support > 0
    m["spec"] = float(np.nanmean(spec[present])) if present.any() else float("nan")

    y_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    per_auc = []
    for i in range(NUM_CLASSES):
        pos = int(y_bin[:, i].sum())
        if 0 < pos < len(labels):
            per_auc.append(float(roc_auc_score(y_bin[:, i], probs[:, i])))
        else:
            per_auc.append(float("nan"))
    m["per_auc"] = per_auc
    m["auc"] = float(np.nanmean(per_auc)) if not all(np.isnan(v) for v in per_auc) else float("nan")
    return m


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """Linear warm-up for warmup_epochs, then cosine decay."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        e = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if e < self.warmup_epochs:
                lr = base_lr * (e + 1) / max(self.warmup_epochs, 1)
            else:
                progress = (e - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                import math

                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            lrs.append(lr)
        return lrs


class Trainer:
    """
    Parameters
    ----------
    backbone       : str    "convnext" | "efficientnet" | "swin" | "vit"
    data_dir       : str    Root data folder (contains images/ and annotations/)
    output_dir     : str    Where checkpoints and logs are saved
    batch_size     : int
    lr             : float  Peak learning rate (after warm-up)
    weight_decay   : float
    warmup_epochs  : int    Epochs of linear warm-up
    freeze_epochs  : int    Epochs to keep backbone frozen
    patience       : int    Early-stopping patience (val loss)
    label_smoothing: float
    use_amp        : bool   Mixed precision (auto-disabled on CPU)
    num_workers    : int
    resume         : str    None = fresh run; "auto" = continue from
                            <output_dir>/<backbone>/last.pt if it exists;
                            otherwise a path to a checkpoint to continue from
    """

    LOG_FIELDS = (
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"]
        + ["train_f1", "train_auc", "val_auc"]
        + ["train_sens", "val_sens", "train_spec", "val_spec"]
        + [f"val_auc_{c}" for c in CLASS_NAMES]
        + [f"val_f1_{c}" for c in CLASS_NAMES]
        + [f"train_auc_{c}" for c in CLASS_NAMES]
    )

    def __init__(
        self,
        backbone: str = "convnext",
        data_dir: str = "data",
        output_dir: str = "runs",
        batch_size: int = 192,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        freeze_epochs: int = 5,
        patience: int = 5,
        label_smoothing: float = 0.1,
        use_amp: bool = True,
        num_workers: int = 8,
        resume: str | None = None,
    ):
        self.backbone = backbone
        self.data_dir = data_dir
        self.output_dir = os.path.join(output_dir, backbone)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.freeze_epochs = freeze_epochs
        self.patience = patience
        self.use_amp = use_amp and torch.cuda.is_available()

        os.makedirs(self.output_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self.resume_path = None
        if resume:
            cand = os.path.join(self.output_dir, "last.pt") if resume == "auto" else resume
            if os.path.isfile(cand):
                self.resume_path = cand
            elif resume == "auto":
                print(f"No last.pt found in {self.output_dir}; starting fresh")
            else:
                raise FileNotFoundError(f"Resume checkpoint not found: {cand}")

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"\n{'='*60}")
        print(f" Training backbone : {backbone}")
        print(f" Device            : {self.device}")
        print(f" Output dir        : {self.output_dir}")
        print(f"{'='*60}\n")

        # Datasets
        self.train_ds = ThyroidDataset(data_dir, "train", backbone)
        self.val_ds = ThyroidDataset(data_dir, "val", backbone)
        sampler = self.train_ds.get_weighted_sampler()
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Model, loss, optimiser
        self.model = build_model(backbone, freeze_epochs=freeze_epochs)
        self.model.to(self.device)

        self.criterion = LabelSmoothingCE(smoothing=label_smoothing)
        self.scaler = GradScaler() if self.use_amp else None
        self.scheduler = None

        # Separate param groups: lower LR for backbone, higher for head
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.head.parameters())
        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": lr * 0.1},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )

        # CSV log (kept when resuming so training curves stay continuous)
        self.log_path = os.path.join(self.output_dir, "metrics.csv")
        if self.resume_path is None or not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(self.LOG_FIELDS)

    def _train_epoch(self, epoch: int) -> tuple[float, dict]:
        self.model.train()
        self.model.on_epoch_start(epoch)

        total_loss, total = 0.0, 0
        all_preds, all_labels, all_probs = [], [], []

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type="cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            probs = torch.softmax(logits.detach().float(), dim=-1)
            all_preds.extend(probs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch:3d} | step {batch_idx+1:4d}" f" | loss {loss.item():.4f}")

        return total_loss / total, compute_epoch_metrics(all_labels, all_preds, all_probs)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast(device_type="cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits.float(), dim=-1)
            all_preds.extend(probs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        val_loss = total_loss / len(all_labels)
        return val_loss, compute_epoch_metrics(all_labels, all_preds, all_probs)

    def fit(self, epochs: int = 30) -> str:
        """
        Train for `epochs` epochs with early stopping.
        Returns path to the best checkpoint.
        """
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=epochs,
        )

        best_val_loss = float("inf")
        patience_count = 0
        start_epoch = 1
        best_ckpt_path = os.path.join(self.output_dir, "best.pt")

        if self.resume_path:
            start_epoch, best_val_loss, patience_count = self._load_checkpoint()

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()

            train_loss, tm = self._train_epoch(epoch)
            val_loss, vm = self._val_epoch()
            self.scheduler.step()

            val_acc = vm["acc"]
            current_lr = self.optimizer.param_groups[-1]["lr"]
            elapsed = time.time() - t0

            print(
                f"\nEpoch {epoch:3d}/{epochs} | {elapsed:.1f}s | "
                f"train loss {train_loss:.4f}  acc {tm['acc']:.4f} | "
                f"val   loss {val_loss:.4f}  acc {val_acc:.4f}  "
                f"f1 {vm['f1']:.4f}  auc {vm['auc']:.4f} | lr {current_lr:.2e}"
            )

            # Log metrics
            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.5f}",
                "train_acc": f"{tm['acc']:.5f}",
                "val_loss": f"{val_loss:.5f}",
                "val_acc": f"{val_acc:.5f}",
                "val_f1": f"{vm['f1']:.5f}",
                "lr": f"{current_lr:.2e}",
                "train_f1": f"{tm['f1']:.5f}",
                "train_auc": f"{tm['auc']:.5f}",
                "val_auc": f"{vm['auc']:.5f}",
                "train_sens": f"{tm['sens']:.5f}",
                "val_sens": f"{vm['sens']:.5f}",
                "train_spec": f"{tm['spec']:.5f}",
                "val_spec": f"{vm['spec']:.5f}",
            }
            for c, v in zip(CLASS_NAMES, vm["per_auc"]):
                row[f"val_auc_{c}"] = f"{v:.5f}"
            for c, v in zip(CLASS_NAMES, vm["per_f1"]):
                row[f"val_f1_{c}"] = f"{v:.5f}"
            for c, v in zip(CLASS_NAMES, tm["per_auc"]):
                row[f"train_auc_{c}"] = f"{v:.5f}"
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([row[k] for k in self.LOG_FIELDS])

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                self._save_checkpoint(
                    epoch, val_loss, val_acc, best_ckpt_path, best_val_loss, patience_count
                )
                print(f"  ✓ New best val loss: {best_val_loss:.4f}")
            else:
                patience_count += 1
                print(f"  No improvement ({patience_count}/{self.patience})")

            # Save last checkpoint always (after the best/patience update so a
            # resumed run continues early stopping from the right state)
            self._save_checkpoint(
                epoch,
                val_loss,
                val_acc,
                os.path.join(self.output_dir, "last.pt"),
                best_val_loss,
                patience_count,
            )

            if patience_count >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print(f"\nTraining complete. Best checkpoint: {best_ckpt_path}")
        try:
            self._plot_training_curves()
        except Exception as e:
            print(f"Could not plot training curves: {e}")
        self._final_eval(best_ckpt_path)
        return best_ckpt_path

    def _save_checkpoint(self, epoch, val_loss, val_acc, path, best_val_loss, patience_count):
        torch.save(
            {
                "epoch": epoch,
                "backbone": self.backbone,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state": self.scaler.state_dict() if self.scaler else None,
                "best_val_loss": best_val_loss,
                "patience_count": patience_count,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            path,
        )

    def _load_checkpoint(self) -> tuple[int, float, int]:
        """
        Restore model/optimizer/scheduler/scaler state from self.resume_path.
        Returns (start_epoch, best_val_loss, patience_count). A checkpoint
        saved at epoch N resumes training at epoch N+1.
        """
        state = torch.load(self.resume_path, map_location=self.device)

        ckpt_backbone = state.get("backbone", self.backbone)
        if ckpt_backbone != self.backbone:
            raise ValueError(
                f"Checkpoint backbone '{ckpt_backbone}' does not match trainer '{self.backbone}'"
            )

        self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scaler is not None and state.get("scaler_state"):
            self.scaler.load_state_dict(state["scaler_state"])

        completed = state["epoch"]

        # Restore scheduler progress but keep the current run's horizon so a
        # larger --epochs at resume time extends the cosine schedule
        sched_state = state.get("scheduler_state")
        if sched_state is not None:
            warm, total = self.scheduler.warmup_epochs, self.scheduler.total_epochs
            self.scheduler.load_state_dict(sched_state)
            self.scheduler.warmup_epochs, self.scheduler.total_epochs = warm, total
        else:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for _ in range(completed):
                    self.scheduler.step()

        start_epoch = completed + 1

        # models.py unfreezes only when epoch == freeze_epochs, so a resume
        # past that point must unfreeze explicitly
        if start_epoch > self.freeze_epochs:
            self.model._unfreeze_backbone()

        best_val_loss = state.get("best_val_loss", state.get("val_loss", float("inf")))
        patience_count = state.get("patience_count", 0)

        self._trim_metrics_log(completed)

        print(
            f"Resumed from {self.resume_path} "
            f"(epoch {completed} done, best val loss {best_val_loss:.4f}, "
            f"patience {patience_count}/{self.patience}) -> continuing at epoch {start_epoch}"
        )
        return start_epoch, best_val_loss, patience_count

    def _trim_metrics_log(self, completed_epoch: int):
        """Drop CSV rows past the resumed epoch so re-trained epochs are not duplicated."""
        rows = []
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= completed_epoch]
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.LOG_FIELDS)
            for r in rows:
                writer.writerow([r.get(k) or "nan" for k in self.LOG_FIELDS])

    @torch.no_grad()
    def _final_eval(self, ckpt_path: str):
        """Load best checkpoint and print full classification report on val set."""
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        all_preds, all_labels, all_probs = [], [], []
        for images, labels in self.val_loader:
            images = images.to(self.device)
            logits = self.model(images)
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

        print("\n--- Final validation report ---")
        unique_labels = sorted(set(all_labels) | set(all_preds))

        # print(classification_report(all_labels, all_preds,
        #                             target_names=CLASS_NAMES, zero_division=0))

        report = classification_report(
            all_labels,
            all_preds,
            labels=unique_labels,
            target_names=[CLASS_NAMES[i] for i in unique_labels],
            zero_division=0,
        )
        print(report)

        # AUC (macro OvR)
        y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
        macro_auc = None
        try:
            macro_auc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")
            print(f"Macro AUC: {macro_auc:.4f}")
        except ValueError:
            print("AUC: not computable (too few classes in val set)")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
        print("\nConfusion matrix:")
        print(cm)

        self._save_eval_plots(
            all_labels, all_preds, all_probs, cm, unique_labels, report, macro_auc
        )

    # ------------------------------------------------------------------
    # Metric plots
    # ------------------------------------------------------------------

    def _plot_training_curves(self):
        with open(self.log_path) as f:
            history = list(csv.DictReader(f))
        if not history:
            return

        def col(name):
            return np.array([float(r.get(name) or "nan") for r in history])

        epochs = col("epoch")
        fig, axes = plt.subplots(3, 3, figsize=(16, 12), facecolor=PLOT_SURFACE)

        pair_panels = (
            (axes[0][0], "Loss", "train_loss", "val_loss"),
            (axes[0][1], "Accuracy", "train_acc", "val_acc"),
            (axes[0][2], "Macro AUC", "train_auc", "val_auc"),
            (axes[1][0], "Macro F1", "train_f1", "val_f1"),
            (axes[1][1], "Sensitivity (macro)", "train_sens", "val_sens"),
            (axes[1][2], "Specificity (macro)", "train_spec", "val_spec"),
        )
        for ax, title, tcol, vcol in pair_panels:
            ax.plot(
                epochs, col(tcol), color=PLOT_SERIES[0], lw=2, marker="o", ms=3.5, label="train"
            )
            ax.plot(epochs, col(vcol), color=PLOT_SERIES[1], lw=2, marker="o", ms=3.5, label="val")
            ax.set_title(title, color=PLOT_INK, fontsize=11)
            ax.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK_2)

        class_panels = (
            (axes[2][0], "Per-class AUC (val)", "val_auc_"),
            (axes[2][1], "Per-class F1 (val)", "val_f1_"),
            (axes[2][2], "Per-class AUC (train)", "train_auc_"),
        )
        for ax, title, prefix in class_panels:
            for i, c in enumerate(CLASS_NAMES):
                ax.plot(
                    epochs,
                    col(f"{prefix}{c}"),
                    color=PLOT_SERIES[i % len(PLOT_SERIES)],
                    lw=2,
                    marker="o",
                    ms=3.5,
                    label=c,
                )
            ax.set_title(title, color=PLOT_INK, fontsize=11)
            ax.legend(frameon=False, fontsize=8, labelcolor=PLOT_INK_2, ncols=2)

        for row in axes:
            for ax in row:
                _style_axis(ax)
                ax.set_xlabel("epoch", color=PLOT_INK_2, fontsize=9)

        fig.suptitle(
            f"{self.backbone} — training metrics", color=PLOT_INK, fontsize=15, fontweight="bold"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(self.plots_dir, "training_curves.png")
        fig.savefig(out, dpi=150, facecolor=PLOT_SURFACE)
        plt.close(fig)
        print(f"  Saved {out}")

    def _save_eval_plots(
        self, all_labels, all_preds, all_probs, cm, unique_labels, report, macro_auc
    ):
        plots = (
            ("confusion_matrix.png", lambda: self._plot_confusion_matrix(cm, unique_labels)),
            ("roc_curves.png", lambda: self._plot_roc_curves(all_labels, all_probs, macro_auc)),
            ("pr_curves.png", lambda: self._plot_pr_curves(all_labels, all_probs)),
            (
                "per_class_metrics.png",
                lambda: self._plot_per_class_bars(all_labels, all_preds, unique_labels),
            ),
        )
        for name, fn in plots:
            try:
                fn()
            except Exception as e:
                print(f"  Could not plot {name}: {e}")

        try:
            report_path = os.path.join(self.plots_dir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report + "\n")
                if macro_auc is not None:
                    f.write(f"\nMacro AUC (OvR): {macro_auc:.4f}\n")
            print(f"  Saved {report_path}")
        except Exception as e:
            print(f"  Could not save classification report: {e}")

    def _plot_confusion_matrix(self, cm, unique_labels):
        names = [CLASS_NAMES[i] for i in unique_labels]
        cm = np.asarray(cm)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cmap = LinearSegmentedColormap.from_list("seq_blue", PLOT_SEQ_RAMP)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PLOT_SURFACE)
        panels = (
            (axes[0], cm, "Counts", "d", max(int(cm.max()), 1)),
            (axes[1], cm_norm, "Row-normalized", ".2f", 1.0),
        )
        for ax, mat, title, fmt, vmax in panels:
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xticks(range(len(names)), labels=names)
            ax.set_yticks(range(len(names)), labels=names)
            ax.tick_params(colors=PLOT_MUTED, labelsize=9)
            ax.set_xlabel("Predicted", color=PLOT_INK_2, fontsize=10)
            ax.set_ylabel("True", color=PLOT_INK_2, fontsize=10)
            ax.set_title(title, color=PLOT_INK, fontsize=11)
            for spine in ax.spines.values():
                spine.set_color(PLOT_GRID)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    color = "#ffffff" if mat[i, j] > 0.55 * vmax else PLOT_INK
                    ax.text(
                        j,
                        i,
                        format(mat[i, j], fmt),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=color,
                    )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors=PLOT_MUTED, labelsize=8)
            cbar.outline.set_edgecolor(PLOT_GRID)

        fig.suptitle(f"{self.backbone} — validation confusion matrix", color=PLOT_INK, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = os.path.join(self.plots_dir, "confusion_matrix.png")
        fig.savefig(out, dpi=150, facecolor=PLOT_SURFACE)
        plt.close(fig)
        print(f"  Saved {out}")

    def _plot_roc_curves(self, all_labels, all_probs, macro_auc):
        probs = np.asarray(all_probs)
        y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        fig, ax = plt.subplots(figsize=(7, 6), facecolor=PLOT_SURFACE)
        for i in range(NUM_CLASSES):
            pos = int(y_bin[:, i].sum())
            if pos == 0 or pos == len(all_labels):
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            ax.plot(
                fpr,
                tpr,
                color=PLOT_SERIES[i % len(PLOT_SERIES)],
                lw=2,
                label=f"{CLASS_NAMES[i]} (AUC {auc(fpr, tpr):.3f})",
            )
        ax.plot([0, 1], [0, 1], color=PLOT_AXIS, lw=1, ls="--")
        _style_axis(ax)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("False positive rate", color=PLOT_INK_2, fontsize=10)
        ax.set_ylabel("True positive rate", color=PLOT_INK_2, fontsize=10)
        title = f"{self.backbone} — validation ROC (one-vs-rest)"
        if macro_auc is not None:
            title += f"\nmacro AUC {macro_auc:.3f}"
        ax.set_title(title, color=PLOT_INK, fontsize=11)
        ax.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK_2, loc="lower right")
        fig.tight_layout()
        out = os.path.join(self.plots_dir, "roc_curves.png")
        fig.savefig(out, dpi=150, facecolor=PLOT_SURFACE)
        plt.close(fig)
        print(f"  Saved {out}")

    def _plot_pr_curves(self, all_labels, all_probs):
        probs = np.asarray(all_probs)
        y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        fig, ax = plt.subplots(figsize=(7, 6), facecolor=PLOT_SURFACE)
        for i in range(NUM_CLASSES):
            pos = int(y_bin[:, i].sum())
            if pos == 0 or pos == len(all_labels):
                continue
            prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
            ap = average_precision_score(y_bin[:, i], probs[:, i])
            ax.plot(
                rec,
                prec,
                color=PLOT_SERIES[i % len(PLOT_SERIES)],
                lw=2,
                label=f"{CLASS_NAMES[i]} (AP {ap:.3f})",
            )
        _style_axis(ax)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Recall", color=PLOT_INK_2, fontsize=10)
        ax.set_ylabel("Precision", color=PLOT_INK_2, fontsize=10)
        ax.set_title(
            f"{self.backbone} — validation precision-recall (one-vs-rest)",
            color=PLOT_INK,
            fontsize=11,
        )
        ax.legend(frameon=False, fontsize=9, labelcolor=PLOT_INK_2, loc="lower left")
        fig.tight_layout()
        out = os.path.join(self.plots_dir, "pr_curves.png")
        fig.savefig(out, dpi=150, facecolor=PLOT_SURFACE)
        plt.close(fig)
        print(f"  Saved {out}")

    def _plot_per_class_bars(self, all_labels, all_preds, unique_labels):
        p, r, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=unique_labels, zero_division=0
        )
        names = [f"{CLASS_NAMES[i]}\n(n={s})" for i, s in zip(unique_labels, support)]
        x = np.arange(len(names))
        w = 0.26

        fig, ax = plt.subplots(figsize=(8, 5), facecolor=PLOT_SURFACE)
        for offset, vals, label, color in (
            (-w, p, "precision", PLOT_SERIES[0]),
            (0.0, r, "recall", PLOT_SERIES[1]),
            (w, f1, "F1", PLOT_SERIES[2]),
        ):
            ax.bar(x + offset, vals, width=w - 0.04, color=color, label=label)
        _style_axis(ax)
        ax.xaxis.grid(False)
        ax.set_xticks(x, labels=names)
        ax.set_ylim(0, 1.05)
        ax.legend(
            frameon=False,
            fontsize=9,
            labelcolor=PLOT_INK_2,
            ncols=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.suptitle(f"{self.backbone} — per-class validation metrics", color=PLOT_INK, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        out = os.path.join(self.plots_dir, "per_class_metrics.png")
        fig.savefig(out, dpi=150, facecolor=PLOT_SURFACE)
        plt.close(fig)
        print(f"  Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TI-RADS backbone models")
    parser.add_argument(
        "--backbone", type=str, default="all", help="convnext|efficientnet|swin|vit|all"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="artifacts/ckpts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="'auto' = continue each backbone from its <output_dir>/<backbone>/last.pt; "
        "or a checkpoint path (single-backbone runs only)",
    )
    args = parser.parse_args()

    backbones = (
        ["convnext", "efficientnet", "swin", "vit"] if args.backbone == "all" else [args.backbone]
    )

    if args.resume and args.resume != "auto" and len(backbones) > 1:
        parser.error("--resume <path> only works with a single --backbone; use --resume auto")

    for bb in backbones:
        trainer = Trainer(
            backbone=bb,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            resume=args.resume,
        )
        trainer.fit(epochs=args.epochs)
        print(f"\n{'='*60}\n")
