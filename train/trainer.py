"""
trainer.py
----------
Training engine for individual BackboneModels.

Features
--------
  - Label-smoothed cross-entropy loss
  - Cosine annealing LR scheduler with linear warm-up
  - WeightedRandomSampler to handle class imbalance
  - Gradient clipping
  - Early stopping on val loss
  - Per-epoch checkpoint saving (best + last)
  - CSV metrics log
  - Grad-CAM hook registration (used by inference.py)
  - Supports mixed precision (torch.amp) when CUDA available

Usage
-----
    from trainer import Trainer
    trainer = Trainer(backbone="convnext", data_dir="data", output_dir="runs")
    trainer.fit(epochs=30)
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from data_pipeline.create_dataset import ThyroidDataset, CLASS_NAMES, NUM_CLASSES
from models.models import build_model
import argparse


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
    """

    def __init__(
        self,
        backbone: str = "convnext",
        data_dir: str = "data",
        output_dir: str = "runs",
        batch_size: int = 16,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        freeze_epochs: int = 5,
        patience: int = 10,
        label_smoothing: float = 0.1,
        use_amp: bool = True,
        num_workers: int = 4,
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
            sampler=sampler,  # handles class imbalance
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

        # CSV log
        self.log_path = os.path.join(self.output_dir, "metrics.csv")
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"]
            )

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        self.model.on_epoch_start(epoch)

        total_loss, correct, total = 0.0, 0, 0

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

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch:3d} | step {batch_idx+1:4d}" f" | loss {loss.item():.4f}")

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []

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
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        n = len(all_labels)
        val_loss = total_loss / n
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return val_loss, val_acc, val_f1

    def fit(self, epochs: int = 30) -> str:
        """
        Train for `epochs` epochs with early stopping.
        Returns path to the best checkpoint.
        """
        scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=epochs,
        )

        best_val_loss = float("inf")
        patience_count = 0
        best_ckpt_path = os.path.join(self.output_dir, "best.pt")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc, val_f1 = self._val_epoch()
            scheduler.step()

            current_lr = self.optimizer.param_groups[-1]["lr"]
            elapsed = time.time() - t0

            print(
                f"\nEpoch {epoch:3d}/{epochs} | {elapsed:.1f}s | "
                f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
                f"val   loss {val_loss:.4f}  acc {val_acc:.4f}  "
                f"f1 {val_f1:.4f} | lr {current_lr:.2e}"
            )

            # Log metrics
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        f"{train_loss:.5f}",
                        f"{train_acc:.5f}",
                        f"{val_loss:.5f}",
                        f"{val_acc:.5f}",
                        f"{val_f1:.5f}",
                        f"{current_lr:.2e}",
                    ]
                )

            # Save last checkpoint always
            self._save_checkpoint(
                epoch, val_loss, val_acc, os.path.join(self.output_dir, "last.pt")
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                self._save_checkpoint(epoch, val_loss, val_acc, best_ckpt_path)
                print(f"  ✓ New best val loss: {best_val_loss:.4f}")
            else:
                patience_count += 1
                print(f"  No improvement ({patience_count}/{self.patience})")
                if patience_count >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        print(f"\nTraining complete. Best checkpoint: {best_ckpt_path}")
        self._final_eval(best_ckpt_path)
        return best_ckpt_path

    def _save_checkpoint(self, epoch, val_loss, val_acc, path):
        torch.save(
            {
                "epoch": epoch,
                "backbone": self.backbone,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            path,
        )

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

        print(
            classification_report(
                all_labels,
                all_preds,
                labels=unique_labels,
                target_names=[CLASS_NAMES[i] for i in unique_labels],
                zero_division=0,
            )
        )

        # AUC (macro OvR)
        y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
        try:
            auc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")
            print(f"Macro AUC: {auc:.4f}")
        except ValueError:
            print("AUC: not computable (too few classes in val set)")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion matrix:")
        print(cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TI-RADS backbone models")
    parser.add_argument(
        "--backbone", type=str, default="all", help="convnext|efficientnet|swin|vit|all"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="artifacts/ckpts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    backbones = (
        ["convnext", "efficientnet", "swin", "vit"] if args.backbone == "all" else [args.backbone]
    )

    for bb in backbones:
        trainer = Trainer(
            backbone=bb,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        trainer.fit(epochs=args.epochs)
        print(f"\n{'='*60}\n")
