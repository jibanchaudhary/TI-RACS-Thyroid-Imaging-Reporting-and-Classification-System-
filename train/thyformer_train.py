"""
ThyFormer — Training Engine

Full training loop:
  • AdamW with differential LR (backbone vs head)
  • Cosine LR schedule + linear warmup
  • FP16 mixed-precision (torch.cuda.amp)
  • Gradient clipping
  • Early stopping on val_auc
  • Checkpoint manager (top-k by AUC)
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from configs.thyformer_config import ThyFormerConfig
from utils.thyformer_loss import ThyFormerLoss
from models.thyformer_models import ThyFormer
from utils.thyformer_metrics import compute_metrics
from utils.thyformer_logging import MetricLogger


def build_optimizer(model: ThyFormer, cfg: ThyFormerConfig) -> AdamW:
    groups = model.get_param_groups()
    return AdamW(
        [
            {"params": groups["backbone"], "lr": cfg.training.lr_backbone},
            {"params": groups["head"], "lr": cfg.training.lr_head},
        ],
        weight_decay=cfg.training.weight_decay,
        betas=cfg.training.betas,
    )


def build_scheduler(opt: AdamW, cfg: ThyFormerConfig, steps_per_epoch: int) -> CosineAnnealingLR:
    warmup = cfg.training.warmup_epochs * steps_per_epoch
    total = cfg.training.epochs * steps_per_epoch
    return CosineAnnealingLR(opt, T_max=total - warmup, eta_min=cfg.training.min_lr)


def apply_warmup(opt: AdamW, step: int, warmup_steps: int, lr_bb: float, lr_h: float):
    if step < warmup_steps:
        f = step / max(warmup_steps, 1)
        opt.param_groups[0]["lr"] = lr_bb * f
        opt.param_groups[1]["lr"] = lr_h * f


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = -1e9 if mode == "max" else 1e9
        self.counter = 0
        self.stop = False

    def step(self, val: float) -> bool:
        improved = val > self.best if self.mode == "max" else val < self.best
        if improved:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class CheckpointManager:
    def __init__(self, save_dir: str, top_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.saved: List[tuple] = []  # (score, path)

    def save(
        self, model: ThyFormer, opt: AdamW, epoch: int, metrics: Dict, cfg: ThyFormerConfig
    ) -> Path:
        score = metrics.get("val_auc", 0.0)
        path = self.save_dir / f"ep{epoch:03d}_auc{score:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "metrics": metrics,
                "config": cfg,
            },
            path,
        )
        self.saved.append((score, path))
        self.saved.sort(key=lambda x: x[0], reverse=True)
        while len(self.saved) > self.top_k:
            _, old = self.saved.pop()
            if old.exists():
                old.unlink()
        return path

    @property
    def best(self) -> Optional[Path]:
        return self.saved[0][1] if self.saved else None


def train_one_epoch(
    model, loader, opt, loss_fn, scaler, scheduler, epoch, cfg, global_step, logger
):
    model.train()
    device = next(model.parameters()).device
    warmup_steps = cfg.training.warmup_epochs * len(loader)
    total_loss = 0.0
    all_logits, all_labels = [], []

    for step, batch in enumerate(loader):
        apply_warmup(opt, global_step, warmup_steps, cfg.training.lr_backbone, cfg.training.lr_head)

        imgs = batch["image"].to(device, non_blocking=True)
        lbs = batch["label"].to(device, non_blocking=True)
        msks = batch["mask"].to(device, non_blocking=True)
        bnds = batch["boundary"].to(device, non_blocking=True)

        opt.zero_grad()
        with autocast(device_type="cuda", enabled=cfg.training.fp16):
            preds = model(imgs)
            losses = loss_fn(preds, {"label": lbs, "mask": msks, "boundary": bnds}, epoch=epoch)
            loss = losses["loss_total"]

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
        scaler.step(opt)
        scaler.update()

        if global_step >= warmup_steps:
            scheduler.step()

        total_loss += loss.item()
        all_logits.append(preds["cls_logits"].detach().cpu())
        hard = lbs.argmax(1) if lbs.dim() == 2 else lbs
        all_labels.append(hard.cpu())

        if step % cfg.training.log_interval == 0:
            logger.log_step(epoch, step, len(loader), losses, opt.param_groups[0]["lr"])
        global_step += 1

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    m = compute_metrics(logits, labels, prefix="train")
    m["train_loss"] = total_loss / len(loader)
    return m, global_step


@torch.no_grad()
def evaluate(model, loader, loss_fn, epoch, prefix="val", cfg=None):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        lbs = batch["label"].to(device, non_blocking=True)
        msks = batch["mask"].to(device, non_blocking=True)
        bnds = batch["boundary"].to(device, non_blocking=True)
        fp16 = cfg.training.fp16 if cfg else False
        with autocast(device_type="cuda", enabled=fp16):
            preds = model(imgs)
            losses = loss_fn(preds, {"label": lbs, "mask": msks, "boundary": bnds}, epoch=epoch)
        total_loss += losses["loss_total"].item()
        all_logits.append(preds["cls_logits"].cpu())
        hard = lbs.argmax(1) if lbs.dim() == 2 else lbs
        all_labels.append(hard.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    m = compute_metrics(logits, labels, prefix=prefix)
    m[f"{prefix}_loss"] = total_loss / len(loader)
    return m


def train(
    model: ThyFormer,
    dataloaders: Dict[str, DataLoader],
    loss_fn: ThyFormerLoss,
    cfg: ThyFormerConfig,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, len(dataloaders["train"]))
    scaler = GradScaler(enabled=cfg.training.fp16)
    stopper = EarlyStopping(cfg.training.early_stopping_patience, cfg.training.early_stopping_mode)
    ckpt_mgr = CheckpointManager(cfg.training.checkpoint_dir, cfg.training.save_top_k)
    logger = MetricLogger(
        cfg.training.log_dir,
        cfg.training.use_wandb,
        cfg.training.project_name,
        cfg.training.experiment_name,
    )
    best_metrics = {}
    global_step = 0

    print(f"\nTraining ThyFormer on {device}")
    print(
        f"Epochs {cfg.training.epochs}  |  "
        f"Batch {cfg.training.batch_size}  |  "
        f"FP16 {cfg.training.fp16}"
    )
    print("─" * 60)

    for epoch in range(cfg.training.epochs):
        t0 = time.time()

        train_m, global_step = train_one_epoch(
            model,
            dataloaders["train"],
            opt,
            loss_fn,
            scaler,
            sched,
            epoch,
            cfg,
            global_step,
            logger,
        )

        val_m = evaluate(model, dataloaders["val"], loss_fn, epoch, "val", cfg)

        logger.log_epoch({**train_m, **val_m, "epoch": epoch})
        _print_epoch(epoch, time.time() - t0, train_m, val_m)

        ckpt_mgr.save(model, opt, epoch, val_m, cfg)

        watch = val_m.get(cfg.training.early_stopping_metric, 0.0)
        if watch == stopper.best:
            best_metrics = val_m.copy()
        if stopper.step(watch):
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best {cfg.training.early_stopping_metric}: {stopper.best:.4f}"
            )
            break

    # ── Final test ───────────────────────────────────────────────
    print("\nFinal test evaluation …")
    if ckpt_mgr.best and ckpt_mgr.best.exists():
        ckpt = torch.load(ckpt_mgr.best, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint: {ckpt_mgr.best}")

    test_m = evaluate(model, dataloaders["test"], loss_fn, 0, "test", cfg)
    logger.log_epoch(test_m)
    _print_test(test_m)
    logger.close()
    return {**best_metrics, **test_m}


def _print_epoch(epoch, elapsed, tm, vm):
    print(
        f"[{epoch:3d}] {elapsed:5.1f}s | "
        f"train loss={tm.get('train_loss',0):.4f} "
        f"AUC={tm.get('train_auc',0):.4f} | "
        f"val loss={vm.get('val_loss',0):.4f} "
        f"AUC={vm.get('val_auc',0):.4f} "
        f"F1={vm.get('val_f1',0):.4f}"
    )


def _print_test(m):
    print("\n" + "=" * 55)
    print("  TEST RESULTS")
    print("=" * 55)
    for k, v in sorted(m.items()):
        if isinstance(v, float):
            print(f"  {k:<35s} {v:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ThyFormer")
    parser.add_argument("--data_root", default=None, help="Path to image root dir")
    parser.add_argument("--train_csv", default=None, help="Path to train.csv")
    parser.add_argument("--val_csv", default=None, help="Path to val.csv")
    parser.add_argument("--test_csv", default=None, help="Path to test.csv")
    parser.add_argument("--medsam_dir", default=None, help="Path to MedSAM boundary .npy dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr_backbone", type=float, default=None)
    parser.add_argument("--lr_head", type=float, default=None)
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--experiment", default=None, help="Experiment name")
    parser.add_argument("--checkpoint_dir", default=None)
    args = parser.parse_args()

    # Load config and apply CLI overrides
    cfg = ThyFormerConfig()

    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.train_csv:
        cfg.data.train_csv = args.train_csv
    if args.val_csv:
        cfg.data.val_csv = args.val_csv
    if args.test_csv:
        cfg.data.test_csv = args.test_csv
    if args.medsam_dir:
        cfg.data.medsam_masks_dir = args.medsam_dir
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr_backbone:
        cfg.training.lr_backbone = args.lr_backbone
    if args.lr_head:
        cfg.training.lr_head = args.lr_head
    if args.no_fp16:
        cfg.training.fp16 = False
    if args.wandb:
        cfg.training.use_wandb = True
    if args.experiment:
        cfg.training.experiment_name = args.experiment
    if args.checkpoint_dir:
        cfg.training.checkpoint_dir = args.checkpoint_dir

    # Import here to avoid circular imports at module level
    from data_pipeline.thyformer_create_dataset import build_dataloaders
    from models.thyformer_models import build_model
    from utils.thyformer_loss import build_loss

    # Build everything
    model = build_model(cfg.model)
    loss_fn = build_loss(cfg.loss)
    loaders = build_dataloaders(
        cfg.data, cfg.augmentation, cfg.training.batch_size, cfg.training.num_workers
    )

    # Train
    results = train(model, loaders, loss_fn, cfg)
    print("\nDone. Final results:", results)
