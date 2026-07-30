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


def amp_settings(cfg: ThyFormerConfig):
    """Returns (enabled, dtype) for autocast based on config.

    bf16 is range-safe (no loss scaling); fp16 needs a GradScaler.
    """
    enabled = cfg.training.fp16
    dtype = (
        torch.bfloat16 if getattr(cfg.training, "amp_dtype", "fp16") == "bf16" else torch.float16
    )
    return enabled, dtype


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


def accum_steps(cfg: ThyFormerConfig) -> int:
    """Micro-batches per optimiser step (>=1). batch_size * accum_steps is the
    effective batch the optimiser actually sees."""
    return max(int(getattr(cfg.training, "accum_steps", 1)), 1)


def opt_steps_per_epoch(cfg: ThyFormerConfig, n_batches: int) -> int:
    """Optimiser steps per epoch under gradient accumulation. The LR schedule and
    warmup are counted in OPTIMISER steps, not micro-batches — counting
    micro-batches would run the warmup accum_steps times too fast."""
    return max(n_batches // accum_steps(cfg), 1)


def build_scheduler(opt: AdamW, cfg: ThyFormerConfig, steps_per_epoch: int) -> CosineAnnealingLR:
    """steps_per_epoch must be OPTIMISER steps per epoch (see opt_steps_per_epoch)."""
    warmup = cfg.training.warmup_epochs * steps_per_epoch
    total = cfg.training.epochs * steps_per_epoch
    return CosineAnnealingLR(opt, T_max=max(total - warmup, 1), eta_min=cfg.training.min_lr)


def set_backbone_frozen(model: ThyFormer, frozen: bool) -> int:
    """
    Freeze/unfreeze the pretrained Swin stages, returning the number of
    parameters toggled. Matches get_param_groups' own convention of identifying
    backbone parameters by the substring "swin" in their name.

    Called after the optimiser is built, so frozen parameters stay in their param
    group; their .grad simply stays None and AdamW skips them (including weight
    decay). That lets them resume cleanly on unfreeze.
    """
    n = 0
    for name, p in model.named_parameters():
        if "swin" in name:
            p.requires_grad = not frozen
            n += p.numel()
    return n


def apply_warmup(opt: AdamW, step: int, warmup_steps: int, lr_bb: float, lr_h: float):
    if step < warmup_steps:
        f = step / max(warmup_steps, 1)
        opt.param_groups[0]["lr"] = lr_bb * f
        opt.param_groups[1]["lr"] = lr_h * f


class EarlyStopping:
    """
    Tracks the best value of the watched metric and stops after `patience`
    non-improving epochs.

    `step` exposes whether this epoch improved via `self.improved`, so callers no
    longer have to compare against `self.best` themselves. The previous code did
    `if watch == stopper.best` *before* calling `step`, which compares against
    the PREVIOUS best and therefore recorded the wrong epoch's metrics as best.
    """

    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = -1e9 if mode == "max" else 1e9
        self.counter = 0
        self.stop = False
        self.improved = False

    def is_better(self, val: float) -> bool:
        return val > self.best if self.mode == "max" else val < self.best

    def step(self, val: float) -> bool:
        self.improved = self.is_better(val)
        if self.improved:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class MetricSmoother:
    """
    Trailing mean of the selection metric.

    With only 29 validation clips a single epoch's macro AUC carries a standard
    error of roughly +/-0.10, so ranking checkpoints on the raw per-epoch value
    reliably selects the luckiest epoch rather than the best model — ep005's
    val_auc 0.8223 sat between 0.745 and 0.689 and tested at 0.5798. Averaging
    over a short window suppresses that spike.

    window <= 1 disables smoothing (returns the raw value).
    """

    def __init__(self, window: int = 1):
        self.window = max(int(window), 1)
        self.history: List[float] = []

    def __call__(self, val: float) -> float:
        if not (val == val) or val in (float("inf"), float("-inf")):
            # Non-finite epochs carry no information; don't poison the window.
            return val
        self.history.append(float(val))
        w = self.history[-self.window :]
        return sum(w) / len(w)


class CheckpointManager:
    """
    Keeps the top-k checkpoints ranked by the SAME metric early stopping watches.

    Previously this ranked by `val_auc` while EarlyStopping watched `val_loss`, so
    `best` was the peak of a noisy 29-clip AUC and disagreed with the stopping
    criterion. `metric`/`mode` now come from cfg.training.early_stopping_metric /
    _mode, and the score written into each filename is the (optionally smoothed)
    selection score, not a raw AUC.
    """

    def __init__(
        self,
        save_dir: str,
        top_k: int = 3,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.metric = metric
        self.mode = mode
        self.saved: List[tuple] = []  # (score, path), best first
        self._scan_existing()

    def _sort(self):
        # Best first: descending for "max", ascending for "min".
        self.saved.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

    def _scan_existing(self):
        """Rebuild saved list from any .pt files already on disk (needed for resume).
        Accepts the current `ep000_score1.2345.pt` scheme and the legacy
        `ep000_auc0.8223.pt` one so old runs remain resumable."""
        for token in ("_score", "_auc"):
            for p in sorted(self.save_dir.glob(f"ep*{token}*.pt")):
                if any(p == q for _, q in self.saved):
                    continue
                try:
                    self.saved.append((float(p.stem.split(token)[-1]), p))
                except ValueError:
                    pass
        self._sort()

    def save(
        self,
        model: ThyFormer,
        opt: AdamW,
        epoch: int,
        metrics: Dict,
        cfg: ThyFormerConfig,
        scheduler=None,
        scaler=None,
        score: Optional[float] = None,
    ) -> Path:
        """`score` is the selection score to rank on (smoothed, when smoothing is
        enabled). Falls back to the raw configured metric."""
        if score is None:
            score = metrics.get(self.metric, 0.0)
        path = self.save_dir / f"ep{epoch:03d}_score{score:.4f}.pt"
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "metrics": metrics,
            "config": cfg,
            # Provenance so a checkpoint can never be misread as "best AUC" again.
            "selection_metric": self.metric,
            "selection_mode": self.mode,
            "selection_score": float(score),
        }
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        torch.save(payload, path)
        self.saved.append((float(score), path))
        self._sort()
        while len(self.saved) > self.top_k:
            _, old = self.saved.pop()  # worst, since _sort puts best first
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
    amp_enabled, amp_dtype = amp_settings(cfg)
    accum = accum_steps(cfg)
    # Warmup and the cosine schedule advance per OPTIMISER step, so their horizons
    # are expressed in optimiser steps too. `global_step` is an optimiser step
    # counter (it used to count micro-batches).
    warmup_steps = cfg.training.warmup_epochs * opt_steps_per_epoch(cfg, len(loader))
    total_loss = 0.0
    n_batches = 0
    n_nonfinite = 0
    all_logits, all_labels = [], []

    def optimizer_step(grad_rescale: float = 1.0):
        """One optimiser step on the accumulated gradients.

        `grad_rescale` corrects a short final group: each micro-batch contributed
        loss/accum, so a group of m < accum micro-batches is under-weighted by
        accum/m. Rescaling after unscale_ makes the partial group equivalent to a
        full one instead of silently taking a smaller step.
        """
        nonlocal global_step
        apply_warmup(opt, global_step, warmup_steps, cfg.training.lr_backbone, cfg.training.lr_head)
        scaler.unscale_(opt)
        if grad_rescale != 1.0:
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.mul_(grad_rescale)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if global_step >= warmup_steps:
            scheduler.step()
        global_step += 1

    opt.zero_grad(set_to_none=True)
    micro = 0  # micro-batches accumulated into the current optimiser step

    for step, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True)
        lbs = batch["label"].to(device, non_blocking=True)
        msks = batch["mask"].to(device, non_blocking=True)
        bnds = batch["boundary"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            preds = model(imgs)
            losses = loss_fn(preds, {"label": lbs, "mask": msks, "boundary": bnds}, epoch=epoch)
            loss = losses["loss_total"]

        # Skip non-finite batches so one bad step can't corrupt the weights. A
        # skipped micro-batch contributes no gradient, so it must not count
        # toward the accumulation group either.
        if not torch.isfinite(loss):
            n_nonfinite += 1
            continue

        # Scale by 1/accum so the accumulated gradient is the MEAN over the
        # effective batch, matching what batch_size=accum*batch_size would give.
        scaler.scale(loss / accum).backward()
        micro += 1

        total_loss += loss.item()
        n_batches += 1
        all_logits.append(preds["cls_logits"].detach().float().cpu())
        hard = lbs.argmax(1) if lbs.dim() == 2 else lbs
        all_labels.append(hard.cpu())

        if micro == accum:
            optimizer_step()
            micro = 0

        if step % cfg.training.log_interval == 0:
            logger.log_step(epoch, step, len(loader), losses, opt.param_groups[0]["lr"])

    # Flush a trailing partial group (len(loader) need not divide by accum).
    if micro > 0:
        optimizer_step(grad_rescale=accum / micro)

    if n_nonfinite:
        print(f"  ⚠ skipped {n_nonfinite} non-finite loss batch(es) this epoch")

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    m = compute_metrics(logits, labels, prefix="train")
    m["train_loss"] = total_loss / max(n_batches, 1)
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
        amp_enabled, amp_dtype = amp_settings(cfg) if cfg else (False, torch.float16)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            preds = model(imgs)
            losses = loss_fn(preds, {"label": lbs, "mask": msks, "boundary": bnds}, epoch=epoch)
        total_loss += losses["loss_total"].item()
        all_logits.append(preds["cls_logits"].float().cpu())
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
    resume_from: Optional[str] = None,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = build_optimizer(model, cfg)
    accum = accum_steps(cfg)
    steps_per_epoch = opt_steps_per_epoch(cfg, len(dataloaders["train"]))
    sched = build_scheduler(opt, cfg, steps_per_epoch)
    # GradScaler is only needed for fp16 (bf16 has fp32 range; scaling is a no-op/harmful).
    _, amp_dtype = amp_settings(cfg)
    scaler = GradScaler(enabled=cfg.training.fp16 and amp_dtype == torch.float16)
    sel_metric = cfg.training.early_stopping_metric
    sel_mode = cfg.training.early_stopping_mode
    stopper = EarlyStopping(cfg.training.early_stopping_patience, sel_mode)
    smoother = MetricSmoother(getattr(cfg.training, "selection_smoothing_epochs", 1))
    # Checkpoint ranking and early stopping now share one metric and one mode.
    ckpt_mgr = CheckpointManager(
        cfg.training.checkpoint_dir, cfg.training.save_top_k, sel_metric, sel_mode
    )
    freeze_epochs = int(getattr(cfg.training, "freeze_backbone_epochs", 0))
    backbone_frozen = False
    logger = MetricLogger(
        cfg.training.log_dir,
        cfg.training.use_wandb,
        cfg.training.project_name,
        cfg.training.experiment_name,
        resume=bool(resume_from),
    )
    best_metrics = {}
    global_step = 0
    start_epoch = 0

    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        # global_step counts OPTIMISER steps (see train_one_epoch).
        global_step = start_epoch * steps_per_epoch
        prev_metric = ckpt.get("selection_score", ckpt["metrics"].get(sel_metric, stopper.best))
        stopper.best = prev_metric
        print(
            f"Resumed from {resume_path.name}  (epoch {ckpt['epoch']} → continuing at {start_epoch})"
        )

    print(f"\nTraining ThyFormer on {device}")
    print(
        f"Epochs {cfg.training.epochs}  |  "
        f"Batch {cfg.training.batch_size} x {accum} accum "
        f"= effective {cfg.training.batch_size * accum}  |  "
        f"FP16 {cfg.training.fp16} ({getattr(cfg.training, 'amp_dtype', 'fp16')})"
    )
    print(
        f"Selection: {sel_metric} ({sel_mode}), "
        f"smoothed over {smoother.window} epoch(s)  |  "
        f"backbone frozen for first {freeze_epochs} epoch(s)"
    )
    print("─" * 60)

    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        # Freeze the pretrained Swin stages while the random head/stem settles,
        # then release them — matching the baseline's freeze_epochs=3.
        want_frozen = epoch < freeze_epochs
        if want_frozen != backbone_frozen:
            n = set_backbone_frozen(model, want_frozen)
            print(
                f"  {'❄ froze' if want_frozen else '🔥 unfroze'} Swin backbone "
                f"({n / 1e6:.1f}M params) at epoch {epoch}"
            )
            backbone_frozen = want_frozen

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

        # One selection signal for both checkpoint ranking and early stopping:
        # the configured metric, smoothed over a short window.
        raw = val_m.get(sel_metric, 0.0)
        watch = smoother(raw)
        val_m[f"{sel_metric}_smoothed"] = watch
        if smoother.window > 1:
            print(f"      selection {sel_metric}: raw {raw:.4f} → smoothed {watch:.4f}")

        ckpt_mgr.save(model, opt, epoch, val_m, cfg, scheduler=sched, scaler=scaler, score=watch)

        # `step` sets `improved` internally, so best_metrics is recorded for the
        # epoch that actually improved. The old `watch == stopper.best` test ran
        # BEFORE step() and so compared against the previous best.
        stop = stopper.step(watch)
        if stopper.improved:
            best_metrics = val_m.copy()
        if stop:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best {sel_metric} ({sel_mode}, smoothed): {stopper.best:.4f}"
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
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pt file to resume from")
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
    results = train(model, loaders, loss_fn, cfg, resume_from=args.resume)
    print("\nDone. Final results:", results)
