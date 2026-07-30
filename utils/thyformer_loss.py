"""
ThyFormer — Composite Loss  ★ NOVEL (Stage 5)

L_total = α · L_ce  +  β · L_dice  +  γ(t) · L_boundary

γ(t) is linearly annealed 0 → γ over boundary_warmup_epochs,
allowing stable early convergence before boundary precision is enforced.

The boundary loss forces the model to attend to nodule margins —
the critical region for T2↔T3 confusion in TI-RADS scoring.
"""
import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.thyformer_config import LossConfig


# ─────────────────────────────────────────────────────────────────
# L_ce — Soft cross-entropy (supports Mixup one-hot targets)
# ─────────────────────────────────────────────────────────────────


class SoftCrossEntropyLoss(nn.Module):
    """
    Cross-entropy accepting both hard integer and soft (Mixup) one-hot labels.
    Optional class weighting + label smoothing.
    """

    def __init__(
        self,
        num_classes: int = 5,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else torch.ones(num_classes)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure one-hot
        if targets.dim() == 1:
            targets = F.one_hot(targets, self.num_classes).float()

        # Label smoothing
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / self.num_classes
            targets = targets * (1 - self.label_smoothing) + smooth

        # log_softmax in fp32: under bf16 autocast the logits carry only ~2-3
        # decimal digits, which shows up directly in the reported loss.
        log_p = F.log_softmax(logits.float(), dim=-1)
        w = self.class_weights.to(log_p.device, dtype=log_p.dtype)
        loss = -(targets.to(log_p.dtype) * log_p * w.unsqueeze(0)).sum(dim=-1).mean()
        return loss


# ─────────────────────────────────────────────────────────────────
# L_dice — Dice loss for binary segmentation
# ─────────────────────────────────────────────────────────────────


class DiceLoss(nn.Module):
    """
    Dice = 1 - (2·|X∩Y| + ε) / (|X| + |Y| + ε)
    Operates on sigmoid-activated logits.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # fp32 throughout: under bf16 autocast the reductions below run over
        # ~500k pixels per sample, where bf16's 8-bit mantissa loses accuracy and
        # near-1.0 values are not resolvable. Cost is negligible (one cast).
        probs = torch.sigmoid(logits.float())
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.float().view(targets.size(0), -1).clamp(0.0, 1.0)
        inter = (p_flat * t_flat).sum(dim=1)
        # clamp_min keeps the ratio finite even if both sums underflow to 0 (an
        # empty target mask paired with an all-zero prediction).
        denom = (p_flat.sum(dim=1) + t_flat.sum(dim=1)).clamp_min(0.0)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


# ─────────────────────────────────────────────────────────────────
# L_boundary — MedSAM-guided boundary loss  ★ NOVEL
# ─────────────────────────────────────────────────────────────────


class MedSAMBoundaryLoss(nn.Module):
    """
    Penalises the segmentation head when its predicted boundary
    deviates from the MedSAM-precomputed boundary map.

    Boundary of the predicted mask is extracted morphologically
    (dilation − erosion), then compared to the MedSAM boundary via
    weighted BCE — boundary pixels receive higher weight.

    This forces the classification head to attend to nodule edges,
    directly addressing T2↔T3 margin-based confusion in TI-RADS.
    """

    def __init__(self, boundary_weight: float = 5.0, kernel_size: int = 3):
        super().__init__()
        self.bw = boundary_weight
        self.ksize = kernel_size

    def _boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary: [B,1,H,W] float → [B,1,H,W] boundary map."""
        pad = self.ksize // 2
        dilated = F.max_pool2d(mask, self.ksize, 1, pad)
        eroded = -F.max_pool2d(-mask, self.ksize, 1, pad)
        return (dilated - eroded).clamp(0, 1)

    def forward(self, seg_logits: torch.Tensor, medsam_boundary: torch.Tensor) -> torch.Tensor:
        # ── fp32 is REQUIRED here, not an optimisation ────────────────────────
        # This term was the source of `val_loss = inf` at epochs 6/8/9. Under
        # bf16 autocast the Python float `1 - 1e-6` rounds to *exactly 1.0*
        # (bf16 has 8 mantissa bits, so the spacing near 1.0 is 2^-8), which
        # makes the upper clamp below a no-op. `pred_bd` reaches exactly 1.0 as
        # soon as sigmoid(seg_logits) saturates — which is what happens once the
        # seg head grows confident, hence the failure appearing only at later
        # epochs — and then log(1 - 1.0) = -inf poisons the whole composite loss.
        # fp16 has the same defect; fp32 resolves 1-1e-6 correctly.
        pred_mask = torch.sigmoid(seg_logits.float())  # [B,1,H,W]
        pred_bd = self._boundary(pred_mask)
        target_bd = medsam_boundary.float().to(pred_bd.device).clamp(0.0, 1.0)
        weight = 1.0 + (self.bw - 1.0) * target_bd
        eps = 1e-6
        pred_bd = pred_bd.clamp(eps, 1.0 - eps)
        # log1p(-p) is the accurate form of log(1-p) as p -> 0.
        bce = -(target_bd * torch.log(pred_bd) + (1.0 - target_bd) * torch.log1p(-pred_bd))
        return (weight * bce).mean()


# ─────────────────────────────────────────────────────────────────
# Composite loss  ★ NOVEL
# ─────────────────────────────────────────────────────────────────


class ThyFormerLoss(nn.Module):
    """
    L_total = α·L_ce  +  β·L_dice  +  γ(t)·L_boundary

    Args:
        cfg:           LossConfig
        num_classes:   5 (T1–T5)
        class_weights: [5] tensor for imbalance correction
    """

    # Class-level so the empty-boundary warning fires once per process, not once
    # per batch (it would otherwise print thousands of times per epoch).
    _warned_empty_boundary = False

    def __init__(
        self, cfg: LossConfig, num_classes: int = 5, class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.gamma_max = cfg.gamma
        self.warmup = cfg.boundary_warmup_epochs
        self.ce = SoftCrossEntropyLoss(num_classes, cfg.label_smoothing, class_weights)
        self.dice = DiceLoss()
        self.bnd = MedSAMBoundaryLoss()

    def _gamma(self, epoch: int) -> float:
        """Linear warmup: 0 at epoch 0 → gamma_max at warmup."""
        if self.warmup <= 0:
            return self.gamma_max
        return self.gamma_max * min(epoch / self.warmup, 1.0)

    def forward(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        preds:
            cls_logits  [B,4]
            seg_logits  [B,1,H,W]
        targets:
            label       [B,4] soft | [B] hard
            mask        [B,1,H,W]
            boundary    [B,1,H,W]
        """
        l_ce = self.ce(preds["cls_logits"], targets["label"])
        l_dice = self.dice(preds["seg_logits"], targets["mask"])
        gamma = self._gamma(epoch)
        l_bnd = self.bnd(preds["seg_logits"], targets["boundary"])

        # ── Guard: an all-zero boundary target supervises nothing ─────────────
        # Every file in stanford_dataset/med_sam is identically zero, so this
        # term degenerates to "drive the predicted boundary to 0 everywhere",
        # i.e. predict a spatially uniform mask — which directly opposes L_dice.
        # Rather than train against a vacuous target, drop the term and say so
        # once. Regenerate the MedSAM maps (data_pipeline/thyformer_medsam.py) or
        # set LossConfig.gamma = 0.0 to silence this deliberately.
        if gamma > 0.0 and float(targets["boundary"].abs().max()) == 0.0:
            if not ThyFormerLoss._warned_empty_boundary:
                ThyFormerLoss._warned_empty_boundary = True
                warnings.warn(
                    "MedSAM boundary target is all zeros — L_boundary is supervising "
                    "against an empty map and would push the seg head toward a "
                    "boundary-free mask. Dropping the boundary term for this run. "
                    "Regenerate stanford_dataset/med_sam or set LossConfig.gamma=0.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            gamma = 0.0

        l_tot = self.alpha * l_ce + self.beta * l_dice + gamma * l_bnd

        return {
            "loss_total": l_tot,
            "loss_ce": l_ce,
            "loss_dice": l_dice,
            "loss_boundary": l_bnd,
            "boundary_weight": torch.tensor(gamma),
        }


def build_loss(
    cfg: LossConfig, num_classes: int = 5, class_weights: Optional[torch.Tensor] = None
) -> ThyFormerLoss:
    return ThyFormerLoss(cfg, num_classes, class_weights)
