"""
ThyFormer — Composite Loss  ★ NOVEL (Stage 5)

L_total = α · L_ce  +  β · L_dice  +  γ(t) · L_boundary

γ(t) is linearly annealed 0 → γ over boundary_warmup_epochs,
allowing stable early convergence before boundary precision is enforced.

The boundary loss forces the model to attend to nodule margins —
the critical region for T2↔T3 confusion in TI-RADS scoring.
"""
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
        num_classes: int = 4,
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

        log_p = F.log_softmax(logits, dim=-1)
        w = self.class_weights.to(logits.device)
        loss = -(targets * log_p * w.unsqueeze(0)).sum(dim=-1).mean()
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
        probs = torch.sigmoid(logits)
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.float().view(targets.size(0), -1)
        inter = (p_flat * t_flat).sum(dim=1)
        denom = p_flat.sum(dim=1) + t_flat.sum(dim=1)
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
        pred_mask = torch.sigmoid(seg_logits)  # [B,1,H,W]
        pred_bd = self._boundary(pred_mask)
        target_bd = medsam_boundary.float().to(seg_logits.device)
        weight = 1.0 + (self.bw - 1.0) * target_bd
        pred_bd = pred_bd.clamp(1e-6, 1 - 1e-6)
        bce = -(target_bd * torch.log(pred_bd) + (1 - target_bd) * torch.log(1 - pred_bd))
        return (weight * bce).mean()


# ─────────────────────────────────────────────────────────────────
# Composite loss  ★ NOVEL
# ─────────────────────────────────────────────────────────────────


class ThyFormerLoss(nn.Module):
    """
    L_total = α·L_ce  +  β·L_dice  +  γ(t)·L_boundary

    Args:
        cfg:           LossConfig
        num_classes:   4 (T1–T4)
        class_weights: [4] tensor for imbalance correction
    """

    def __init__(
        self, cfg: LossConfig, num_classes: int = 4, class_weights: Optional[torch.Tensor] = None
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
            seg_logits  [B,1,224,224]
        targets:
            label       [B,4] soft | [B] hard
            mask        [B,1,224,224]
            boundary    [B,1,224,224]
        """
        l_ce = self.ce(preds["cls_logits"], targets["label"])
        l_dice = self.dice(preds["seg_logits"], targets["mask"])
        gamma = self._gamma(epoch)
        l_bnd = self.bnd(preds["seg_logits"], targets["boundary"])
        l_tot = self.alpha * l_ce + self.beta * l_dice + gamma * l_bnd

        return {
            "loss_total": l_tot,
            "loss_ce": l_ce,
            "loss_dice": l_dice,
            "loss_boundary": l_bnd,
            "boundary_weight": torch.tensor(gamma),
        }


def build_loss(
    cfg: LossConfig, num_classes: int = 4, class_weights: Optional[torch.Tensor] = None
) -> ThyFormerLoss:
    return ThyFormerLoss(cfg, num_classes, class_weights)
