"""
ThyFormer — Full Model Architecture

Stage 1 : DespecklingCNNStem         ★ Novel
Stage 2 : EchogenicityChannelAttn    ★ Novel
Stage 3 : Swin Transformer encoder   (timm backbone)
Stage 4a: ClassificationHead
Stage 4b: FPNDecoder (segmentation)
"""
from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.thyformer_config import ModelConfig


class DespecklingCNNStem(nn.Module):
    """
    Pre-tokenisation stem for US speckle suppression.

        DWConv3×3 (speckle-aware) → GroupNorm → GELU
        → PatchEmbed4×4 → LayerNorm
        → token sequence [B, H*W, C]

    DW weights initialised to Gaussian kernel (soft denoising prior).
    LayerNorm replaces BatchNorm — no batch-size dependency.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 96, kernel: int = 3, patch: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, padding=kernel // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.norm = nn.GroupNorm(num_groups=in_ch, num_channels=in_ch)
        self.act = nn.GELU()
        self.pe = nn.Conv2d(in_ch, out_ch, patch, stride=patch, bias=False)
        self.ln = nn.LayerNorm(out_ch)
        self._init_gaussian(kernel)

    def _init_gaussian(self, k: int):
        with torch.no_grad():
            c = torch.arange(k, dtype=torch.float32) - k // 2
            g = torch.exp(-(c**2) / 2.0)
            ker = g.unsqueeze(0) * g.unsqueeze(1)
            ker = ker / ker.sum()
            self.dw.weight.data.copy_(ker.unsqueeze(0).unsqueeze(0).expand(self.dw.weight.shape))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """x: [B,3,H,W]  →  tokens: [B,(H/4)*(W/4),C], H/4, W/4"""
        x = self.act(self.norm(self.pw(self.dw(x))))
        x = self.pe(x)  # [B,C,H/4,W/4]
        B, C, H, W = x.shape
        x = self.ln(x.flatten(2).transpose(1, 2))  # [B,H*W,C]
        return x, H, W


class EchogenicityChannelAttention(nn.Module):
    """
    Channel attention whose hidden layer maps to 3 echogenicity bins:
        hypoechoic / isoechoic / hyperechoic
    (directly correspond to ACR TI-RADS echogenicity feature language)

    Unlike standard SE attention (arbitrary excitation), ECA weights
    are clinically interpretable — each bin maps to a US category.

    Math:
        w = σ( FC₂( ReLU( FC₁( GAP(F) ) ) ) )
        F_out = F ⊗ w
    """

    def __init__(self, channels: int, echo_bins: int = 3, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, echo_bins)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        self.gate = nn.Sigmoid()
        # Named echo-bin projection for interpretability / ablation
        self.echo_proj = nn.Linear(mid, echo_bins)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : [B, N, C]
        returns:
            x_out        [B, N, C]  — recalibrated tokens
            echo_weights [B, 3]     — echogenicity activations
        """
        B, N, C = x.shape
        sq = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B,C]
        h = self.act(self.fc1(sq))  # [B,mid]
        echo_weights = self.echo_proj(h)  # [B,3]
        w = self.gate(self.fc2(h))  # [B,C]
        return x * w.unsqueeze(1), echo_weights


class ClassificationHead(nn.Module):
    """GAP(F4) → LayerNorm → Dropout → FC(4)"""

    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,H,W,C]  →  logits [B,num_classes]"""
        x = x.mean(dim=[1, 2])  # GAP over spatial dims
        return self.fc(self.drop(self.norm(x)))


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network fusing F1–F4 (top-down pathway).
    Outputs binary nodule mask at the input image resolution.

    Swin-T channels: F1=96, F2=192, F3=384, F4=768
    """

    def __init__(self, in_channels: List[int], fpn_ch: int = 128, seg_cls: int = 1):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in in_channels])
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1),
                    nn.BatchNorm2d(fpn_ch),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch // 2, 3, padding=1),
            nn.BatchNorm2d(fpn_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_ch // 2, seg_cls, 1),
        )

    def forward(self, features: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        """
        features: [F1,F2,F3,F4] each [B,Hi,Wi,Ci] (channel-last from Swin)
        out_size: (H, W) of the input image
        returns:  [B,1,H,W]
        """
        fs = [f.permute(0, 3, 1, 2).contiguous() for f in features]
        lats = [lat(f) for lat, f in zip(self.lateral, fs)]

        # Top-down fusion
        for i in range(len(lats) - 2, -1, -1):
            lats[i] = lats[i] + F.interpolate(
                lats[i + 1], size=lats[i].shape[-2:], mode="bilinear", align_corners=False
            )

        p1 = self.smooth[0](lats[0])  # [B,fpn,H/4,W/4]
        p1 = F.interpolate(p1, out_size, mode="bilinear", align_corners=False)
        return self.head(p1)  # [B,1,H,W]


class ThyFormer(nn.Module):
    """
    ThyFormer — hybrid CNN-Transformer for thyroid nodule TI-RADS classification.

    Forward returns:
        cls_logits   [B,4]       — TI-RADS class probabilities
        seg_logits   [B,1,H,W]   — nodule mask logits (input resolution)
        echo_weights [B,3]       — echogenicity bin activations
    """

    SWIN_T_CH = [96, 192, 384, 768]  # Swin-T stage output channels

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # Stage 1
        self.stem = DespecklingCNNStem(
            in_ch=3,
            out_ch=self.SWIN_T_CH[0],
            kernel=cfg.stem_kernel_size,
            patch=cfg.stem_patch_size,
        )

        # Stage 2
        self.eca = EchogenicityChannelAttention(
            channels=self.SWIN_T_CH[0],
            echo_bins=cfg.eca_echo_bins,
            reduction=cfg.eca_reduction_ratio,
        )

        # Stage 3 — timm Swin backbone (features_only bypasses cls head)
        # img_size must match the training resolution: Swin precomputes its
        # shifted-window attention masks at init for this resolution.
        self.swin = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            features_only=True,
            drop_path_rate=cfg.drop_path_rate,
            img_size=cfg.img_size,
        )

        if cfg.freeze_stages > 0:
            for name, p in self.swin.named_parameters():
                if any(f"layers.{i}" in name for i in range(cfg.freeze_stages)):
                    p.requires_grad = False

        # Stage 4a
        self.cls_head = ClassificationHead(
            self.SWIN_T_CH[-1], num_classes=5, dropout=cfg.cls_dropout
        )

        # Stage 4b
        self.fpn = FPNDecoder(
            self.SWIN_T_CH, fpn_ch=cfg.fpn_out_channels, seg_cls=cfg.seg_num_classes
        )

    def _run_swin_stages(self, tokens: torch.Tensor, H: int, W: int) -> List[torch.Tensor]:
        """
        Feed pre-computed tokens through Swin stages, bypassing
        the backbone's built-in patch embedding.
        """
        B, N, C = tokens.shape
        x = tokens.view(B, H, W, C)
        features = []

        for i in range(4):
            layer = getattr(self.swin, f"layers_{i}")
            x = layer(x)
            features.append(x)
        return features

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B,3,H,W] with H,W divisible by the stem patch size"""
        # Stage 1 — Despeckling stem
        tokens, H, W = self.stem(x)  # [B,(H/4)*(W/4),96]

        # Stage 2 — Echogenicity attention
        tokens, echo_w = self.eca(tokens)

        # Stage 3 — Swin encoder
        features = self._run_swin_stages(tokens, H, W)

        # Stage 4a — Classification
        cls_logits = self.cls_head(features[-1])  # [B,4]

        # Stage 4b — Segmentation
        seg_logits = self.fpn(features, x.shape[-2:])  # [B,1,H,W]

        return {
            "cls_logits": cls_logits,
            "seg_logits": seg_logits,
            "echo_weights": echo_w,
        }

    def get_param_groups(self) -> Dict[str, list]:
        """Separate backbone / head param groups for differential LR."""
        backbone, head = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (backbone if "swin" in name else head).append(p)
        return {"backbone": backbone, "head": head}


def build_model(cfg: ModelConfig) -> ThyFormer:
    model = ThyFormer(cfg)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ThyFormer  total={total/1e6:.1f}M  trainable={train/1e6:.1f}M")
    return model


class TimmClassifier(nn.Module):
    """
    Thin wrapper around a plain timm classification backbone (EfficientNet,
    ResNet, ConvNeXt, ViT, …) that exposes the same output contract as
    ThyFormer — a dict with a `cls_logits` key — so it can be dropped into the
    evaluation pipeline as a baseline for the DeLong test.

    No segmentation / echogenicity heads: a baseline only needs class scores.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int = 5,
        pretrained: bool = False,
        img_size: int = None,
    ):
        super().__init__()
        kwargs = dict(pretrained=pretrained, num_classes=num_classes)
        # Transformer backbones (ViT/Swin) need img_size baked in at build time;
        # CNNs are resolution-agnostic and reject the kwarg, so only pass it when
        # the factory accepts it.
        if img_size is not None:
            try:
                self.net = timm.create_model(backbone, img_size=img_size, **kwargs)
            except TypeError:
                self.net = timm.create_model(backbone, **kwargs)
        else:
            self.net = timm.create_model(backbone, **kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"cls_logits": self.net(x)}

    def load_compatible(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """
        Load weights saved either from this wrapper (keys prefixed ``net.``) or
        from a bare timm model (no prefix, e.g. a separately trained baseline).
        """
        if any(k.startswith("net.") for k in state_dict):
            return self.load_state_dict(state_dict, strict=strict)
        return self.net.load_state_dict(state_dict, strict=strict)


def build_baseline_model(arch: str, cfg: ModelConfig, num_classes: int = 5) -> nn.Module:
    """
    Build a baseline model for head-to-head comparison (e.g. DeLong's test).

        arch == "thyformer" / "same"  → a full ThyFormer (same architecture)
        any other timm model name      → that backbone wrapped in TimmClassifier
                                          (e.g. "efficientnet_b0", "resnet50",
                                          "convnext_tiny")

    The returned module always emits a `cls_logits` tensor of shape
    [B, num_classes], matching ThyFormer's classification output.
    """
    if arch.lower() in ("thyformer", "same"):
        return ThyFormer(cfg)

    model = TimmClassifier(
        arch,
        num_classes=num_classes,
        pretrained=False,
        img_size=cfg.img_size,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"Baseline [{arch}]  params={n/1e6:.1f}M")
    return model
