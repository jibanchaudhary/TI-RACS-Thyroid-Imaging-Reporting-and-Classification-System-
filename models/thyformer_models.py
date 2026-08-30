"""
ThyFormer — Full Model Architecture

Stage 1 : DespecklingCNNStem         ★ Novel
Stage 2 : EchogenicityChannelAttn    ★ Novel
Stage 3 : Swin Transformer encoder   (timm backbone)
Stage 4a: ClassificationHead
Stage 4b: FPNDecoder (segmentation)
"""
from typing import Dict, List, Sequence, Tuple

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
        # self.norm = nn.GroupNorm(num_groups=in_ch, num_channels=in_ch)
        self.act = nn.GELU()
        # self.pe = nn.Conv2d(in_ch, out_ch, patch, stride=patch, bias=False)
        # self.ln = nn.LayerNorm(out_ch)
        self.out = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self._init_gaussian(kernel)
        with torch.no_grad():
            self.pw.weight.copy_(torch.eye(in_ch).view(in_ch, in_ch, 1, 1))
            self.out.weight.zero_()

    def _init_gaussian(self, k: int):
        with torch.no_grad():
            c = torch.arange(k, dtype=torch.float32) - k // 2
            g = torch.exp(-(c**2) / 2.0)
            ker = g.unsqueeze(0) * g.unsqueeze(1)
            ker = ker / ker.sum()
            self.dw.weight.data.copy_(ker.unsqueeze(0).unsqueeze(0).expand(self.dw.weight.shape))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # """x: [B,3,H,W]  →  tokens: [B,(H/4)*(W/4),C], H/4, W/4"""
        # x = self.act(self.norm(self.pw(self.dw(x))))
        # x = self.pe(x)  # [B,C,H/4,W/4]
        # B, C, H, W = x.shape
        # x = self.ln(x.flatten(2).transpose(1, 2))  # [B,H*W,C]
        # return x, H, W
        return x + self.out(self.act(self.pw(self.dw(x))))


class EchogenicityChannelAttention(nn.Module):
    """
    Channel attention whose hidden layer maps to 3 echogenicity bins:
        hypoechoic / isoechoic / hyperechoic
    (directly correspond to ACR TI-RADS echogenicity feature language)

    Unlike standard SE attention (arbitrary excitation), ECA weights
    are clinically interpretable — each bin maps to a US category.

    Math:
        F_out = F ⊗ w,  w = g( FC₂( ReLU( FC₁( GAP(F) ) ) ) )

    Gating form (``gate``):
        "sigmoid"       — legacy w = σ(FC₂(h)). With zero bias and Xavier weights
                          this sits at 0.500 at init, so it HALVES every token
                          before the pretrained Swin sees it (measured
                          output/input norm ratio 0.5005). The backbone then
                          spends its frozen warmup epochs re-adapting to an input
                          scale shift that carries no information.
        "residual_tanh" — w = 1 + tanh(FC₂(h)) with FC₂ zero-initialised: exactly
                          identity at init, still bounded in [0,2] so the block
                          can suppress or amplify once it has a reason to.

    Both forms use identical parameter shapes, so a checkpoint trained under one
    loads under the other.
    """

    def __init__(
        self,
        channels: int,
        echo_bins: int = 3,
        reduction: int = 16,
        gate: str = "residual_tanh",
    ):
        super().__init__()
        if gate not in ("residual_tanh", "sigmoid"):
            raise ValueError(f"eca_gate must be 'residual_tanh' or 'sigmoid', got {gate!r}")
        mid = max(channels // reduction, echo_bins)
        self.gate_mode = gate
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        # Named echo-bin projection. Supervised via LossConfig.delta — without it
        # this layer's gradient is None and it stays at its random init.
        self.echo_proj = nn.Linear(mid, echo_bins)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        if gate == "residual_tanh":
            nn.init.zeros_(self.fc2.weight)  # -> tanh(0) = 0 -> scale exactly 1.0
        else:
            nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : [B, N, C]
        returns:
            x_out        [B, N, C]  — recalibrated tokens
            echo_logits  [B, 3]     — echogenicity bin logits (hypo/iso/hyper)
        """
        sq = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B,C]
        h = self.act(self.fc1(sq))  # [B,mid]
        echo_logits = self.echo_proj(h)  # [B,3]
        raw = self.fc2(h)  # [B,C]
        w = 1.0 + torch.tanh(raw) if self.gate_mode == "residual_tanh" else torch.sigmoid(raw)
        return x * w.unsqueeze(1), echo_logits


class ClassificationHead(nn.Module):
    """GAP(F4) → LayerNorm → Dropout → FC(num_classes)"""

    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,H,W,C]  →  logits [B,num_classes]"""
        x = x.mean(dim=[1, 2])  # GAP over spatial dims
        return self.fc(self.drop(self.norm(x)))


class TiradsDescriptorHead(nn.Module):
    """
    Auxiliary head predicting the five ACR TI-RADS descriptors ★ NOVEL

        composition / echogenicity / shape / margin / echogenic foci

    The ACR TI-RADS grade is not an opaque class — it is the banded sum of the
    points these five findings carry. metadata.csv supplies all five, from the
    reading radiologist, for every clip. Supervising them decomposes one 5-way
    label over 134 clips into five easier sub-problems that share an encoder, and
    it makes the model's reasoning inspectable against the rubric a clinician
    actually applies: you can ask *which finding* it disagreed with, not just
    that it got the grade wrong.

    One linear head per descriptor over the shared pooled feature — deliberately
    small, so the auxiliary task shapes the encoder rather than being solved in
    the head.
    """

    def __init__(self, in_ch: int, bins: Sequence[int], dropout: float = 0.2):
        super().__init__()
        self.bins = list(bins)
        self.norm = nn.LayerNorm(in_ch)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(in_ch, b) for b in self.bins])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: [B,H,W,C] → list of 5 logit tensors, [B,bins_i] each"""
        z = self.drop(self.norm(x.mean(dim=[1, 2])))
        return [h(z) for h in self.heads]


class ACRScoringLayer(nn.Module):
    """
    Differentiable ACR TI-RADS scoring table ★ NOVEL (concept bottleneck)

    Turns the five predicted descriptor distributions into a grade distribution
    by the *published* ACR rule, with no learned parameters at all:

        total points = sum of the five descriptors' point values
        0 pts -> TR1,  1-2 -> TR2,  3 -> TR3,  4-6 -> TR4,  >=7 -> TR5

    Because the descriptors are conditionally independent given the image
    features, the distribution over the total is the discrete convolution of the
    five per-descriptor point distributions — computed exactly here, not
    approximated by the expected score. P(grade) is then the total's mass inside
    each band.

    This is what makes the head a concept *bottleneck* rather than a parallel
    auxiliary task: the grade cannot be predicted except through the concepts, so
    the concept predictions are necessarily faithful, and a clinician can
    overwrite one concept and read off the new grade (see `intervene`).

    `point_values[j]` lists descriptor j's ACR point value per class index, in the
    same class order the descriptor head emits. metadata.csv already stores the
    ACR point values themselves, so these come straight from its observed levels.

    Note on 1 point: the published table jumps 0 -> 2, so a 1-point total is not
    reachable by any real combination of findings, but a soft prediction can put
    mass there. It is banded into TR2 as the nearest category.
    """

    BANDS = [(0, 0), (1, 2), (3, 3), (4, 6), (7, 10**6)]  # TR1 … TR5

    def __init__(self, point_values: Sequence[Sequence[int]]):
        super().__init__()
        self.point_values = [list(pv) for pv in point_values]
        max_pts = int(sum(max(pv) for pv in self.point_values))
        self.max_pts = max_pts
        # band_matrix[p, g] = 1 when a total of p points falls in grade g.
        band = torch.zeros(max_pts + 1, len(self.BANDS))
        for g, (lo, hi) in enumerate(self.BANDS):
            for p in range(max_pts + 1):
                if lo <= p <= hi:
                    band[p, g] = 1.0
        self.register_buffer("band_matrix", band)

    def point_distribution(self, descriptor_logits: List[torch.Tensor]) -> torch.Tensor:
        """[B, max_pts+1] distribution over the total ACR score."""
        b = descriptor_logits[0].shape[0]
        dev = descriptor_logits[0].device
        total = torch.zeros(b, self.max_pts + 1, device=dev, dtype=torch.float32)
        total[:, 0] = 1.0
        for lg, pv in zip(descriptor_logits, self.point_values):
            p = F.softmax(lg.float(), dim=-1)  # [B, n_levels]
            nxt = torch.zeros_like(total)
            for k, pts in enumerate(pv):
                # Shift this descriptor's mass up by its point value.
                nxt[:, pts:] = nxt[:, pts:] + total[:, : self.max_pts + 1 - pts] * p[:, k : k + 1]
            total = nxt
        return total

    def forward(self, descriptor_logits: List[torch.Tensor]) -> torch.Tensor:
        """Five descriptor logit tensors → grade log-probabilities [B,5].

        Returns log-probs so downstream softmax(cls_logits) recovers exactly this
        distribution, matching the contract CornHead and the eval pipeline use.
        """
        pts = self.point_distribution(descriptor_logits)
        grade = pts @ self.band_matrix  # [B,5]
        return grade.clamp_min(1e-12).log()

    @torch.no_grad()
    def intervene(self, descriptor_logits: List[torch.Tensor], j: int, level: int) -> torch.Tensor:
        """
        Clinician intervention: force descriptor `j` to `level` and re-score.

        The signature capability of a concept bottleneck — the grade must move in
        the way the ACR table says it should, because that table *is* the head.
        """
        forced = [lg.clone() for lg in descriptor_logits]
        onehot = torch.full_like(forced[j], -30.0)
        onehot[:, level] = 30.0
        forced[j] = onehot
        return self(forced)


class CornHead(nn.Module):
    """
    Ordinal classification head (CORN — Cao, Mirjalili & Raschka, 2020).

    TI-RADS is a graded scale, not a set of unrelated categories: confusing TR-3
    with TR-4 is a smaller error than confusing TR-3 with TR-5, and a softmax
    head is blind to that. CORN predicts K-1 *conditional* binary logits

        q_k = P(y > k | y > k-1),   k = 0 … K-2

    from which the cumulative probabilities are the running product
    P(y > k) = Π_{j<=k} σ(q_j) — monotone by construction, so the model cannot
    emit the incoherent "probably >TR-4 but probably not >TR-3".

    Emits both:
        ordinal_logits [B,K-1] — raw q_k, what the CORN loss consumes
        cls_logits     [B,K]   — log of the implied class probabilities, so every
                                 downstream consumer that does softmax(cls_logits)
                                 (evaluate.py, the metric utils, Grad-CAM) keeps
                                 working unchanged and recovers exactly these
                                 probabilities.
    """

    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(in_ch)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes - 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B,H,W,C]  →  (cls_logits [B,K], ordinal_logits [B,K-1])"""
        x = x.mean(dim=[1, 2])
        q = self.fc(self.drop(self.norm(x))).float()  # [B,K-1]
        return corn_to_class_logits(q), q


def corn_to_class_logits(q: torch.Tensor) -> torch.Tensor:
    """
    CORN conditional logits [B,K-1] → log class probabilities [B,K].

    cum_k = P(y > k) = Π_{j<=k} σ(q_j), computed as a cumulative sum in log space
    so the product stays stable when K grows or σ(q_j) saturates. Then

        P(y=0)   = 1 - cum_0
        P(y=k)   = cum_{k-1} - cum_k
        P(y=K-1) = cum_{K-2}
    """
    log_cum = F.logsigmoid(q.float()).cumsum(dim=1)  # [B,K-1]
    cum = log_cum.exp()
    ones = torch.ones_like(cum[:, :1])
    upper = torch.cat([ones, cum], dim=1)  # P(y > k-1), k = 0…K-1
    lower = torch.cat([cum, torch.zeros_like(cum[:, :1])], dim=1)  # P(y > k)
    probs = (upper - lower).clamp_min(1e-12)
    return probs.log()


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
            gate=getattr(cfg, "eca_gate", "residual_tanh"),
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
        self.head_type = getattr(cfg, "head_type", "softmax")
        if self.head_type == "concept_bottleneck":
            # The grade comes only from the ACR scoring layer, so there is no
            # direct feature->grade path at all. The descriptor head below is
            # mandatory in this mode, not auxiliary.
            self.cls_head = None
            self.acr = ACRScoringLayer(cfg.descriptor_point_values)
        else:
            head_cls = CornHead if self.head_type == "corn" else ClassificationHead
            self.cls_head = head_cls(self.SWIN_T_CH[-1], num_classes=5, dropout=cfg.cls_dropout)
            self.acr = None

        # Stage 4b
        self.fpn = FPNDecoder(
            self.SWIN_T_CH, fpn_ch=cfg.fpn_out_channels, seg_cls=cfg.seg_num_classes
        )

        # Stage 4c — ACR descriptor head. Auxiliary in the softmax/corn modes;
        # the only route to a grade under concept_bottleneck.
        want_desc = getattr(cfg, "descriptor_head", False) or self.head_type == "concept_bottleneck"
        self.desc_head = (
            TiradsDescriptorHead(self.SWIN_T_CH[-1], cfg.descriptor_bins, dropout=cfg.cls_dropout)
            if want_desc
            else None
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
        """x: [B,3,H,W] with H,W divisible by the Swin patch size (4)"""

        # Stage 1 — Despeckling stem
        x_dn = self.stem(x)

        # Stage 2 — Swin's pretrained patch embedding
        tokens = self.swin.patch_embed(x_dn)
        B, H, W, C = tokens.shape

        # Stage 3 — (variable) echo_w: Any
        tokens, echo_w = self.eca(tokens.flatten(1, 2))

        # Stage 4 - Swin encoder stages
        features = self._run_swin_stages(tokens, H, W)

        # Stage 4c — ACR descriptors (needed before 4a in bottleneck mode)
        desc_logits = self.desc_head(features[-1]) if self.desc_head is not None else None

        # Stage 4a — Classification
        ordinal_logits = None
        if self.head_type == "concept_bottleneck":
            cls_logits = self.acr(desc_logits)
        else:
            head_out = self.cls_head(features[-1])
            # CornHead returns (cls_logits, ordinal_logits); the softmax head
            # returns cls_logits alone. Either way `cls_logits` is softmax-able,
            # so every downstream consumer sees one contract.
            if isinstance(head_out, tuple):
                cls_logits, ordinal_logits = head_out
            else:
                cls_logits = head_out

        # Stage 4b — Segmentation
        seg_logits = self.fpn(features, x.shape[-2:])  # [B,1,H,W]

        out = {
            "cls_logits": cls_logits,
            "seg_logits": seg_logits,
            "echo_weights": echo_w,
        }
        if ordinal_logits is not None:
            out["ordinal_logits"] = ordinal_logits
        if desc_logits is not None:
            out["descriptor_logits"] = desc_logits
        return out

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
    print(
        f"ThyFormer  total={total/1e6:.1f}M  trainable={train/1e6:.1f}M  "
        f"eca_gate={getattr(cfg, 'eca_gate', 'residual_tanh')}  "
        f"head={getattr(cfg, 'head_type', 'softmax')}"
    )
    return model


# Architecture-defining fields: two checkpoints that disagree on any of these are
# different models, even where the state-dict shapes happen to match.
ARCH_FIELDS = ("backbone", "img_size", "eca_gate", "head_type", "eca_echo_bins", "fpn_out_channels")

# Values that predate a config field. A checkpoint saved before `eca_gate`
# existed was necessarily trained with the old sigmoid gate, and one saved before
# `head_type` existed used the softmax head.
_LEGACY_DEFAULTS = {"eca_gate": "sigmoid", "head_type": "softmax"}


def model_cfg_for_checkpoint(ckpt: Dict, cfg: ModelConfig) -> ModelConfig:
    """
    Reconcile the live ModelConfig with the architecture a checkpoint was trained
    under, so a checkpoint is always rebuilt as the model that produced it.

    This matters because `eca_gate` changes the *function* the ECA block computes
    while leaving every parameter shape identical: load_state_dict would accept a
    sigmoid-trained checkpoint into a residual_tanh model without complaint and
    quietly return different predictions. Fields absent from the saved config
    fall back to _LEGACY_DEFAULTS rather than to today's defaults.

    Returns a copy; the caller's config is not mutated.
    """
    import copy

    out = copy.deepcopy(cfg)
    saved = ckpt.get("config") if isinstance(ckpt, dict) else None
    saved_model = getattr(saved, "model", None)
    if saved_model is None:
        return out

    # Read the pickled INSTANCE dict, not getattr: a dataclass field with a
    # default is also a class attribute, so getattr on a config pickled before
    # the field existed silently returns *today's* default and the whole
    # reconciliation no-ops. Absence from __dict__ is the only reliable signal
    # that a checkpoint predates a field.
    saved_fields = vars(saved_model)

    changed = []
    for f in ARCH_FIELDS:
        if f in saved_fields:
            wanted = saved_fields[f]
        elif f in _LEGACY_DEFAULTS:
            wanted = _LEGACY_DEFAULTS[f]
        else:
            continue
        if wanted is not None and getattr(out, f, None) != wanted:
            changed.append(f"{f}: {getattr(out, f, None)} → {wanted}")
            setattr(out, f, wanted)
    if changed:
        print("  checkpoint architecture differs from the live config; rebuilding with:")
        for c in changed:
            print(f"    {c}")
    return out


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
