"""
models.py
---------
Four backbone models for TI-RADS classification, each fine-tuned via
transfer learning from ImageNet weights.

Architectures
-------------
  convnext    → ConvNeXt-Tiny    (CNN)                 feature dim: 768
  efficientnet→ EfficientNetV2-S (CNN)                 feature dim: 1280
  swin        → Swin-Transformer-Tiny (hybrid)         feature dim: 768
  vit         → ViT-B/16 (pure transformer)            feature dim: 768

Each model returns either:
  - class logits  [B, num_classes]  (default, for training)
  - feature vector [B, feat_dim]   (with return_features=True, for ensemble)

EnsembleModel wraps all four and fuses via soft-voting or concatenation.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from data_pipeline.create_dataset import NUM_CLASSES

FEATURE_DIM = {"convnext": 768, "efficientnet": 1280, "swin": 768, "vit": 768}


class BackboneModel(nn.Module):
    """
    Wraps any timm model with a custom classification head.

    Head: GlobalAvgPool → Dropout(0.3) → Linear(feat_dim, 256)
          → GELU → Dropout(0.2) → Linear(256, num_classes)

    Parameters
    ----------
    backbone_name : str     Key from FEATURE_DIM
    num_classes   : int     Output classes (5 for TR-1..TR-5)
    pretrained    : bool    Load ImageNet weights
    freeze_epochs : int     Freeze backbone for this many epochs (warmup)
    """

    TIMM_NAMES = {
        "convnext": "convnext_tiny.fb_in22k_ft_in1k",
        "efficientnet": "tf_efficientnetv2_s.in21k_ft_in1k",
        "swin": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "vit": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    }

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_epochs: int = 3,
    ):
        super().__init__()
        assert backbone_name in self.TIMM_NAMES, f"Unknown backbone: {backbone_name}"
        self.backbone_name = backbone_name
        self.freeze_epochs = freeze_epochs
        self._epoch = 0

        feat_dim = FEATURE_DIM[backbone_name]

        """Loading backbone"""
        self.backbone = timm.create_model(
            self.TIMM_NAMES[backbone_name], pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

        self._init_head()

        if freeze_epochs > 0:
            self._freeze_backbone()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def on_epoch_start(self, epoch: int):
        self._epoch = epoch
        if epoch == self.freeze_epochs:
            self._unfreeze_backbone()
        print(f"[{self.backbone_name}] Backbone unfrozen at epoch {epoch}")

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        Returns [B, num_classes] logits  or  [B, feat_dim] features.
        """
        features = self.backbone(x)  # [B, feat_dim]
        if return_features:
            return features
        return self.head(features)  # [B, num_classes]

    def get_features_and_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        return features, logits


def build_model(backbone_name: str, **kwargs) -> BackboneModel:
    model = BackboneModel(backbone_name, **kwargs)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[build_model] {backbone_name:15s} | {n_params:.1f}M params")
    return model


class EnsembleModel(nn.Module):
    """
    Loads 4 pre-trained BackboneModels from checkpoint files and
    fuses their predictions.

    Fusion modes
    ------------
    "soft_vote"    : average softmax probabilities             (default)
    "concat"       : concatenate logits → small linear head    (trainable)
    """

    def __init__(
        self,
        checkpoint_paths: dict[str, str],
        num_classes: int = NUM_CLASSES,
        fusion_mode: str = "soft_vote",
        device: str = "cpu",
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.models = nn.ModuleDict()
        self.device = device

        for name, ckpt_path in checkpoint_paths.items():
            model = BackboneModel(name, pretrained=False)
            state = torch.load(ckpt_path, map_location=device)

            if "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            model.eval()
            self.models[name] = model

        if fusion_mode == "concat":
            n_models = len(checkpoint_paths)
            self.meta = nn.Linear(n_models * num_classes, num_classes)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]  — caller is responsible for per-backbone resize
        Returns [B, num_classes] final logits.
        """

        all_probs = []
        for name, model in self.models.items():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

        if self.fusion_mode == "soft_vote":
            stacked = torch.stack(all_probs, dim=0)
            return stacked.mean(dim=0)

        elif self.fusion_mode == "concat":
            concat = torch.cat(all_probs, dim=-1)
            return self.meta(concat)

        raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
