"""
ThyFormer — Explainability
GradCAM (CNN layers) + Attention Rollout (Swin) + heatmap export.
"""
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thyformer_models import ThyFormer

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
TIRADS = {0: "T1", 1: "T2", 2: "T3", 3: "T4"}


class GradCAM:
    """
    GradCAM hooked on the last Swin stage's norm layer.
    Works for any nn.Module target layer.
    """

    def __init__(self, model: ThyFormer, target_layer: Optional[nn.Module] = None):
        self.model = model.eval()
        if target_layer is None:
            target_layer = model.swin.layers_3.blocks[-1].norm1
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._hooks = [
            target_layer.register_forward_hook(lambda m, i, o: setattr(self, "_acts", o.detach())),
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(
                    self, "_grads", go[0].detach() if go[0] is not None else None
                )
            ),
        ]

    def remove(self):
        for h in self._hooks:
            h.remove()

    def generate(self, image: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """image: [1,3,224,224] → heatmap [224,224] float32 [0,1]"""
        self.model.zero_grad()
        device = next(self.model.parameters()).device
        image = image.to(device).requires_grad_(True)
        out = self.model(image)
        logits = out["cls_logits"]
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        logits[0, class_idx].backward()

        acts = self._acts
        grads = self._grads
        if acts is None or grads is None:
            return np.zeros((224, 224), dtype=np.float32)

        # Handle channel-last (Swin) and token formats
        if acts.dim() == 3:
            B, N, C = acts.shape
            H = W = int(N**0.5)
            acts = acts.view(B, H, W, C)
            grads = grads.view(B, H, W, C)

        weights = grads.mean(dim=(1, 2))  # [1,C]
        cam = (weights.unsqueeze(1).unsqueeze(1) * acts).sum(-1)  # [1,H,W]
        cam = F.relu(cam).squeeze(0).cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        cam = cv2.GaussianBlur(cam, (9, 9), 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.astype(np.float32)


class AttentionRollout:
    """
    Rollout of Swin attention maps across all layers.
    Reference: Abnar & Zuidema, ACL 2020.
    """

    def __init__(self, model: ThyFormer, discard_ratio: float = 0.9):
        self.model = model.eval()
        self.ratio = discard_ratio
        self._atts: List[torch.Tensor] = []
        self._hooks = []
        for _, m in model.swin.named_modules():
            if hasattr(m, "attn"):

                def hook(mod, inp, out):
                    if isinstance(out, torch.Tensor) and out.dim() == 4:
                        self._atts.append(out.detach().cpu())

                self._hooks.append(m.attn.register_forward_hook(hook))

    def remove(self):
        for h in self._hooks:
            h.remove()

    def generate(self, image: torch.Tensor) -> np.ndarray:
        """image: [1,3,224,224] → rollout map [224,224] float32 [0,1]"""
        self._atts.clear()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            self.model(image.to(device))
        if not self._atts:
            return np.zeros((224, 224), dtype=np.float32)

        result = None
        for attn in self._atts:
            avg = attn.mean(1)  # [B,N,N]
            flat = avg.view(avg.size(0), -1)
            thresh = flat.quantile(self.ratio, dim=-1, keepdim=True).view(avg.size(0), 1, 1)
            attn_t = (avg > thresh).float() * avg
            attn_t = attn_t / (attn_t.sum(-1, keepdim=True) + 1e-8)
            I_eye = torch.eye(attn_t.size(-1)).unsqueeze(0)
            ar = (attn_t + I_eye) / 2.0
            if result is None:
                result = ar[0]
            elif result.shape == ar[0].shape:
                result = ar[0] @ result

        if result is None:
            return np.zeros((224, 224), dtype=np.float32)

        row = result[0].numpy()
        side = int(len(row) ** 0.5)
        if side * side == len(row):
            mask = row.reshape(side, side)
        else:
            mask = row[:49].reshape(7, 7)
        mask = (mask - mask.min()) / (mask.max() + 1e-8)
        return cv2.resize(mask, (224, 224)).astype(np.float32)


def denorm(t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation → uint8 [H,W,3]."""
    img = t.permute(1, 2, 0).cpu().numpy()
    img = (img * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
    return (img * 255).astype(np.uint8)


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h_u8 = (heatmap * 255).astype(np.uint8)
    h_rgb = cv2.cvtColor(cv2.applyColorMap(h_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    if image.shape[:2] != h_rgb.shape[:2]:
        h_rgb = cv2.resize(h_rgb, (image.shape[1], image.shape[0]))
    return (alpha * h_rgb + (1 - alpha) * image).astype(np.uint8)


def save_figure(
    orig: np.ndarray,
    heatmap: np.ndarray,
    pred: int,
    true: int,
    stem: str,
    out_dir: str,
    score: float = 0.0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ov = overlay(orig, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, img in zip(
        axes, ["Original US", "GradCAM", "Overlay"], [orig, plt.cm.jet(heatmap)[:, :, :3], ov]
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    colour = "green" if pred == true else "red"
    fig.suptitle(
        f"{stem}  pred:{TIRADS[pred]}  true:{TIRADS[true]}  score:{score:.3f}",
        color=colour,
        fontsize=11,
    )
    plt.tight_layout()
    path = out_dir / f"{stem}_gradcam.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def run_gradcam_batch(model: ThyFormer, loader, out_dir: str, n: int = 50, device: str = "cuda"):
    model.to(device).eval()
    gcam = GradCAM(model)
    saved = 0

    for batch in loader:
        if saved >= n:
            break
        images = batch["image"]
        labels = batch["label"]
        stems = batch["stem"]

        for i in range(images.size(0)):
            if saved >= n:
                break
            img_t = images[i : i + 1]
            true_l = int(labels[i].argmax() if labels.dim() == 2 else labels[i])
            heat = gcam.generate(img_t, class_idx=true_l)
            with torch.no_grad():
                out = model(img_t.to(device))
            probs = torch.softmax(out["cls_logits"], 1)[0].cpu().numpy()
            pred = int(probs.argmax())
            orig = denorm(images[i])
            save_figure(orig, heat, pred, true_l, stems[i], out_dir, float(probs[pred]))
            saved += 1

    gcam.remove()
    print(f"Saved {saved} GradCAM figures → {out_dir}")
