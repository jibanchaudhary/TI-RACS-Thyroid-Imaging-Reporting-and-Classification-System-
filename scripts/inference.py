"""
inference.py
------------
Inference for trained TI-RADS classification models.

Supports
--------
  1. Single-image prediction   (single backbone or ensemble)
  2. Batch prediction          (from a folder or a list of image paths)
  3. Test-set evaluation       (full metrics: accuracy, F1, AUC, confusion matrix)
  4. Grad-CAM heatmap          (per backbone, overlaid on the original image)

Usage examples
--------------
  # Single image with Grad-CAM
  python inference.py --mode single --image path/to/us.png \
      --checkpoint runs/convnext/best.pt --backbone convnext --gradcam

  # Ensemble over four checkpoints
  python inference.py --mode ensemble --image path/to/us.png \
      --checkpoints runs/convnext/best.pt runs/efficientnet/best.pt \
                    runs/swin/best.pt runs/vit/best.pt

  # Full test-set evaluation
  python inference.py --mode test --data_dir data \
      --backbone convnext --checkpoint runs/convnext/best.pt
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from data_pipeline.create_dataset import (
    ThyroidDataset,
    CLASS_NAMES,
    NUM_CLASSES,
    BACKBONE_SIZE,
    build_val_transforms,
)
from data_pipeline.dataset_conversion import polygon_to_mask, crop_nodule

from models.models import BackboneModel, EnsembleModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def preprocess_image(
    img_path: str,
    backbone: str,
    polygon: list[dict] | None = None,
    padding: int = 24,
) -> torch.Tensor:
    """
    Load, (optionally crop to polygon ROI), resize, normalise.
    Returns [1, 3, H, W] float32 tensor.
    """

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if polygon is not None:
        h, w = img_rgb.shape[:2]
        mask = polygon_to_mask(polygon, h, w)
        img_rgb, _ = crop_nodule(img_rgb, mask, padding)

    transform = build_val_transforms(BACKBONE_SIZE[backbone])
    tensor = transform(image=img_rgb)["image"]  # [3, H, W]
    return tensor.unsqueeze(0)  # [1, 3, H, W]


class GradCAM:
    """
    Grad-CAM for any timm model with a feature extractor ending in
    a Conv2d layer (ConvNeXt, EfficientNet, Swin).
    For ViT we target the last attention norm layer.

    Usage
    -----
        gcam = GradCAM(model, backbone_name)
        heatmap = gcam(image_tensor, class_idx)   # numpy [H, W] 0..1
        gcam.remove_hooks()
    """

    def __init__(self, model: BackboneModel, backbone_name: str):
        self.model = model
        self.handles = []
        self._grads = None
        self._acts = None

        # Select the target layer per architecture
        target = self._get_target_layer(model.backbone, backbone_name)
        self.handles.append(target.register_forward_hook(self._save_activation))
        self.handles.append(target.register_full_backward_hook(self._save_gradient))

    @staticmethod
    def _get_target_layer(backbone, name: str):
        if name == "convnext":
            # Last stage, last block's depthwise conv
            return backbone.stages[-1].blocks[-1].conv_dw
        elif name == "efficientnet":
            return backbone.blocks[-1][-1].conv_pwl
        elif name == "swin":
            # Last swin block's norm
            return backbone.layers[-1].blocks[-1].norm1
        elif name == "vit":
            return backbone.blocks[-1].norm1
        else:
            raise ValueError(f"Unknown backbone for Grad-CAM: {name}")

    def _save_activation(self, module, input, output):
        self._acts = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._grads = grad_output[0].detach()

    def __call__(
        self,
        image_tensor: torch.Tensor,  # [1, 3, H, W]
        class_idx: int | None = None,
    ) -> np.ndarray:
        """Returns heatmap as [H, W] float32 array, normalised to [0,1]."""
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)  # [1, C]
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Grad-CAM formula: weight = mean(grads over spatial dims)
        # Then heatmap = ReLU(sum(weight_k * activation_k))
        grads = self._grads  # [1, C, H', W'] or [1, N, C] for ViT
        acts = self._acts

        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks did not fire. " "Check target layer selection.")

        # Handle ViT patch tokens: shape [1, N, C]
        if grads.dim() == 3:
            weights = grads.mean(dim=1, keepdim=True)  # [1, 1, C]
            cam = (weights * acts).sum(dim=-1)  # [1, N]
            # Reshape to sqrt(N) × sqrt(N) spatial grid
            n = cam.shape[1]
            s = int(n**0.5)
            cam = cam.reshape(1, s, s)
        else:
            # CNN / Swin: [1, C, H', W']
            weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            cam = (weights * acts).sum(dim=1)  # [1, H', W']

        cam = F.relu(cam.squeeze(0))  # [H', W']
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


def overlay_gradcam(
    original_img_path: str,
    heatmap: np.ndarray,  # [H', W'] float in [0,1]
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Resize heatmap to original image size, overlay as a coloured mask.
    Returns BGR uint8 image (suitable for cv2.imwrite).
    """
    orig = cv2.imread(original_img_path)
    h, w = orig.shape[:2]
    heat = cv2.resize(heatmap, (w, h))
    heat8 = (heat * 255).astype(np.uint8)
    color = cv2.applyColorMap(heat8, colormap)
    blend = cv2.addWeighted(orig, 1 - alpha, color, alpha, 0)
    return blend


@torch.no_grad()
def predict_single(
    model: BackboneModel | EnsembleModel,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Returns dict with keys:
        predicted_class  : str   e.g. "TR-4"
        predicted_idx    : int   0-4
        probabilities    : list  [p0..p4] floats summing to 1
        confidence       : float probability of predicted class
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    out = model(image_tensor)
    if out.shape[-1] == NUM_CLASSES:
        # Already probabilities (EnsembleModel with soft_vote)
        probs = out if out.max() <= 1.0 else F.softmax(out, dim=-1)
    else:
        probs = F.softmax(out, dim=-1)

    probs_list = probs.squeeze(0).cpu().tolist()
    pred_idx = int(np.argmax(probs_list))

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_idx": pred_idx,
        "probabilities": {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs_list)},
        "confidence": round(probs_list[pred_idx], 4),
    }


@torch.no_grad()
def evaluate_test_set(
    model: BackboneModel | EnsembleModel,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run inference on all batches in test_loader.
    Returns dict with accuracy, macro-F1, AUC, confusion matrix,
    and per-class report string.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in test_loader:
        images = images.to(device)
        out = model(images)
        probs = F.softmax(out, dim=-1) if out.max() > 1.0 else out
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    unique_labels = sorted(set(all_labels) | set(all_preds))
    # report = classification_report(all_labels, all_preds,
    #                                target_names=CLASS_NAMES, zero_division=0)
    report = classification_report(
        all_labels,
        all_preds,
        labels=unique_labels,
        target_names=[CLASS_NAMES[i] for i in unique_labels],
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    try:
        auc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "macro_auc": round(auc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def load_single_backbone(
    checkpoint_path: str, backbone: str, device: torch.device
) -> BackboneModel:
    model = BackboneModel(backbone, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="TI-RADS inference")
    parser.add_argument("--mode", choices=["single", "ensemble", "test"], required=True)
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input ultrasound image (single/ensemble mode)",
    )
    parser.add_argument("--backbone", type=str, default="convnext")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to single backbone checkpoint (single/test mode)",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Paths to multiple checkpoints (ensemble mode). "
        "Order: convnext efficientnet swin vit",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="inference_out")
    parser.add_argument(
        "--gradcam", action="store_true", help="Generate Grad-CAM heatmap (single mode only)"
    )
    parser.add_argument(
        "--polygon", type=str, default=None, help="JSON string of SVG polygon points for ROI crop"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if args.mode == "single":
        assert (
            args.image and args.checkpoint and args.backbone
        ), "--image, --checkpoint, and --backbone required for single mode"

        polygon = None
        if args.polygon:
            import json

            polygon = json.loads(args.polygon)

        image_tensor = preprocess_image(args.image, args.backbone, polygon=polygon)

        model = load_single_backbone(args.checkpoint, args.backbone, device)
        result = predict_single(model, image_tensor, device)

        print("\n--- Prediction ---")
        print(f"  Predicted class : {result['predicted_class']}")
        print(f"  Confidence      : {result['confidence']:.1%}")
        print("  Per-class probs :")
        for cls, p in result["probabilities"].items():
            bar = "█" * int(p * 30)
            print(f"    {cls}: {p:.4f} {bar}")

        if args.gradcam:
            gcam = GradCAM(model, args.backbone)
            heat = gcam(image_tensor.to(device), class_idx=result["predicted_idx"])
            gcam.remove_hooks()

            blend = overlay_gradcam(args.image, heat)
            out_p = os.path.join(args.output_dir, f"gradcam_{args.backbone}.png")
            cv2.imwrite(out_p, blend)
            print(f"\n  Grad-CAM saved → {out_p}")

    elif args.mode == "ensemble":
        assert (
            args.image and args.checkpoints
        ), "--image and --checkpoints required for ensemble mode"

        backbone_names = ["convnext", "efficientnet", "swin", "vit"]
        assert (
            len(args.checkpoints) == 4
        ), "Provide exactly 4 checkpoints: convnext efficientnet swin vit"

        ckpt_map = dict(zip(backbone_names, args.checkpoints))
        polygon = None
        if args.polygon:
            polygon = json.loads(args.polygon)

        # Ensemble uses a common size (224); resize handled inside each model
        # Here we use convnext size as default input since all 224 backbones
        # share the same spatial resolution
        image_tensor = preprocess_image(args.image, "convnext", polygon=polygon)

        ensemble = EnsembleModel(ckpt_map, fusion_mode="soft_vote", device=str(device))
        ensemble.to(device)

        result = predict_single(ensemble, image_tensor, device)

        print("\n--- Ensemble Prediction ---")
        print(f"  Predicted class : {result['predicted_class']}")
        print(f"  Confidence      : {result['confidence']:.1%}")
        print("  Per-class probs :")
        for cls, p in result["probabilities"].items():
            bar = "█" * int(p * 30)
            print(f"    {cls}: {p:.4f} {bar}")

        if args.gradcam:
            gcam = GradCAM(model, args.backbone)
            heat = gcam(image_tensor.to(device), class_idx=result["predicted_idx"])
            gcam.remove_hooks()

            blend = overlay_gradcam(args.image, heat)
            out_p = os.path.join(args.output_dir, f"gradcam_{args.backbone}.png")
            cv2.imwrite(out_p, blend)
            print(f"\n  Grad-CAM saved → {out_p}")

    elif args.mode == "test":
        assert (
            args.checkpoint and args.backbone
        ), "--checkpoint and --backbone required for test mode"

        model = load_single_backbone(args.checkpoint, args.backbone, device)

        test_ds = ThyroidDataset(args.data_dir, "test", args.backbone)
        test_loader = DataLoader(
            test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )
        results = evaluate_test_set(model, test_loader, device)
        print("\n--- Test Set Evaluation ---")
        print(f"  Accuracy  : {results['accuracy']:.4f}")
        print(f"  Macro F1  : {results['macro_f1']:.4f}")
        print(f"  Macro AUC : {results['macro_auc']:.4f}")
        print("\n" + results["classification_report"])
        print("Confusion matrix:")
        print(results["confusion_matrix"])

        cm = results["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(6, 6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["TR-2", "TR-3", "TR-4", "TR-5"],
        )

        disp.plot(ax=ax, cmap="Blues", values_format="d")

        plt.title(f"Confusion Matrix - {args.backbone}")
        plt.tight_layout()

        cm_png_path = os.path.join(
            args.output_dir,
            f"confusion_matrix_{args.backbone}.png",
        )

        plt.savefig(cm_png_path, dpi=300)
        plt.close()

        out_p = os.path.join(args.output_dir, f"test_results_{args.backbone}.txt")
        with open(out_p, "w") as f:
            f.write(f"Backbone:  {args.backbone}\n")
            f.write(f"Accuracy:  {results['accuracy']}\n")
            f.write(f"Macro F1:  {results['macro_f1']}\n")
            f.write(f"Macro AUC: {results['macro_auc']}\n\n")
            f.write(results["classification_report"])
            f.write("\nConfusion matrix:\n")
            f.write(str(results["confusion_matrix"]))
        print(f"\nResults saved → {out_p}")


if __name__ == "__main__":
    main()
