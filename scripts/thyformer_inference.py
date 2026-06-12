"""
ThyFormer — Inference
Run a trained ThyFormer checkpoint on:
  1. A single image
  2. A folder of images
  3. A CSV of image paths (same format as test.csv)

Outputs per image:
  - TI-RADS class (T1–T4) + confidence scores
  - Nodule segmentation mask (optional)
  - GradCAM heatmap overlay (optional)
  - CSV summary report

Usage examples:

  # Single image
  python scripts/inference.py \
      --checkpoint checkpoints/best.pt \
      --image path/to/us_image.png

  # Folder
  python scripts/inference.py \
      --checkpoint checkpoints/best.pt \
      --input_dir data/new_cases/ \
      --output_dir outputs/predictions/

  # CSV
  python scripts/inference.py \
      --checkpoint checkpoints/best.pt \
      --csv data/splits/test.csv \
      --output_dir outputs/predictions/ \
      --gradcam \
      --save_masks
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from configs.thyformer_config import get_config
from models.thyformer_models import ThyFormer, build_model
from utils.thyformer_explainability import GradCAM, overlay

# ── TI-RADS label info ─────────────────────────────────────────────
TIRADS = {
    0: {"name": "T1", "desc": "Benign", "action": "No FNA needed"},
    1: {"name": "T2", "desc": "Not suspicious", "action": "No FNA needed"},
    2: {"name": "T3", "desc": "Mildly suspicious", "action": "FNA if ≥2.5 cm"},
    3: {"name": "T4", "desc": "Moderately suspicious", "action": "FNA if ≥1.5 cm"},
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(path: str, image_size: int = 720) -> torch.Tensor:
    """
    Load and preprocess a single ultrasound image.
    Returns: float32 tensor [1, 3, H, W] ready for the model.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Bilateral filter (speckle suppression, mirrors training preprocessing)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Resize
    img = cv2.resize(img, (image_size, image_size))

    # Normalise to float32 [0, 1], replicate to 3 channels
    img = img.astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=2)  # [H, W, 3]

    # ImageNet normalisation
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW → [1, 3, H, W]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.float()


def load_original_rgb(path: str, image_size: int = 720) -> np.ndarray:
    """Load original image as uint8 RGB [H, W, 3] for overlay."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    img = cv2.resize(img, (image_size, image_size))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def load_model(checkpoint_path: str, device: str = "cuda") -> Tuple[ThyFormer, dict]:
    """Load ThyFormer from checkpoint. Returns (model, checkpoint_dict)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model config from checkpoint if available
    if "config" in ckpt and hasattr(ckpt["config"], "model"):
        model_cfg = ckpt["config"].model
    else:
        model_cfg = get_config().model

    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    val_auc = metrics.get("val_auc", 0.0)
    print(f"Loaded checkpoint — epoch {epoch}  val_auc {val_auc:.4f}")
    return model, ckpt


@torch.no_grad()
def predict_single(
    model: ThyFormer,
    image_path: str,
    device: str = "cuda",
    fp16: bool = True,
    image_size: int = 720,
) -> Dict:
    """
    Run inference on one image.

    Returns dict:
        image_path     : str
        pred_class     : int   (0–3)
        pred_label     : str   ("T1"–"T4")
        pred_desc      : str   clinical description
        pred_action    : str   ACR TI-RADS recommended action
        confidence     : float (0–1)  softmax score for predicted class
        scores         : list  [T1_score, T2_score, T3_score, T4_score]
        echo_weights   : list  [hypo, iso, hyper] ECA activations
        seg_mask       : np.ndarray [H, W] binary mask (sigmoid > 0.5)
        inference_ms   : float
    """
    tensor = preprocess_image(image_path, image_size).to(device)

    t0 = time.perf_counter()
    with autocast(device_type=device, enabled=fp16):
        out = model(tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = F.softmax(out["cls_logits"], dim=-1)[0].cpu().numpy()
    pred_class = int(probs.argmax())
    seg_mask = (torch.sigmoid(out["seg_logits"])[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    echo_w = out["echo_weights"][0].cpu().numpy().tolist()

    return {
        "image_path": image_path,
        "pred_class": pred_class,
        "pred_label": TIRADS[pred_class]["name"],
        "pred_desc": TIRADS[pred_class]["desc"],
        "pred_action": TIRADS[pred_class]["action"],
        "confidence": float(probs[pred_class]),
        "scores": {
            "T1": float(probs[0]),
            "T2": float(probs[1]),
            "T3": float(probs[2]),
            "T4": float(probs[3]),
        },
        "echo_weights": {
            "hypoechoic": float(echo_w[0]),
            "isoechoic": float(echo_w[1]),
            "hyperechoic": float(echo_w[2]),
        },
        "seg_mask": seg_mask,
        "inference_ms": round(elapsed_ms, 2),
    }


def collect_image_paths(
    input_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    single: Optional[str] = None,
    image_root: Optional[str] = None,
) -> List[str]:
    """Collect all image paths from any of the three input modes."""
    if single:
        return [single]
    if csv_path:
        import pandas as pd

        df = pd.read_csv(csv_path)
        col = "image_path" if "image_path" in df.columns else "img_path"
        paths = df[col].tolist()
        if image_root:
            paths = [str(Path(image_root) / p) for p in paths]
        return paths
    if input_dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths = sorted([str(p) for p in Path(input_dir).rglob("*") if p.suffix.lower() in exts])
        return paths
    raise ValueError("Provide --image, --input_dir, or --csv")


def run_batch(
    model: ThyFormer,
    image_paths: List[str],
    output_dir: str,
    device: str = "cuda",
    fp16: bool = True,
    save_masks: bool = False,
    gradcam: bool = False,
    image_size: int = 720,
) -> List[Dict]:
    """
    Run inference on a list of images.
    Saves:
      - outputs/predictions/<stem>_mask.png  (if save_masks)
      - outputs/gradcam/<stem>_gradcam.png   (if gradcam)
      - outputs/report.csv

    Returns list of result dicts.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    gcam_dir = out_dir / "gradcam"
    if save_masks:
        mask_dir.mkdir(exist_ok=True)
    if gradcam:
        gcam_dir.mkdir(exist_ok=True)
        gcam_engine = GradCAM(model)

    results = []
    total = len(image_paths)

    print(f"\nRunning inference on {total} image(s) …")
    print(f"  Device: {device}  |  FP16: {fp16}  |  " f"GradCAM: {gradcam}  |  Masks: {save_masks}")
    print("─" * 55)

    for i, path in enumerate(image_paths):
        stem = Path(path).stem

        try:
            result = predict_single(model, path, device, fp16, image_size)
        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR {Path(path).name}: {e}")
            results.append({"image_path": path, "error": str(e)})
            continue

        # ── Save segmentation mask ────────────────────────────────
        if save_masks:
            mask_img = (result["seg_mask"] * 255).astype(np.uint8)
            cv2.imwrite(str(mask_dir / f"{stem}_mask.png"), mask_img)

        # ── Save GradCAM 3-panel ──────────────────────────────────
        if gradcam:
            tensor = preprocess_image(path, image_size).to(device)
            heatmap = gcam_engine.generate(tensor, result["pred_class"])
            orig = load_original_rgb(path, image_size)
            _save_gradcam_panel(orig, heatmap, result, gcam_dir / f"{stem}_gradcam.png")

        # ── Print progress ────────────────────────────────────────
        s = result["scores"]
        print(
            f"  [{i+1:4d}/{total}] {Path(path).name:<30s} → "
            f"{result['pred_label']} ({result['confidence']:.3f})  "
            f"[T1:{s['T1']:.2f} T2:{s['T2']:.2f} "
            f"T3:{s['T3']:.2f} T4:{s['T4']:.2f}]  "
            f"{result['inference_ms']:.1f}ms"
        )

        # Strip numpy array before storing (not CSV-serialisable)
        result_clean = {k: v for k, v in result.items() if k != "seg_mask"}
        results.append(result_clean)

    if gradcam:
        gcam_engine.remove()

    # ── Write CSV report ──────────────────────────────────────────
    _write_csv_report(results, out_dir / "report.csv")

    # ── Print summary ─────────────────────────────────────────────
    _print_summary(results, out_dir)

    return results


def _write_csv_report(results: List[Dict], csv_path: Path):
    """Write flat CSV report — one row per image."""
    fieldnames = [
        "image_path",
        "pred_label",
        "pred_desc",
        "pred_action",
        "confidence",
        "T1_score",
        "T2_score",
        "T3_score",
        "T4_score",
        "echo_hypo",
        "echo_iso",
        "echo_hyper",
        "inference_ms",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            if "error" in r:
                writer.writerow({"image_path": r["image_path"], "error": r["error"]})
                continue
            scores = r.get("scores", {})
            echo = r.get("echo_weights", {})
            writer.writerow(
                {
                    "image_path": r["image_path"],
                    "pred_label": r["pred_label"],
                    "pred_desc": r["pred_desc"],
                    "pred_action": r["pred_action"],
                    "confidence": f"{r['confidence']:.4f}",
                    "T1_score": f"{scores.get('T1', 0):.4f}",
                    "T2_score": f"{scores.get('T2', 0):.4f}",
                    "T3_score": f"{scores.get('T3', 0):.4f}",
                    "T4_score": f"{scores.get('T4', 0):.4f}",
                    "echo_hypo": f"{echo.get('hypoechoic', 0):.4f}",
                    "echo_iso": f"{echo.get('isoechoic', 0):.4f}",
                    "echo_hyper": f"{echo.get('hyperechoic', 0):.4f}",
                    "inference_ms": r["inference_ms"],
                    "error": "",
                }
            )
    print(f"\nReport saved → {csv_path}")


def _print_summary(results: List[Dict], out_dir: Path):
    """Print class distribution summary."""
    counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
    errors = 0
    times = []

    for r in results:
        if "error" in r:
            errors += 1
            continue
        counts[r["pred_label"]] += 1
        times.append(r["inference_ms"])

    valid = sum(counts.values())
    avg_t = sum(times) / len(times) if times else 0.0

    print("\n" + "=" * 55)
    print("  INFERENCE SUMMARY")
    print("=" * 55)
    print(f"  Total processed : {valid + errors}")
    print(f"  Errors          : {errors}")
    print(f"  Avg latency     : {avg_t:.1f} ms / image")
    print()
    for label, count in counts.items():
        pct = 100.0 * count / max(valid, 1)
        bar = "█" * int(pct / 4)
        print(f"  {label}  {count:5d}  ({pct:5.1f}%)  {bar}")
    print()
    print(f"  Output directory: {out_dir}")
    print("=" * 55)


def _save_gradcam_panel(orig: np.ndarray, heatmap: np.ndarray, result: Dict, save_path: Path):
    """Save a 3-panel figure: original | GradCAM heatmap | overlay."""
    ov = overlay(orig, heatmap)
    heatmap_rgb = plt.cm.jet(heatmap)[:, :, :3]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, img in zip(
        axes,
        ["Original US", "GradCAM", "Overlay"],
        [orig, heatmap_rgb, ov],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    s = result["scores"]
    fig.suptitle(
        f"{Path(result['image_path']).name}  "
        f"pred:{result['pred_label']} ({result['confidence']:.1%})  "
        f"[T1:{s['T1']:.2f} T2:{s['T2']:.2f} T3:{s['T3']:.2f} T4:{s['T4']:.2f}]",
        fontsize=10,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="ThyFormer inference — TI-RADS T1–T4 classification")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")

    # Input modes (choose one)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Path to a single image")
    g.add_argument("--input_dir", help="Directory of images")
    g.add_argument("--csv", help="CSV with image_path column")

    # Output
    p.add_argument(
        "--output_dir",
        default="outputs/inference",
        help="Where to save results (default: outputs/inference)",
    )

    # Options
    p.add_argument("--gradcam", action="store_true", help="Save GradCAM heatmap overlays")
    p.add_argument("--save_masks", action="store_true", help="Save predicted segmentation masks")
    p.add_argument("--no_fp16", action="store_true", help="Disable FP16 mixed precision")
    p.add_argument("--device", default=None, help="cuda / cpu  (auto-detected if not set)")
    p.add_argument("--image_size", type=int, default=720)
    p.add_argument(
        "--image_root", default=None, help="Root directory prepended to relative paths in --csv"
    )

    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = not args.no_fp16 and device == "cuda"

    # Load model
    model, _ = load_model(args.checkpoint, device)

    # Collect paths
    paths = collect_image_paths(
        input_dir=args.input_dir,
        csv_path=args.csv,
        single=args.image,
        image_root=args.image_root,
    )

    if not paths:
        print("No images found. Check your --image / --input_dir / --csv argument.")
        return

    # Single-image: print detailed result to terminal
    if args.image:
        result = predict_single(model, args.image, device, fp16, args.image_size)
        _print_single_result(result)
        # Still save report if output_dir given
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            _write_csv_report([result], Path(args.output_dir) / "report.csv")
            if args.gradcam:
                gcam = GradCAM(model)
                tensor = preprocess_image(args.image, args.image_size).to(device)
                heatmap = gcam.generate(tensor, result["pred_class"])
                orig = load_original_rgb(args.image, args.image_size)
                gcam_path = Path(args.output_dir) / f"{Path(args.image).stem}_gradcam.png"
                _save_gradcam_panel(orig, heatmap, result, gcam_path)
                print(f"GradCAM saved → {gcam_path}")
                gcam.remove()
    else:
        run_batch(
            model=model,
            image_paths=paths,
            output_dir=args.output_dir,
            device=device,
            fp16=fp16,
            save_masks=args.save_masks,
            gradcam=args.gradcam,
            image_size=args.image_size,
        )


def _print_single_result(r: Dict):
    """Detailed terminal output for a single image prediction."""
    print("\n" + "=" * 55)
    print("  THYFORMER PREDICTION")
    print("=" * 55)
    print(f"  Image       : {Path(r['image_path']).name}")
    print(f"  Prediction  : {r['pred_label']} — {r['pred_desc']}")
    print(f"  Confidence  : {r['confidence']:.1%}")
    print(f"  Action      : {r['pred_action']}")
    print()
    print("  Class probabilities:")
    for label, score in r["scores"].items():
        bar = "█" * int(score * 30)
        mark = " ←" if label == r["pred_label"] else ""
        print(f"    {label}  {score:.4f}  {bar}{mark}")
    print()
    print("  Echogenicity activations (ECA module):")
    for name, val in r["echo_weights"].items():
        print(f"    {name:<14s} {val:.4f}")
    print()
    print(f"  Inference time : {r['inference_ms']:.1f} ms")
    print("=" * 55)


if __name__ == "__main__":
    main()
