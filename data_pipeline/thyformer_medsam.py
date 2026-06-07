import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

MEDSAM_CKPT = "checkpoints/medsam_vit_b.pth"


def create_masks(dataset_dir: str, masks_dir: str):
    dataset_dir = Path(dataset_dir)
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg"}
    total_created = 0
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        out_class_dir = masks_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for json_path in class_dir.glob("*.json"):
            stem = json_path.stem

            # Find matching image
            image_path = None
            for ext in image_exts:
                candidate = class_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                print(f"  Skipping {json_path}: image not found")
                continue

            # Read annotation
            with open(json_path, "r") as f:
                ann = json.load(f)

            polygon = ann.get("polygon", [])
            if not polygon:
                print(f"  Skipping {json_path}: polygon missing")
                continue

            # Read image for dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  Skipping {image_path}: cannot read image")
                continue

            # Create binary mask from polygon
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(
                [[p["x"], p["y"]] for p in polygon],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 255)

            out_path = out_class_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), mask)
            total_created += 1

    print(f"  Created {total_created} masks → {masks_dir}/")
    return masks_dir


def collect_image_files(directory: Path):
    """
    Recursively collect all image files from a directory.
    Supports both flat (images/) and nested (class_A/, class_B/) layouts.
    """
    exts = {".png", ".jpg", ".jpeg"}
    files = []
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() in exts:
            files.append(p)
    return files


def find_mask_for_image(image_path: Path, masks_dir: Path) -> Optional[Path]:
    """
    Find the corresponding mask for an image, checking both flat and nested layouts.
    Tries: masks_dir/{stem}.png, masks_dir/{parent_name}/{stem}.png
    """
    stem = image_path.stem
    parent_name = image_path.parent.name

    c = masks_dir / parent_name / f"{stem}.png"
    if c.exists():
        return c

    return None


def extract_boundary(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Extract boundary of a binary mask via morphological dilation - erosion.
    Returns float32 [H,W] boundary map.
    """
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = (dilated.astype(np.float32) - eroded.astype(np.float32)) / 255.0
    return boundary.clip(0, 1)


def run_medsam_on_image(model, image_path: Path, device: str = "cuda") -> np.ndarray:
    """
    Run MedSAM on a single image and return a binary mask.
    Falls back to empty mask if MedSAM is unavailable.
    """
    try:
        import torch.nn.functional as F

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((224, 224), dtype=np.float32)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)

        img_1024 = F.interpolate(img_t, size=(1024, 1024), mode="bilinear", align_corners=False)

        H, W = img.shape
        box = torch.tensor([[0, 0, W, H]], dtype=torch.float32).to(device)

        with torch.no_grad():
            image_embed = model.image_encoder(img_1024)
            sparse_emb, dense_emb = model.prompt_encoder(points=None, boxes=box, masks=None)
            low_res_logits, _ = model.mask_decoder(
                image_embeddings=image_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
        mask = torch.sigmoid(low_res_logits[0, 0]).cpu().numpy()
        mask = cv2.resize(mask, (224, 224))
        return (mask > 0.5).astype(np.float32)

    except Exception as e:
        print(f"  MedSAM failed for {image_path.name}: {e} — using empty mask")
        return np.zeros((224, 224), dtype=np.float32)


def precompute(
    images_dir: str,
    masks_dir: Optional[str],
    out_dir: str,
    device: str = "cuda",
):
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_path = Path(masks_dir) if masks_dir else None

    # Try to load MedSAM
    model = None
    ckpt = Path(MEDSAM_CKPT)
    if ckpt.exists():
        try:
            sys.path.insert(0, ".")
            from segment_anything import sam_model_registry

            model = sam_model_registry["vit_b"](checkpoint=str(ckpt))
            model = model.to(device).eval()
            print(f"Loaded MedSAM from {ckpt}")
        except ImportError:
            print("WARNING: segment_anything not installed — using GT masks only")
    else:
        print(f"WARNING: MedSAM checkpoint not found at {ckpt}")
        print("         Using morphological boundary extraction from GT masks.")

    image_files = collect_image_files(images_dir)
    print(f"Processing {len(image_files)} images → {out_dir}/")

    created, skipped = 0, 0
    for i, img_path in enumerate(image_files):
        stem = img_path.stem
        out_path = out_dir / f"{stem}.npy"

        if out_path.exists():
            skipped += 1
            continue

        mask = None
        if masks_path:
            mask_file = find_mask_for_image(img_path, masks_path)
            if mask_file:
                m = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    mask = (cv2.resize(m, (224, 224)) > 127).astype(np.float32)

        # Fall back to MedSAM if no mask is found
        if mask is None and model is not None:
            mask = run_medsam_on_image(model, img_path, device)
        elif mask is None:
            mask = np.zeros((224, 224), dtype=np.float32)

        boundary = extract_boundary(mask)
        np.save(str(out_path), boundary)
        created += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(image_files)}] done")

    print(f"Done. Created {created} boundary maps, skipped {skipped} existing → {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Precompute MedSAM boundary maps for ThyFormer training"
    )
    p.add_argument(
        "--data_dir",
        default=None,
        help="Single dataset directory containing class folders with images + JSON annotations. "
        "Masks will be auto-generated from JSON polygons. "
        "If provided, --images_dir and --masks_dir are derived automatically.",
    )
    p.add_argument("--images_dir", default=None, help="Image directory (if not using --data_dir)")
    p.add_argument(
        "--masks_dir", default=None, help="Pre-existing masks directory (if not using --data_dir)"
    )
    p.add_argument("--out_dir", default="data/medsam_boundaries")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if a.data_dir:
        data_dir = Path(a.data_dir)
        auto_masks_dir = data_dir.parent / "generated_masks"

        print(f"═══ Step 1: Generating masks from JSON annotations in {data_dir} ═══")
        create_masks(str(data_dir), str(auto_masks_dir))
        # Step 2: Precompute boundaries using the generated masks
        print("\n═══ Step 2: Computing boundary maps ═══")
        precompute(str(data_dir), str(auto_masks_dir), a.out_dir, a.device)

    elif a.images_dir:
        precompute(a.images_dir, a.masks_dir, a.out_dir, a.device)

    else:
        p.error("Provide either --data_dir or --images_dir")
