"""
ThyFormer — Data Pipeline
Dataset, preprocessing, augmentation, and DataLoader factory.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from configs.thyformer_config import AugmentationConfig, DataConfig

TIRADS_MAP = {
    "TR-1": 0,
    "TR-2": 1,
    "TR-3": 2,
    "TR-4": 3,
    "TR-5": 4,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_clahe(image: np.ndarray, clip: float = 2.0, grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast-limited AHE for low-contrast ultrasound regions."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(image)


def apply_bilateral(image: np.ndarray) -> np.ndarray:
    """Edge-preserving bilateral filter to suppress speckle noise."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def preprocess_us(image: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
    """Full preprocessing: CLAHE → bilateral → normalise to float32 [0,1]."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)
    image = apply_clahe(image, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    image = apply_bilateral(image)
    return image.astype(np.float32) / 255.0


def build_train_transforms(cfg: AugmentationConfig, size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=cfg.random_flip_p),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=cfg.random_rotate_limit, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.ElasticTransform(
                alpha=cfg.elastic_alpha,
                sigma=cfg.elastic_sigma,
                p=cfg.elastic_p,
                border_mode=cv2.BORDER_REFLECT,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit, contrast_limit=cfg.contrast_limit, p=0.4
            ),
            # Ultrasound speckle noise injection (multiplicative Gaussian)
            A.GaussNoise(var_limit=(0, int(cfg.speckle_noise_var * 255**2)), p=cfg.speckle_noise_p),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def build_val_transforms(size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


class ThyroidDataset(Dataset):
    """
    Thyroid ultrasound dataset for TI-RADS T1–T4 classification.

    CSV columns required:
        image_path  – path to image (relative to data_root)
        label       – integer 0..3 (T1..T4)
    Optional:
        mask_path   – binary nodule segmentation mask
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        transform: Optional[A.Compose],
        medsam_masks_dir: Optional[str] = None,
        aug_cfg: Optional[AugmentationConfig] = None,
        is_train: bool = True,
        image_size: int = 224,
    ):
        self.df = pd.read_csv(csv_path)
        self.root = Path(data_root)
        self.transform = transform
        self.medsam_dir = Path(medsam_masks_dir) if medsam_masks_dir else None
        self.aug_cfg = aug_cfg or AugmentationConfig()
        self.is_train = is_train
        self.image_size = image_size
        self.has_masks = "mask_path" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> np.ndarray:
        path = self.root / rel_path
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = preprocess_us(img, self.aug_cfg)  # float32 [H,W]
        img = np.stack([img, img, img], axis=-1)  # [H,W,3]
        return (img * 255).astype(np.uint8)

    def _load_mask(self, row: pd.Series, img_shape) -> np.ndarray:
        img_h, img_w = img_shape
        mask_str = row.get("mask", "")
        if pd.isna(mask_str) or not mask_str:
            return np.zeros((img_h, img_w), dtype=np.uint8)

        try:
            pairs = mask_str.split(";")
            pts = np.array(
                [[int(c) for c in p.split(",")] for p in pairs if "," in p],
                dtype=np.int32,
            )
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            return (mask > 127).astype(np.uint8)
        except Exception:
            return np.zeros((img_h, img_w), dtype=np.uint8)

    def _load_boundary(self, stem: str) -> np.ndarray:
        if self.medsam_dir is not None:
            p = self.medsam_dir / f"{stem}.npy"
            if p.exists():
                return np.load(str(p)).astype(np.float32)
        return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        stem = Path(row["img_path"]).stem
        image = self._load_image(row["img_path"])
        mask = self._load_mask(row, image.shape[:2])
        label = TIRADS_MAP[row["label"]]

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]  # Tensor [3,H,W]
            mask = aug["mask"]  # Tensor [H,W]

        boundary = (
            torch.from_numpy(
                cv2.resize(self._load_boundary(stem), (self.image_size, self.image_size))
            )
            .float()
            .unsqueeze(0)
        )

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        else:
            mask = mask.float()

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask.unsqueeze(0),  # [1,H,W]
            "boundary": boundary,  # [1,H,W]
            "stem": stem,
        }


def mixup_collate(batch: List[Dict], alpha: float = 0.2, p: float = 0.3, num_classes: int = 4):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    boundaries = torch.stack([b["boundary"] for b in batch])
    stems = [b["stem"] for b in batch]

    one_hot = torch.zeros(len(batch), num_classes).scatter_(1, labels.unsqueeze(1), 1.0)

    if random.random() < p:
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(len(batch))
        images = lam * images + (1 - lam) * images[perm]
        masks = lam * masks + (1 - lam) * masks[perm]
        boundaries = lam * boundaries + (1 - lam) * boundaries[perm]
        one_hot_p = torch.zeros_like(one_hot).scatter_(1, labels[perm].unsqueeze(1), 1.0)
        one_hot = lam * one_hot + (1 - lam) * one_hot_p

    return {"image": images, "label": one_hot, "mask": masks, "boundary": boundaries, "stem": stems}


def build_weighted_sampler(ds: ThyroidDataset) -> WeightedRandomSampler:
    labels = ds.df["label"].values
    numeric_labels = np.array([TIRADS_MAP[label] for label in labels])
    counts = np.bincount(numeric_labels, minlength=5)
    weights = 1.0 / np.maximum(counts, 1)
    s_weights = torch.from_numpy(weights[numeric_labels]).float()
    return WeightedRandomSampler(s_weights, len(s_weights), replacement=True)


def build_dataloaders(
    data_cfg: DataConfig,
    aug_cfg: AugmentationConfig,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    train_tf = build_train_transforms(aug_cfg, data_cfg.image_size)
    val_tf = build_val_transforms(data_cfg.image_size)

    train_ds = ThyroidDataset(
        data_cfg.train_csv,
        data_cfg.data_root,
        train_tf,
        data_cfg.medsam_masks_dir,
        aug_cfg,
        True,
        image_size=data_cfg.image_size,
    )
    val_ds = ThyroidDataset(
        data_cfg.val_csv,
        data_cfg.data_root,
        val_tf,
        data_cfg.medsam_masks_dir,
        aug_cfg,
        False,
        image_size=data_cfg.image_size,
    )
    test_ds = ThyroidDataset(
        data_cfg.test_csv,
        data_cfg.data_root,
        val_tf,
        data_cfg.medsam_masks_dir,
        aug_cfg,
        False,
        image_size=data_cfg.image_size,
    )

    sampler = build_weighted_sampler(train_ds)

    # Mixup needs enough samples in a micro-batch to blend distinct images. Below
    # aug_cfg.mixup_min_batch_size it is turned off (mixup_p = 0) and the collate
    # still runs, so batches keep their one-hot soft-label format and the rest of
    # the pipeline is unchanged. Note this is the DataLoader batch size, not the
    # accumulated effective batch — collate_fn only ever sees one micro-batch.
    min_bs = getattr(aug_cfg, "mixup_min_batch_size", 0)
    mixup_p = aug_cfg.mixup_p
    if batch_size < min_bs and mixup_p > 0:
        print(
            f"[dataloader] mixup disabled: batch_size={batch_size} < "
            f"mixup_min_batch_size={min_bs} (randperm({batch_size}) is too "
            f"degenerate to regularise; it only injects label noise)"
        )
        mixup_p = 0.0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=lambda b: mixup_collate(b, aug_cfg.mixup_alpha, mixup_p, data_cfg.num_classes),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
