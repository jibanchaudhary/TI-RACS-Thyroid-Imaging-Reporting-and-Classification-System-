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

# ── ACR TI-RADS descriptors ──────────────────────────────────────────────────
# stanford_dataset/metadata.csv carries the radiologist's five ACR descriptors for
# all 192 clips with zero missing values, and its `ti_rads_level` agrees with the
# training labels on every clip. The final TI-RADS grade is a deterministic
# function of these five (sum the points, then band), so supervising them hands
# the model the decomposition of its own target instead of asking it to infer the
# whole rubric from one 5-way label over 134 clips.
DESCRIPTOR_COLS = [
    "ti_rads_composition",
    "ti_rads_echogenicity",
    "ti_rads_shape",
    "ti_rads_margin",
    "ti_rads_echogenicfoci",
]
# Column values are ACR *point* values, which are sparse (shape is only ever 0 or
# 3). Levels are the sorted observed values over the whole metadata file, so the
# value->class-index mapping is fixed and identical across every split and fold.
# Populated in place by load_descriptors — never rebound, so `from … import
# DESCRIPTOR_LEVELS` keeps seeing updates.
DESCRIPTOR_LEVELS: Dict[str, List[int]] = {}
# Widest descriptor, so all five fit one padded [5, MAX_BINS] one-hot tensor that
# mixup can interpolate in a single lerp.
MAX_DESCRIPTOR_BINS = 8


def load_descriptors(metadata_csv: str) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """
    Read metadata.csv into {clip_id: [5] class indices} plus the per-descriptor
    bin counts.

    `annot_id` is "8_" / "112_", so the clip id is the part before the underscore
    — the same id space as the frame stems.
    """
    md = pd.read_csv(metadata_csv)
    md["_clip"] = md["annot_id"].astype(str).str.split("_").str[0]

    DESCRIPTOR_LEVELS.clear()
    DESCRIPTOR_LEVELS.update({c: sorted(md[c].dropna().unique().tolist()) for c in DESCRIPTOR_COLS})
    bins = [len(DESCRIPTOR_LEVELS[c]) for c in DESCRIPTOR_COLS]
    if max(bins) > MAX_DESCRIPTOR_BINS:
        raise ValueError(f"descriptor with {max(bins)} levels exceeds MAX_DESCRIPTOR_BINS")

    lut = {c: {v: i for i, v in enumerate(DESCRIPTOR_LEVELS[c])} for c in DESCRIPTOR_COLS}
    table = {
        row["_clip"]: np.array([lut[c].get(row[c], -1) for c in DESCRIPTOR_COLS], dtype=np.int64)
        for _, row in md.iterrows()
    }
    return table, bins


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


def polygon_points(mask_str) -> Optional[np.ndarray]:
    """Parse the CSV's "x,y;x,y;…" polygon into [N,2] int32, or None if absent."""
    if mask_str is None or pd.isna(mask_str) or not str(mask_str).strip():
        return None
    try:
        pts = np.array(
            [[int(c) for c in p.split(",")] for p in str(mask_str).split(";") if "," in p],
            dtype=np.int32,
        )
        return pts if len(pts) >= 3 else None
    except (ValueError, TypeError):
        return None


def echogenicity_label(
    raw_gray: np.ndarray,
    pts: Optional[np.ndarray],
    iso_band: Tuple[float, float] = (0.90, 1.10),
    ring_px: int = 40,
) -> int:
    """
    Pseudo-label the nodule's echogenicity: 0 hypo / 1 iso / 2 hyper.

    Echogenicity is defined *relative to adjacent thyroid parenchyma*, so this
    compares the mean intensity inside the polygon against a ring of the same
    image just outside it, and bins the ratio.

    Takes the RAW grayscale frame, never the CLAHE'd one: CLAHE equalises within
    8x8 tiles, which is precisely the local intensity relationship echogenicity
    is measured on. Running this after preprocess_us would read the equaliser's
    output rather than the tissue.

    Returns 1 (iso, the neutral bin) when there is no polygon to measure.
    """
    if pts is None:
        return 1
    h, w = raw_gray.shape[:2]
    inside = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(inside, [pts], 1)
    if inside.sum() == 0:
        return 1

    k = 2 * ring_px + 1
    dilated = cv2.dilate(inside, np.ones((k, k), np.uint8))
    ring = dilated - inside
    if ring.sum() < 32:  # nodule fills the frame; no parenchyma to compare against
        return 1

    g = raw_gray.astype(np.float32)
    mean_in = float(g[inside > 0].mean())
    mean_ring = float(g[ring > 0].mean())
    if mean_ring < 1e-3:
        return 1
    ratio = mean_in / mean_ring
    lo, hi = iso_band
    return 0 if ratio < lo else (2 if ratio > hi else 1)


def roi_bbox(pts: Optional[np.ndarray], shape: Tuple[int, int], pad: float = 1.6):
    """
    Nodule bbox expanded by `pad` about its centre and clipped to the frame.
    Returns (x0, y0, x1, y1), or the full frame when there is no polygon.
    """
    h, w = shape[:2]
    if pts is None:
        return 0, 0, w, h
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = max(x1 - x0, 8) * pad / 2.0, max(y1 - y0, 8) * pad / 2.0
    return (
        int(max(cx - bw, 0)),
        int(max(cy - bh, 0)),
        int(min(cx + bw, w)),
        int(min(cy + bh, h)),
    )


def _resize_ops(size: int, mode: str = "letterbox") -> List:
    """
    Geometry stage of the transform.

    "stretch" is the legacy A.Resize(size, size): on 802x1054 frames it compresses
    the x axis 1.3x more than the y axis, so a taller-than-wide nodule can come
    out wider-than-tall — and taller-than-wide is a scored ACR TI-RADS shape
    criterion, worth a point on its own. "letterbox" scales the long side and
    zero-pads the short one (black is the US background), leaving the ratio
    intact.
    """
    if mode == "stretch":
        return [A.Resize(size, size)]
    return [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,  # albumentations >=2.0 renamed value/mask_value to fill/fill_mask
            fill_mask=0,
            position="center",
        ),
    ]


def build_train_transforms(
    cfg: AugmentationConfig, size: int = 224, resize_mode: str = "letterbox"
) -> A.Compose:
    return A.Compose(
        [
            *_resize_ops(size, resize_mode),
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
            # Ultrasound speckle noise injection.
            # albumentations >=2.0 dropped `var_limit` (a variance in 0-255^2
            # units) for `std_range` (a std as a FRACTION of the max value), and
            # silently ignores the old kwarg — so cfg.speckle_noise_var was doing
            # nothing and the transform ran at its default strength. sqrt() takes
            # the configured variance to the std the new API wants.
            A.GaussNoise(
                std_range=(0.0, float(cfg.speckle_noise_var) ** 0.5),
                p=cfg.speckle_noise_p,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def build_val_transforms(size: int = 224, resize_mode: str = "letterbox") -> A.Compose:
    return A.Compose(
        [
            *_resize_ops(size, resize_mode),
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
        data_cfg: Optional[DataConfig] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.root = Path(data_root)
        self.transform = transform
        self.medsam_dir = Path(medsam_masks_dir) if medsam_masks_dir else None
        self.aug_cfg = aug_cfg or AugmentationConfig()
        self.is_train = is_train
        self.image_size = image_size
        self.has_masks = "mask_path" in self.df.columns
        self.data_cfg = data_cfg or DataConfig()
        # Frames of one clip are near-duplicates of a single nodule, so the clip
        # — not the frame — is the independent unit for sampling and for CV
        # folds. Same "138_69" -> "138" convention as utils.thyformer_metrics.
        self.clips = np.array([str(Path(p).stem).split("_")[0] for p in self.df["img_path"]])
        self.labels = np.array([TIRADS_MAP[x] for x in self.df["label"]])

        # ACR descriptor targets, joined by clip. -1 marks "no annotation" and is
        # ignored by the loss, so a partially-annotated dataset degrades to
        # supervising only the frames it can.
        self.descriptors = None
        self.descriptor_bins = None
        if self.data_cfg.echo_source == "metadata" and self.data_cfg.metadata_csv:
            table, self.descriptor_bins = load_descriptors(self.data_cfg.metadata_csv)
            miss = np.full(len(DESCRIPTOR_COLS), -1, dtype=np.int64)
            self.descriptors = np.stack([table.get(c, miss) for c in self.clips])
            n_cov = int((self.descriptors[:, 0] >= 0).sum())
            if n_cov < len(self.df):
                print(
                    f"{len(self.df) - n_cov}/{len(self.df)} frames have no ACR descriptor "
                    f"row in {self.data_cfg.metadata_csv}; their aux loss is masked out.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def __len__(self) -> int:
        return len(self.df)

    def _load_raw(self, rel_path: str) -> np.ndarray:
        path = self.root / rel_path
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img

    def _enhance(self, raw: np.ndarray) -> np.ndarray:
        """CLAHE + bilateral, replicated to 3 channels as uint8."""
        img = preprocess_us(raw, self.aug_cfg)  # float32 [H,W] in [0,1]
        img = np.stack([img, img, img], axis=-1)  # [H,W,3]
        return (img * 255).astype(np.uint8)

    @staticmethod
    def _fill_mask(pts: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if pts is not None:
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def _load_boundary(self, stem: str) -> np.ndarray:
        if self.medsam_dir is not None:
            p = self.medsam_dir / f"{stem}.npy"
            if p.exists():
                return np.load(str(p)).astype(np.float32)
        return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        stem = Path(row["img_path"]).stem
        raw = self._load_raw(row["img_path"])
        pts = polygon_points(row.get("mask", ""))
        label = int(self.labels[idx])

        # ACR descriptors: the radiologist's labels when metadata.csv is
        # available, otherwise the intensity-derived echogenicity fallback (read
        # off the RAW frame, before CLAHE — see echogenicity_label — and before
        # any crop, so the parenchyma ring is still in view).
        if self.descriptors is not None:
            desc = self.descriptors[idx]
        elif self.data_cfg.echo_source == "derived":
            e = echogenicity_label(
                raw, pts, self.data_cfg.echo_iso_band, self.data_cfg.echo_ring_px
            )
            desc = np.full(len(DESCRIPTOR_COLS), -1, dtype=np.int64)
            desc[DESCRIPTOR_COLS.index("ti_rads_echogenicity")] = e
        else:
            desc = np.full(len(DESCRIPTOR_COLS), -1, dtype=np.int64)

        image = self._enhance(raw)
        mask = self._fill_mask(pts, image.shape)
        if self.data_cfg.roi_crop == "crop":
            x0, y0, x1, y1 = roi_bbox(pts, image.shape, self.data_cfg.roi_pad)
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                image, mask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

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
            "descriptors": torch.from_numpy(np.ascontiguousarray(desc)),  # [5], -1 = missing
            "mask": mask.unsqueeze(0),  # [1,H,W]
            "boundary": boundary,  # [1,H,W]
            "stem": stem,
        }


def mixup_collate(batch: List[Dict], alpha: float = 0.2, p: float = 0.3, num_classes: int = 4):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    boundaries = torch.stack([b["boundary"] for b in batch])
    desc = torch.stack([b["descriptors"] for b in batch])  # [B,5], -1 = missing
    stems = [b["stem"] for b in batch]

    one_hot = torch.zeros(len(batch), num_classes).scatter_(1, labels.unsqueeze(1), 1.0)
    desc_oh = _descriptor_one_hot(desc)

    if random.random() < p:
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(len(batch))
        images = lam * images + (1 - lam) * images[perm]
        masks = lam * masks + (1 - lam) * masks[perm]
        boundaries = lam * boundaries + (1 - lam) * boundaries[perm]
        one_hot_p = torch.zeros_like(one_hot).scatter_(1, labels[perm].unsqueeze(1), 1.0)
        one_hot = lam * one_hot + (1 - lam) * one_hot_p
        # The ddata_cfgescriptors describe the blended image too, so they mix on the same
        # lambda — otherwise the aux heads are trained against the un-blended
        # frame's findings. A missing descriptor stays all-zero on its row, so a
        # blend of annotated and unannotated partially supervises, which is what
        # the row-sum mask in the loss expects.
        desc_oh = lam * desc_oh + (1 - lam) * _descriptor_one_hot(desc[perm])

    return {
        "image": images,
        "label": one_hot,
        "descriptors": desc_oh,
        "mask": masks,
        "boundary": boundaries,
        "stem": stems,
    }


def _descriptor_one_hot(desc: torch.Tensor) -> torch.Tensor:
    """
    [B,5] class indices (-1 = missing) → [B,5,MAX_DESCRIPTOR_BINS] one-hot.

    Padding every descriptor to one width lets mixup interpolate all five with a
    single lerp, and lets a missing descriptor be encoded as an all-zero row that
    the loss masks by row sum rather than by a sentinel it has to special-case.
    """
    b, d = desc.shape
    oh = torch.zeros(b, d, MAX_DESCRIPTOR_BINS, dtype=torch.float32)
    valid = desc >= 0
    if valid.any():
        idx = desc.clamp_min(0).unsqueeze(-1)
        oh.scatter_(2, idx, valid.unsqueeze(-1).float())
    return oh


def _cap_clip_share(
    w: np.ndarray, clips: np.ndarray, max_share: float, iters: int = 50
) -> np.ndarray:
    """
    Iteratively cap any clip's total sampling probability at `max_share`,
    redistributing the surplus proportionally over the uncapped clips.

    Needed because class balancing has no answer for a class backed by a single
    clip — TR-1 is one clip, so a perfectly class-balanced sampler still spends
    20% of every epoch on that one nodule and the model memorises it. Repeated
    because redistributing surplus can push another clip over the cap.
    """
    if max_share is None or max_share <= 0 or max_share >= 1:
        return w
    w = w / w.sum()
    uniq, inv = np.unique(clips, return_inverse=True)
    if max_share * len(uniq) <= 1.0:  # cap is below uniform; nothing to enforce
        return w
    for _ in range(iters):
        share = np.bincount(inv, weights=w, minlength=len(uniq))
        over = share > max_share + 1e-12
        if not over.any():
            break
        # Rescale each over-cap clip's frames down to exactly the cap, then give
        # the freed mass to the rest in proportion to what they already hold.
        scale = np.ones(len(uniq))
        scale[over] = max_share / share[over]
        w = w * scale[inv]
        freed = 1.0 - w.sum()
        under = ~over
        pool = float(w[under[inv]].sum())
        if pool <= 0 or freed <= 0:
            break
        w[under[inv]] *= 1.0 + freed / pool
    return w / w.sum()


def build_weighted_sampler(
    ds: ThyroidDataset, mode: str = "clip", max_clip_share: float = 0.0
) -> WeightedRandomSampler:
    """
    Class-balanced sampler over the training frames.

    "frame" (legacy) weights each frame by 1/count(its class). Because the draw
    count is len(ds), every class receives len(ds)/5 draws per epoch regardless of
    how many *clips* back it — and TR-1 is a single clip of 52 frames. That gave
    each of those 52 frames ~45 appearances per epoch, making one nodule 20% of
    every epoch while TR-4's 58 clips were seen 0.5x. The model memorises the
    clip, not the class.

    "clip" splits the weight budget twice: uniformly across classes, then
    uniformly across the clips within a class, then uniformly across the frames
    within a clip. A class is still balanced, but no clip can dominate it, and
    frame count no longer buys a clip extra exposure.
    """
    labels, clips = ds.labels, ds.clips
    if mode == "frame":
        counts = np.bincount(labels, minlength=5)
        w = (1.0 / np.maximum(counts, 1))[labels].astype(np.float64)
    else:
        n_cls = np.bincount(labels, minlength=5)
        clips_per_class = {c: len(np.unique(clips[labels == c])) for c in range(5) if n_cls[c] > 0}
        frames_per_clip = dict(zip(*np.unique(clips, return_counts=True)))
        w = np.array(
            [
                1.0 / (clips_per_class[lab] * frames_per_clip[clip])
                for lab, clip in zip(labels, clips)
            ],
            dtype=np.float64,
        )

    w = _cap_clip_share(w, clips, max_clip_share)
    return WeightedRandomSampler(torch.from_numpy(w).float(), len(labels), replacement=True)


def build_dataloaders(
    data_cfg: DataConfig,
    aug_cfg: AugmentationConfig,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    train_tf = build_train_transforms(aug_cfg, data_cfg.image_size, data_cfg.resize_mode)
    val_tf = build_val_transforms(data_cfg.image_size, data_cfg.resize_mode)

    def _ds(csv, tf, is_train):
        return ThyroidDataset(
            csv,
            data_cfg.data_root,
            tf,
            data_cfg.medsam_masks_dir,
            aug_cfg,
            is_train,
            image_size=data_cfg.image_size,
            data_cfg=data_cfg,
        )

    train_ds = _ds(data_cfg.train_csv, train_tf, True)
    val_ds = _ds(data_cfg.val_csv, val_tf, False)
    test_ds = _ds(data_cfg.test_csv, val_tf, False)

    sampler = (
        None
        if data_cfg.sampler == "none"
        else build_weighted_sampler(train_ds, data_cfg.sampler, data_cfg.max_clip_share)
    )
    print(
        f"[dataloader] resize={data_cfg.resize_mode}  sampler={data_cfg.sampler}  "
        f"roi_crop={data_cfg.roi_crop}  "
        f"train={len(train_ds)} frames / {len(np.unique(train_ds.clips))} clips"
    )

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
        shuffle=sampler is None,
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
