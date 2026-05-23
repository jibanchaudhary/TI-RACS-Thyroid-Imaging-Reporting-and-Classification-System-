import os
import glob

import cv2
import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


from data_pipeline.dataset_conversion import (
    build_train_transforms,
    build_val_transforms,
    parse_binary_xml_case,
)

CLASS_NAMES = ["Benign", "Malignant"]
NUM_CLASSES = 2


BACKBONE_SIZE = {
    "convnext": 224,
    "efficientnet": 300,
    "swin": 224,
    "vit": 224,
}


class BinaryThyroidDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        backbone: str = "convnext",
        crop_nodule: bool = True,
        padding: int = 24,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 17,
    ):
        assert split in ("train", "val", "test")
        assert backbone in BACKBONE_SIZE, f"backbone must be one of {list(BACKBONE_SIZE.keys())}"

        self.data_dir = data_dir
        self.split = split
        self.backbone = backbone
        self.crop = crop_nodule
        self.padding = padding
        self.img_size = BACKBONE_SIZE[backbone]

        if split == "train":
            self.transform = build_train_transforms(self.img_size)
        elif split == "val":
            self.transform = build_val_transforms(self.img_size)
        elif split == "test":
            self.transform = build_val_transforms(self.img_size)

        all_cases = self._load_all_cases()
        self.cases = self._split(all_cases, train_ratio, val_ratio, seed, split)

        print(
            f"[ThyroidDataset] {split:5s} | {len(self.cases):4d} cases | "
            f"backbone={backbone} | img_size={self.img_size}"
        )
        self._print_class_dist()

    def _load_all_cases(self) -> List[dict]:
        ann_dir = os.path.join(self.data_dir, "annotations")
        img_dir = os.path.join(self.data_dir, "images")

        xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        if not xml_files:
            raise FileNotFoundError(f"No xml files found in the {ann_dir}")

        cases = []
        for xml_path in xml_files:
            case = parse_binary_xml_case(xml_path)
            if case is None:
                continue

            img_id = case["case_id"]
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                img_matches = glob.glob(os.path.join(img_dir, f"{img_id}*{ext}"))
                img_path = img_matches[0]
                break

            if img_path is None:
                continue

            case["img_path"] = img_path

            cases.append(case)
        if not cases:
            raise RuntimeError("No valid cases found. Check data_dir layout.")
        return cases

    @staticmethod
    def _split(cases, train_r, val_r, seed, which):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(cases))
        n = len(cases)
        n_tr = int(n * train_r)
        n_va = int(n * val_r)
        splits = {
            "train": idx[:n_tr],
            "val": idx[n_tr : n_tr + n_va],
            "test": idx[n_tr + n_va :],
        }
        return [cases[i] for i in splits[which]]

    def _print_class_dist(self):
        dist = None
        from collections import Counter

        counts = Counter(c["label"] for c in self.cases)
        dist = " | ".join(f"{CLASS_NAMES[k]}: {counts[k]}" for k in sorted(counts))
        print(f"  Class dist → {dist}")

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        from collections import Counter

        labels = [c["label"] for c in self.cases]
        counts = Counter(labels)
        weights = [1.0 / counts[label] for label in labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        case = self.cases[idx]

        # 1. Load image as RGB numpy array
        img_bgr = cv2.imread(case["img_path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Cannot read image: {case['img_path']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        if "bbox" in case:
            x_min, y_min, x_max, y_max = case["bbox"]

        if self.crop:
            img_rgb = img_rgb[y_min:y_max, x_min:x_max]

        transformed = self.transform(image=img_rgb)
        image_tensor = transformed["image"]  # shape [3, H, W] float32

        return image_tensor, case["label"]


if __name__ == "__main__":
    data_dir = "/home/jiban/Documents/TI-RACS/anees/dataset/filtered_dataset"

    train_dataset = BinaryThyroidDataset(
        data_dir=data_dir,
        split="train",
        backbone="convnext",
        crop_nodule=True,
        padding=24,
    )

    val_dataset = BinaryThyroidDataset(
        data_dir=data_dir,
        split="val",
        backbone="convnext",
        crop_nodule=True,
        padding=24,
    )

    test_dataset = BinaryThyroidDataset(
        data_dir=data_dir,
        split="test",
        backbone="convnext",
        crop_nodule=True,
        padding=24,
    )
