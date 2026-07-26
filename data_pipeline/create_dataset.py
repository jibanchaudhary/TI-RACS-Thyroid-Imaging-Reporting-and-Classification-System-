import os
import glob

import cv2
import json
import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


from data_pipeline.dataset_conversion import (
    build_train_transforms,
    build_val_transforms,
    polygon_to_mask,
    crop_nodule,
)


########Constants####################
TIRADS_MAP = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "4a": 3,
    "4b": 3,
    "4c": 3,
    "5": 4,
}
CLASS_NAMES = ["TR-1", "TR-2", "TR-3", "TR-4", "TR-5"]
NUM_CLASSES = 5


BACKBONE_SIZE = {
    "convnext": 720,
    "efficientnet": 720,
    "swin": 720,
    "vit": 720,
}


class ThyroidDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        backbone: str = "convnext",
        crop_nodule: bool = False,
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
        ann_dir = os.path.join(self.data_dir, "jsons")
        img_dir = os.path.join(self.data_dir, "images")

        json_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))

        if not json_files:
            raise FileNotFoundError(f"No json files found in {ann_dir}")

        cases = []

        for json_path in json_files:
            with open(json_path, "r") as f:
                content = json.load(f)

            label = TIRADS_MAP[str(content["tirads_raw"])]
            content["label"] = label

            img_path = os.path.join(img_dir, f'{content["frame_id"]}.jpg')
            content["img_path"] = img_path
            cases.append(content)

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

        # 2. Build binary mask from polygon
        mask = polygon_to_mask(case["polygon"], h, w)

        # 3. (Optional) crop to nodule ROI
        if self.crop:
            img_rgb, mask = crop_nodule(img_rgb, mask, self.padding)

        # 4. Apply transforms (resize, augment, normalize → tensor)
        transformed = self.transform(image=img_rgb)
        image_tensor = transformed["image"]  # shape [3, H, W] float32

        return image_tensor, case["label"]


if __name__ == "__main__":
    data_dir = "/home/jiban/Documents/TI-RACS/anees/dataset/filtered_dataset"

    # train_dataset = ThyroidDataset(
    #     data_dir=data_dir,
    #     split="train",
    #     backbone="convnext",
    #     crop_nodule=True,
    #     padding=24,
    # )

    val_dataset = ThyroidDataset(
        data_dir=data_dir,
        split="val",
        backbone="convnext",
        crop_nodule=True,
        padding=24,
    )

    # test_dataset = ThyroidDataset(
    #     data_dir=data_dir,
    #     split="test",
    #     backbone="convnext",
    #     crop_nodule=True,
    #     padding=24,
    # )
    # export_dataset_to_json(
    #     train_dataset, output_dir="/home/jiban/Documents/TI-RACS/artifacts/test_v1/json_dataset"
    # )

    # export_dataset_to_json(
    #     val_dataset, output_dir="/home/jiban/Documents/TI-RACS/artifacts/test_v1/json_dataset"
    # )

    # export_dataset_to_json(
    #     test_dataset, output_dir="/home/jiban/Documents/TI-RACS/artifacts/test_v1/json_dataset"
    # )
    # print("\nFirst case:")
    # print(dataset.cases[0])
