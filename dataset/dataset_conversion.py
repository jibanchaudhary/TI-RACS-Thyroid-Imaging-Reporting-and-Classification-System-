import os
import json
import glob

import cv2
import numpy as np

import torch

import albumentations as albun
from albumentations.pytorch import ToTensorV2

import xml.etree.ElementTree as ET

######## ImageNet statistics ##########
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



TIRADS_MAP = {
    "":   0,
    "1":  1,
    "2":  2,
    "3":  3,
    "4":  4, "4a": 4, "4b": 4, "4c": 4,
    "5":  5,
}

def build_train_transforms(img_size: int) -> albun.Compose:
    return albun.Compose([
        albun.Resize(img_size, img_size),
        albun.HorizontalFlip(p=0.5),
        albun.VerticalFlip(p=0.3),
        albun.Rotate(limit=15, p=0.5),
        albun.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=0, p=0.4),
        # Ultrasound-specific: simulate speckle / gain variation
        albun.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        albun.GaussNoise(var_limit=(5, 25), p=0.3),
        albun.CLAHE(clip_limit=2.0, p=0.3),
        albun.CoarseDropout(max_holes=4, max_height=16,
                        max_width=16, p=0.2),
        albun.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def build_val_transforms(img_size: int) -> albun.Compose:
    return albun.Compose([
        albun.Resize(img_size, img_size),
        albun.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
        ToTensorV2(),
    ])


def parse_xml_case(xml_path: str) -> dict | None:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot() 
    except ET.ParseError:
        return None
    

    tirads_raw = (root.findtext("tirads") or "").strip().lower()
    mark = root.find("mark")

    img_id = (mark.findtext("image") or "").strip()
    svg_text  = (mark.findtext("svg")   or "").strip()

    try:
        svg_data = json.loads(svg_text)
        polygon = svg_data[0]["points"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
    
    return {
    "case_id":        root.findtext("number", "").strip(),
    "image_id":       img_id,
    "label":          TIRADS_MAP[tirads_raw],
    "tirads_raw":     tirads_raw,
    "polygon":        polygon,
    "age":            root.findtext("age", "0").strip(),
    "sex":            root.findtext("sex", "").strip(),
    "composition":    root.findtext("composition", "").strip(),
    "echogenicity":   root.findtext("echogenicity", "").strip(),
    "margins":        root.findtext("margins", "").strip(),
    "calcifications": root.findtext("calcifications", "").strip(),
}



def polygon_to_mask(points: list[dict], height: int, width: int) -> np.ndarray:
    """Convert list of {x, y} dicts to a uint8 binary mask (0/1)."""
    pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def crop_nodule(image: np.ndarray,
                mask: np.ndarray,
                padding: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop the bounding box of the mask (+ padding) from image and mask.
    Falls back to the full image if no mask pixels found.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return image, mask

    h, w = image.shape[:2]
    x1 = max(0,  xs.min() - padding)
    x2 = min(w,  xs.max() + padding)
    y1 = max(0,  ys.min() - padding)
    y2 = min(h,  ys.max() + padding)

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]