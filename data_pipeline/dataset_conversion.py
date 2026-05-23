import json

import os
import cv2
import shutil
import numpy as np


import albumentations as albun
from albumentations.pytorch import ToTensorV2

import xml.etree.ElementTree as ET

######## ImageNet statistics ##########
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
BINARY_TIRADS_MAP = {"0": 0, "1": 1}

CLASS_NAMES = ["TR-1", "TR-2", "TR-3", "TR-4", "TR-5"]


def build_train_transforms(img_size: int) -> albun.Compose:
    return albun.Compose(
        [
            albun.Resize(img_size, img_size),
            albun.HorizontalFlip(p=0.5),
            albun.VerticalFlip(p=0.3),
            albun.Rotate(limit=15, p=0.5),
            albun.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.4),
            # Ultrasound-specific: simulate speckle / gain variation
            albun.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            albun.GaussNoise(var_limit=(5, 25), p=0.3),
            albun.CLAHE(clip_limit=2.0, p=0.3),
            albun.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.2),
            albun.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_val_transforms(img_size: int) -> albun.Compose:
    return albun.Compose(
        [
            albun.Resize(img_size, img_size),
            albun.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


"""For binary classes"""


def parse_binary_xml_case(xml_path: str) -> dict | None:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    obj = root.find("object")
    if obj is None:
        return None

    label_text = obj.findtext("name", "").strip()
    if label_text == "":
        return None
    bbox = obj.find("bndbox")
    if bbox is None:
        return None
    try:
        xmin = int(float(bbox.findtext("xmin", "0")))
        ymin = int(float(bbox.findtext("ymin", "0")))
        xmax = int(float(bbox.findtext("xmax", "0")))
        ymax = int(float(bbox.findtext("ymax", "0")))
    except ValueError:
        return None
    filename = root.findtext("filename", "").strip()
    return {
        "case_id": os.path.splitext(filename)[0],
        "image_id": filename,
        "label": BINARY_TIRADS_MAP[label_text],
        "bbox": [xmin, ymin, xmax, ymax],
        "width": int(root.findtext("size/width", "0")),
        "height": int(root.findtext("size/height", "0")),
    }


"""For multiple classes"""


def parse_xml_case(xml_path: str) -> dict | None:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    tirads_raw = (root.findtext("tirads") or "").strip().lower()
    if tirads_raw == "":
        return None
    mark = root.find("mark")

    # img_id = (mark.findtext("image") or "").strip()
    svg_text = (mark.findtext("svg") or "").strip()

    try:
        svg_data = json.loads(svg_text)
        polygon = svg_data[0]["points"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None

    return {
        "case_id": root.findtext("number", "").strip(),
        # "image_id": img_id, #######just the croppeed image_id, not the image ########
        "image_id": root.findtext("number", "").strip(),
        "label": TIRADS_MAP[tirads_raw],
        "tirads_raw": tirads_raw,
        "polygon": polygon,
        "age": root.findtext("age", "0").strip(),
        "sex": root.findtext("sex", "").strip(),
        "composition": root.findtext("composition", "").strip(),
        "echogenicity": root.findtext("echogenicity", "").strip(),
        "margins": root.findtext("margins", "").strip(),
        "calcifications": root.findtext("calcifications", "").strip(),
    }


def polygon_to_mask(points: list[dict], height: int, width: int) -> np.ndarray:
    """Convert list of {x, y} dicts to a uint8 binary mask (0/1)."""
    pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def crop_nodule(
    image: np.ndarray, mask: np.ndarray, padding: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop the bounding box of the mask (+ padding) from image and mask.
    Falls back to the full image if no mask pixels found.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return image, mask

    h, w = image.shape[:2]
    x1 = max(0, xs.min() - padding)
    x2 = min(w, xs.max() + padding)
    y1 = max(0, ys.min() - padding)
    y2 = min(h, ys.max() + padding)

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


######## Exporting dataset ####################
def export_dataset_to_json(dataset, output_dir):
    split_dir = os.path.join(output_dir, dataset.split)
    os.makedirs(split_dir, exist_ok=True)

    for case in dataset.cases:
        class_name = CLASS_NAMES[case["label"]]
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_src = case["img_path"]
        img_name = os.path.basename(img_src)

        img_dst = os.path.join(class_dir, img_name)

        shutil.copy2(img_src, img_dst)
        json_name = os.path.splitext(img_name)[0] + ".json"
        json_dst = os.path.join(class_dir, json_name)

        json_data = {
            "case_id": case["case_id"],
            "image_id": case["image_id"],
            "label": case["label"],
            "class_name": class_name,
            "tirads_raw": case["tirads_raw"],
            "polygon": case["polygon"],
            "age": case["age"],
            "sex": case["sex"],
            "composition": case["composition"],
            "echogenicity": case["echogenicity"],
            "margins": case["margins"],
            "calcifications": case["calcifications"],
        }
        with open(json_dst, "w") as f:
            json.dump(json_data, f, indent=4)

    print(f"\nSaved {dataset.split} split to:")
    print(split_dir)
