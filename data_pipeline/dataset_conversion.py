import json
import csv
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


####Create a csv file from the the images and json data file ########
def create_dataset_csv(data_dir, output_csv):
    rows = []
    for subfolder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, subfolder)
        if not os.path.isdir(folder_path):
            continue
        for f in sorted(os.listdir(folder_path)):
            if not f.endswith(".json"):
                continue
            img_path = os.path.join(subfolder, f.replace(".json", ".jpg"))
            with open(os.path.join(folder_path, f)) as jf:
                data = json.load(jf)
            label = data.get("class_name", subfolder)
            polygon = data.get("polygon", [])
            mask = ";".join(f"{p['x']},{p['y']}" for p in polygon)
            rows.append([img_path, label, mask])

    with open(output_csv, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["img_path", "label", "mask"])
        w.writerows(rows)
    print(f"Done: {len(rows)} rows written to {output_csv}")


#### split the dataset into train, val and test csv files ########
def split_dataset_csv(
    input_csv, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
):
    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "val.csv")
    test_csv = os.path.join(output_dir, "test.csv")

    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    np.random.seed(random_seed)
    np.random.shuffle(rows)

    total = len(rows)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }
    for split_name, split_rows in splits.items():
        output_csv = {"train": train_csv, "val": val_csv, "test": test_csv}[split_name]
        with open(output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["img_path", "label", "mask"])
            w.writerows([r["img_path"], r["label"], r["mask"]] for r in split_rows)
        print(f"{split_name}: {len(split_rows)} rows → {output_csv}")
