"""COCO to YOLO Dataset Converter for PKLot Roboflow Export.

This module converts a Roboflow COCO export format dataset into standard YOLO TXT format:
- Split folders (train, valid, test) containing images and _annotations.coco.json.
- Creates `images/` and `labels/` subdirectories inside each split.
- Remaps COCO category IDs:
    - 0 ('spaces'): dropped (supercategory)
    - 1 ('space-empty'): remapped to YOLO class 0
    - 2 ('space-occupied'): remapped to YOLO class 1
- Writes normalized YOLO bounding boxes: <class_id> <x_center> <y_center> <width> <height>
- Updates data.yaml with 2 classes and correct dataset paths.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("coco2yolo")


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, val))


def coco_bbox_to_yolo(
    bbox: List[float], img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Convert COCO bbox [x_min, y_min, w, h] to YOLO normalized [x_center, y_center, w, h].

    Returns None if width or height <= 0 or image dimensions invalid.
    """
    x_min, y_min, w, h = bbox
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None

    x_center = (x_min + w / 2.0) / img_w
    y_center = (y_min + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    x_center = clamp(x_center)
    y_center = clamp(y_center)
    w_norm = clamp(w_norm)
    h_norm = clamp(h_norm)

    return (round(x_center, 6), round(y_center, 6), round(w_norm, 6), round(h_norm, 6))


def convert_split(
    split_dir: Path,
    annotation_filename: str = "_annotations.coco.json",
    class_remap: Optional[Dict[int, Optional[int]]] = None,
) -> Dict[str, Any]:
    """Convert a single split from COCO to YOLO format.

    Parameters
    ----------
    split_dir : Path
        Directory for the split (e.g. data/train).
    annotation_filename : str
        Name of COCO annotation json file.
    class_remap : Dict[int, Optional[int]], optional
        Mapping from raw COCO category_id to final YOLO class_id.
        If class_id is None, the annotation is skipped.

    Returns
    -------
    Dict[str, Any]
        Conversion stats for this split.
    """
    if class_remap is None:
        class_remap = {0: None, 1: 0, 2: 1}

    json_path = split_dir / annotation_filename
    if not json_path.exists():
        logger.warning(f"Annotation file not found: {json_path}")
        return {"converted_images": 0, "converted_boxes": 0}

    logger.info(f"Loading COCO annotations from {json_path}")
    with open(json_path, encoding="utf-8") as f:
        coco_data = json.load(f)

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Build image_id -> image info map
    images_info = {img["id"]: img for img in coco_data.get("images", [])}

    # Group annotations by image_id
    img_annotations: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in images_info}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id in img_annotations:
            img_annotations[img_id].append(ann)

    converted_images = 0
    converted_boxes = 0
    skipped_boxes = 0

    for img_id, img_info in tqdm(images_info.items(), desc=f"Converting {split_dir.name}"):
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Move/Copy image into images/ folder if it's in split_dir
        src_img_path = split_dir / file_name
        dest_img_path = images_dir / file_name

        if src_img_path.exists() and src_img_path != dest_img_path:
            shutil.move(str(src_img_path), str(dest_img_path))

        stem = Path(file_name).stem
        label_file_path = labels_dir / f"{stem}.txt"

        yolo_lines = []
        for ann in img_annotations.get(img_id, []):
            raw_cat_id = ann["category_id"]
            if raw_cat_id not in class_remap or class_remap[raw_cat_id] is None:
                skipped_boxes += 1
                continue

            final_cls_id = class_remap[raw_cat_id]
            yolo_box = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            if yolo_box is None:
                skipped_boxes += 1
                continue

            x_c, y_c, w_n, h_n = yolo_box
            yolo_lines.append(f"{final_cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
            converted_boxes += 1

        with open(label_file_path, "w", encoding="utf-8") as f_lbl:
            if yolo_lines:
                f_lbl.write("\n".join(yolo_lines) + "\n")

        converted_images += 1

    stats = {
        "converted_images": converted_images,
        "converted_boxes": converted_boxes,
        "skipped_boxes": skipped_boxes,
    }
    logger.info(f"[{split_dir.name}] Summary: {stats}")
    return stats


def update_data_yaml(
    dataset_root: Path,
    class_names: Tuple[str, ...] = ("space-empty", "space-occupied"),
) -> Path:
    """Generate/Update data.yaml for Ultralytics YOLO training."""
    dataset_root = dataset_root.resolve()
    yaml_data = {
        "path": str(dataset_root).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    logger.info(f"Updated data.yaml at {yaml_path}")
    return yaml_path


def run_conversion(
    dataset_root: Path = Path("data"),
    class_remap: Optional[Dict[int, Optional[int]]] = None,
    class_names: Tuple[str, ...] = ("space-empty", "space-occupied"),
) -> Dict[str, Dict[str, Any]]:
    """Convert all splits in dataset_root to YOLO TXT format."""
    if class_remap is None:
        class_remap = {0: None, 1: 0, 2: 1}

    dataset_root = Path(dataset_root)
    all_stats = {}

    for split in ["train", "valid", "test"]:
        split_dir = dataset_root / split
        if split_dir.exists():
            all_stats[split] = convert_split(split_dir, class_remap=class_remap)

    update_data_yaml(dataset_root, class_names=class_names)
    return all_stats


if __name__ == "__main__":
    run_conversion()
