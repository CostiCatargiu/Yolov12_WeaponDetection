#!/usr/bin/env python3
"""
YOLO -> COCO annotation converter.

Handles both YOLO label formats:
    detection     : class cx cy w h                 (normalized)
    segmentation  : class x1 y1 x2 y2 ... xn yn     (normalized polygon)
Polygons are reduced to their axis-aligned bounding box.

Every split listed in SPLITS is converted in one run, producing one COCO
JSON per split. Category ids start at 1, which is what pycocotools expects.
"""

import json
from pathlib import Path

# =============================================================================
# EDIT THESE
# =============================================================================

BASE = Path("/path/to/dataset")

# Class names in YOLO class-index order (index 0 first).
# Must match the `names:` list of your data.yaml.
CLASS_NAMES = ["class_a", "class_b", "class_c"]

# One entry per split to convert. Paths are relative to BASE.
SPLITS = {
    "valid": {
        "images": "valid/images",
        "labels": "valid/labels",
        "output": "annotations_coco_valid.json",
    },
    "test": {
        "images": "test/images",
        "labels": "test/labels",
        "output": "annotations_coco_test.json",
    },
}

# COCO category id = YOLO class index + this offset. Keep at 1 unless you
# know your prediction files use 0-based ids.
CATEGORY_ID_OFFSET = 1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# HELPERS
# =============================================================================

def read_image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) without decoding the whole image where possible."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except ImportError:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return w, h
    except Exception:
        return None


def parse_label_line(parts: list[str]) -> tuple[int, float, float, float, float, bool] | None:
    """
    Parse one YOLO label line into (class_idx, cx, cy, w, h, was_polygon),
    all box values normalized. Returns None if the line can't be parsed.
    """
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    if len(coords) == 4:
        cx, cy, bw, bh = coords
        return cls, cx, cy, bw, bh, False

    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs, ys = coords[0::2], coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return (cls, (x_min + x_max) / 2, (y_min + y_max) / 2,
                x_max - x_min, y_max - y_min, True)

    return None


# =============================================================================
# CONVERSION
# =============================================================================

def convert_split(images_dir: Path, labels_dir: Path, output_json: Path) -> dict:
    images: list = []
    annotations: list = []
    stats = {
        "unreadable_images": 0,
        "missing_labels": 0,
        "empty_lines": 0,
        "invalid_lines": 0,
        "degenerate_boxes": 0,
        "bad_class_ids": 0,
        "polygons": 0,
    }

    img_id = 1
    ann_id = 1

    image_paths = sorted(p for p in images_dir.glob("*.*")
                         if p.suffix.lower() in IMAGE_EXTS)

    for img_path in image_paths:
        size = read_image_size(img_path)
        if size is None:
            print(f"⚠ Skipping unreadable image: {img_path}")
            stats["unreadable_images"] += 1
            continue
        w, h = size

        images.append({"id": img_id, "file_name": img_path.name,
                       "width": w, "height": h})

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            stats["missing_labels"] += 1
            img_id += 1
            continue

        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    stats["empty_lines"] += 1
                    continue

                parsed = parse_label_line(line.split())
                if parsed is None:
                    print(f"⚠ Invalid label in {label_path}: {line}")
                    stats["invalid_lines"] += 1
                    continue

                cls, cx, cy, bw, bh, was_polygon = parsed
                if was_polygon:
                    stats["polygons"] += 1

                if not 0 <= cls < len(CLASS_NAMES):
                    print(f"⚠ Class id {cls} out of range in {label_path}")
                    stats["bad_class_ids"] += 1
                    continue

                # normalized -> absolute, then clamp the corners (clamping the
                # corners rather than the width keeps the box where it was)
                x1 = max(0.0, (cx - bw / 2) * w)
                y1 = max(0.0, (cy - bh / 2) * h)
                x2 = min(float(w), (cx + bw / 2) * w)
                y2 = min(float(h), (cy + bh / 2) * h)
                box_w, box_h = x2 - x1, y2 - y1

                if box_w <= 0 or box_h <= 0:
                    stats["degenerate_boxes"] += 1
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + CATEGORY_ID_OFFSET,
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                })
                ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + CATEGORY_ID_OFFSET, "name": name}
                       for i, name in enumerate(CLASS_NAMES)],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f)

    stats["images"] = len(images)
    stats["annotations"] = len(annotations)
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    for split_name, info in SPLITS.items():
        images_dir = BASE / info["images"]
        labels_dir = BASE / info["labels"]
        output_json = BASE / info["output"]

        print("\n" + "=" * 60)
        print(f"SPLIT: {split_name}")
        print("=" * 60)
        print(f"Images dir : {images_dir}")
        print(f"Labels dir : {labels_dir}")
        print(f"Output     : {output_json}")

        if not images_dir.is_dir():
            print(f"⚠ Images dir not found — skipping split '{split_name}'.")
            continue
        if not labels_dir.is_dir():
            print(f"⚠ Labels dir not found — skipping split '{split_name}'.")
            continue

        stats = convert_split(images_dir, labels_dir, output_json)

        print("-" * 60)
        print(f"Images:              {stats['images']}")
        print(f"Annotations:         {stats['annotations']}")
        print(f"  of which polygons: {stats['polygons']}")
        print("-" * 60)
        print(f"Unreadable images:   {stats['unreadable_images']}")
        print(f"Missing label files: {stats['missing_labels']}")
        print(f"Empty lines skipped: {stats['empty_lines']}")
        print(f"Invalid lines:       {stats['invalid_lines']}")
        print(f"Degenerate boxes:    {stats['degenerate_boxes']}")
        print(f"Out-of-range classes:{stats['bad_class_ids']}")
        print("=" * 60)

    print("\n✅ COCO conversion complete")


if __name__ == "__main__":
    main()