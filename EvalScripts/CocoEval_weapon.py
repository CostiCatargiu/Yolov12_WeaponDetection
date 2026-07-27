#!/usr/bin/env python3
"""
COCO evaluation with class remapping.
Model classes (knife, long_gun, pistol) → GT classes
No coordinate rescaling (Ultralytics already outputs original-resolution coords).
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# =============================================================================
# ✏️  EDIT THESE
# =============================================================================

WEIGHTS = "/home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/v12custom_100e/top2_auxiliary6/weights/best.pt"
DATA_YAML = "/home/constantin/Downloads/YouTube-GDD/images/data.yaml"
COCO_ANN_FILE = "/home/constantin/Downloads/YouTube-GDD/annotations_coco_matched.json"
OUTPUT_JSON = "/home/constantin/Downloads/YouTube-GDD/images/test/results.json"
IMG_SIZE = 640
SMALL_THRESH = 32
LARGE_THRESH = 96

# Model prediction category IDs (1-indexed from Ultralytics JSON)
# → mapped to GT category IDs (from COCO annotation file)
# Set to None to skip that class
# Will be auto-built, but you can override:
# CATEGORY_MAP = {1: 0, 2: 1, 3: None, 4: 3}
#   1(knife)→0(knife), 2(long_gun)→1(long_gun), 3(no_weapon)→skip, 4(pistol)→3(pistol)


# =============================================================================
# FIX PREDICTIONS: category remapping only, NO coordinate changes
# =============================================================================

def fix_predictions(pred_json: Path, coco_ann_file: str) -> Path:
    from pycocotools.coco import COCO

    coco_gt = COCO(coco_ann_file)

    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
    print(f"[FIX] GT categories: {cat_id_to_name}")

    # Model pred categories (1-indexed): 1=knife, 2=long_gun, 3=no_weapon, 4=pistol
    model_classes = {1: "knife", 2: "long_gun", 3: "no_weapon", 4: "pistol"}

    # Build mapping: pred_cat_id → gt_cat_id
    category_map = {}
    for pred_cat_id, pred_name in model_classes.items():
        if pred_name == "no_weapon":
            category_map[pred_cat_id] = None
        elif pred_name in cat_name_to_id:
            category_map[pred_cat_id] = cat_name_to_id[pred_name]
        else:
            category_map[pred_cat_id] = None

    print(f"[FIX] Category mapping: {category_map}")
    for pid, gid in category_map.items():
        pname = model_classes[pid]
        gname = cat_id_to_name.get(gid, "SKIP") if gid is not None else "SKIP"
        print(f"       pred {pid} ({pname}) → gt {gid} ({gname})")

    # Build filename → image_id lookup
    filename_to_id = {}
    for img_id, info in coco_gt.imgs.items():
        fname = info["file_name"]
        stem = Path(fname).stem
        clean = stem.split(".rf")[0]
        for key in [fname, stem, clean, clean + ".jpg"]:
            filename_to_id[key] = img_id

    with open(pred_json) as f:
        preds = json.load(f)
    print(f"[FIX] Total raw predictions: {len(preds)}")

    # Show raw predictions before any changes
    print(f"[FIX] First 3 RAW predictions (no changes):")
    for p in preds[:3]:
        print(f"       img={p['image_id']}  cat={p['category_id']}  "
              f"score={p['score']:.3f}  bbox={[round(x,1) for x in p['bbox']]}")

    fixed = []
    skipped_no_match = 0
    skipped_no_cat = 0
    counts = defaultdict(int)

    for p in preds:
        img_id_raw = p["image_id"]

        # Resolve string → int image_id
        if isinstance(img_id_raw, str):
            stem = Path(img_id_raw).stem
            clean = stem.split(".rf")[0]
            key = Path(img_id_raw).name
            resolved_id = (filename_to_id.get(key)
                           or filename_to_id.get(stem)
                           or filename_to_id.get(clean)
                           or filename_to_id.get(clean + ".jpg"))
            if resolved_id is None:
                skipped_no_match += 1
                continue
        elif isinstance(img_id_raw, int):
            if img_id_raw in coco_gt.imgs:
                resolved_id = img_id_raw
            else:
                skipped_no_match += 1
                continue
        else:
            skipped_no_match += 1
            continue

        # Map category
        pred_cat = int(p["category_id"])
        new_cat = category_map.get(pred_cat)
        if new_cat is None:
            skipped_no_cat += 1
            continue

        counts[model_classes.get(pred_cat, str(pred_cat))] += 1

        bx, by, bw, bh = p["bbox"]

        fixed.append({
            "image_id": resolved_id,
            "category_id": new_cat,
            "bbox": [bx, by, bw, bh],  # NO rescaling
            "score": p["score"],
            "area": bw * bh,
        })

    print(f"[FIX] Kept: {len(fixed)}, skipped (no img): {skipped_no_match}, skipped (no cat): {skipped_no_cat}")
    print(f"[FIX] Per-class counts: {dict(counts)}")

    # Spot check: find an image with both GT and predictions, compute IoU
    print(f"\n[FIX] ─── Spot check ───")
    checked = False
    for fp in fixed:
        img_id = fp["image_id"]
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        # Find GT annotation with same category
        matching_gt = [a for a in gt_anns if a["category_id"] == fp["category_id"]]
        if not matching_gt:
            continue

        fname = coco_gt.imgs[img_id]["file_name"]
        orig_w = coco_gt.imgs[img_id]["width"]
        orig_h = coco_gt.imgs[img_id]["height"]
        gt_a = matching_gt[0]

        b1, b2 = gt_a["bbox"], fp["bbox"]
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0]+b1[2], b2[0]+b2[2])
        y2 = min(b1[1]+b1[3], b2[1]+b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        union = b1[2]*b1[3] + b2[2]*b2[3] - inter
        iou = inter/union if union > 0 else 0

        print(f"  Image: {fname} ({orig_w}x{orig_h})")
        print(f"  GT:   cat={gt_a['category_id']} ({cat_id_to_name[gt_a['category_id']]})  "
              f"bbox={[round(x,1) for x in gt_a['bbox']]}")
        print(f"  Pred: cat={fp['category_id']} ({cat_id_to_name[fp['category_id']]})  "
              f"score={fp['score']:.3f}  bbox={[round(x,1) for x in fp['bbox']]}")
        print(f"  IoU = {iou:.4f}")

        if iou < 0.1:
            print(f"\n  ⚠️  IoU is very low! Checking coordinate ranges...")
            all_pred_x = [p["bbox"][0] + p["bbox"][2] for p in fixed[:100]]
            all_pred_y = [p["bbox"][1] + p["bbox"][3] for p in fixed[:100]]
            print(f"  Pred coord ranges: x=[0..{max(all_pred_x):.0f}], y=[0..{max(all_pred_y):.0f}]")
            print(f"  Image size: {orig_w}x{orig_h}")
            print(f"  If pred coords are ~half of image size, Ultralytics used inference resolution.")
        checked = True
        break

    if not checked:
        print("  No matching GT/pred pair found for spot check")
    print(f"[FIX] ─── End spot check ───\n")

    fixed_path = pred_json.parent / "predictions_fixed.json"
    with open(fixed_path, "w") as f:
        json.dump(fixed, f)
    print(f"[FIX] Saved → {fixed_path}\n")
    return fixed_path


# =============================================================================
# COCO EVALUATION
# =============================================================================

def run_coco_evaluation(fixed_json: Path, ann_file: str,
                        small_thresh: int, large_thresh: int) -> tuple:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)

    cat_ids = sorted(coco_gt.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    # Only evaluate classes that have GT annotations
    gt_ann_cats = set(a["category_id"] for a in coco_gt.loadAnns(coco_gt.getAnnIds()))
    eval_cat_ids = [cid for cid in cat_ids if cid in gt_ann_cats]
    eval_names = {cid: cat_id_to_name[cid] for cid in eval_cat_ids}
    print(f"[EVAL] Evaluating: {eval_names}")

    cat_id_to_idx = {cid: i for i, cid in enumerate(eval_cat_ids)}

    coco_dt = coco_gt.loadRes(str(fixed_json))

    area_ranges = [
        [0, 1e5 ** 2],
        [0, small_thresh ** 2],
        [small_thresh ** 2, large_thresh ** 2],
        [large_thresh ** 2, 1e5 ** 2],
    ]
    area_labels = ["all", "small", "medium", "large"]

    metrics = {}
    per_class = {}

    for tag, iou_thrs in [("50", np.array([0.5])),
                           ("50_95", np.linspace(0.5, 0.95, 10))]:
        print(f"\n{'─'*50}\nCOCO Eval @ IoU={tag.replace('_',':')}\n{'─'*50}")

        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.iouThrs = iou_thrs
        ev.params.areaRng = area_ranges
        ev.params.areaRngLbl = area_labels
        ev.params.catIds = eval_cat_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        prec = ev.eval["precision"]
        rec = ev.eval["recall"]

        prefix = f"mAP{tag}"
        ar_prefix = f"AR{tag}"

        for ai, aname in enumerate(area_labels):
            vals = prec[:, :, :, ai, -1]
            vals = vals[vals > -1]
            metrics[f"{prefix}_{aname}"] = float(np.mean(vals)) if len(vals) else -1.0

            rvals = rec[:, :, ai, -1] if rec.ndim == 4 else rec[:, :, ai, -1]
            rvals = rvals[rvals > -1]
            metrics[f"{ar_prefix}_{aname}"] = float(np.mean(rvals)) if len(rvals) else -1.0

        for cid in eval_cat_ids:
            cname = cat_id_to_name[cid]
            cidx = cat_id_to_idx[cid]
            if cname not in per_class:
                per_class[cname] = {}
            for ai, aname in enumerate(area_labels):
                vals = prec[:, :, cidx, ai, -1]
                vals = vals[vals > -1]
                per_class[cname][f"AP{tag}_{aname}"] = float(np.mean(vals)) if len(vals) else -1.0

                if rec.ndim == 4:
                    rvals = rec[:, cidx, ai, -1]
                else:
                    rvals = rec[:, cidx, ai, -1]
                rvals_clean = rvals[rvals > -1] if hasattr(rvals, '__len__') else np.array([rvals])
                per_class[cname][f"AR{tag}_{aname}"] = float(np.mean(rvals_clean)) if len(rvals_clean) else -1.0

    return metrics, per_class


# =============================================================================
# MAIN
# =============================================================================

def main():
    from ultralytics import YOLO

    weights = Path(WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"[INFO] Loading model: {weights}")
    model = YOLO(str(weights))

    val_results = model.val(data=DATA_YAML, imgsz=IMG_SIZE, save_json=True, verbose=True)
    save_dir = Path(val_results.save_dir)
    pred_json = save_dir / "predictions.json"

    if not pred_json.exists():
        raise FileNotFoundError(f"predictions.json not found in {save_dir}")

    # Fix predictions (category remap only)
    fixed_json = fix_predictions(pred_json, COCO_ANN_FILE)

    # Ultralytics metrics
    box = val_results.box
    class_names = list(model.names.values())

    print("\n" + "=" * 70)
    print("ULTRALYTICS METRICS (model's own eval)")
    print("=" * 70)
    print(f"  {'mAP50':<20} {box.map50:>8.4f}")
    print(f"  {'mAP50-95':<20} {box.map:>8.4f}")
    print(f"  {'Precision':<20} {box.mp:>8.4f}")
    print(f"  {'Recall':<20} {box.mr:>8.4f}")
    for i, name in enumerate(class_names):
        if i < len(box.ap50):
            print(f"  {name:<15} AP50={box.ap50[i]:.4f}  AP50-95={box.ap[i]:.4f}  "
                  f"P={box.p[i]:.4f}  R={box.r[i]:.4f}")

    # COCO evaluation
    coco_metrics, coco_per_class = run_coco_evaluation(
        fixed_json, COCO_ANN_FILE, SMALL_THRESH, LARGE_THRESH
    )

    print("\n" + "=" * 110)
    print("COCO METRICS (per-class, with size breakdown)")
    print("=" * 110)
    for k, v in sorted(coco_metrics.items()):
        print(f"  {k:<25} {v:>8.4f}" if v != -1 else f"  {k:<25} {'N/A':>8}")

    print("\n" + "=" * 110)
    print("PER-CLASS COCO METRICS")
    print("=" * 110)
    header = (f"{'Class':<12} │ {'AP50':>6} │ {'AP50_S':>6} │ {'AP50_M':>6} │ "
              f"{'AP50_L':>6} │ {'AP5095':>6} │ {'AP5095_S':>8} │ "
              f"{'AP5095_M':>8} │ {'AP5095_L':>8}")
    print(header)
    print("─" * 110)

    for cls in sorted(coco_per_class):
        m = coco_per_class[cls]
        fmt = lambda v: f"{v:>6.3f}" if v is not None and v != -1 else f"{'---':>6}"
        fmt8 = lambda v: f"{v:>8.3f}" if v is not None and v != -1 else f"{'---':>8}"
        print(
            f"{cls:<12} │ {fmt(m.get('AP50_all'))} │ {fmt(m.get('AP50_small'))} │ "
            f"{fmt(m.get('AP50_medium'))} │ {fmt(m.get('AP50_large'))} │ "
            f"{fmt(m.get('AP50_95_all'))} │ {fmt8(m.get('AP50_95_small'))} │ "
            f"{fmt8(m.get('AP50_95_medium'))} │ {fmt8(m.get('AP50_95_large'))}"
        )
    print("─" * 110)

    output = {
        "weights": str(WEIGHTS),
        "img_size": IMG_SIZE,
        "small_thresh": SMALL_THRESH,
        "large_thresh": LARGE_THRESH,
        "ultralytics": {
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        },
        "coco_metrics": coco_metrics,
        "coco_per_class": coco_per_class,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
