#!/usr/bin/env python3
"""
Seed validation on the FULL revised dataset — fills the seed gap with REAL runs.

You already have seed 0 for each finalist. This trains seed 1 and seed 2 (real
training, not estimated) so the paper can report mean ± std over 3 seeds.

Configurations (the clean 2x2: {stock, globalctx} x {default, best-TAL}):
  stock_full_default     : stock YOLOv12s  + vanilla default loss   (baseline)
  stock_full_besttal     : stock YOLOv12s  + best-TAL               (loss only)
  globalctx_full_default : globalctx arch  + vanilla default loss   (arch only)
  globalctx_full_besttal : globalctx arch  + best-TAL               (headline)

Each is run at seed 1 and seed 2 here (seed 0 = your existing runs). All on the
full revised dataset, 90 ep, batch 48 — identical to the seed-0 runs, so the three
seeds are directly comparable. Runs are ordered headline-first so a truncated night
still yields the most important seeds.

Output run names: <config>_seed<N>  (e.g. globalctx_full_besttal_seed1).
Verify the head at startup: stock prints 'Detect', globalctx prints 'DetectAuxDual'
— this is also the ground-truth check for which checkpoint is which.
"""

import time
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = "/home/constantin/Doctorat/GunDatasetNoAugSplit/data.yaml"   # FULL revised dataset
PROJECT_DIR = "runs_noaug_weapon_full_review"
YAML_DIR = "arch_yamls"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 8
IMG_SIZE = 640
PRETRAINED = "yolov12s.pt"
DETECT_SRC_IDX = 21
BATCH = 50
EPOCHS = 90
AUX_W = 0.5

SEEDS = [1, 2]   # seed 0 already exists; add the two missing seeds

# globalctx architecture (the verified design) -- Detect @ 27
ARCH_GLOBALCTX = f"""nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 11
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]        # 14 -- P3 neck (raw; aux anchor)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]        # 17 -- P4 neck (raw; aux anchor)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]             # 20 -- P5 neck (raw; aux anchor)
  - [17, 1, ZGLSKAWideFuseV2, [512, 11, 23, 3, 5]]   # 21 -- P4 hybrid
  - [14, 1, ZGSmallDetail, [256, 3, 5]]              # 22 -- P3 detail
  - [20, 1, ZGLSKAWideFuse, [1024, 11, 23]]          # 23 -- P5 context
  - [22, 1, ZGGlobalContext, [256]]                  # 24 -- P3 + global context
  - [21, 1, ZGGlobalContext, [512]]                  # 25 -- P4 + global context
  - [23, 1, ZGGlobalContext, [1024]]                 # 26 -- P5 + global context
  - [[24, 25, 26, 14, 17, 20], 1, DetectAuxDual, [nc, {AUX_W}]]  # 27
"""

# TRUE vanilla default (matches the seed-0 default runs exactly)
TAL_DEFAULT = dict(
    tal_topk=10, tal_alpha=0.5, tal_beta=6.0,
    alpha_start=0.0, alpha_end=0.0, alpha_min=0.0, alpha_max=0.0,
    iou_clip_start=999.0, iou_clip_end=999.0, dfl_clip_start=999.0, dfl_clip_end=999.0,
    small_obj_boost=1.0, small_obj_px=0,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, use_vfl=False,
)
# best-TAL recipe (matches the seed-0 best-TAL runs exactly)
TAL_BEST_LOOSE = dict(
    cls=1.2,
    alpha_start=0.7, alpha_end=0.3, alpha_min=0.2, alpha_max=0.8,
    small_obj_px=40, small_obj_boost=2.5,
    center_loss_weight_init=0.0, center_loss_weight_min=0.0, center_loss_decay_epochs=35,
    iou_clip_start=50.0, iou_clip_end=20.0, dfl_clip_start=25.0, dfl_clip_end=10.0,
    tal_topk=13, tal_alpha=0.7, tal_beta=4.0, iou_type="DIoU", use_vfl=False,
)

# Three finalists, each seeded (1 & 2) -> 6 runs:
#   globalctx + best TAL (headline) | globalctx + default (pure arch) | stock + best TAL (loss only)
CONFIGS = [
    # {"name": "globalctx_full_besttal", "arch": "globalctx", "loss": TAL_BEST_LOOSE},  # best arch + best TAL
    # {"name": "globalctx_full_default", "arch": "globalctx", "loss": TAL_DEFAULT},     # best arch (default loss)
    {"name": "stock_full_besttal",     "arch": "stock",     "loss": TAL_BEST_LOOSE},  # stock + best TAL
]


def save_yaml(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def load_pretrained_with_detect_remap(model, weights=PRETRAINED):
    model.load(weights)
    det_dst = len(model.model.model) - 1
    if det_dst == DETECT_SRC_IDX:
        return model
    ckpt = torch.load(weights, map_location="cpu")
    src = ckpt.get("model", ckpt)
    csd = (src.float() if hasattr(src, "float") else src).state_dict() \
        if hasattr(src, "state_dict") else src
    pfx_src, pfx_dst = f"model.{DETECT_SRC_IDX}.", f"model.{det_dst}."
    remapped = {pfx_dst + k[len(pfx_src):]: v for k, v in csd.items() if k.startswith(pfx_src)}
    matched = intersect_dicts(remapped, model.model.state_dict())
    model.model.load_state_dict(matched, strict=False)
    print(f"  [detect-remap] Detect {DETECT_SRC_IDX} -> {det_dst}: {len(matched)}/{len(remapped)} keys")
    return model


def build_model(arch, name):
    if arch == "globalctx":
        yaml_path = os.path.join(YAML_DIR, f"{name}.yaml")
        save_yaml(ARCH_GLOBALCTX, yaml_path)
        model = YOLO(yaml_path)
        load_pretrained_with_detect_remap(model)
    else:  # stock yolov12s
        model = YOLO(PRETRAINED)
    return model


def on_train_epoch_start(trainer):
    epoch = trainer.epoch
    try:
        if getattr(trainer, "criterion", None) is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, "_sync_bbox_loss_state"):
                trainer.criterion._sync_bbox_loss_state()
    except Exception:
        pass
    try:
        trainer.model.current_epoch = epoch
    except Exception:
        pass


def run_one(cfg, seed):
    run_name = f"{cfg['name']}_seed{seed}"
    print(f"\n{'#' * 80}\n# {run_name}  ({cfg['arch']})\n# FULL data  Batch {BATCH}  Epochs {EPOCHS}  seed {seed}\n{'#' * 80}\n")
    start = time.time()
    try:
        model = build_model(cfg["arch"], run_name)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        head = type(model.model.model[-1]).__name__
        print(f"  head = {head}  (expect 'Detect' for stock, 'DetectAuxDual' for globalctx)")
        kw = dict(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE,
                  workers=WORKERS, project=PROJECT_DIR, name=run_name, patience=100,
                  close_mosaic=10, seed=seed, deterministic=True)
        kw.update(cfg["loss"])
        model.train(**kw)
        el = (time.time() - start) / 3600
        print(f"\n  DONE: {run_name} ({el:.2f}h)")
        return {"name": run_name, "status": "OK", "time": el}
    except Exception as e:
        el = (time.time() - start) / 3600
        print(f"\n  FAILED: {run_name} ({el:.2f}h) -- {e}")
        import traceback; traceback.print_exc()
        return {"name": run_name, "status": f"FAILED: {e}", "time": el}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    os.makedirs(YAML_DIR, exist_ok=True)
    t0 = time.time()
    print(f"\n{'=' * 80}\n  SEED VALIDATION (real runs) on FULL revised data — seeds {SEEDS}")
    print(f"  data: {DATA_YAML}")
    print(f"{'=' * 80}")
    results = []
    for seed in SEEDS:
        for cfg in CONFIGS:
            results.append(run_one(cfg, seed))
    print(f"\n{'=' * 80}\n  ALL DONE ({(time.time()-t0)/3600:.2f}h)")
    for r in results:
        print(f"  [{'OK' if r['status']=='OK' else 'FAIL'}] {r['name']:<32} {r['time']:.2f}h  {r['status']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
