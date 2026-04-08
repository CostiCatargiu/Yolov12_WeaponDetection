#!/usr/bin/env python3
"""
Phase B Ablation: Complete Combined Configuration Study
========================================================

Overview
--------
Phase B tests ALL COMBINATIONS of best configurations from Phase A1, A2, A3,
and A4 to understand interactions and find the optimal combined
configuration for small object detection.

IMPORTANT: This script SKIPS experiments that already have results:
    - B00 (baseline)  - from Phase A1 (A1d_01_baseline)
    - B01 (A4 only)   - from Phase A4 (A4c_03)
    - B02 (A3b only)  - from Phase A3 (A3b_20)
    - B04 (A3a only)  - from Phase A3 (A3a_01)
    - B08 (A2 only)   - from Phase A2 (A2b_03)
    - B16 (A1 only)   - from Phase A1 (A1a_20)

Only COMBINATION experiments (B03, B05-B07, B09-B15, B17-B31) are run = 26 experiments.

Best Configurations from Phase A Steps
---------------------------------------
Phase A1 (A1a_20): Alpha Schedule
    - alpha_start: 0.9, alpha_end: 0.4, alpha_min: 0.3, alpha_max: 1.0
    - small_obj_px: 48, small_obj_boost: 1.5
    - Improvement: +15.2% mAP50-95, +10.7% mAP_small

Phase A2 (A2b_03): Center Loss Weight
    - center_loss_weight_init: 0.05, center_loss_weight_min: 0.01
    - Improvement: +1.8% mAP50-95, +4.2% mAP_small

Phase A3a (A3a_01): IoU Loss Clipping
    - iou_clip_start: 6.0, iou_clip_end: 2.0
    - Improvement: +1.0% mAP50-95, +4.2% mAP_small

Phase A3b (A3b_20): DFL Loss Clipping
    - dfl_clip_start: 8.0, dfl_clip_end: 5.0
    - Improvement: +1.1% mAP50-95, +7.4% mAP_small

Phase A4 (A4c_03): TAL Parameters
    - tal_topk: 8, tal_alpha: 0.5, tal_beta: 6.0
    - Improvement: -0.6% mAP50-95, +9.1% mAP_small

Combination Matrix (5 components = 32 combinations)
----------------------------------------------------
Each experiment is a binary combination of components:

    B00: None (baseline)                          [already exists]
    B01: A4                                       [already exists]
    B02: A3b                                      [already exists]
    B03: A3b + A4
    B04: A3a                                      [already exists]
    B05: A3a + A4
    B06: A3a + A3b
    B07: A3a + A3b + A4
    B08: A2                                       [already exists]
    B09: A2 + A4
    B10: A2 + A3b
    B11: A2 + A3b + A4
    B12: A2 + A3a
    B13: A2 + A3a + A4
    B14: A2 + A3a + A3b
    B15: A2 + A3a + A3b + A4
    B16: A1                                       [already exists]
    B17: A1 + A4
    B18: A1 + A3b
    B19: A1 + A3b + A4
    B20: A1 + A3a
    B21: A1 + A3a + A4
    B22: A1 + A3a + A3b
    B23: A1 + A3a + A3b + A4
    B24: A1 + A2
    B25: A1 + A2 + A4
    B26: A1 + A2 + A3b
    B27: A1 + A2 + A3b + A4
    B28: A1 + A2 + A3a
    B29: A1 + A2 + A3a + A4
    B30: A1 + A2 + A3a + A3b
    B31: A1 + A2 + A3a + A3b + A4 (all combined)

Experiments to Run (26 total)
-----------------------------
Combinations of 2+ components that don't already exist.

Statistics
----------
    Total combinations:     32
    Already completed:       6 (single-component + baseline)
    New experiments:        26
    Time per experiment:    ~1 hour
    Total estimated time:   ~26 hours

Overall Ablation Structure
--------------------------
    Phase A:  Isolated ablation studies
      ├── A1: Size-Aware Weighting    (phase_a1_ablation.py)
      ├── A2: Center Loss             (phase_a2_ablation.py)
      ├── A3: Adaptive Clipping       (phase_a3_ablation.py)
      └── A4: TAL Parameters          (phase_a4_ablation.py)
    Phase B:  Combined configuration   (this script)

Usage
-----
    python phase_b_ablation.py                    # Run all 26 combinations
    python phase_b_ablation.py --show-matrix      # Show combination matrix
    python phase_b_ablation.py --analyze          # Show results analysis
    python phase_b_ablation.py --experiment B31   # Run specific experiment

Author: Constantin
Project: YOLOv12 Small Object Detection Optimization
"""


import os
import json
import time
import gc
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_YAML = "/home/constantin/Doctorat/GunDataset_matched_cleanedHist/data.yaml"
COCO_ANN_FILE = "/home/constantin/Doctorat/GunDataset_matched_cleanedHist/annotations_coco_val.json"

MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_phaseE"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 64
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# BASELINE CONFIGURATION (No modifications - default YOLO)
# =============================================================================

BASELINE_CONFIG = {
    # Phase A: Alpha Schedule (DISABLED - default values)
    "alpha_start": 1.0,
    "alpha_end": 1.0,
    "alpha_min": 0.3,
    "alpha_max": 1.0,
    "small_obj_px": 48,
    "small_obj_boost": 1.0,  # No boost in baseline

    # Phase B: Center Loss Weight (DISABLED)
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,

    # Phase C1: IoU Clipping (DISABLED - very high = no clipping)
    "iou_clip_start": 1000.0,
    "iou_clip_end": 1000.0,

    # Phase C2: DFL Clipping (DISABLED - very high = no clipping)
    "dfl_clip_start": 1000.0,
    "dfl_clip_end": 1000.0,

    # Phase D: TAL Parameters (DEFAULT YOLO values)
    "tal_topk": 10,
    "tal_alpha": 0.5,
    "tal_beta": 6.0,
}

# =============================================================================
# PHASE MODIFICATIONS (Best results from each phase)
# =============================================================================

PHASE_MODIFICATIONS = {
    "A": {
        "name": "Alpha Schedule",
        "description": "Knowledge distillation (α: 0.9→0.4) + small obj boost",
        "params": {
            "alpha_start": 0.9,
            "alpha_end": 0.4,
            "alpha_min": 0.3,
            "alpha_max": 1.0,
            "small_obj_px": 48,
            "small_obj_boost": 1.0,
        },
        "bit": 4,  # Bit position (2^4 = 16)
    },
    "B": {
        "name": "Center Loss Weight",
        "description": "Center loss schedule (0.05→0.01)",
        "params": {
            "center_loss_weight_init": 0.05,
            "center_loss_weight_min": 0.01,
        },
        "bit": 3,  # Bit position (2^3 = 8)
    },
    "C1": {
        "name": "IoU Clipping",
        "description": "IoU loss clipping (6→2)",
        "params": {
            "iou_clip_start": 6.0,
            "iou_clip_end": 2.0,
        },
        "bit": 2,  # Bit position (2^2 = 4)
    },
    "C2": {
        "name": "DFL Clipping",
        "description": "DFL loss clipping (8→5)",
        "params": {
            "dfl_clip_start": 8.0,
            "dfl_clip_end": 5.0,
        },
        "bit": 1,  # Bit position (2^1 = 2)
    },
    "D": {
        "name": "TAL Parameters",
        "description": "Task-Aligned Learning (topk=8, α=0.5, β=6.0)",
        "params": {
            "tal_topk": 8,
            "tal_alpha": 0.5,
            "tal_beta": 6.0,
        },
        "bit": 0,  # Bit position (2^0 = 1)
    },
}

# Phase order (MSB to LSB)
PHASE_ORDER = ["A", "B", "C1", "C2", "D"]


def build_config_for_phases(phases: List[str]) -> Dict[str, Any]:
    """Build configuration by applying phase modifications to baseline."""
    config = BASELINE_CONFIG.copy()
    for phase in phases:
        if phase in PHASE_MODIFICATIONS:
            config.update(PHASE_MODIFICATIONS[phase]["params"])
    return config


# =============================================================================
# EXISTING RESULTS FROM PREVIOUS PHASES
# =============================================================================

EXISTING_RESULTS = {
    0: {  # E00 - baseline (binary 00000)
        "name": "E00_baseline",
        "phases": [],
        "short_name": "baseline",
        "description": "Baseline (no modifications)",
        "source": "A4_01_baseline",
        "metrics": {
            "mAP50": 0.6840,
            "mAP50_95": 0.3510,
            "precision": 0.7692,
            "recall": 0.6441,
            "mAP_small": 0.2385,
            "mAP_medium": 0.3120,
            "mAP_large": 0.4087,
        },
    },
    16: {  # E16 - A only (binary 10000)
        "name": "E16_A",
        "phases": ["A"],
        "short_name": "A",
        "description": "Alpha Schedule",
        "source": "A1_20",
        "metrics": {
            "mAP50": 0.7094,
            "mAP50_95": 0.4043,
            "precision": 0.7614,
            "recall": 0.6866,
            "mAP_small": 0.2639,
            "mAP_medium": 0.3471,
            "mAP_large": 0.4699,
        },
    },
    8: {  # E08 - B only (binary 01000)
        "name": "E08_B",
        "phases": ["B"],
        "short_name": "B",
        "description": "Center Loss Weight",
        "source": "B3_03",
        "metrics": {
            "mAP50": 0.6918152709114648,
            "mAP50_95": 0.3568002135337611,
            "precision": 0.7636518686315,
            "recall": 0.6666677633124232,
            "mAP_small": 0.24858614374844326,
            "mAP_medium": 0.3220632005313519,
            "mAP_large": 0.404996933366447,
        },
    },
    4: {  # E04 - C1 only (binary 00100)
        "name": "E04_C1",
        "phases": ["C1"],
        "short_name": "C1",
        "description": "IoU Clipping",
        "source": "C1_01",
        "metrics": {
            "mAP50": 0.6957686830743368,
            "mAP50_95": 0.35462942989366414,
            "precision": 0.7807273905978538,
            "recall": 0.6500676786879359,
            "mAP_small": 0.24855839154708875,
            "mAP_medium": 0.32181639206220886,
            "mAP_large": 0.4063432385489556,
        },
    },
    2: {  # E02 - C2 only (binary 00010)
        "name": "E02_C2",
        "phases": ["C2"],
        "short_name": "C2",
        "description": "DFL Clipping",
        "source": "C2_20",
        "metrics": {
            "mAP50": 0.6866,
            "mAP50_95": 0.3548,
            "precision": 0.7677,
            "recall": 0.6601,
            "mAP_small": 0.2561,
            "mAP_medium": 0.3139,
            "mAP_large": 0.4117,
        },
    },
    1: {  # E01 - D only (binary 00001)
        "name": "E01_D",
        "phases": ["D"],
        "short_name": "D",
        "description": "TAL Parameters",
        "source": "D2_03",
        "metrics": {
            "mAP50": 0.6875,
            "mAP50_95": 0.3490,
            "precision": 0.7514,
            "recall": 0.6790,
            "mAP_small": 0.2602,
            "mAP_medium": 0.3111,
            "mAP_large": 0.4013,
        },
    },
}

# Baseline metrics for comparison
BASELINE_METRICS = EXISTING_RESULTS[0]["metrics"]


# =============================================================================
# COMBINATION GENERATOR
# =============================================================================

def generate_all_combinations() -> List[Dict[str, Any]]:
    """
    Generate all 32 combinations using proper binary encoding.

    Binary encoding (5 bits):
        Bit 4 (value 16): A  - Alpha Schedule
        Bit 3 (value 8):  B  - Center Loss Weight
        Bit 2 (value 4):  C1 - IoU Clipping
        Bit 1 (value 2):  C2 - DFL Clipping
        Bit 0 (value 1):  D  - TAL Parameters

    Examples:
        0  = 00000 = baseline (no phases)
        1  = 00001 = D only
        2  = 00010 = C2 only
        3  = 00011 = C2 + D
        4  = 00100 = C1 only
        8  = 01000 = B only
        16 = 10000 = A only
        31 = 11111 = A + B + C1 + C2 + D (all)
    """
    experiments = []

    for i in range(32):
        # Determine active phases based on bit flags
        active_phases = []
        for phase in PHASE_ORDER:
            bit_pos = PHASE_MODIFICATIONS[phase]["bit"]
            if i & (1 << bit_pos):
                active_phases.append(phase)

        # Build config from baseline + active modifications
        config = build_config_for_phases(active_phases)

        # Generate name
        if not active_phases:
            name = "E00_baseline"
            description = "Baseline (no modifications)"
            short_name = "baseline"
        else:
            phases_str = "_".join(active_phases)
            name = f"E{i:02d}_{phases_str}"
            phase_descs = [PHASE_MODIFICATIONS[p]["name"] for p in active_phases]
            description = " + ".join(phase_descs)
            short_name = "+".join(active_phases)

        # Check if we already have results
        needs_training = i not in EXISTING_RESULTS

        experiments.append({
            "name": name,
            "index": i,
            "binary": format(i, '05b'),
            "phases": active_phases,
            "short_name": short_name,
            "description": description,
            "config": config,
            "num_components": len(active_phases),
            "needs_training": needs_training,
        })

    return experiments


def get_experiments_to_run() -> List[Dict[str, Any]]:
    """Get only experiments that need training."""
    all_exps = generate_all_combinations()
    return [e for e in all_exps if e["needs_training"]]


def print_experiment_matrix():
    """Print the complete experiment matrix."""

    experiments = generate_all_combinations()

    print("\n" + "=" * 120)
    print("PHASE E: COMPLETE COMBINATION MATRIX")
    print("=" * 120)

    print("\nBinary Encoding (5 bits): A B C1 C2 D")
    print("  Bit 4 (16): A  = Alpha Schedule (α: 0.9→0.4, boost=1.5)")
    print("  Bit 3 (8):  B  = Center Loss Weight (0.05→0.01)")
    print("  Bit 2 (4):  C1 = IoU Clipping (6→2)")
    print("  Bit 1 (2):  C2 = DFL Clipping (8→5)")
    print("  Bit 0 (1):  D  = TAL Parameters (topk=8)")

    print("\n" + "-" * 120)
    print(f"{'Index':<6} {'Name':<20} {'Binary':<8} {'Phases':<20} {'Description':<40} {'Status'}")
    print("-" * 120)

    to_run = 0
    existing = 0

    for exp in experiments:
        if exp["needs_training"]:
            status = "🔄 TO RUN"
            to_run += 1
        else:
            source = EXISTING_RESULTS.get(exp["index"], {}).get("source", "")
            status = f"✅ SKIP ({source})"
            existing += 1

        print(f"{exp['index']:<6} {exp['name']:<20} {exp['binary']:<8} {exp['short_name']:<20} "
              f"{exp['description']:<40} {status}")

    print("-" * 120)
    print(f"\nSummary:")
    print(f"  ✅ Existing results (skip): {existing}")
    print(f"  🔄 New experiments (run):   {to_run}")
    print(f"  📊 Total combinations:      32")
    print(f"  ⏱️  Estimated time:          ~{to_run} hours")
    print("=" * 120)


# =============================================================================
# EPOCH SYNC CALLBACK
# =============================================================================

def on_train_epoch_start(trainer):
    """Sync current epoch to loss function for dynamic scheduling."""
    epoch = trainer.epoch
    try:
        if hasattr(trainer, 'criterion') and trainer.criterion is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, '_sync_bbox_loss_state'):
                trainer.criterion._sync_bbox_loss_state()
            if epoch % 10 == 0:
                print(f"    [Epoch Sync] Epoch {epoch}")
            return
    except Exception:
        pass
    try:
        trainer.model.current_epoch = epoch
        if epoch % 10 == 0:
            print(f"    [Epoch Sync] Epoch {epoch}")
    except Exception:
        pass


# =============================================================================
# RESULTS MANAGER
# =============================================================================

class PhaseEResults:
    """Manages Phase E results, including existing results from previous phases."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.project_dir / "results.json"
        self.data = self._load()
        self._ensure_existing_results()

    def _load(self) -> Dict:
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {
            "experiments": [],
            "failed": [],
            "best_config": None,
            "analysis_timestamp": None,
        }

    def _ensure_existing_results(self):
        """Add existing results from previous phases if not present."""
        existing_names = {e["name"] for e in self.data["experiments"]}

        for index, result in EXISTING_RESULTS.items():
            if result["name"] not in existing_names:
                # Build the correct config for this experiment's phases
                config = build_config_for_phases(result["phases"])

                self.data["experiments"].append({
                    "name": result["name"],
                    "index": index,
                    "binary": format(index, '05b'),
                    "phases": result["phases"],
                    "short_name": result["short_name"],
                    "description": result["description"],
                    "num_components": len(result["phases"]),
                    "source": result["source"],
                    "config": config,
                    "metrics": result["metrics"],
                    "time_hours": 0,
                    "timestamp": "from_previous_phase",
                })
            else:
                # Update existing entry with correct config if needed
                for exp in self.data["experiments"]:
                    if exp["name"] == result["name"]:
                        exp["config"] = build_config_for_phases(result["phases"])
                        break

        self.save()

    def save(self):
        with open(self.results_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_completed(self, name: str) -> bool:
        return any(e["name"] == name for e in self.data["experiments"])

    def add_experiment(self, experiment: Dict, metrics: Dict, time_hours: float):
        # Remove if exists (for re-runs)
        self.data["experiments"] = [
            e for e in self.data["experiments"] if e["name"] != experiment["name"]
        ]

        self.data["experiments"].append({
            "name": experiment["name"],
            "index": experiment["index"],
            "binary": experiment["binary"],
            "phases": experiment["phases"],
            "short_name": experiment["short_name"],
            "description": experiment["description"],
            "num_components": experiment["num_components"],
            "config": experiment["config"],
            "metrics": metrics,
            "time_hours": time_hours,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def add_failed(self, name: str, error: str):
        self.data["failed"].append({
            "name": name,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def get_experiment(self, name: str) -> Optional[Dict]:
        for e in self.data["experiments"]:
            if e["name"] == name:
                return e
        return None

    def get_progress(self) -> Tuple[int, int]:
        """Get progress for NEW experiments only."""
        new_completed = sum(
            1 for e in self.data["experiments"]
            if e["index"] not in EXISTING_RESULTS and e.get("timestamp") != "from_previous_phase"
        )
        return new_completed, 26  # 32 - 6 existing = 26 new


# =============================================================================
# COCO EVALUATION
# =============================================================================

def run_coco_evaluation(pred_json: Path, ann_file: str) -> Dict[str, float]:
    """Run COCO evaluation with size-stratified metrics."""
    default_metrics = {"mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0}

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not Path(ann_file).exists() or not pred_json.exists():
            return default_metrics

        coco_gt = COCO(str(ann_file))

        filename_to_id = {}
        for img_id, img_info in coco_gt.imgs.items():
            filename = img_info['file_name']
            filename_to_id[filename] = img_id
            filename_to_id[Path(filename).stem] = img_id

        with open(pred_json, 'r') as f:
            predictions = json.load(f)

        converted = []
        for pred in predictions:
            img_id = pred.get('image_id')
            if isinstance(img_id, str):
                if img_id in filename_to_id:
                    pred['image_id'] = filename_to_id[img_id]
                elif Path(img_id).stem in filename_to_id:
                    pred['image_id'] = filename_to_id[Path(img_id).stem]
                else:
                    continue
            converted.append(pred)

        if not converted:
            return default_metrics

        converted_json = pred_json.parent / 'predictions_converted.json'
        with open(converted_json, 'w') as f:
            json.dump(converted, f)

        coco_dt = coco_gt.loadRes(str(converted_json))
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

        coco_eval.params.areaRng = [
            [0, 1e5 ** 2], [0, 48 ** 2], [48 ** 2, 128 ** 2], [128 ** 2, 1e5 ** 2]
        ]
        coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large']

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        return {
            "mAP_small": float(stats[3]) if len(stats) > 3 else 0.0,
            "mAP_medium": float(stats[4]) if len(stats) > 4 else 0.0,
            "mAP_large": float(stats[5]) if len(stats) > 5 else 0.0,
        }

    except Exception as e:
        print(f"    Warning: COCO eval error: {e}")
        return default_metrics


def compute_size_based_map(model, data_yaml: str, imgsz: int) -> Dict[str, float]:
    """Compute size-stratified mAP."""
    size_metrics = {"mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0}

    try:
        print("    Running validation with save_json=True...")
        val_results = model.val(data=data_yaml, imgsz=imgsz, save_json=True, verbose=False)

        pred_json = None
        if hasattr(val_results, 'save_dir'):
            save_dir = Path(val_results.save_dir)
            for name in ['predictions.json', 'results.json', 'detections.json']:
                candidate = save_dir / name
                if candidate.exists():
                    pred_json = candidate
                    break

        if pred_json and pred_json.exists():
            size_metrics = run_coco_evaluation(pred_json, COCO_ANN_FILE)

    except Exception as e:
        print(f"    Warning: Size-based metrics failed: {e}")

    return size_metrics


# =============================================================================
# TRAINING
# =============================================================================

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_experiment(experiment: Dict) -> Tuple[Dict[str, float], float]:
    """Run a single training experiment."""

    name = experiment["name"]
    config = experiment["config"]

    print(f"\n{'=' * 90}")
    print(f"EXPERIMENT: {name} (index={experiment['index']}, binary={experiment['binary']})")
    print(f"{'=' * 90}")
    print(f"  Phases: {experiment['short_name']}")
    print(f"  Description: {experiment['description']}")
    print(f"  Components: {experiment['num_components']}/5")
    print(f"{'=' * 90}")

    if experiment['phases']:
        print("\n  Active Modifications:")
        for phase in experiment['phases']:
            info = PHASE_MODIFICATIONS[phase]
            print(f"    [{phase}] {info['name']}: {info['description']}")
            for param, value in info['params'].items():
                print(f"         {param}: {value}")

    print(f"\n  Full Configuration:")
    print(f"    alpha_start: {config['alpha_start']}, alpha_end: {config['alpha_end']}")
    print(f"    alpha_min: {config['alpha_min']}, alpha_max: {config['alpha_max']}")
    print(f"    small_obj_px: {config['small_obj_px']}, small_obj_boost: {config['small_obj_boost']}")
    print(f"    center_loss_weight_init: {config['center_loss_weight_init']}, "
          f"center_loss_weight_min: {config['center_loss_weight_min']}")
    print(f"    iou_clip: {config['iou_clip_start']}→{config['iou_clip_end']}")
    print(f"    dfl_clip: {config['dfl_clip_start']}→{config['dfl_clip_end']}")
    print(f"    tal_topk: {config['tal_topk']}, tal_alpha: {config['tal_alpha']}, "
          f"tal_beta: {config['tal_beta']}")

    print(f"\n{'=' * 90}\n")

    start_time = time.time()

    try:
        model = YOLO(MODEL_WEIGHTS)
        model.add_callback('on_train_epoch_start', on_train_epoch_start)
        print("  [OK] Epoch sync callback registered")

        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=name,

            # Phase A: Alpha Schedule (all parameters)
            alpha_start=config["alpha_start"],
            alpha_end=config["alpha_end"],
            alpha_min=config["alpha_min"],
            alpha_max=config["alpha_max"],
            small_obj_px=config["small_obj_px"],
            small_obj_boost=config["small_obj_boost"],

            # Phase B: Center Loss Weight
            center_loss_weight_init=config["center_loss_weight_init"],
            center_loss_weight_min=config["center_loss_weight_min"],
            center_loss_decay_epochs=config["center_loss_decay_epochs"],

            # Phase C1 & C2: Loss Clipping
            iou_clip_start=config["iou_clip_start"],
            iou_clip_end=config["iou_clip_end"],
            dfl_clip_start=config["dfl_clip_start"],
            dfl_clip_end=config["dfl_clip_end"],

            # Phase D: TAL Parameters
            tal_topk=config["tal_topk"],
            tal_alpha=config["tal_alpha"],
            tal_beta=config["tal_beta"],

            # Reproducibility
            seed=0,
            deterministic=True,
        )

        r = results.results_dict
        metrics = {
            "mAP50": float(r.get("metrics/mAP50(B)", 0)),
            "mAP50_95": float(r.get("metrics/mAP50-95(B)", 0)),
            "precision": float(r.get("metrics/precision(B)", 0)),
            "recall": float(r.get("metrics/recall(B)", 0)),
        }

        print("\n  Computing size-based mAP...")
        size_metrics = compute_size_based_map(model, DATA_YAML, IMG_SIZE)
        metrics.update(size_metrics)

        training_time = (time.time() - start_time) / 3600

        # Results with comparison to baseline
        print(f"\n{'=' * 70}")
        print(f"COMPLETED: {name}")
        print(f"{'=' * 70}")

        bm = BASELINE_METRICS
        for metric, value in metrics.items():
            baseline_val = bm.get(metric, 0)
            if baseline_val > 0:
                delta = ((value / baseline_val) - 1) * 100
                print(f"    {metric:<12} {value:.4f}  ({delta:+.1f}% vs baseline)")
            else:
                print(f"    {metric:<12} {value:.4f}")

        print(f"    {'Time':<12} {training_time:.2f}h")
        print(f"{'=' * 70}\n")

        cleanup()
        return metrics, training_time

    except Exception as e:
        print(f"\n  ERROR: {e}")
        cleanup()
        raise


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(results: PhaseEResults):
    """Analyze all results including existing ones."""

    experiments = results.data["experiments"]
    bm = BASELINE_METRICS

    print("\n" + "=" * 120)
    print("PHASE E: COMPLETE ANALYSIS REPORT")
    print("=" * 120)

    # Progress
    new_completed, new_total = results.get_progress()
    total_with_existing = len(experiments)
    print(f"\nNew experiments completed: {new_completed}/{new_total}")
    print(f"Total experiments (with existing): {total_with_existing}/32")

    # Sort by mAP_small
    sorted_exps = sorted(
        experiments,
        key=lambda x: x.get("metrics", {}).get("mAP_small", 0),
        reverse=True
    )

    print(f"\n{'=' * 120}")
    print("ALL RESULTS (sorted by mAP_small)")
    print(f"{'=' * 120}")

    print(f"\n{'#':<4} {'Name':<22} {'Phases':<20} {'mAP_small':<12} {'Δ%':<10} "
          f"{'mAP50-95':<12} {'Recall':<10} {'Source'}")
    print("-" * 120)

    for i, exp in enumerate(sorted_exps, 1):
        m = exp.get("metrics", {})
        mAP_small = m.get("mAP_small", 0)
        delta = ((mAP_small / bm["mAP_small"]) - 1) * 100 if bm["mAP_small"] > 0 else 0

        source = exp.get("source", "Phase E")
        marker = " ★" if i == 1 else ""

        print(f"{i:<4} {exp['name']:<22} {exp.get('short_name', ''):<20} "
              f"{mAP_small:<12.4f} {delta:>+8.1f}% "
              f"{m.get('mAP50_95', 0):<12.4f} {m.get('recall', 0):<10.4f} "
              f"{source}{marker}")

    # Top 5 tables
    for metric in ["mAP_small", "mAP50_95", "recall"]:
        print(f"\n{'=' * 90}")
        print(f"TOP 5 BY {metric.upper()}")
        print(f"{'=' * 90}")

        sorted_by = sorted(
            experiments,
            key=lambda x: x.get("metrics", {}).get(metric, 0),
            reverse=True
        )[:5]

        for i, exp in enumerate(sorted_by, 1):
            m = exp.get("metrics", {})
            val = m.get(metric, 0)
            baseline_val = bm.get(metric, 0)
            delta = ((val / baseline_val) - 1) * 100 if baseline_val > 0 else 0
            print(f"  {i}. {exp['name']:<22} {exp.get('short_name', ''):<20} "
                  f"{val:.4f} ({delta:+.1f}%)")

    # Best overall
    best = sorted_exps[0] if sorted_exps else None
    if best:
        print(f"\n{'=' * 120}")
        print("🏆 BEST CONFIGURATION FOR SMALL OBJECTS")
        print(f"{'=' * 120}")
        print(f"\n  Experiment: {best['name']}")
        print(f"  Phases: {best.get('short_name', '')}")
        print(f"  Description: {best.get('description', '')}")
        print(f"\n  Performance vs Baseline:")

        for metric, value in best.get("metrics", {}).items():
            baseline_val = bm.get(metric, 0)
            if baseline_val > 0:
                delta = ((value / baseline_val) - 1) * 100
                print(f"    {metric:<15} {value:.4f} ({delta:+.1f}%)")

        # Print the winning configuration
        if best.get("phases"):
            print(f"\n  Winning Configuration:")
            for phase in best["phases"]:
                info = PHASE_MODIFICATIONS[phase]
                print(f"    [{phase}] {info['name']}:")
                for param, value in info['params'].items():
                    print(f"         {param}: {value}")

        results.data["best_config"] = best["name"]

    results.data["analysis_timestamp"] = datetime.now().isoformat()
    results.save()

    print(f"\n{'=' * 120}")
    print(f"Results saved to: {results.results_file}")
    print(f"{'=' * 120}\n")


def print_interaction_analysis(results: PhaseEResults):
    """Analyze interactions between phases."""

    experiments = results.data["experiments"]
    bm = BASELINE_METRICS["mAP_small"]

    print("\n" + "=" * 100)
    print("INTERACTION ANALYSIS (mAP_small)")
    print("=" * 100)

    # Get individual effects from existing results
    individual_effects = {}
    for phase in PHASE_ORDER:
        bit = PHASE_MODIFICATIONS[phase]["bit"]
        index = 1 << bit  # 2^bit
        exp = next((e for e in experiments if e.get("index") == index), None)
        if exp:
            individual_effects[phase] = exp["metrics"]["mAP_small"] - bm

    print("\n1. Individual Phase Effects (from previous phases):")
    print("-" * 60)
    for phase, effect in individual_effects.items():
        pct = (effect / bm) * 100 if bm > 0 else 0
        info = PHASE_MODIFICATIONS[phase]
        print(f"   {phase} ({info['name']:<20}): {effect:+.4f} ({pct:+.1f}%)")

    # Pairwise interactions
    print("\n2. Pairwise Interactions:")
    print("-" * 90)
    print(f"   {'Phases':<15} {'Expected':<12} {'Actual':<12} {'Interaction':<15} {'Type'}")
    print("   " + "-" * 85)

    for i, p1 in enumerate(PHASE_ORDER):
        for p2 in PHASE_ORDER[i + 1:]:
            # Calculate expected index for combination
            bit1 = PHASE_MODIFICATIONS[p1]["bit"]
            bit2 = PHASE_MODIFICATIONS[p2]["bit"]
            combo_index = (1 << bit1) | (1 << bit2)

            combo_exp = next((e for e in experiments if e.get("index") == combo_index), None)

            if combo_exp and p1 in individual_effects and p2 in individual_effects:
                expected = bm + individual_effects[p1] + individual_effects[p2]
                actual = combo_exp["metrics"].get("mAP_small", 0)
                interaction = actual - expected

                if interaction > 0.005:
                    int_type = "SYNERGY ✓"
                elif interaction < -0.005:
                    int_type = "ANTAGONISM ✗"
                else:
                    int_type = "NEUTRAL"

                print(f"   {p1}+{p2:<10} {expected:<12.4f} {actual:<12.4f} "
                      f"{interaction:>+12.4f}   {int_type}")
            elif combo_exp is None:
                print(f"   {p1}+{p2:<10} {'(not yet run)':<50}")

    print("\n" + "=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def run_all_experiments(results: PhaseEResults):
    """Run all NEW experiments."""

    to_run = get_experiments_to_run()

    print(f"\n{'=' * 90}")
    print(f"PHASE E: RUNNING {len(to_run)} NEW COMBINATION EXPERIMENTS")
    print(f"{'=' * 90}")

    existing_indices = sorted(EXISTING_RESULTS.keys())
    existing_names = [EXISTING_RESULTS[i]["name"] for i in existing_indices]
    print(f"\n✅ Skipping {len(EXISTING_RESULTS)} experiments with existing results:")
    print(f"   {', '.join(existing_names)}")
    print(f"\n🔄 Running {len(to_run)} new combinations")

    for exp in to_run:
        if results.is_completed(exp["name"]):
            print(f"\n>> Skipping {exp['name']} (already completed)")
            continue

        try:
            metrics, time_hours = train_experiment(exp)
            results.add_experiment(exp, metrics, time_hours)

            completed, total = results.get_progress()
            print(f"\n[Progress: {completed}/{total} new experiments completed]")

        except Exception as e:
            print(f"\n>> FAILED: {exp['name']} - {e}")
            results.add_failed(exp["name"], str(e))
            continue

    analyze_results(results)
    print_interaction_analysis(results)


def main():
    parser = argparse.ArgumentParser(
        description="Phase E: Combined Configuration Ablation (26 new experiments)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python phase_e_combined_ablation.py              # Run all 26 new experiments
    python phase_e_combined_ablation.py --show-matrix
    python phase_e_combined_ablation.py --analyze
    python phase_e_combined_ablation.py --experiment E31
        """
    )

    parser.add_argument("--experiment", type=str, help="Run specific experiment (e.g., E31 or 31)")
    parser.add_argument("--resume", action="store_true", help="Resume (skip completed)")
    parser.add_argument("--show-matrix", action="store_true", help="Show experiment matrix")
    parser.add_argument("--analyze", action="store_true", help="Analyze only")
    parser.add_argument("--reset-existing", action="store_true",
                        help="Reset and reload existing results from previous phases")

    args = parser.parse_args()

    if args.show_matrix:
        print_experiment_matrix()
        return

    results = PhaseEResults(PROJECT_DIR)

    # Reset existing results if requested
    if args.reset_existing:
        print("\n🔄 Resetting existing results from previous phases...")
        # Remove existing entries and re-add them
        results.data["experiments"] = [
            e for e in results.data["experiments"]
            if e.get("timestamp") != "from_previous_phase"
        ]
        results._ensure_existing_results()
        print("✅ Existing results reloaded with correct configurations")
        return

    if args.analyze:
        analyze_results(results)
        print_interaction_analysis(results)
        return

    # Header
    print("\n" + "=" * 90)
    print("PHASE E: COMBINED CONFIGURATION ABLATION STUDY")
    print("=" * 90)

    print("\nBinary Encoding (5 bits): A B C1 C2 D")
    print("  Bit 4 (16): A  = Alpha Schedule (α: 0.9→0.4, boost=1.5)")
    print("  Bit 3 (8):  B  = Center Loss Weight (0.05→0.01)")
    print("  Bit 2 (4):  C1 = IoU Clipping (6→2)")
    print("  Bit 1 (2):  C2 = DFL Clipping (8→5)")
    print("  Bit 0 (1):  D  = TAL Parameters (topk=8)")

    existing_indices = sorted(EXISTING_RESULTS.keys())
    existing_info = [f"E{i:02d}={EXISTING_RESULTS[i]['short_name']}" for i in existing_indices]
    print(f"\n✅ Using existing results: {', '.join(existing_info)}")
    print(f"🔄 Running 26 new combination experiments")

    completed, total = results.get_progress()
    print(f"\nProgress: {completed}/{total} new experiments completed")

    # Run specific experiment
    if args.experiment:
        # Parse experiment number
        exp_str = args.experiment.upper().replace("E", "")
        try:
            exp_index = int(exp_str)
        except ValueError:
            print(f"\nERROR: Invalid experiment format: {args.experiment}")
            return

        all_exps = generate_all_combinations()
        exp = next((e for e in all_exps if e["index"] == exp_index), None)

        if not exp:
            print(f"\nERROR: Experiment E{exp_index:02d} not found (valid range: 0-31)")
            return

        if not exp["needs_training"]:
            source = EXISTING_RESULTS.get(exp["index"], {}).get("source", "")
            print(f"\n{exp['name']} already has results from {source}. No need to run.")
            return

        if results.is_completed(exp["name"]) and not args.resume:
            print(f"\n{exp['name']} already completed. Use --resume to re-run.")
            return

        metrics, time_hours = train_experiment(exp)
        results.add_experiment(exp, metrics, time_hours)
        analyze_results(results)
        return

    # Run all new experiments
    run_all_experiments(results)


if __name__ == "__main__":
    main()
