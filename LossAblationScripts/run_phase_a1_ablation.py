#!/usr/bin/env python3
"""
Phase A1 Ablation: Size-Aware Weighting Grid Search
====================================================

Overview
--------
Phase A1 performs a sequential ablation study over size-aware weighting
parameters to optimize small object detection performance.

NOTE: Phase A2, A3, A4 parameters are DISABLED (set to neutral/default values)
      to test size-aware weighting parameters in isolation.

Training Configuration
----------------------
    Epochs:     70
    Image Size: 640
    Batch Size: 64
    Model:      YOLOv12s

COCO Evaluation Ranges (Standard)
---------------------------------
    Small:  area < 32² pixels   (< 1024 px²)
    Medium: 32² ≤ area < 96²    (1024 - 9216 px²)
    Large:  area ≥ 96² pixels   (≥ 9216 px²)

Fixed Hyperparameters (Disabled/Neutral)
----------------------------------------
Phase A2 (Center Loss) - DISABLED:
    center_loss_weight_init:  0.0   (no center loss)
    center_loss_weight_min:   0.0   (no center loss)
    center_loss_decay_epochs: 35    (irrelevant when weight=0.0)

Phase A3 (Adaptive Clipping) - DISABLED:
    iou_clip_start:  100.0   (no clipping)
    iou_clip_end:    100.0   (no clipping)
    dfl_clip_start:  100.0   (no clipping)
    dfl_clip_end:    100.0   (no clipping)

Phase A4 (TAL Parameters) - DEFAULT:
    tal_topk:  10
    tal_alpha: 0.5
    tal_beta:  6.0

Phase A1 Fixed Parameters:
    alpha_min: 0.3
    alpha_max: 1.0

Step A1a: Alpha Schedule Grid (6×6 = 36 runs)
----------------------------------------------
Grid search over alpha_start and alpha_end values.

6 alpha_start values: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
6 alpha_end values:   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

Constraint: alpha_end <= alpha_start (alpha decays during training)

Grid Structure A1a:

                alpha_end
                0.3    0.4    0.5    0.6    0.7    0.8
             +------------------------------------------
       0.5   |   X      X      X      -      -      -
       0.6   |   X      X      X      X      -      -
alpha  0.7   |   X      X      X      X      X      -
_start 0.8   |   X      X      X      X      X      X
       0.9   |   X      X      X      X      X      X
       1.0   |   X      X      X      X      X      X

Legend: X = valid experiment (end <= start)
        - = invalid (end > start)

Fixed: small_obj_boost=1.0, small_obj_px=48

Valid combinations: 27 experiments
Estimated time: ~35 hours (at ~1.3 hours per 70-epoch run)

Step A1b: Small Object Boost (5 runs)
--------------------------------------
Using best alpha schedule from A1a, tests small object boost multipliers.

5 boost values: [1.0, 1.5, 2.0, 2.5, 3.0]
  - 1.0: No boost (baseline)
  - 1.5-3.0: Increasing emphasis on small objects

Fixed: small_obj_px=48, uses best alpha from A1a

Total: 5 experiments
Estimated time: ~7 hours (at ~1.3 hours per 70-epoch run)

Step A1c: Pixel Threshold (6 runs)
-----------------------------------
Using best alpha and boost from A1a/A1b, tests pixel thresholds defining
"small" objects.

6 px values: [24, 32, 40, 48, 56, 64]
  - Lower values: Only very small objects get boosted
  - Higher values: More objects qualify for boost

Uses best alpha from A1a and best boost from A1b.

Total: 6 experiments
Estimated time: ~8 hours (at ~1.3 hours per 70-epoch run)

Step A1d: Validation (7 runs)
------------------------------
Validates best configuration against baseline and tests sensitivity.

Experiments:
1. Baseline - Pure YOLO (no modifications)
2. Alpha schedule only (no boost)
3. Best configuration from A1a+A1b+A1c
4. Sensitivity: boost + 0.5
5. Sensitivity: boost - 0.5
6. Sensitivity: px + 8
7. Sensitivity: px - 8

Total: 7 experiments
Estimated time: ~9 hours (at ~1.3 hours per 70-epoch run)

Total Phase A1 Statistics
-------------------------
    Step A1a: 27 runs (alpha schedule grid, valid combinations only)
    Step A1b:  5 runs (small object boost)
    Step A1c:  6 runs (pixel threshold)
    Step A1d:  7 runs (validation)
    ---------------------------------
    Total:    45 runs
    Estimated: ~59 hours (~2.5 days at 70 epochs per run)

Overall Ablation Phases
-----------------------
    Phase A1: Size-Aware Weighting    (this script)
    Phase A2: Center Loss             (phase_a2_ablation.py)
    Phase A3: Adaptive Clipping       (phase_a3_ablation.py)
    Phase A4: TAL Parameters          (phase_a4_ablation.py)

Usage
-----
    python phase_a1_ablation.py               # Run full Phase A1
    python phase_a1_ablation.py --step A1a    # Run only A1a (alpha grid)
    python phase_a1_ablation.py --step A1b    # Run only A1b (boost)
    python phase_a1_ablation.py --step A1c    # Run only A1c (px threshold)
    python phase_a1_ablation.py --step A1d    # Run only A1d (validation)
    python phase_a1_ablation.py --status      # Show current progress
    python phase_a1_ablation.py --results     # Show results analysis

Author: Constantin
Project: YOLOv12 Small Object Detection Optimization
"""

import json
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_YAML = "/home/constantin/Doctorat/GunDataset_matched_cleanedHist/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_phaseA"

PREVIOUS_RESULTS_FILE = "runs_phaseAprev/results.json"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 64
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# PHASE A PARAMETERS
# =============================================================================

# Fixed parameters for Phase A
ALPHA_MIN = 0.3
ALPHA_MAX = 1.0

# Phase A1: Alpha schedule grid
ALPHA_START_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHA_END_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Phase A2: Small object boost
BOOST_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]

# Phase A3: Pixel threshold
PX_VALUES = [24, 32, 40, 48, 56, 64]

# Defaults for A1
DEFAULT_BOOST = 1.0
DEFAULT_PX = 48

# =============================================================================
# PHASE B, C, D: DISABLED (NEUTRAL VALUES)
# =============================================================================

PHASE_B_DISABLED = {
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
}

PHASE_C_DISABLED = {
    "iou_clip_start": 100.0,
    "iou_clip_end": 100.0,
    "dfl_clip_start": 100.0,
    "dfl_clip_end": 100.0,
}

PHASE_D_DEFAULT = {
    "tal_topk": 10,
    "tal_alpha": 0.5,
    "tal_beta": 6.0,
}

# =============================================================================
# COCO EVALUATION
# =============================================================================

COCO_ANN_FILE = "/home/constantin/Doctorat/GunDataset_matched_cleanedHist/annotations_coco_val.json"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def float_eq(a: float, b: float, eps: float = 1e-6) -> bool:
    """Compare two floats with tolerance."""
    return abs(a - b) <= eps


def is_valid_alpha_pair(start: float, end: float) -> bool:
    """Check if alpha_end <= alpha_start (valid configuration)."""
    return end <= start or float_eq(end, start)


def load_previous_results() -> List[Dict]:
    """Load experiments from previous results file."""
    if Path(PREVIOUS_RESULTS_FILE).exists():
        try:
            with open(PREVIOUS_RESULTS_FILE, 'r') as f:
                prev_data = json.load(f)
                return prev_data.get("experiments", [])
        except Exception as e:
            print(f"Warning: Could not load previous results: {e}")
    return []


# =============================================================================
# EPOCH SYNC CALLBACK
# =============================================================================

def on_train_epoch_start(trainer):
    """Sync epoch to loss function for dynamic alpha scheduling."""
    epoch = trainer.epoch
    try:
        if hasattr(trainer, 'criterion') and trainer.criterion is not None:
            trainer.criterion.epoch = epoch
            if hasattr(trainer.criterion, '_sync_bbox_loss_state'):
                trainer.criterion._sync_bbox_loss_state()
            if epoch % 10 == 0:
                print(f"[Epoch Sync] Epoch {epoch}")
    except:
        pass
    try:
        trainer.model.current_epoch = epoch
    except:
        pass


# =============================================================================
# RESULTS MANAGER
# =============================================================================

class PhaseAResults:
    """Manages Phase A results and tracks best values."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.project_dir / "results.json"
        self.data = self._load()

        self.previous_experiments = load_previous_results()
        if self.previous_experiments:
            print(f"Loaded {len(self.previous_experiments)} experiments from previous runs")

    def _load(self) -> Dict:
        """Load existing results."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {
            "phase_b_disabled": PHASE_B_DISABLED,
            "phase_c_disabled": PHASE_C_DISABLED,
            "phase_d_default": PHASE_D_DEFAULT,
            "grid_config": {
                "alpha_start_values": ALPHA_START_VALUES,
                "alpha_end_values": ALPHA_END_VALUES,
                "boost_values": BOOST_VALUES,
                "px_values": PX_VALUES,
            },
            "experiments": [],
            "a1_best_config": None,
            "a1_best_metrics": None,
            "a2_best_config": None,
            "a2_best_metrics": None,
            "a3_best_config": None,
            "a3_best_metrics": None,
            "final_best_config": None,
            "final_best_metrics": None,
            "current_phase": "A1",
        }

    def save(self):
        """Save results."""
        with open(self.results_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_all_experiments(self) -> List[Dict]:
        """Get all experiments (current + previous)."""
        return self.data["experiments"] + self.previous_experiments

    def is_completed(self, name: str) -> bool:
        """Check if experiment completed by name."""
        return any(e["name"] == name for e in self.data["experiments"])

    def add_experiment(self, name: str, phase: str, config: Dict, metrics: Dict, time_hours: float):
        """Add experiment result."""
        self.data["experiments"].append({
            "name": name,
            "phase": phase,
            "config": config,
            "metrics": metrics,
            "time_hours": time_hours,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()
        self._update_best(phase)

    def _update_best(self, phase: str, metric: str = "mAP_small"):
        """Update best configuration for a phase."""
        all_experiments = self.get_all_experiments()
        phase_results = [e for e in all_experiments if e.get("phase") == phase]

        if not phase_results:
            return

        sorted_results = sorted(
            phase_results,
            key=lambda x: x["metrics"].get(metric, 0),
            reverse=True
        )

        best = sorted_results[0]

        if phase == "A1":
            self.data["a1_best_config"] = best["config"]
            self.data["a1_best_metrics"] = best["metrics"]
        elif phase == "A2":
            self.data["a2_best_config"] = best["config"]
            self.data["a2_best_metrics"] = best["metrics"]
        elif phase == "A3":
            self.data["a3_best_config"] = best["config"]
            self.data["a3_best_metrics"] = best["metrics"]

        # Update final best across all phases
        if all_experiments:
            final_best = max(all_experiments, key=lambda x: x["metrics"].get(metric, 0))
            self.data["final_best_config"] = final_best["config"]
            self.data["final_best_metrics"] = final_best["metrics"]

        self.save()

    def get_a1_best(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best alpha_start and alpha_end from A1."""
        if self.data.get("a1_best_config"):
            cfg = self.data["a1_best_config"]
            return cfg.get("alpha_start"), cfg.get("alpha_end")

        all_experiments = self.get_all_experiments()
        a1_results = [e for e in all_experiments if e.get("phase") == "A1"]

        if a1_results:
            best = max(a1_results, key=lambda x: x["metrics"].get("mAP_small", 0))
            cfg = best["config"]
            return cfg.get("alpha_start"), cfg.get("alpha_end")

        return None, None

    def get_a2_best(self) -> Optional[float]:
        """Get best boost from A2."""
        if self.data.get("a2_best_config"):
            return self.data["a2_best_config"].get("small_obj_boost")

        all_experiments = self.get_all_experiments()
        a2_results = [e for e in all_experiments if e.get("phase") == "A2"]

        if a2_results:
            best = max(a2_results, key=lambda x: x["metrics"].get("mAP_small", 0))
            return best["config"].get("small_obj_boost")

        return None

    def get_a3_best(self) -> Optional[int]:
        """Get best px from A3."""
        if self.data.get("a3_best_config"):
            return self.data["a3_best_config"].get("small_obj_px")

        all_experiments = self.get_all_experiments()
        a3_results = [e for e in all_experiments if e.get("phase") == "A3"]

        if a3_results:
            best = max(a3_results, key=lambda x: x["metrics"].get("mAP_small", 0))
            return best["config"].get("small_obj_px")

        return None

    def get_phase_results(self, phase: str) -> List[Dict]:
        """Get all results for a phase."""
        return [e for e in self.get_all_experiments() if e["phase"] == phase]

    def print_a1_grid_status(self):
        """Print A1 (alpha) grid status."""
        print(f"\n{'=' * 90}")
        print("PHASE A1 GRID STATUS (Alpha Start × End)")
        print(f"Fixed: boost={DEFAULT_BOOST}, px={DEFAULT_PX}")
        print(f"Phase B/C: DISABLED, Phase D: DEFAULT")
        print(f"{'=' * 90}")

        status_map = {}
        source_map = {}

        for exp in self.get_all_experiments():
            if exp.get("phase") != "A1":
                continue
            config = exp.get("config", {})
            alpha_start = config.get("alpha_start")
            alpha_end = config.get("alpha_end")

            if alpha_start is not None and alpha_end is not None:
                mAP = exp["metrics"].get("mAP_small", 0)
                key = (alpha_start, alpha_end)

                if key not in status_map or exp in self.data["experiments"]:
                    status_map[key] = f"{mAP:.3f}"
                    source_map[key] = "new" if exp in self.data["experiments"] else "prev"

        # Header
        header = f"{'start\\end':<12}"
        for end_val in ALPHA_END_VALUES:
            header += f"{end_val:<10}"
        print(header)
        print("-" * 90)

        # Rows
        valid_count = 0
        completed_count = 0
        for start_val in ALPHA_START_VALUES:
            row = f"{start_val:<12}"
            for end_val in ALPHA_END_VALUES:
                key = (start_val, end_val)
                if not is_valid_alpha_pair(start_val, end_val):
                    row += f"{'--':<10}"
                elif key in status_map:
                    marker = "*" if source_map.get(key) == "prev" else ""
                    val_str = f"{status_map[key]}{marker}"
                    row += f"{val_str:<10}"
                    completed_count += 1
                    valid_count += 1
                else:
                    row += f"{'.':<10}"
                    valid_count += 1
            print(row)

        print(f"\nLegend: 0.XXX = mAP_small | * = from previous | . = Pending | -- = Invalid (end > start)")

        remaining = valid_count - completed_count
        print(f"Progress: {completed_count}/{valid_count} valid | Remaining: {remaining} (~{remaining * 1.3:.1f} hours)")

        # Show best
        best_start, best_end = self.get_a1_best()
        if best_start is not None:
            print(f"\nBest Alpha config: start={best_start}, end={best_end}")

    def print_a2_status(self):
        """Print A2 (boost) status."""
        best_start, best_end = self.get_a1_best()

        print(f"\n{'=' * 70}")
        print("PHASE A2 STATUS (Small Object Boost)")
        if best_start is not None:
            print(f"Using A1 best: alpha={best_start} → {best_end}")
        print(f"Fixed: px={DEFAULT_PX}")
        print(f"{'=' * 70}")

        results_map = {}
        source_map = {}

        for exp in self.get_all_experiments():
            if exp.get("phase") != "A2":
                continue
            config = exp.get("config", {})
            boost = config.get("small_obj_boost")

            if boost is not None:
                mAP = exp["metrics"].get("mAP_small", 0)
                if boost not in results_map or exp in self.data["experiments"]:
                    results_map[boost] = mAP
                    source_map[boost] = "new" if exp in self.data["experiments"] else "prev"

        print(f"{'Boost':<12} {'mAP_small':<12} {'Status':<10}")
        print("-" * 40)

        for boost in BOOST_VALUES:
            if boost in results_map:
                marker = "*" if source_map.get(boost) == "prev" else ""
                print(f"{boost:<12} {results_map[boost]:<12.4f} {'Done' + marker:<10}")
            else:
                print(f"{boost:<12} {'--':<12} {'Pending':<10}")

        # Show best
        best_boost = self.get_a2_best()
        if best_boost is not None:
            print(f"\nBest boost: {best_boost}")

    def print_a3_status(self):
        """Print A3 (px) status."""
        best_start, best_end = self.get_a1_best()
        best_boost = self.get_a2_best()

        print(f"\n{'=' * 70}")
        print("PHASE A3 STATUS (Pixel Threshold)")
        if best_start is not None:
            print(f"Using A1 best: alpha={best_start} → {best_end}")
        if best_boost is not None:
            print(f"Using A2 best: boost={best_boost}")
        print(f"{'=' * 70}")

        results_map = {}
        source_map = {}

        for exp in self.get_all_experiments():
            if exp.get("phase") != "A3":
                continue
            config = exp.get("config", {})
            px = config.get("small_obj_px")

            if px is not None:
                mAP = exp["metrics"].get("mAP_small", 0)
                if px not in results_map or exp in self.data["experiments"]:
                    results_map[px] = mAP
                    source_map[px] = "new" if exp in self.data["experiments"] else "prev"

        print(f"{'PX':<12} {'mAP_small':<12} {'Status':<10}")
        print("-" * 40)

        for px in PX_VALUES:
            if px in results_map:
                marker = "*" if source_map.get(px) == "prev" else ""
                print(f"{px:<12} {results_map[px]:<12.4f} {'Done' + marker:<10}")
            else:
                print(f"{px:<12} {'--':<12} {'Pending':<10}")

        # Show best
        best_px = self.get_a3_best()
        if best_px is not None:
            print(f"\nBest px: {best_px}")

    def analyze_results(self, phase: str = None, top_n: int = 10):
        """Analyze and print top results."""
        all_experiments = self.get_all_experiments()

        if phase:
            results = [e for e in all_experiments if e.get("phase") == phase]
            title = f"PHASE {phase} RESULTS"
        else:
            results = all_experiments
            title = "ALL PHASE A RESULTS"

        if not results:
            print(f"No results for {phase if phase else 'Phase A'} yet.")
            return

        sorted_results = sorted(
            results,
            key=lambda x: x["metrics"].get("mAP_small", 0),
            reverse=True
        )

        print(f"\n{'=' * 130}")
        print(f"TOP {min(top_n, len(sorted_results))} {title} (sorted by mAP_small)")
        print(f"{'=' * 130}")
        print(f"{'#':<4} {'Name':<30} {'α_start':<10} {'α_end':<10} {'boost':<8} {'px':<6} "
              f"{'mAP_small':<11} {'mAP_med':<11} {'mAP_large':<11} {'mAP50-95':<11}")
        print("-" * 130)

        for i, r in enumerate(sorted_results[:top_n], 1):
            m = r["metrics"]
            c = r["config"]
            marker = " **BEST**" if i == 1 else ""
            print(
                f"{i:<4} {r['name']:<30} "
                f"{c.get('alpha_start', 0):<10.2f} "
                f"{c.get('alpha_end', 0):<10.2f} "
                f"{c.get('small_obj_boost', 0):<8.1f} "
                f"{c.get('small_obj_px', 0):<6} "
                f"{m.get('mAP_small', 0):<11.4f} "
                f"{m.get('mAP_medium', 0):<11.4f} "
                f"{m.get('mAP_large', 0):<11.4f} "
                f"{m.get('mAP50_95', 0):<11.4f}{marker}"
            )


# =============================================================================
# COCO EVALUATION
# =============================================================================

def run_coco_evaluation(pred_json: Path, ann_file: str) -> Dict:
    """Run COCO evaluation with size-based metrics."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not Path(ann_file).exists() or not pred_json.exists():
            return {"mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0}

        coco_gt = COCO(str(ann_file))

        filename_to_id = {}
        for img_id, img_info in coco_gt.imgs.items():
            filename_to_id[img_info['file_name']] = img_id
            filename_to_id[Path(img_info['file_name']).stem] = img_id

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
            return {"mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0}

        converted_json = pred_json.parent / 'predictions_converted.json'
        with open(converted_json, 'w') as f:
            json.dump(converted, f)

        coco_dt = coco_gt.loadRes(str(converted_json))
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.areaRng = [
            [0, 1e5 ** 2],
            [0, 32 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2]
        ]
        coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large']
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "mAP_small": float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large": float(coco_eval.stats[5]),
        }
    except Exception as e:
        print(f"    Warning: COCO eval error: {e}")
        return {"mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0}


def compute_size_based_map(model, data_yaml: str, imgsz: int) -> Dict:
    """Compute size-based mAP using COCO evaluation."""
    try:
        val_results = model.val(data=data_yaml, imgsz=imgsz, save_json=True, verbose=False)
        if hasattr(val_results, 'save_dir'):
            pred_json = Path(val_results.save_dir) / 'predictions.json'
            if pred_json.exists():
                return run_coco_evaluation(pred_json, COCO_ANN_FILE)
    except Exception as e:
        print(f"    Warning: Size-based metrics failed: {e}")
    return {"mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0}


# =============================================================================
# TRAINING
# =============================================================================

def cleanup():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_experiment(
        name: str,
        alpha_start: float,
        alpha_end: float,
        small_obj_boost: float,
        small_obj_px: int,
) -> Tuple[Dict, float]:
    """Run a single training experiment."""

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 70}")
    print(f"  alpha_start:     {alpha_start}")
    print(f"  alpha_end:       {alpha_end}")
    print(f"  small_obj_boost: {small_obj_boost}")
    print(f"  small_obj_px:    {small_obj_px}")
    print(f"  (Phase B/C: DISABLED, Phase D: DEFAULT)")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    print("  [OK] Epoch sync callback registered")

    pb = PHASE_B_DISABLED
    pc = PHASE_C_DISABLED
    pd = PHASE_D_DEFAULT

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=name,
        # Phase A parameters (ACTIVE - what we're testing)
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_min=ALPHA_MIN,
        alpha_max=ALPHA_MAX,
        small_obj_px=small_obj_px,
        small_obj_boost=small_obj_boost,
        # Phase B parameters (DISABLED - no center loss)
        center_loss_weight_init=pb["center_loss_weight_init"],
        center_loss_weight_min=pb["center_loss_weight_min"],
        center_loss_decay_epochs=pb["center_loss_decay_epochs"],
        # Phase C parameters (DISABLED - no clipping)
        iou_clip_start=pc["iou_clip_start"],
        iou_clip_end=pc["iou_clip_end"],
        dfl_clip_start=pc["dfl_clip_start"],
        dfl_clip_end=pc["dfl_clip_end"],
        # Phase D parameters (DEFAULT)
        tal_topk=pd["tal_topk"],
        tal_alpha=pd["tal_alpha"],
        tal_beta=pd["tal_beta"],
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

    print("\n  Computing size-based mAP (COCO evaluation)...")
    size_metrics = compute_size_based_map(model, DATA_YAML, IMG_SIZE)
    metrics.update(size_metrics)

    training_time = (time.time() - start_time) / 3600

    print(f"\n{'=' * 50}")
    print(f"COMPLETED: {name}")
    print(f"{'=' * 50}")
    print(f"    mAP50:      {metrics['mAP50']:.4f}")
    print(f"    mAP50-95:   {metrics['mAP50_95']:.4f}")
    print(f"    mAP_small:  {metrics['mAP_small']:.4f}")
    print(f"    mAP_medium: {metrics['mAP_medium']:.4f}")
    print(f"    mAP_large:  {metrics['mAP_large']:.4f}")
    print(f"    time:       {training_time:.2f}h")
    print(f"{'=' * 50}\n")

    cleanup()
    return metrics, training_time


# =============================================================================
# PHASE A1: ALPHA SCHEDULE GRID SEARCH (6×6)
# =============================================================================

def run_phase_A1(results: PhaseAResults):
    """
    Phase A1: Grid Search over alpha_start × alpha_end

    6×6 Grid with boost and px fixed (neutral values)
    Only valid combinations where end <= start are tested.
    """

    print("\n" + "#" * 80)
    print("# PHASE A1: Alpha Schedule Grid Search (6×6)")
    print(f"# Fixed: boost={DEFAULT_BOOST}, px={DEFAULT_PX}")
    print("# Phase B/C: DISABLED, Phase D: DEFAULT")
    print("#" * 80)

    results.print_a1_grid_status()

    experiments = []
    exp_num = 1
    skipped_count = 0

    for alpha_start in ALPHA_START_VALUES:
        for alpha_end in ALPHA_END_VALUES:
            # Skip invalid combinations (end > start)
            if not is_valid_alpha_pair(alpha_start, alpha_end):
                continue

            name = f"A1_{exp_num:02d}_as{alpha_start}_ae{alpha_end}"

            if results.is_completed(name):
                print(f"  Skipping {name} (completed)")
                skipped_count += 1
                continue

            experiments.append({
                "name": name,
                "alpha_start": alpha_start,
                "alpha_end": alpha_end,
                "small_obj_boost": DEFAULT_BOOST,
                "small_obj_px": DEFAULT_PX,
            })
            exp_num += 1

    print(f"\n{len(experiments)} new experiments to run ({skipped_count} skipped)")

    if not experiments:
        print("All A1 experiments already completed!")
        results.analyze_results(phase="A1", top_n=10)
        return

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")

        config = {
            "alpha_start": exp["alpha_start"],
            "alpha_end": exp["alpha_end"],
            "small_obj_boost": exp["small_obj_boost"],
            "small_obj_px": exp["small_obj_px"],
            "phase_bcd_disabled": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            alpha_start=exp["alpha_start"],
            alpha_end=exp["alpha_end"],
            small_obj_boost=exp["small_obj_boost"],
            small_obj_px=exp["small_obj_px"],
        )

        results.add_experiment(name, "A1", config, metrics, time_hours)

        if i % 5 == 0:
            results.print_a1_grid_status()

    print("\n" + "=" * 80)
    print("PHASE A1 COMPLETE")
    print("=" * 80)
    results.print_a1_grid_status()
    results.analyze_results(phase="A1", top_n=10)

    # Update current phase
    results.data["current_phase"] = "A2"
    results.save()


# =============================================================================
# PHASE A2: SMALL OBJECT BOOST (5 runs)
# =============================================================================

def run_phase_A2(results: PhaseAResults):
    """
    Phase A2: Search over small_obj_boost

    Uses best alpha from A1.
    Tests: [1.0, 1.5, 2.0, 2.5, 3.0]
    """

    best_start, best_end = results.get_a1_best()

    if best_start is None:
        print("\nWarning: A1 not complete. Using default alpha (0.7 → 0.5).")
        best_start, best_end = 0.7, 0.5

    print("\n" + "#" * 80)
    print("# PHASE A2: Small Object Boost Search (5 runs)")
    print(f"# Using A1 best: alpha={best_start} → {best_end}")
    print(f"# Fixed: px={DEFAULT_PX}")
    print("# Phase B/C: DISABLED, Phase D: DEFAULT")
    print("#" * 80)

    results.print_a2_status()

    experiments = []
    skipped_count = 0

    for i, boost in enumerate(BOOST_VALUES, 1):
        name = f"A2_{i:02d}_boost{boost}"

        if results.is_completed(name):
            print(f"  Skipping {name} (completed)")
            skipped_count += 1
            continue

        experiments.append({
            "name": name,
            "alpha_start": best_start,
            "alpha_end": best_end,
            "small_obj_boost": boost,
            "small_obj_px": DEFAULT_PX,
        })

    print(f"\n{len(experiments)} new experiments to run ({skipped_count} skipped)")

    if not experiments:
        print("All A2 experiments already completed!")
        results.analyze_results(phase="A2", top_n=5)
        return

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")

        config = {
            "alpha_start": exp["alpha_start"],
            "alpha_end": exp["alpha_end"],
            "small_obj_boost": exp["small_obj_boost"],
            "small_obj_px": exp["small_obj_px"],
            "phase_bcd_disabled": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            alpha_start=exp["alpha_start"],
            alpha_end=exp["alpha_end"],
            small_obj_boost=exp["small_obj_boost"],
            small_obj_px=exp["small_obj_px"],
        )

        results.add_experiment(name, "A2", config, metrics, time_hours)

    print("\n" + "=" * 80)
    print("PHASE A2 COMPLETE")
    print("=" * 80)
    results.print_a2_status()
    results.analyze_results(phase="A2", top_n=5)

    # Update current phase
    results.data["current_phase"] = "A3"
    results.save()


# =============================================================================
# PHASE A3: PIXEL THRESHOLD (6 runs)
# =============================================================================

def run_phase_A3(results: PhaseAResults):
    """
    Phase A3: Search over small_obj_px

    Uses best alpha from A1 and best boost from A2.
    Tests: [24, 32, 40, 48, 56, 64]
    """

    best_start, best_end = results.get_a1_best()
    best_boost = results.get_a2_best()

    if best_start is None:
        print("\nWarning: A1 not complete. Using default alpha (0.7 → 0.5).")
        best_start, best_end = 0.7, 0.5

    if best_boost is None:
        print("\nWarning: A2 not complete. Using default boost (1.5).")
        best_boost = 1.5

    print("\n" + "#" * 80)
    print("# PHASE A3: Pixel Threshold Search (6 runs)")
    print(f"# Using A1 best: alpha={best_start} → {best_end}")
    print(f"# Using A2 best: boost={best_boost}")
    print("# Phase B/C: DISABLED, Phase D: DEFAULT")
    print("#" * 80)

    results.print_a3_status()

    experiments = []
    skipped_count = 0

    for i, px in enumerate(PX_VALUES, 1):
        name = f"A3_{i:02d}_px{px}"

        if results.is_completed(name):
            print(f"  Skipping {name} (completed)")
            skipped_count += 1
            continue

        experiments.append({
            "name": name,
            "alpha_start": best_start,
            "alpha_end": best_end,
            "small_obj_boost": best_boost,
            "small_obj_px": px,
        })

    print(f"\n{len(experiments)} new experiments to run ({skipped_count} skipped)")

    if not experiments:
        print("All A3 experiments already completed!")
        results.analyze_results(phase="A3", top_n=6)
        return

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")

        config = {
            "alpha_start": exp["alpha_start"],
            "alpha_end": exp["alpha_end"],
            "small_obj_boost": exp["small_obj_boost"],
            "small_obj_px": exp["small_obj_px"],
            "phase_bcd_disabled": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            alpha_start=exp["alpha_start"],
            alpha_end=exp["alpha_end"],
            small_obj_boost=exp["small_obj_boost"],
            small_obj_px=exp["small_obj_px"],
        )

        results.add_experiment(name, "A3", config, metrics, time_hours)

    print("\n" + "=" * 80)
    print("PHASE A3 COMPLETE")
    print("=" * 80)
    results.print_a3_status()
    results.analyze_results(phase="A3", top_n=6)

    # Update current phase
    results.data["current_phase"] = "A4"
    results.save()


# =============================================================================
# PHASE A4: VALIDATION (7 runs)
# =============================================================================

def run_phase_A4(results: PhaseAResults):
    """
    Phase A4: Validation combining best configs with variations.

    Experiments:
    1. Baseline - Pure YOLO (no modifications)
    2. Alpha schedule only (no boost)
    3. Best configuration from A1+A2+A3
    4. Sensitivity: boost + 0.5
    5. Sensitivity: boost - 0.5
    6. Sensitivity: px + 8
    7. Sensitivity: px - 8
    """

    best_start, best_end = results.get_a1_best()
    best_boost = results.get_a2_best()
    best_px = results.get_a3_best()

    if best_start is None:
        print("\nWarning: A1 not complete. Using default alpha (0.7 → 0.5).")
        best_start, best_end = 0.7, 0.5

    if best_boost is None:
        print("\nWarning: A2 not complete. Using default boost (1.5).")
        best_boost = 1.5

    if best_px is None:
        print("\nWarning: A3 not complete. Using default px (48).")
        best_px = 48

    print("\n" + "#" * 80)
    print("# PHASE A4: Validation (7 runs)")
    print(f"# Best config: α={best_start}→{best_end}, boost={best_boost}, px={best_px}")
    print("# Phase B/C: DISABLED, Phase D: DEFAULT")
    print("#" * 80)

    # Build experiment list
    experiments = [
        {
            "name": "A4_01_baseline",
            "desc": "Baseline - Pure YOLO (no modifications)",
            "alpha_start": 1.0,
            "alpha_end": 1.0,
            "small_obj_boost": 1.0,
            "small_obj_px": 64,
        },
        {
            "name": "A4_02_alpha_only",
            "desc": "Alpha schedule only (no boost)",
            "alpha_start": best_start,
            "alpha_end": best_end,
            "small_obj_boost": 1.0,
            "small_obj_px": 64,
        },
        {
            "name": "A4_03_best",
            "desc": "Best configuration from A1+A2+A3",
            "alpha_start": best_start,
            "alpha_end": best_end,
            "small_obj_boost": best_boost,
            "small_obj_px": best_px,
        },
        {
            "name": "A4_04_boost_up",
            "desc": f"Sensitivity: boost + 0.5 ({best_boost} → {best_boost + 0.5})",
            "alpha_start": best_start,
            "alpha_end": best_end,
            "small_obj_boost": best_boost + 0.5,
            "small_obj_px": best_px,
        },
        {
            "name": "A4_05_boost_down",
            "desc": f"Sensitivity: boost - 0.5 ({best_boost} → {max(1.0, best_boost - 0.5)})",
            "alpha_start": best_start,
            "alpha_end": best_end,
            "small_obj_boost": max(1.0, best_boost - 0.5),
            "small_obj_px": best_px,
        },
        {
            "name": "A4_06_px_up",
            "desc": f"Sensitivity: px + 8 ({best_px} → {best_px + 8})",
            "alpha_start": best_start,
            "alpha_end": best_end,
            "small_obj_boost": best_boost,
            "small_obj_px": best_px + 8,
        },
        {
            "name": "A4_07_px_down",
            "desc": f"Sensitivity: px - 8 ({best_px} → {max(16, best_px - 8)})",
            "alpha_start": best_start,
            "alpha_end": best_end,
            "small_obj_boost": best_boost,
            "small_obj_px": max(16, best_px - 8),
        },
    ]

    print(f"\n{len(experiments)} validation experiments:")
    for exp in experiments:
        print(f"  - {exp['name']}: {exp['desc']}")

    # Run experiments
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        if results.is_completed(name):
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (completed)")
            continue

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")
        print(f"   {exp['desc']}")

        config = {
            "alpha_start": exp["alpha_start"],
            "alpha_end": exp["alpha_end"],
            "small_obj_boost": exp["small_obj_boost"],
            "small_obj_px": exp["small_obj_px"],
            "phase_bcd_disabled": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            alpha_start=exp["alpha_start"],
            alpha_end=exp["alpha_end"],
            small_obj_boost=exp["small_obj_boost"],
            small_obj_px=exp["small_obj_px"],
        )

        results.add_experiment(name, "A4", config, metrics, time_hours)

    print("\n" + "=" * 80)
    print("PHASE A4 COMPLETE")
    print("=" * 80)
    results.analyze_results(phase="A4", top_n=7)

    # Print validation comparison
    print_validation_comparison(results)


def print_validation_comparison(results: PhaseAResults):
    """Print validation comparison table."""

    print(f"\n{'=' * 80}")
    print("VALIDATION COMPARISON")
    print(f"{'=' * 80}")

    all_exp = results.get_all_experiments()
    baseline = next((e for e in all_exp if e["name"] == "A4_01_baseline"), None)
    alpha_only = next((e for e in all_exp if e["name"] == "A4_02_alpha_only"), None)
    best_exp = next((e for e in all_exp if e["name"] == "A4_03_best"), None)

    if not baseline or not best_exp:
        print("  Validation experiments not complete yet.")
        return

    print(f"\n{'Config':<25} {'mAP_small':<12} {'mAP_medium':<12} {'mAP_large':<12} {'mAP50-95':<12}")
    print("-" * 75)

    b = baseline["metrics"]
    print(f"{'Baseline (pure YOLO)':<25} {b.get('mAP_small', 0):<12.4f} {b.get('mAP_medium', 0):<12.4f} {b.get('mAP_large', 0):<12.4f} {b.get('mAP50_95', 0):<12.4f}")

    if alpha_only:
        a = alpha_only["metrics"]
        print(f"{'Alpha only':<25} {a.get('mAP_small', 0):<12.4f} {a.get('mAP_medium', 0):<12.4f} {a.get('mAP_large', 0):<12.4f} {a.get('mAP50_95', 0):<12.4f}")

    best = best_exp["metrics"]
    print(f"{'Best config':<25} {best.get('mAP_small', 0):<12.4f} {best.get('mAP_medium', 0):<12.4f} {best.get('mAP_large', 0):<12.4f} {best.get('mAP50_95', 0):<12.4f}")

    # Improvement analysis
    print(f"\n{'=' * 80}")
    print("IMPROVEMENT OVER BASELINE")
    print(f"{'=' * 80}")

    metrics_to_compare = ["mAP_small", "mAP_medium", "mAP_large", "mAP50_95"]

    for metric in metrics_to_compare:
        b_val = baseline["metrics"].get(metric, 0)
        best_val = best_exp["metrics"].get(metric, 0)

        diff = best_val - b_val
        pct = (diff / max(b_val, 1e-6)) * 100

        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"  {metric:<12}: {b_val:.4f} → {best_val:.4f} ({arrow} {diff:+.4f}, {pct:+.1f}%)")


# =============================================================================
# SUMMARY
# =============================================================================

def print_final_summary(results: PhaseAResults):
    """Print final Phase A summary."""

    print("\n" + "=" * 100)
    print("PHASE A GRID SEARCH: FINAL SUMMARY")
    print("=" * 100)
    print("\nNOTE: Phase B/C parameters were DISABLED for this ablation study.")
    print("      Phase D (TAL) parameters were set to DEFAULT values.")
    print("      Only size-aware weighting parameters were tested.")

    # Count experiments per phase
    all_exp = results.get_all_experiments()
    a1_count = len([e for e in all_exp if e.get("phase") == "A1"])
    a2_count = len([e for e in all_exp if e.get("phase") == "A2"])
    a3_count = len([e for e in all_exp if e.get("phase") == "A3"])
    a4_count = len([e for e in all_exp if e.get("phase") == "A4"])

    # Calculate valid combinations
    a1_valid = sum(1 for s in ALPHA_START_VALUES for e in ALPHA_END_VALUES if e <= s)

    print(f"\nCompleted experiments:")
    print(f"  A1 (alpha grid):    {a1_count}/{a1_valid}")
    print(f"  A2 (boost):         {a2_count}/{len(BOOST_VALUES)}")
    print(f"  A3 (px threshold):  {a3_count}/{len(PX_VALUES)}")
    print(f"  A4 (validation):    {a4_count}/7")

    # Best from each phase
    if results.data.get("a1_best_config"):
        bc = results.data["a1_best_config"]
        bm = results.data["a1_best_metrics"]
        print(f"\nBest A1 (alpha schedule):")
        print(f"  alpha: {bc.get('alpha_start')} → {bc.get('alpha_end')}")
        print(f"  mAP_small={bm.get('mAP_small', 0):.4f}")

    if results.data.get("a2_best_config"):
        bc = results.data["a2_best_config"]
        bm = results.data["a2_best_metrics"]
        print(f"\nBest A2 (boost):")
        print(f"  boost={bc.get('small_obj_boost')}")
        print(f"  mAP_small={bm.get('mAP_small', 0):.4f}")

    if results.data.get("a3_best_config"):
        bc = results.data["a3_best_config"]
        bm = results.data["a3_best_metrics"]
        print(f"\nBest A3 (px threshold):")
        print(f"  px={bc.get('small_obj_px')}")
        print(f"  mAP_small={bm.get('mAP_small', 0):.4f}")

    # Final best
    if results.data.get("final_best_config"):
        bc = results.data["final_best_config"]
        bm = results.data["final_best_metrics"]

        print(f"""
+------------------------------------------------------------------------------+
|                         FINAL BEST CONFIGURATION                             |
+------------------------------------------------------------------------------+
|  alpha_start:     {str(bc.get('alpha_start', 'N/A')):<58}|
|  alpha_end:       {str(bc.get('alpha_end', 'N/A')):<58}|
|  small_obj_boost: {str(bc.get('small_obj_boost', 'N/A')):<58}|
|  small_obj_px:    {str(bc.get('small_obj_px', 'N/A')):<58}|
+------------------------------------------------------------------------------+
|  mAP_small:  {str(round(bm.get('mAP_small', 0), 4)):<63}|
|  mAP_medium: {str(round(bm.get('mAP_medium', 0), 4)):<63}|
|  mAP_large:  {str(round(bm.get('mAP_large', 0), 4)):<63}|
|  mAP50-95:   {str(round(bm.get('mAP50_95', 0), 4)):<63}|
+------------------------------------------------------------------------------+
""")

    total_time = sum(e.get("time_hours", 0) for e in results.data["experiments"])
    print(f"Total training time (this run): {total_time:.1f} hours")
    print(f"Results saved to: {results.results_file}")

    # Overall analysis
    results.analyze_results(top_n=15)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run Phase A grid search."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Phase A: Size-Aware Weighting Grid Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase_a_ablation.py              # Run full Phase A
  python phase_a_ablation.py --phase A1   # Run only A1 (alpha grid)
  python phase_a_ablation.py --phase A2   # Run only A2 (boost)
  python phase_a_ablation.py --phase A3   # Run only A3 (px threshold)
  python phase_a_ablation.py --phase A4   # Run only A4 (validation)
  python phase_a_ablation.py --status     # Show current progress
  python phase_a_ablation.py --results    # Show results analysis
        """
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "A1", "A2", "A3", "A4"],
                        help="Which phase to run (default: all)")
    parser.add_argument("--status", action="store_true",
                        help="Show current grid status and exit")
    parser.add_argument("--results", action="store_true",
                        help="Show results analysis and exit")

    args = parser.parse_args()

    # Check COCO annotation file
    if not Path(COCO_ANN_FILE).exists():
        print(f"\nERROR: COCO annotation file not found!")
        print(f"  Expected: {COCO_ANN_FILE}")
        return

    # Initialize results manager
    results = PhaseAResults(PROJECT_DIR)

    print(f"\nPhase A: Size-Aware Weighting Grid Search (ISOLATED)")
    print(f"Project: {PROJECT_DIR}")
    print(f"Previous results: {PREVIOUS_RESULTS_FILE}")

    # Show disabled parameters
    print(f"\n{'=' * 80}")
    print("FIXED HYPERPARAMETERS")
    print(f"{'=' * 80}")
    print(f"  Phase B: DISABLED (center_loss=0.0)")
    print(f"  Phase C: DISABLED (clip=100.0, no clipping)")
    print(f"  Phase D: DEFAULT (tal_topk={PHASE_D_DEFAULT['tal_topk']}, "
          f"tal_alpha={PHASE_D_DEFAULT['tal_alpha']}, tal_beta={PHASE_D_DEFAULT['tal_beta']})")

    # Calculate valid combinations
    a1_valid = sum(1 for s in ALPHA_START_VALUES for e in ALPHA_END_VALUES if e <= s)

    # Show grid configuration
    print(f"\n{'=' * 80}")
    print("PHASE A GRID CONFIGURATION (ACTIVE)")
    print(f"{'=' * 80}")
    print(f"  A1: Alpha Schedule Grid ({a1_valid} valid combinations)")
    print(f"      alpha_start: {ALPHA_START_VALUES}")
    print(f"      alpha_end:   {ALPHA_END_VALUES}")
    print(f"      Fixed: boost={DEFAULT_BOOST}, px={DEFAULT_PX}")
    print(f"")
    print(f"  A2: Small Object Boost ({len(BOOST_VALUES)} experiments)")
    print(f"      boost: {BOOST_VALUES}")
    print(f"      Uses best alpha from A1")
    print(f"")
    print(f"  A3: Pixel Threshold ({len(PX_VALUES)} experiments)")
    print(f"      px: {PX_VALUES}")
    print(f"      Uses best alpha from A1, best boost from A2")
    print(f"")
    print(f"  A4: Validation (7 experiments)")
    print(f"      Baseline + best config + sensitivity analysis")
    print(f"")
    print(f"  Total: {a1_valid + len(BOOST_VALUES) + len(PX_VALUES) + 7} experiments "
          f"(~{(a1_valid + len(BOOST_VALUES) + len(PX_VALUES) + 7) * 1.3:.0f} hours)")

    if args.status:
        results.print_a1_grid_status()
        results.print_a2_status()
        results.print_a3_status()
        return

    if args.results:
        results.analyze_results(top_n=20)
        return

    # Run phases
    if args.phase == "all":
        run_phase_A1(results)
        run_phase_A2(results)
        run_phase_A3(results)
        run_phase_A4(results)
    elif args.phase == "A1":
        run_phase_A1(results)
    elif args.phase == "A2":
        run_phase_A2(results)
    elif args.phase == "A3":
        run_phase_A3(results)
    elif args.phase == "A4":
        run_phase_A4(results)

    print_final_summary(results)


if __name__ == "__main__":
    main()
