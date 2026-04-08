#!/usr/bin/env python3
"""
Phase A2 Ablation: Center Loss Grid Search
===========================================

Overview
--------
Phase A2 performs a comprehensive grid search over center loss hyperparameters
to optimize small object detection performance. The center loss encourages
predicted bounding box centers to align with ground truth centers, which is
particularly beneficial for small objects where precise localization is critical.

NOTE: Phase A1, A3, A4 parameters are DISABLED (set to neutral/default values)
      to test center loss parameters in isolation.

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
Phase A1 (Size-Aware Weighting) - DISABLED:
    alpha_start:     1.0   (no area weighting)
    alpha_end:       1.0   (no area weighting)
    alpha_min:       1.0   (no area weighting)
    alpha_max:       1.0   (no area weighting)
    small_obj_boost: 1.0   (no boost)
    small_obj_px:    32    (irrelevant when boost=1.0)

Phase A3 (Adaptive Clipping) - DISABLED:
    iou_clip_start:  100.0   (no clipping)
    iou_clip_end:    100.0   (no clipping)
    dfl_clip_start:  100.0   (no clipping)
    dfl_clip_end:    100.0   (no clipping)

Phase A4 (TAL Parameters) - DEFAULT:
    tal_topk:  10
    tal_alpha: 0.5
    tal_beta:  6.0

Step A2a: Init × Min Grid (6×6 = 36 runs)
------------------------------------------
Explores the interaction between initial center loss weight and its minimum
(floor) value after decay.

6 init values: [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
6 min values:  [0.000, 0.005, 0.010, 0.015, 0.020, 0.025]

Constraint: min <= init (diagonal where min = init gives constant weight)

Grid Structure A2a:

             center_loss_weight_min
             0.000   0.005   0.010   0.015   0.020   0.025
           +-------------------------------------------------------
     0.01  |   X       X       X       -       -       -      (3)
     0.02  |   X       X       X       X       X       -      (5)
init 0.03  |   X       X       X       X       X       X      (6)
     0.05  |   X       X       X       X       X       X      (6)
     0.07  |   X       X       X       X       X       X      (6)
     0.10  |   X       X       X       X       X       X      (6)

Legend: X = valid experiment (min <= init)
        - = invalid (min > init)

Fixed: center_loss_decay_epochs = 35

Valid combinations: 32 experiments
Estimated time: ~42 hours (at ~1.3 hours per 70-epoch run)

Step A2b: Decay Epochs Search (6 runs)
---------------------------------------
Using the best (init, min) combination from A2a, explores different decay
schedules to find optimal training dynamics.

6 decay values: [15, 25, 35, 45, 55, 70]
  - 15: Aggressive decay (center loss reaches min early)
  - 25-35: Moderate decay (standard range)
  - 45-55: Gradual decay (center loss active longer)
  - 70: Very gradual decay (full training duration)

Uses best (init, min) from A2a.

Total: 6 experiments
Estimated time: ~8 hours (at ~1.3 hours per 70-epoch run)

Total Phase A2 Statistics
-------------------------
    Step A2a: 32 runs (init × min grid, valid combinations only)
    Step A2b:  6 runs (decay epochs search)
    ---------------------------------
    Total:    38 runs
    Estimated: ~50 hours (~2 days at 70 epochs per run)

Overall Ablation Phases
-----------------------
    Phase A1: Size-Aware Weighting    (phase_a1_ablation.py)
    Phase A2: Center Loss             (this script)
    Phase A3: Adaptive Clipping       (phase_a3_ablation.py)
    Phase A4: TAL Parameters          (phase_a4_ablation.py)

Usage
-----
    python phase_a2_ablation.py               # Run full Phase A2
    python phase_a2_ablation.py --step A2a    # Run only A2a (init × min)
    python phase_a2_ablation.py --step A2b    # Run only A2b (decay)
    python phase_a2_ablation.py --status      # Show current progress
    python phase_a2_ablation.py --results     # Show results analysis

Author: Constantin
Project: YOLOv12 Small Object Detection Optimization
"""


import json
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Set

import torch
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_YAML = "/home/constantin/Doctorat/GunDataset_matched_cleanedHist/data.yaml"
MODEL_WEIGHTS = "yolov12s.pt"
PROJECT_DIR = "runs_phaseB"

PREVIOUS_RESULTS_FILE = "runs_phaseBprev/results.json"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 64
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# PHASE A, C, D: DISABLED (NEUTRAL VALUES)
# =============================================================================

PHASE_A_DISABLED = {
    "alpha_start": 1.0,
    "alpha_end": 1.0,
    "alpha_min": 1.0,
    "alpha_max": 1.0,
    "small_obj_boost": 1.0,
    "small_obj_px": 32,
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
# PHASE B1 GRID PARAMETERS (init x min)
# =============================================================================

CENTER_LOSS_INIT_VALUES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
CENTER_LOSS_MIN_VALUES = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025]

# Fixed for Phase B1
CENTER_LOSS_DECAY_EPOCHS_DEFAULT = 35

# =============================================================================
# PHASE B2 PARAMETERS (decay epochs)
# =============================================================================

CENTER_LOSS_DECAY_VALUES = [15, 25, 35, 45, 55, 70]

# =============================================================================
# COCO EVALUATION
# =============================================================================

COCO_ANN_FILE = "/home/constantin/Doctorat/GunDataset_matched_cleanedHist/annotations_coco_val.json"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def float_eq(a: float, b: float, eps: float = 1e-9) -> bool:
    """Compare two floats with tolerance."""
    return abs(a - b) <= eps


def is_valid_combination(init_val: float, min_val: float) -> bool:
    """Check if min <= init (valid configuration)."""
    return min_val <= init_val or float_eq(min_val, init_val)


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


def config_matches(config: Dict, init_val: float, min_val: float, decay_val: int) -> bool:
    """Check if a config matches the given parameters."""
    return (
        float_eq(config.get("center_loss_weight_init", -1), init_val) and
        float_eq(config.get("center_loss_weight_min", -1), min_val) and
        config.get("center_loss_decay_epochs", -1) == decay_val
    )


# =============================================================================
# EPOCH SYNC CALLBACK
# =============================================================================

def on_train_epoch_start(trainer):
    """Sync epoch to loss function for dynamic scheduling."""
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

class PhaseBResults:
    """Manages Phase B results and tracks best values."""

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
            "phase_a_disabled": PHASE_A_DISABLED,
            "phase_c_disabled": PHASE_C_DISABLED,
            "phase_d_default": PHASE_D_DEFAULT,
            "grid_config": {
                "init_values": CENTER_LOSS_INIT_VALUES,
                "min_values": CENTER_LOSS_MIN_VALUES,
                "decay_values": CENTER_LOSS_DECAY_VALUES,
            },
            "experiments": [],
            "b1_best_config": None,
            "b1_best_metrics": None,
            "b2_best_config": None,
            "b2_best_metrics": None,
            "final_best_config": None,
            "final_best_metrics": None,
        }

    def save(self):
        """Save results."""
        with open(self.results_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_all_experiments(self) -> List[Dict]:
        """Get all experiments (current + previous)."""
        return self.data["experiments"] + self.previous_experiments

    def is_config_tested(self, init_val: float, min_val: float, decay_val: int) -> Tuple[bool, Optional[str]]:
        """Check if a configuration was already tested."""
        for exp in self.get_all_experiments():
            config = exp.get("config", {})
            if config_matches(config, init_val, min_val, decay_val):
                return True, exp.get("name", "unknown")
        return False, None

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

        best = max(phase_results, key=lambda x: x["metrics"].get(metric, 0))

        if phase == "B1":
            self.data["b1_best_config"] = best["config"]
            self.data["b1_best_metrics"] = best["metrics"]
        elif phase == "B2":
            self.data["b2_best_config"] = best["config"]
            self.data["b2_best_metrics"] = best["metrics"]

        # Update final best
        if all_experiments:
            final_best = max(all_experiments, key=lambda x: x["metrics"].get(metric, 0))
            self.data["final_best_config"] = final_best["config"]
            self.data["final_best_metrics"] = final_best["metrics"]

        self.save()

    def get_b1_best(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best init and min from B1."""
        if self.data.get("b1_best_config"):
            cfg = self.data["b1_best_config"]
            return cfg.get("center_loss_weight_init"), cfg.get("center_loss_weight_min")

        all_experiments = self.get_all_experiments()
        b1_results = [e for e in all_experiments if e.get("phase") == "B1"]

        if b1_results:
            best = max(b1_results, key=lambda x: x["metrics"].get("mAP_small", 0))
            cfg = best["config"]
            return cfg.get("center_loss_weight_init"), cfg.get("center_loss_weight_min")

        return None, None

    def get_phase_progress(self, phase: str) -> Tuple[int, int, int]:
        """Get progress for a phase."""
        if phase == "B1":
            total_valid = sum(
                1 for init_val in CENTER_LOSS_INIT_VALUES
                for min_val in CENTER_LOSS_MIN_VALUES
                if is_valid_combination(init_val, min_val)
            )

            completed_new = len([e for e in self.data["experiments"] if e.get("phase") == "B1"])

            completed_prev = sum(
                1 for init_val in CENTER_LOSS_INIT_VALUES
                for min_val in CENTER_LOSS_MIN_VALUES
                if is_valid_combination(init_val, min_val) and
                self.is_config_tested(init_val, min_val, CENTER_LOSS_DECAY_EPOCHS_DEFAULT)[0] and
                not any(
                    config_matches(e.get("config", {}), init_val, min_val, CENTER_LOSS_DECAY_EPOCHS_DEFAULT)
                    for e in self.data["experiments"]
                )
            )

            return completed_new, completed_prev, total_valid

        elif phase == "B2":
            best_init, best_min = self.get_b1_best()
            if best_init is None:
                return 0, 0, len(CENTER_LOSS_DECAY_VALUES)

            completed_new = len([e for e in self.data["experiments"] if e.get("phase") == "B2"])

            completed_prev = sum(
                1 for decay in CENTER_LOSS_DECAY_VALUES
                if self.is_config_tested(best_init, best_min, decay)[0] and
                not any(
                    config_matches(e.get("config", {}), best_init, best_min, decay)
                    for e in self.data["experiments"]
                )
            )

            return completed_new, completed_prev, len(CENTER_LOSS_DECAY_VALUES)

        return 0, 0, 0

    def print_b1_grid_status(self):
        """Print current B1 grid status."""
        print(f"\n{'=' * 90}")
        print("PHASE B1 GRID STATUS (Init × Min)")
        print(f"Fixed: decay_epochs={CENTER_LOSS_DECAY_EPOCHS_DEFAULT}")
        print(f"Phase A/C: DISABLED, Phase D: DEFAULT")
        print(f"{'=' * 90}")

        status_map = {}
        source_map = {}

        for exp in self.get_all_experiments():
            config = exp.get("config", {})
            if config.get("center_loss_decay_epochs") != CENTER_LOSS_DECAY_EPOCHS_DEFAULT:
                continue

            init_val = config.get("center_loss_weight_init")
            min_val = config.get("center_loss_weight_min")

            if init_val is not None and min_val is not None:
                mAP = exp["metrics"].get("mAP_small", 0)
                key = (init_val, min_val)

                if key not in status_map or exp in self.data["experiments"]:
                    status_map[key] = f"{mAP:.3f}"
                    source_map[key] = "new" if exp in self.data["experiments"] else "prev"

        # Header
        header = f"{'init\\min':<12}"
        for min_val in CENTER_LOSS_MIN_VALUES:
            header += f"{min_val:<10}"
        print(header)
        print("-" * 90)

        # Rows
        valid_count = 0
        completed_count = 0
        for init_val in CENTER_LOSS_INIT_VALUES:
            row = f"{init_val:<12}"
            for min_val in CENTER_LOSS_MIN_VALUES:
                key = (init_val, min_val)
                if not is_valid_combination(init_val, min_val):
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

        print(f"\nLegend: 0.XXX = mAP_small | * = from previous | . = Pending | -- = Invalid (min > init)")

        remaining = valid_count - completed_count
        print(f"Progress: {completed_count}/{valid_count} valid | Remaining: {remaining} (~{remaining * 1.3:.1f} hours)")

        # Show best
        best_init, best_min = self.get_b1_best()
        if best_init is not None:
            print(f"\nBest B1 config: init={best_init}, min={best_min}")

    def print_b2_status(self):
        """Print current B2 status."""
        best_init, best_min = self.get_b1_best()

        print(f"\n{'=' * 70}")
        print("PHASE B2 STATUS (Decay Epochs)")
        if best_init is not None:
            print(f"Using B1 best: init={best_init}, min={best_min}")
        else:
            print("Waiting for B1 to complete...")
        print(f"Phase A/C: DISABLED, Phase D: DEFAULT")
        print(f"{'=' * 70}")

        if best_init is None:
            return

        results_map = {}
        source_map = {}

        for exp in self.get_all_experiments():
            if exp.get("phase") != "B2":
                continue
            config = exp.get("config", {})
            decay = config.get("center_loss_decay_epochs")

            if decay is not None:
                mAP = exp["metrics"].get("mAP_small", 0)
                if decay not in results_map or exp in self.data["experiments"]:
                    results_map[decay] = mAP
                    source_map[decay] = "new" if exp in self.data["experiments"] else "prev"

        print(f"{'Decay':<12} {'mAP_small':<12} {'Status':<10}")
        print("-" * 40)

        for decay in CENTER_LOSS_DECAY_VALUES:
            if decay in results_map:
                marker = "*" if source_map.get(decay) == "prev" else ""
                print(f"{decay:<12} {results_map[decay]:<12.4f} {'Done' + marker:<10}")
            else:
                print(f"{decay:<12} {'--':<12} {'Pending':<10}")

        # Show best
        if self.data.get("b2_best_config"):
            best_decay = self.data["b2_best_config"].get("center_loss_decay_epochs")
            print(f"\nBest decay: {best_decay}")

    def analyze_results(self, phase: str = None, top_n: int = 10):
        """Analyze and print top results."""
        all_experiments = self.get_all_experiments()

        if phase:
            results = [e for e in all_experiments if e.get("phase") == phase]
            title = f"PHASE {phase} RESULTS"
        else:
            results = all_experiments
            title = "ALL PHASE B RESULTS"

        if not results:
            print(f"No results for {phase if phase else 'Phase B'} yet.")
            return

        sorted_results = sorted(
            results,
            key=lambda x: x["metrics"].get("mAP_small", 0),
            reverse=True
        )

        print(f"\n{'=' * 130}")
        print(f"TOP {min(top_n, len(sorted_results))} {title} (sorted by mAP_small)")
        print(f"{'=' * 130}")
        print(f"{'#':<4} {'Name':<30} {'init':<8} {'min':<8} {'decay':<8} "
              f"{'mAP_small':<11} {'mAP_med':<11} {'mAP_large':<11} {'mAP50-95':<11}")
        print("-" * 130)

        for i, r in enumerate(sorted_results[:top_n], 1):
            m = r["metrics"]
            c = r["config"]
            marker = " **BEST**" if i == 1 else ""
            print(
                f"{i:<4} {r['name']:<30} "
                f"{c.get('center_loss_weight_init', 0):<8.3f} "
                f"{c.get('center_loss_weight_min', 0):<8.3f} "
                f"{c.get('center_loss_decay_epochs', 0):<8} "
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
        center_loss_weight_init: float,
        center_loss_weight_min: float,
        center_loss_decay_epochs: int,
) -> Tuple[Dict, float]:
    """Run a single training experiment."""

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 70}")
    print(f"  center_loss_weight_init:  {center_loss_weight_init}")
    print(f"  center_loss_weight_min:   {center_loss_weight_min}")
    print(f"  center_loss_decay_epochs: {center_loss_decay_epochs}")
    print(f"  (Phase A/C: DISABLED, Phase D: DEFAULT)")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    print("  [OK] Epoch sync callback registered")

    pa = PHASE_A_DISABLED
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
        # Phase A parameters (DISABLED)
        alpha_start=pa["alpha_start"],
        alpha_end=pa["alpha_end"],
        alpha_min=pa["alpha_min"],
        alpha_max=pa["alpha_max"],
        small_obj_px=pa["small_obj_px"],
        small_obj_boost=pa["small_obj_boost"],
        # Phase B parameters (ACTIVE - what we're testing)
        center_loss_weight_init=center_loss_weight_init,
        center_loss_weight_min=center_loss_weight_min,
        center_loss_decay_epochs=center_loss_decay_epochs,
        # Phase C parameters (DISABLED)
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
# PHASE B1: INIT × MIN GRID SEARCH (6×6)
# =============================================================================

def run_phase_B1(results: PhaseBResults):
    """
    Phase B1: Grid Search over center_loss_weight_init × center_loss_weight_min

    6×6 Grid with decay_epochs fixed (35)
    Only valid combinations where min <= init are tested.
    """

    print("\n" + "#" * 80)
    print("# PHASE B1: Center Loss Grid Search (Init × Min)")
    print(f"# Fixed: decay_epochs={CENTER_LOSS_DECAY_EPOCHS_DEFAULT}")
    print("# Phase A/C: DISABLED, Phase D: DEFAULT")
    print("#" * 80)

    results.print_b1_grid_status()

    experiments = []
    exp_num = 1
    skipped_count = 0

    for init_val in CENTER_LOSS_INIT_VALUES:
        for min_val in CENTER_LOSS_MIN_VALUES:
            if not is_valid_combination(init_val, min_val):
                continue

            is_tested, prev_name = results.is_config_tested(
                init_val, min_val, CENTER_LOSS_DECAY_EPOCHS_DEFAULT
            )

            if is_tested:
                print(f"  Skipping init={init_val}, min={min_val} (tested as {prev_name})")
                skipped_count += 1
                continue

            experiments.append({
                "name": f"B1_{exp_num:02d}_init{init_val}_min{min_val}",
                "center_loss_weight_init": init_val,
                "center_loss_weight_min": min_val,
                "center_loss_decay_epochs": CENTER_LOSS_DECAY_EPOCHS_DEFAULT,
            })
            exp_num += 1

    print(f"\n{len(experiments)} new experiments to run ({skipped_count} skipped)")

    if not experiments:
        print("All B1 experiments already completed!")
        results.analyze_results(phase="B1", top_n=10)
        return

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        if results.is_completed(name):
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (completed)")
            continue

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")

        config = {
            "center_loss_weight_init": exp["center_loss_weight_init"],
            "center_loss_weight_min": exp["center_loss_weight_min"],
            "center_loss_decay_epochs": exp["center_loss_decay_epochs"],
            "phase_acd_disabled": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            center_loss_weight_init=exp["center_loss_weight_init"],
            center_loss_weight_min=exp["center_loss_weight_min"],
            center_loss_decay_epochs=exp["center_loss_decay_epochs"],
        )

        results.add_experiment(name, "B1", config, metrics, time_hours)

        if i % 5 == 0:
            results.print_b1_grid_status()

    print("\n" + "=" * 80)
    print("PHASE B1 COMPLETE")
    print("=" * 80)
    results.print_b1_grid_status()
    results.analyze_results(phase="B1", top_n=10)


# =============================================================================
# PHASE B2: DECAY EPOCHS SEARCH (6 runs)
# =============================================================================

def run_phase_B2(results: PhaseBResults):
    """
    Phase B2: Search over center_loss_decay_epochs

    Uses best (init, min) from B1.
    Tests: [15, 25, 35, 45, 55, 70]
    """

    best_init, best_min = results.get_b1_best()

    if best_init is None:
        print("\nWarning: B1 not complete. Using default init=0.05, min=0.01.")
        best_init, best_min = 0.05, 0.01

    print("\n" + "#" * 80)
    print("# PHASE B2: Center Loss Decay Epochs Search (6 runs)")
    print(f"# Using B1 best: init={best_init}, min={best_min}")
    print("# Phase A/C: DISABLED, Phase D: DEFAULT")
    print("#" * 80)

    results.print_b2_status()

    experiments = []
    skipped_count = 0

    for i, decay in enumerate(CENTER_LOSS_DECAY_VALUES, 1):
        is_tested, prev_name = results.is_config_tested(best_init, best_min, decay)

        if is_tested:
            print(f"  Skipping decay={decay} (tested as {prev_name})")
            skipped_count += 1
            continue

        experiments.append({
            "name": f"B2_{i:02d}_decay{decay}",
            "center_loss_weight_init": best_init,
            "center_loss_weight_min": best_min,
            "center_loss_decay_epochs": decay,
        })

    print(f"\n{len(experiments)} new experiments to run ({skipped_count} skipped)")

    if not experiments:
        print("All B2 experiments already completed!")
        results.analyze_results(phase="B2", top_n=6)
        return

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        if results.is_completed(name):
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (completed)")
            continue

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")

        config = {
            "center_loss_weight_init": exp["center_loss_weight_init"],
            "center_loss_weight_min": exp["center_loss_weight_min"],
            "center_loss_decay_epochs": exp["center_loss_decay_epochs"],
            "phase_acd_disabled": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            center_loss_weight_init=exp["center_loss_weight_init"],
            center_loss_weight_min=exp["center_loss_weight_min"],
            center_loss_decay_epochs=exp["center_loss_decay_epochs"],
        )

        results.add_experiment(name, "B2", config, metrics, time_hours)

    print("\n" + "=" * 80)
    print("PHASE B2 COMPLETE")
    print("=" * 80)
    results.print_b2_status()
    results.analyze_results(phase="B2", top_n=6)


# =============================================================================
# SUMMARY
# =============================================================================

def print_final_summary(results: PhaseBResults):
    """Print final Phase B summary."""

    print("\n" + "=" * 100)
    print("PHASE B GRID SEARCH: FINAL SUMMARY")
    print("=" * 100)
    print("\nNOTE: Phase A/C parameters were DISABLED for this ablation study.")
    print("      Phase D (TAL) parameters were set to DEFAULT values.")
    print("      Only center loss parameters were tested.")

    # Count experiments per phase
    all_exp = results.get_all_experiments()
    b1_count = len([e for e in all_exp if e.get("phase") == "B1"])
    b2_count = len([e for e in all_exp if e.get("phase") == "B2"])

    # Calculate valid combinations
    b1_valid = sum(
        1 for i in CENTER_LOSS_INIT_VALUES
        for m in CENTER_LOSS_MIN_VALUES
        if is_valid_combination(i, m)
    )

    print(f"\nCompleted experiments:")
    print(f"  B1 (init × min, decay disabled): {b1_count}/{b1_valid}")
    print(f"  B2 (decay epochs):               {b2_count}/{len(CENTER_LOSS_DECAY_VALUES)}")

    # Best from each phase
    if results.data.get("b1_best_config"):
        bc = results.data["b1_best_config"]
        bm = results.data["b1_best_metrics"]
        print(f"\nBest B1 (init × min):")
        print(f"  init={bc.get('center_loss_weight_init')}, min={bc.get('center_loss_weight_min')}")
        print(f"  mAP_small={bm.get('mAP_small', 0):.4f}")

    if results.data.get("b2_best_config"):
        bc = results.data["b2_best_config"]
        bm = results.data["b2_best_metrics"]
        print(f"\nBest B2 (decay epochs):")
        print(f"  decay_epochs={bc.get('center_loss_decay_epochs')}")
        print(f"  mAP_small={bm.get('mAP_small', 0):.4f}")

    # Final best
    if results.data.get("final_best_config"):
        bc = results.data["final_best_config"]
        bm = results.data["final_best_metrics"]

        print(f"""
+------------------------------------------------------------------------------+
|                         FINAL BEST CONFIGURATION                             |
+------------------------------------------------------------------------------+
|  center_loss_weight_init:  {str(bc.get('center_loss_weight_init', 'N/A')):<50}|
|  center_loss_weight_min:   {str(bc.get('center_loss_weight_min', 'N/A')):<50}|
|  center_loss_decay_epochs: {str(bc.get('center_loss_decay_epochs', 'N/A')):<50}|
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
    """Run Phase B grid search."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Phase B: Center Loss Grid Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase_b_center_loss_grid.py              # Run full Phase B
  python phase_b_center_loss_grid.py --phase B1   # Run only B1 (init × min)
  python phase_b_center_loss_grid.py --phase B2   # Run only B2 (decay)
  python phase_b_center_loss_grid.py --status     # Show current progress
  python phase_b_center_loss_grid.py --results    # Show results analysis
        """
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "B1", "B2"],
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
    results = PhaseBResults(PROJECT_DIR)

    print(f"\nPhase B: Center Loss Grid Search (ISOLATED)")
    print(f"Project: {PROJECT_DIR}")
    print(f"Previous results: {PREVIOUS_RESULTS_FILE}")

    # Show disabled parameters
    print(f"\n{'=' * 80}")
    print("FIXED HYPERPARAMETERS")
    print(f"{'=' * 80}")
    print(f"  Phase A: DISABLED (alpha=1.0, boost=1.0)")
    print(f"  Phase C: DISABLED (clip=100.0, no clipping)")
    print(f"  Phase D: DEFAULT (tal_topk={PHASE_D_DEFAULT['tal_topk']}, "
          f"tal_alpha={PHASE_D_DEFAULT['tal_alpha']}, tal_beta={PHASE_D_DEFAULT['tal_beta']})")

    # Calculate valid combinations
    b1_valid = sum(
        1 for i in CENTER_LOSS_INIT_VALUES
        for m in CENTER_LOSS_MIN_VALUES
        if is_valid_combination(i, m)
    )

    # Show grid configuration
    print(f"\n{'=' * 80}")
    print("PHASE B GRID CONFIGURATION (ACTIVE)")
    print(f"{'=' * 80}")
    print(f"  B1: Init × Min Grid ({b1_valid} valid combinations)")
    print(f"      init: {CENTER_LOSS_INIT_VALUES}")
    print(f"      min:  {CENTER_LOSS_MIN_VALUES}")
    print(f"      Fixed: decay_epochs={CENTER_LOSS_DECAY_EPOCHS_DEFAULT}")
    print(f"")
    print(f"  B2: Decay Epochs Search ({len(CENTER_LOSS_DECAY_VALUES)} experiments)")
    print(f"      decay: {CENTER_LOSS_DECAY_VALUES}")
    print(f"      Uses best (init, min) from B1")
    print(f"")
    print(f"  Total: {b1_valid + len(CENTER_LOSS_DECAY_VALUES)} experiments "
          f"(~{(b1_valid + len(CENTER_LOSS_DECAY_VALUES)) * 1.3:.0f} hours)")

    if args.status:
        results.print_b1_grid_status()
        results.print_b2_status()
        return

    if args.results:
        results.analyze_results(top_n=20)
        return

    # Run phases
    if args.phase == "all":
        run_phase_B1(results)
        run_phase_B2(results)
    elif args.phase == "B1":
        run_phase_B1(results)
    elif args.phase == "B2":
        run_phase_B2(results)

    print_final_summary(results)


if __name__ == "__main__":
    main()
