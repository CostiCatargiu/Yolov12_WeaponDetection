#!/usr/bin/env python3
"""
Phase A4 Ablation: TAL (Task-Aligned Learning) Grid Search
===========================================================

Overview
--------
Phase A4 performs searches over Task-Aligned Assigner (TAL) hyperparameters
to optimize small object detection.

NOTE: Phase A1, A2, A3 parameters are DISABLED (set to neutral/default values)
      to test TAL parameters in isolation.

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

Phase A2 (Center Loss) - DISABLED:
    center_loss_weight_init:  0.0   (no center loss)
    center_loss_weight_min:   0.0   (no center loss)
    center_loss_decay_epochs: 1     (irrelevant when weight=0.0)

Phase A3 (Adaptive Clipping) - DISABLED:
    iou_clip_start:  100.0   (no clipping)
    iou_clip_end:    100.0   (no clipping)
    dfl_clip_start:  100.0   (no clipping)
    dfl_clip_end:    100.0   (no clipping)

Step A4a: Alpha × Beta Grid (6×6 = 36 runs)
--------------------------------------------
Grid search over alpha and beta with fixed topk=10.

6 alpha values: [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
6 beta values:  [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

The alignment metric is: alignment = cls_score^alpha × iou^beta
- Higher alpha: More weight on classification confidence
- Higher beta: More weight on IoU quality

Grid Structure A4a:

              tal_beta
              4.0    5.0    6.0    7.0    8.0   10.0
           +------------------------------------------
     0.25  |   X      X      X      X      X      X
     0.40  |   X      X      X      X      X      X
tal  0.50  |   X      X      X*     X      X      X
alpha0.60  |   X      X      X      X      X      X
     0.75  |   X      X      X      X      X      X
     1.00  |   X      X      X      X      X      X

Legend: X = experiment | X* = Default (0.5, 6.0)

Total: 36 experiments
Estimated time: ~47 hours (at ~1.3 hours per 70-epoch run)

Step A4b: TopK Search (12 runs)
--------------------------------
Tests topk values with fixed alpha=0.5, beta=6.0.

12 topk values: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
  - Lower topk = More selective (fewer positives per GT)
  - Higher topk = More permissive (more positives per GT)

Total: 12 experiments
Estimated time: ~16 hours (at ~1.3 hours per 70-epoch run)

Step A4c: Validation (6 runs)
------------------------------
Combines best topk from A4b with top-3 alpha/beta from A4a.

Experiments:
1. Best topk + #1 alpha/beta from A4a
2. Best topk + #2 alpha/beta from A4a
3. Best topk + #3 alpha/beta from A4a
4. (Best topk - 2) + best alpha/beta
5. (Best topk + 2) + best alpha/beta
6. Best topk + best alpha/beta (if not already in top-3)

Total: up to 6 experiments
Estimated time: ~8 hours (at ~1.3 hours per 70-epoch run)

Total Phase A4 Statistics
-------------------------
    Step A4a: 36 runs (alpha × beta grid, topk=10 fixed)
    Step A4b: 12 runs (topk search, alpha=0.5/beta=6.0 fixed)
    Step A4c:  6 runs (validation, best combinations)
    ---------------------------------
    Total:    54 runs
    Estimated: ~70 hours (~3 days at 70 epochs per run)

Overall Ablation Phases
-----------------------
    Phase A1: Size-Aware Weighting    (phase_a1_ablation.py)
    Phase A2: Center Loss             (phase_a2_ablation.py)
    Phase A3: Adaptive Clipping       (phase_a3_ablation.py)
    Phase A4: TAL Parameters          (this script)

Usage
-----
    python phase_a4_ablation.py               # Run full Phase A4
    python phase_a4_ablation.py --step A4a    # Run only A4a (alpha × beta)
    python phase_a4_ablation.py --step A4b    # Run only A4b (topk)
    python phase_a4_ablation.py --step A4c    # Run only A4c (validation)
    python phase_a4_ablation.py --status      # Show current progress
    python phase_a4_ablation.py --results     # Show results analysis

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
PROJECT_DIR = "runs_phaseDorig"

PREVIOUS_RESULTS_FILE = "PhaseDinit.json"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 64
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# PHASE A, B, C: DISABLED (NEUTRAL VALUES)
# =============================================================================
# These are set to neutral/default values so they don't affect results.
# We're testing Phase D (TAL) parameters in isolation.

# Phase A: Size-aware weighting - DISABLED
# alpha=1.0 means pure score weighting (no area component)
# boost=1.0 means no boost for small objects
PHASE_A_DISABLED = {
    "alpha_start": 1.0,  # No area weighting
    "alpha_end": 1.0,  # No area weighting
    "alpha_min": 1.0,  # No area weighting
    "alpha_max": 1.0,  # No area weighting
    "small_obj_boost": 1.0,  # No boost (multiplier = 1)
    "small_obj_px": 32,  # Default small object threshold
}

# Phase B: Center loss - DISABLED
# weight=0.0 means center loss contributes nothing
PHASE_B_DISABLED = {
    "center_loss_weight_init": 0.0,  # No center loss
    "center_loss_weight_min": 0.0,  # No center loss
    "center_loss_decay_epochs": 1,  # Irrelevant when weight=0
}

# Phase C: Adaptive clipping - DISABLED
# Very high clip values = effectively no clipping
PHASE_C_DISABLED = {
    "iou_clip_start": 100.0,  # No clipping (very high)
    "iou_clip_end": 100.0,  # No clipping (very high)
    "dfl_clip_start": 100.0,  # No clipping (very high)
    "dfl_clip_end": 100.0,  # No clipping (very high)
}

# =============================================================================
# PHASE D1: ALPHA × BETA GRID (6×6)
# =============================================================================

TAL_ALPHA_VALUES = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
TAL_BETA_VALUES = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
DEFAULT_TAL_TOPK = 10  # Fixed for D1

# =============================================================================
# PHASE D2: TOPK SEARCH (12 values, up to 25)
# =============================================================================

TAL_TOPK_VALUES = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
DEFAULT_TAL_ALPHA = 0.5  # Fixed for D2
DEFAULT_TAL_BETA = 6.0  # Fixed for D2

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


def config_matches(config: Dict, alpha: float, beta: float, topk: int) -> bool:
    """Check if config matches alpha/beta/topk values."""
    return (
            float_eq(config.get("tal_alpha", -1), alpha) and
            float_eq(config.get("tal_beta", -1), beta) and
            config.get("tal_topk", -1) == topk
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

class PhaseDResults:
    """Manages Phase D results and tracks best configurations."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.project_dir / "PhaseDinit.json"
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
            "phase_b_disabled": PHASE_B_DISABLED,
            "phase_c_disabled": PHASE_C_DISABLED,
            "grid_config": {
                "tal_alpha_values": TAL_ALPHA_VALUES,
                "tal_beta_values": TAL_BETA_VALUES,
                "tal_topk_values": TAL_TOPK_VALUES,
            },
            "experiments": [],
            "d1_best_config": None,
            "d1_best_metrics": None,
            "d1_top3": [],
            "d2_best_config": None,
            "d2_best_metrics": None,
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

    def is_config_tested(self, alpha: float, beta: float, topk: int) -> Tuple[bool, Optional[str]]:
        """Check if config was already tested."""
        for exp in self.get_all_experiments():
            cfg = exp.get("config", {})
            if config_matches(cfg, alpha, beta, topk):
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

        # Sort by metric descending
        sorted_results = sorted(
            phase_results,
            key=lambda x: x["metrics"].get(metric, 0),
            reverse=True
        )

        best = sorted_results[0]

        if phase == "D1":
            self.data["d1_best_config"] = best["config"]
            self.data["d1_best_metrics"] = best["metrics"]
            # Store top 3 for D3 validation
            self.data["d1_top3"] = [
                {"config": r["config"], "metrics": r["metrics"]}
                for r in sorted_results[:3]
            ]
        elif phase == "D2":
            self.data["d2_best_config"] = best["config"]
            self.data["d2_best_metrics"] = best["metrics"]

        # Update final best across all phases
        if all_experiments:
            final_best = max(all_experiments, key=lambda x: x["metrics"].get(metric, 0))
            self.data["final_best_config"] = final_best["config"]
            self.data["final_best_metrics"] = final_best["metrics"]

        self.save()

    def get_d1_best(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best alpha/beta from D1."""
        if self.data.get("d1_best_config"):
            cfg = self.data["d1_best_config"]
            return cfg.get("tal_alpha"), cfg.get("tal_beta")

        all_experiments = self.get_all_experiments()
        d1_results = [e for e in all_experiments if e.get("phase") == "D1"]

        if d1_results:
            best = max(d1_results, key=lambda x: x["metrics"].get("mAP_small", 0))
            cfg = best["config"]
            return cfg.get("tal_alpha"), cfg.get("tal_beta")

        return None, None

    def get_d1_top3(self) -> List[Tuple[float, float, float]]:
        """Get top 3 alpha/beta configs from D1 with their mAP_small."""
        if self.data.get("d1_top3"):
            return [
                (
                    r["config"].get("tal_alpha"),
                    r["config"].get("tal_beta"),
                    r["metrics"].get("mAP_small", 0)
                )
                for r in self.data["d1_top3"]
            ]

        all_experiments = self.get_all_experiments()
        d1_results = [e for e in all_experiments if e.get("phase") == "D1"]

        if d1_results:
            sorted_results = sorted(
                d1_results,
                key=lambda x: x["metrics"].get("mAP_small", 0),
                reverse=True
            )[:3]
            return [
                (
                    r["config"].get("tal_alpha"),
                    r["config"].get("tal_beta"),
                    r["metrics"].get("mAP_small", 0)
                )
                for r in sorted_results
            ]

        return []

    def get_d2_best_topk(self) -> Optional[int]:
        """Get best topk from D2."""
        if self.data.get("d2_best_config"):
            return self.data["d2_best_config"].get("tal_topk")

        all_experiments = self.get_all_experiments()
        d2_results = [e for e in all_experiments if e.get("phase") == "D2"]

        if d2_results:
            best = max(d2_results, key=lambda x: x["metrics"].get("mAP_small", 0))
            return best["config"].get("tal_topk")

        return None

    def print_d1_grid_status(self):
        """Print D1 (alpha × beta) grid status."""
        print(f"\n{'=' * 90}")
        print("PHASE D1 GRID STATUS (Alpha × Beta)")
        print(f"Fixed: topk={DEFAULT_TAL_TOPK}")
        print(f"{'=' * 90}")

        status_map = {}
        source_map = {}

        for exp in self.get_all_experiments():
            if exp.get("phase") != "D1":
                continue
            config = exp.get("config", {})
            alpha = config.get("tal_alpha")
            beta = config.get("tal_beta")

            if alpha is not None and beta is not None:
                mAP = exp["metrics"].get("mAP_small", 0)
                key = (alpha, beta)

                if key not in status_map or exp in self.data["experiments"]:
                    status_map[key] = f"{mAP:.3f}"
                    source_map[key] = "new" if exp in self.data["experiments"] else "prev"

        # Header
        header = f"{'alpha\\beta':<12}"
        for beta_val in TAL_BETA_VALUES:
            header += f"{beta_val:<10}"
        print(header)
        print("-" * 90)

        # Rows
        for alpha_val in TAL_ALPHA_VALUES:
            row = f"{alpha_val:<12}"
            for beta_val in TAL_BETA_VALUES:
                key = (alpha_val, beta_val)
                if key in status_map:
                    marker = "*" if source_map.get(key) == "prev" else ""
                    val_str = f"{status_map[key]}{marker}"
                    row += f"{val_str:<10}"
                else:
                    row += f"{'.':<10}"
            print(row)

        print(f"\nLegend: 0.XXX = mAP_small | * = from previous | . = Pending")

        completed = len(status_map)
        total = len(TAL_ALPHA_VALUES) * len(TAL_BETA_VALUES)
        remaining = total - completed
        print(f"Progress: {completed}/{total} | Remaining: {remaining} (~{remaining * 1.3:.1f} hours)")

        # Show top 3
        top3 = self.get_d1_top3()
        if top3:
            print(f"\nTop 3 configurations:")
            for i, (a, b, m) in enumerate(top3, 1):
                print(f"  #{i}: alpha={a}, beta={b}, mAP_small={m:.4f}")

    def print_d2_status(self):
        """Print D2 (topk) status."""
        print(f"\n{'=' * 60}")
        print("PHASE D2 STATUS (TopK Search)")
        print(f"Fixed: alpha={DEFAULT_TAL_ALPHA}, beta={DEFAULT_TAL_BETA}")
        print(f"{'=' * 60}")

        results_map = {}
        source_map = {}

        for exp in self.get_all_experiments():
            if exp.get("phase") != "D2":
                continue
            config = exp.get("config", {})
            topk = config.get("tal_topk")

            if topk is not None:
                mAP = exp["metrics"].get("mAP_small", 0)
                if topk not in results_map or exp in self.data["experiments"]:
                    results_map[topk] = mAP
                    source_map[topk] = "new" if exp in self.data["experiments"] else "prev"

        print(f"{'TopK':<8} {'mAP_small':<12} {'Status':<10}")
        print("-" * 35)

        for topk in TAL_TOPK_VALUES:
            if topk in results_map:
                marker = "*" if source_map.get(topk) == "prev" else ""
                print(f"{topk:<8} {results_map[topk]:<12.4f} {'Done' + marker:<10}")
            else:
                print(f"{topk:<8} {'--':<12} {'Pending':<10}")

        print(f"\nLegend: * = from previous runs")

        completed = len(results_map)
        remaining = len(TAL_TOPK_VALUES) - completed
        print(f"Progress: {completed}/{len(TAL_TOPK_VALUES)} | Remaining: {remaining} (~{remaining * 1.3:.1f} hours)")

        # Show best
        best_topk = self.get_d2_best_topk()
        if best_topk:
            print(f"\nBest topk: {best_topk}")

    def analyze_results(self, phase: str = None, top_n: int = 10):
        """Analyze and print top results."""
        all_experiments = self.get_all_experiments()

        if phase:
            results = [e for e in all_experiments if e.get("phase") == phase]
            title = f"PHASE {phase} RESULTS"
        else:
            results = all_experiments
            title = "ALL PHASE D RESULTS"

        if not results:
            print(f"No results for {phase if phase else 'Phase D'} yet.")
            return

        sorted_results = sorted(
            results,
            key=lambda x: x["metrics"].get("mAP_small", 0),
            reverse=True
        )

        print(f"\n{'=' * 130}")
        print(f"TOP {min(top_n, len(sorted_results))} {title} (sorted by mAP_small)")
        print(f"{'=' * 130}")
        print(f"{'#':<4} {'Name':<35} {'alpha':<8} {'beta':<8} {'topk':<6} "
              f"{'mAP_small':<11} {'mAP_med':<11} {'mAP_large':<11} {'mAP50-95':<11}")
        print("-" * 130)

        for i, r in enumerate(sorted_results[:top_n], 1):
            m = r["metrics"]
            c = r["config"]
            marker = " **BEST**" if i == 1 else ""
            print(
                f"{i:<4} {r['name']:<35} "
                f"{c.get('tal_alpha', 0):<8.2f} "
                f"{c.get('tal_beta', 0):<8.1f} "
                f"{c.get('tal_topk', 0):<6} "
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
        tal_alpha: float,
        tal_beta: float,
        tal_topk: int,
) -> Tuple[Dict, float]:
    """Run a single training experiment."""

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 70}")
    print(f"  tal_alpha: {tal_alpha}")
    print(f"  tal_beta:  {tal_beta}")
    print(f"  tal_topk:  {tal_topk}")
    print(f"  (Phase A/B/C: DISABLED)")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    print("  [OK] Epoch sync callback registered")

    # Get disabled parameters
    pa = PHASE_A_DISABLED
    pb = PHASE_B_DISABLED
    pc = PHASE_C_DISABLED

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=name,
        # Phase A parameters (DISABLED - neutral values)
        alpha_start=pa["alpha_start"],
        alpha_end=pa["alpha_end"],
        alpha_min=pa["alpha_min"],
        alpha_max=pa["alpha_max"],
        small_obj_px=pa["small_obj_px"],
        small_obj_boost=pa["small_obj_boost"],
        # Phase B parameters (DISABLED - no center loss)
        center_loss_weight_init=pb["center_loss_weight_init"],
        center_loss_weight_min=pb["center_loss_weight_min"],
        center_loss_decay_epochs=pb["center_loss_decay_epochs"],
        # Phase C parameters (DISABLED - no clipping)
        iou_clip_start=pc["iou_clip_start"],
        iou_clip_end=pc["iou_clip_end"],
        dfl_clip_start=pc["dfl_clip_start"],
        dfl_clip_end=pc["dfl_clip_end"],
        # Phase D parameters (ACTIVE - what we're testing)
        tal_alpha=tal_alpha,
        tal_beta=tal_beta,
        tal_topk=tal_topk,
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
# PHASE D1: ALPHA × BETA GRID SEARCH (6×6)
# =============================================================================

def run_phase_D1(results: PhaseDResults):
    """
    Phase D1: Grid Search over tal_alpha × tal_beta

    6×6 Grid with topk=10 (default)
    """

    print("\n" + "#" * 80)
    print("# PHASE D1: TAL Alpha × Beta Grid Search (6×6)")
    print(f"# Fixed: topk={DEFAULT_TAL_TOPK}")
    print("# Phase A/B/C: DISABLED")
    print("#" * 80)

    results.print_d1_grid_status()

    experiments = []
    exp_num = 1
    skipped_count = 0

    for alpha in TAL_ALPHA_VALUES:
        for beta in TAL_BETA_VALUES:
            is_tested, prev_name = results.is_config_tested(alpha, beta, DEFAULT_TAL_TOPK)

            if is_tested:
                print(f"  Skipping alpha={alpha}, beta={beta} (tested as {prev_name})")
                skipped_count += 1
                continue

            experiments.append({
                "name": f"D1_{exp_num:02d}_a{alpha}_b{beta}",
                "tal_alpha": alpha,
                "tal_beta": beta,
                "tal_topk": DEFAULT_TAL_TOPK,
            })
            exp_num += 1

    print(f"\n{len(experiments)} new experiments to run ({skipped_count} skipped)")

    if not experiments:
        print("All D1 experiments already completed!")
        results.analyze_results(phase="D1", top_n=10)
        return

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        if results.is_completed(name):
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (completed)")
            continue

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")

        config = {
            "tal_alpha": exp["tal_alpha"],
            "tal_beta": exp["tal_beta"],
            "tal_topk": exp["tal_topk"],
            # Record that A/B/C are disabled
            "phase_abc_disabled": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            tal_alpha=exp["tal_alpha"],
            tal_beta=exp["tal_beta"],
            tal_topk=exp["tal_topk"],
        )

        results.add_experiment(name, "D1", config, metrics, time_hours)

        if i % 6 == 0:
            results.print_d1_grid_status()

    print("\n" + "=" * 80)
    print("PHASE D1 COMPLETE")
    print("=" * 80)
    results.print_d1_grid_status()
    results.analyze_results(phase="D1", top_n=10)


# =============================================================================
# PHASE D2: TOPK SEARCH (12 values)
# =============================================================================

def run_phase_D2(results: PhaseDResults):
    """
    Phase D2: TopK Search with default alpha/beta

    12 topk values: [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
    """

    print("\n" + "#" * 80)
    print("# PHASE D2: TAL TopK Search (12 values)")
    print(f"# Fixed: alpha={DEFAULT_TAL_ALPHA}, beta={DEFAULT_TAL_BETA}")
    print("# Phase A/B/C: DISABLED")
    print("#" * 80)

    results.print_d2_status()

    experiments = []
    exp_num = 1
    skipped_count = 0

    for topk in TAL_TOPK_VALUES:
        is_tested, prev_name = results.is_config_tested(DEFAULT_TAL_ALPHA, DEFAULT_TAL_BETA, topk)

        if is_tested:
            print(f"  Skipping topk={topk} (tested as {prev_name})")
            skipped_count += 1
            continue

        experiments.append({
            "name": f"D2_{exp_num:02d}_topk{topk}",
            "tal_alpha": DEFAULT_TAL_ALPHA,
            "tal_beta": DEFAULT_TAL_BETA,
            "tal_topk": topk,
        })
        exp_num += 1

    print(f"\n{len(experiments)} new experiments to run ({skipped_count} skipped)")

    if not experiments:
        print("All D2 experiments already completed!")
        results.analyze_results(phase="D2", top_n=12)
        return

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        if results.is_completed(name):
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (completed)")
            continue

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")

        config = {
            "tal_alpha": exp["tal_alpha"],
            "tal_beta": exp["tal_beta"],
            "tal_topk": exp["tal_topk"],
            "phase_abc_disabled": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            tal_alpha=exp["tal_alpha"],
            tal_beta=exp["tal_beta"],
            tal_topk=exp["tal_topk"],
        )

        results.add_experiment(name, "D2", config, metrics, time_hours)

    print("\n" + "=" * 80)
    print("PHASE D2 COMPLETE")
    print("=" * 80)
    results.print_d2_status()
    results.analyze_results(phase="D2", top_n=12)


# =============================================================================
# PHASE D3: VALIDATION (best topk × top-3 alpha/beta + variations)
# =============================================================================

def run_phase_D3(results: PhaseDResults):
    """
    Phase D3: Validation combining best topk with top-3 alpha/beta

    Experiments:
    1-3. Best topk + top-3 alpha/beta from D1
    4. (Best topk - 2) + best alpha/beta
    5. (Best topk + 2) + best alpha/beta
    6. Best topk + best alpha/beta (if not already in top-3)
    """

    # Get best topk from D2
    best_topk = results.get_d2_best_topk()
    if best_topk is None:
        print("\nWarning: D2 not complete. Using default topk=10")
        best_topk = DEFAULT_TAL_TOPK

    # Get top-3 alpha/beta from D1
    top3_configs = results.get_d1_top3()
    if not top3_configs:
        print("\nWarning: D1 not complete. Using default alpha/beta")
        top3_configs = [(DEFAULT_TAL_ALPHA, DEFAULT_TAL_BETA, 0.0)]

    # Get absolute best alpha/beta
    best_alpha, best_beta = results.get_d1_best()
    if best_alpha is None:
        best_alpha, best_beta = DEFAULT_TAL_ALPHA, DEFAULT_TAL_BETA

    print("\n" + "#" * 80)
    print("# PHASE D3: Validation (Best TopK × Top-3 Alpha/Beta)")
    print("# Phase A/B/C: DISABLED")
    print("#" * 80)
    print(f"\nBest topk from D2: {best_topk}")
    print(f"Best alpha/beta from D1: alpha={best_alpha}, beta={best_beta}")
    print(f"\nTop-3 alpha/beta from D1:")
    for i, (a, b, m) in enumerate(top3_configs, 1):
        print(f"  #{i}: alpha={a}, beta={b}, mAP_small={m:.4f}")

    # Build experiment list
    experiments = []
    seen_configs = set()

    # 1-3: Best topk + top-3 alpha/beta
    for i, (alpha, beta, _) in enumerate(top3_configs[:3], 1):
        key = (alpha, beta, best_topk)
        if key not in seen_configs:
            experiments.append({
                "name": f"D3_{len(experiments) + 1:02d}_top{i}_k{best_topk}_a{alpha}_b{beta}",
                "desc": f"Best topk + #{i} alpha/beta",
                "tal_alpha": alpha,
                "tal_beta": beta,
                "tal_topk": best_topk,
            })
            seen_configs.add(key)

    # 4: (Best topk - 2) + best alpha/beta
    topk_minus = max(4, best_topk - 2)
    key = (best_alpha, best_beta, topk_minus)
    if key not in seen_configs:
        experiments.append({
            "name": f"D3_{len(experiments) + 1:02d}_k{topk_minus}_a{best_alpha}_b{best_beta}",
            "desc": f"(Best topk - 2) + best alpha/beta",
            "tal_alpha": best_alpha,
            "tal_beta": best_beta,
            "tal_topk": topk_minus,
        })
        seen_configs.add(key)

    # 5: (Best topk + 2) + best alpha/beta
    topk_plus = min(25, best_topk + 2)
    key = (best_alpha, best_beta, topk_plus)
    if key not in seen_configs:
        experiments.append({
            "name": f"D3_{len(experiments) + 1:02d}_k{topk_plus}_a{best_alpha}_b{best_beta}",
            "desc": f"(Best topk + 2) + best alpha/beta",
            "tal_alpha": best_alpha,
            "tal_beta": best_beta,
            "tal_topk": topk_plus,
        })
        seen_configs.add(key)

    # 6: Best topk + best alpha/beta (if not already included)
    key = (best_alpha, best_beta, best_topk)
    if key not in seen_configs:
        experiments.append({
            "name": f"D3_{len(experiments) + 1:02d}_best_k{best_topk}_a{best_alpha}_b{best_beta}",
            "desc": f"Best topk + best alpha/beta",
            "tal_alpha": best_alpha,
            "tal_beta": best_beta,
            "tal_topk": best_topk,
        })
        seen_configs.add(key)

    print(f"\n{len(experiments)} validation experiments to run:")
    for exp in experiments:
        print(f"  - {exp['name']}: {exp['desc']}")

    # Run experiments
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        # Check if already tested (any phase)
        is_tested, prev_name = results.is_config_tested(
            exp["tal_alpha"], exp["tal_beta"], exp["tal_topk"]
        )

        if is_tested:
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (tested as {prev_name})")
            continue

        if results.is_completed(name):
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (completed)")
            continue

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")
        print(f"   {exp['desc']}")

        config = {
            "tal_alpha": exp["tal_alpha"],
            "tal_beta": exp["tal_beta"],
            "tal_topk": exp["tal_topk"],
            "phase_abc_disabled": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            tal_alpha=exp["tal_alpha"],
            tal_beta=exp["tal_beta"],
            tal_topk=exp["tal_topk"],
        )

        results.add_experiment(name, "D3", config, metrics, time_hours)

    print("\n" + "=" * 80)
    print("PHASE D3 COMPLETE")
    print("=" * 80)
    results.analyze_results(phase="D3", top_n=6)


# =============================================================================
# SUMMARY
# =============================================================================

def print_final_summary(results: PhaseDResults):
    """Print final Phase D summary."""

    print("\n" + "=" * 100)
    print("PHASE D GRID SEARCH: FINAL SUMMARY")
    print("=" * 100)
    print("\nNOTE: Phase A/B/C parameters were DISABLED for this ablation study.")
    print("      Only TAL parameters (alpha, beta, topk) were tested.")

    # Count experiments per phase
    all_exp = results.get_all_experiments()
    d1_count = len([e for e in all_exp if e.get("phase") == "D1"])
    d2_count = len([e for e in all_exp if e.get("phase") == "D2"])
    d3_count = len([e for e in all_exp if e.get("phase") == "D3"])

    print(f"\nCompleted experiments:")
    print(f"  D1 (alpha × beta): {d1_count}/36")
    print(f"  D2 (topk):         {d2_count}/12")
    print(f"  D3 (validation):   {d3_count}")

    # Best from each phase
    if results.data.get("d1_best_config"):
        bc = results.data["d1_best_config"]
        bm = results.data["d1_best_metrics"]
        print(f"\nBest D1 (alpha × beta):")
        print(f"  alpha={bc.get('tal_alpha')}, beta={bc.get('tal_beta')}")
        print(f"  mAP_small={bm.get('mAP_small', 0):.4f}")

        # Top 3
        top3 = results.get_d1_top3()
        if len(top3) > 1:
            print(f"  Top-3:")
            for i, (a, b, m) in enumerate(top3, 1):
                print(f"    #{i}: alpha={a}, beta={b}, mAP={m:.4f}")

    if results.data.get("d2_best_config"):
        bc = results.data["d2_best_config"]
        bm = results.data["d2_best_metrics"]
        print(f"\nBest D2 (topk):")
        print(f"  topk={bc.get('tal_topk')}")
        print(f"  mAP_small={bm.get('mAP_small', 0):.4f}")

    # Final best
    if results.data.get("final_best_config"):
        bc = results.data["final_best_config"]
        bm = results.data["final_best_metrics"]

        print(f"""
+------------------------------------------------------------------------------+
|                         FINAL BEST CONFIGURATION                             |
+------------------------------------------------------------------------------+
|  tal_alpha: {str(bc.get('tal_alpha', 'N/A')):<64}|
|  tal_beta:  {str(bc.get('tal_beta', 'N/A')):<64}|
|  tal_topk:  {str(bc.get('tal_topk', 'N/A')):<64}|
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
    """Run Phase D grid search."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Phase D: TAL (Task-Aligned Learning) Grid Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase_d_tal_grid.py              # Run full Phase D
  python phase_d_tal_grid.py --phase D1   # Run only D1 (alpha × beta)
  python phase_d_tal_grid.py --phase D2   # Run only D2 (topk)
  python phase_d_tal_grid.py --phase D3   # Run only D3 (validation)
  python phase_d_tal_grid.py --status     # Show current progress
  python phase_d_tal_grid.py --results    # Show results analysis
        """
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "D1", "D2", "D3"],
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
    results = PhaseDResults(PROJECT_DIR)

    print(f"\nPhase D: TAL Grid Search (ISOLATED)")
    print(f"Project: {PROJECT_DIR}")
    print(f"Previous results: {PREVIOUS_RESULTS_FILE}")

    # Show disabled parameters
    print(f"\n{'=' * 80}")
    print("PHASE A/B/C: DISABLED (Neutral Values)")
    print(f"{'=' * 80}")
    print(f"  Phase A: alpha=1.0 (no area weighting), boost=1.0 (no boost)")
    print(f"  Phase B: center_loss=0.0 (disabled)")
    print(f"  Phase C: clip=100.0 (effectively no clipping)")

    # Show grid configuration
    print(f"\n{'=' * 80}")
    print("PHASE D GRID CONFIGURATION (ACTIVE)")
    print(f"{'=' * 80}")
    print(f"  D1: Alpha × Beta grid (6×6 = 36 experiments)")
    print(f"      alpha: {TAL_ALPHA_VALUES}")
    print(f"      beta:  {TAL_BETA_VALUES}")
    print(f"      fixed: topk={DEFAULT_TAL_TOPK}")
    print(f"")
    print(f"  D2: TopK search (12 experiments)")
    print(f"      topk:  {TAL_TOPK_VALUES}")
    print(f"      fixed: alpha={DEFAULT_TAL_ALPHA}, beta={DEFAULT_TAL_BETA}")
    print(f"")
    print(f"  D3: Validation (up to 6 experiments)")
    print(f"      Best topk + top-3 alpha/beta combinations")
    print(f"")
    print(f"  Total: ~54 experiments (~70 hours)")

    if args.status:
        results.print_d1_grid_status()
        results.print_d2_status()
        return

    if args.results:
        results.analyze_results(top_n=20)
        return

    # Run phases
    if args.phase == "all":
        run_phase_D1(results)
        run_phase_D2(results)
        run_phase_D3(results)
    elif args.phase == "D1":
        run_phase_D1(results)
    elif args.phase == "D2":
        run_phase_D2(results)
    elif args.phase == "D3":
        run_phase_D3(results)

    print_final_summary(results)


if __name__ == "__main__":
    main()
