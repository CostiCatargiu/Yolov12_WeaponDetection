#!/usr/bin/env python3
"""
Phase A3 Ablation: Adaptive Loss Clipping Grid Search
======================================================

Overview
--------
Phase A3 performs grid searches over adaptive loss clipping parameters
(IoU and DFL clip values) to optimize small object detection.

NOTE: Phase A1, A2, A4 parameters are FIXED to neutral/default values
      to test clipping parameters in isolation.

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
    center_loss_decay_epochs: 35    (irrelevant when weight=0.0)

Phase A4 (TAL Parameters) - DEFAULT:
    tal_topk:  10
    tal_alpha: 0.5
    tal_beta:  6.0

Step A3a: IoU Clipping Grid (6×6 = 36 runs)
--------------------------------------------
Grid search over iou_clip_start and iou_clip_end.
DFL clipping is DISABLED (set to neutral values 100.0 → 100.0) during this step.

6 iou_clip_start values: [6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
6 iou_clip_end values:   [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

Constraint: iou_clip_start >= iou_clip_end
            The clip value decays from start to end during training.

Grid Structure A3a:

                    iou_clip_end
                    2.0    3.0    4.0    5.0    6.0    8.0
                 +------------------------------------------
          6.0    |   X      X      X      X      X      -
          8.0    |   X      X      X      X      X      X
iou_clip 10.0    |   X      X      X      X      X      X
_start   12.0    |   X      X      X      X      X      X
         15.0    |   X      X      X      X      X      X
         20.0    |   X      X      X      X      X      X

Legend: X = valid experiment (start >= end)
        - = invalid (start < end)

Valid combinations: 35 experiments
Estimated time: ~46 hours (at ~1.3 hours per 70-epoch run)

Step A3b: DFL Clipping Grid (6×6 = 36 runs)
--------------------------------------------
Grid search over dfl_clip_start and dfl_clip_end.
IoU clipping is DISABLED (set to neutral values 100.0 → 100.0) during this step.

6 dfl_clip_start values: [6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
6 dfl_clip_end values:   [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

Constraint: dfl_clip_start >= dfl_clip_end

Grid Structure A3b:

                    dfl_clip_end
                    2.0    3.0    4.0    5.0    6.0    8.0
                 +------------------------------------------
          6.0    |   X      X      X      X      X      -
          8.0    |   X      X      X      X      X      X
dfl_clip 10.0    |   X      X      X      X      X      X
_start   12.0    |   X      X      X      X      X      X
         15.0    |   X      X      X      X      X      X
         20.0    |   X      X      X      X      X      X

Valid combinations: 35 experiments
Estimated time: ~46 hours (at ~1.3 hours per 70-epoch run)

Step A3c: Validation (6 runs)
------------------------------
Combines best IoU config from A3a with best DFL config from A3b.

Experiments:
1. Best IoU + Best DFL (final optimal)
2. Best config with tighter clipping (×0.8)
3. Best config with looser clipping (×1.3)
4. IoU clipping only (DFL disabled)
5. DFL clipping only (IoU disabled)
6. No clipping baseline (both disabled)

Estimated time: ~8 hours (at ~1.3 hours per 70-epoch run)

Total Phase A3 Statistics
-------------------------
    Step A3a: 35 runs (IoU grid, DFL disabled)
    Step A3b: 35 runs (DFL grid, IoU disabled)
    Step A3c:  6 runs (validation, best combined)
    ---------------------------------
    Total:    76 runs
    Estimated: ~100 hours (~4 days at 70 epochs per run)

Overall Ablation Phases
-----------------------
    Phase A1: Size-Aware Weighting    (phase_a1_ablation.py)
    Phase A2: Center Loss             (phase_a2_ablation.py)
    Phase A3: Adaptive Clipping       (this script)
    Phase A4: TAL Parameters          (phase_a4_ablation.py)

Usage
-----
    python phase_a3_ablation.py               # Run full Phase A3
    python phase_a3_ablation.py --step A3a    # Run only A3a (IoU grid)
    python phase_a3_ablation.py --step A3b    # Run only A3b (DFL grid)
    python phase_a3_ablation.py --step A3c    # Run only A3c (validation)
    python phase_a3_ablation.py --status      # Show current progress
    python phase_a3_ablation.py --results     # Show results analysis

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
PROJECT_DIR = "runs_phaseC"

PREVIOUS_RESULTS_FILE = "runs_phaseCprev/results.json"

EPOCHS = 70
IMG_SIZE = 640
BATCH = 64
WORKERS = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =============================================================================
# PHASE A, B: DISABLED (NEUTRAL VALUES)
# =============================================================================

PHASE_A_DISABLED = {
    "alpha_start": 1.0,
    "alpha_end": 1.0,
    "alpha_min": 1.0,
    "alpha_max": 1.0,
    "small_obj_boost": 1.0,
    "small_obj_px": 32,
}

PHASE_B_DISABLED = {
    "center_loss_weight_init": 0.0,
    "center_loss_weight_min": 0.0,
    "center_loss_decay_epochs": 35,
}

# =============================================================================
# PHASE C1: IOU CLIPPING GRID (6×6)
# =============================================================================

IOU_CLIP_START_VALUES = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
IOU_CLIP_END_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

# Default DFL values for C1 (neutral - no DFL clipping)
DEFAULT_DFL_CLIP_START = 100.0
DEFAULT_DFL_CLIP_END = 100.0

# =============================================================================
# PHASE C2: DFL CLIPPING GRID (6×6)
# =============================================================================

DFL_CLIP_START_VALUES = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
DFL_CLIP_END_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

# Default IoU values for C2 (neutral - no IoU clipping)
DEFAULT_IOU_CLIP_START = 100.0
DEFAULT_IOU_CLIP_END = 100.0

# =============================================================================
# PHASE D: FIXED (TAL PARAMETERS)
# =============================================================================

TAL_TOPK = 10
TAL_ALPHA = 0.5
TAL_BETA = 6.0

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


def is_valid_clip_pair(start: float, end: float) -> bool:
    """Check if clip start >= clip end (valid configuration)."""
    return start >= end


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


def config_matches_iou(config: Dict, iou_start: float, iou_end: float) -> bool:
    """Check if config matches IoU clip values."""
    return (
        float_eq(config.get("iou_clip_start", -1), iou_start) and
        float_eq(config.get("iou_clip_end", -1), iou_end)
    )


def config_matches_dfl(config: Dict, dfl_start: float, dfl_end: float) -> bool:
    """Check if config matches DFL clip values."""
    return (
        float_eq(config.get("dfl_clip_start", -1), dfl_start) and
        float_eq(config.get("dfl_clip_end", -1), dfl_end)
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

class PhaseCResults:
    """Manages Phase C results and tracks best configurations."""

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
            "phase_b_disabled": PHASE_B_DISABLED,
            "grid_config": {
                "iou_clip_start_values": IOU_CLIP_START_VALUES,
                "iou_clip_end_values": IOU_CLIP_END_VALUES,
                "dfl_clip_start_values": DFL_CLIP_START_VALUES,
                "dfl_clip_end_values": DFL_CLIP_END_VALUES,
            },
            "experiments": [],
            "c1_best_config": None,
            "c1_best_metrics": None,
            "c2_best_config": None,
            "c2_best_metrics": None,
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

    def is_iou_config_tested(self, iou_start: float, iou_end: float) -> Tuple[bool, Optional[str]]:
        """Check if IoU config was already tested (for C1)."""
        for exp in self.get_all_experiments():
            if exp.get("phase") != "C1":
                continue
            cfg = exp.get("config", {})
            if config_matches_iou(cfg, iou_start, iou_end):
                return True, exp.get("name", "unknown")
        return False, None

    def is_dfl_config_tested(self, dfl_start: float, dfl_end: float) -> Tuple[bool, Optional[str]]:
        """Check if DFL config was already tested (for C2)."""
        for exp in self.get_all_experiments():
            if exp.get("phase") != "C2":
                continue
            cfg = exp.get("config", {})
            if config_matches_dfl(cfg, dfl_start, dfl_end):
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

        sorted_results = sorted(
            phase_results,
            key=lambda x: x["metrics"].get(metric, 0),
            reverse=True
        )

        best = sorted_results[0]

        if phase == "C1":
            self.data["c1_best_config"] = best["config"]
            self.data["c1_best_metrics"] = best["metrics"]
        elif phase == "C2":
            self.data["c2_best_config"] = best["config"]
            self.data["c2_best_metrics"] = best["metrics"]

        # Update final best across all phases
        if all_experiments:
            final_best = max(all_experiments, key=lambda x: x["metrics"].get(metric, 0))
            self.data["final_best_config"] = final_best["config"]
            self.data["final_best_metrics"] = final_best["metrics"]

        self.save()

    def get_c1_best(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best IoU clip values from C1."""
        if self.data.get("c1_best_config"):
            cfg = self.data["c1_best_config"]
            return cfg.get("iou_clip_start"), cfg.get("iou_clip_end")

        all_experiments = self.get_all_experiments()
        c1_results = [e for e in all_experiments if e.get("phase") == "C1"]

        if c1_results:
            best = max(c1_results, key=lambda x: x["metrics"].get("mAP_small", 0))
            cfg = best["config"]
            return cfg.get("iou_clip_start"), cfg.get("iou_clip_end")

        return None, None

    def get_c2_best(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best DFL clip values from C2."""
        if self.data.get("c2_best_config"):
            cfg = self.data["c2_best_config"]
            return cfg.get("dfl_clip_start"), cfg.get("dfl_clip_end")

        all_experiments = self.get_all_experiments()
        c2_results = [e for e in all_experiments if e.get("phase") == "C2"]

        if c2_results:
            best = max(c2_results, key=lambda x: x["metrics"].get("mAP_small", 0))
            cfg = best["config"]
            return cfg.get("dfl_clip_start"), cfg.get("dfl_clip_end")

        return None, None

    def print_c1_grid_status(self):
        """Print C1 (IoU) grid status."""
        print(f"\n{'=' * 90}")
        print("PHASE C1 GRID STATUS (IoU Clip Start × End)")
        print(f"DFL clipping: DISABLED (100.0 → 100.0)")
        print(f"{'=' * 90}")

        status_map = {}
        source_map = {}

        for exp in self.get_all_experiments():
            if exp.get("phase") != "C1":
                continue
            config = exp.get("config", {})
            iou_start = config.get("iou_clip_start")
            iou_end = config.get("iou_clip_end")

            if iou_start is not None and iou_end is not None:
                mAP = exp["metrics"].get("mAP_small", 0)
                key = (iou_start, iou_end)

                if key not in status_map or exp in self.data["experiments"]:
                    status_map[key] = f"{mAP:.3f}"
                    source_map[key] = "new" if exp in self.data["experiments"] else "prev"

        # Header
        header = f"{'start\\end':<12}"
        for end_val in IOU_CLIP_END_VALUES:
            header += f"{end_val:<10}"
        print(header)
        print("-" * 90)

        # Rows
        valid_count = 0
        completed_count = 0
        for start_val in IOU_CLIP_START_VALUES:
            row = f"{start_val:<12}"
            for end_val in IOU_CLIP_END_VALUES:
                key = (start_val, end_val)
                if not is_valid_clip_pair(start_val, end_val):
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

        print(f"\nLegend: 0.XXX = mAP_small | * = from previous | . = Pending | -- = Invalid (start < end)")

        remaining = valid_count - completed_count
        print(f"Progress: {completed_count}/{valid_count} valid | Remaining: {remaining} (~{remaining * 1.3:.1f} hours)")

        # Show best
        best_start, best_end = self.get_c1_best()
        if best_start is not None:
            print(f"\nBest IoU config: start={best_start}, end={best_end}")

    def print_c2_grid_status(self):
        """Print C2 (DFL) grid status."""
        print(f"\n{'=' * 90}")
        print("PHASE C2 GRID STATUS (DFL Clip Start × End)")
        print(f"IoU clipping: DISABLED (100.0 → 100.0)")
        print(f"{'=' * 90}")

        status_map = {}
        source_map = {}

        for exp in self.get_all_experiments():
            if exp.get("phase") != "C2":
                continue
            config = exp.get("config", {})
            dfl_start = config.get("dfl_clip_start")
            dfl_end = config.get("dfl_clip_end")

            if dfl_start is not None and dfl_end is not None:
                mAP = exp["metrics"].get("mAP_small", 0)
                key = (dfl_start, dfl_end)

                if key not in status_map or exp in self.data["experiments"]:
                    status_map[key] = f"{mAP:.3f}"
                    source_map[key] = "new" if exp in self.data["experiments"] else "prev"

        # Header
        header = f"{'start\\end':<12}"
        for end_val in DFL_CLIP_END_VALUES:
            header += f"{end_val:<10}"
        print(header)
        print("-" * 90)

        # Rows
        valid_count = 0
        completed_count = 0
        for start_val in DFL_CLIP_START_VALUES:
            row = f"{start_val:<12}"
            for end_val in DFL_CLIP_END_VALUES:
                key = (start_val, end_val)
                if not is_valid_clip_pair(start_val, end_val):
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

        print(f"\nLegend: 0.XXX = mAP_small | * = from previous | . = Pending | -- = Invalid (start < end)")

        remaining = valid_count - completed_count
        print(f"Progress: {completed_count}/{valid_count} valid | Remaining: {remaining} (~{remaining * 1.3:.1f} hours)")

        # Show best
        best_start, best_end = self.get_c2_best()
        if best_start is not None:
            print(f"\nBest DFL config: start={best_start}, end={best_end}")

    def analyze_results(self, phase: str = None, top_n: int = 10):
        """Analyze and print top results."""
        all_experiments = self.get_all_experiments()

        if phase:
            results = [e for e in all_experiments if e.get("phase") == phase]
            title = f"PHASE {phase} RESULTS"
        else:
            results = all_experiments
            title = "ALL PHASE C RESULTS"

        if not results:
            print(f"No results for {phase if phase else 'Phase C'} yet.")
            return

        sorted_results = sorted(
            results,
            key=lambda x: x["metrics"].get("mAP_small", 0),
            reverse=True
        )

        print(f"\n{'=' * 140}")
        print(f"TOP {min(top_n, len(sorted_results))} {title} (sorted by mAP_small)")
        print(f"{'=' * 140}")
        print(f"{'#':<4} {'Name':<35} {'IoU_s':<8} {'IoU_e':<8} {'DFL_s':<8} {'DFL_e':<8} "
              f"{'mAP_small':<11} {'mAP_med':<11} {'mAP_large':<11} {'mAP50-95':<11}")
        print("-" * 140)

        for i, r in enumerate(sorted_results[:top_n], 1):
            m = r["metrics"]
            c = r["config"]
            marker = " **BEST**" if i == 1 else ""
            print(
                f"{i:<4} {r['name']:<35} "
                f"{c.get('iou_clip_start', 0):<8.1f} "
                f"{c.get('iou_clip_end', 0):<8.1f} "
                f"{c.get('dfl_clip_start', 0):<8.1f} "
                f"{c.get('dfl_clip_end', 0):<8.1f} "
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
        iou_clip_start: float,
        iou_clip_end: float,
        dfl_clip_start: float,
        dfl_clip_end: float,
) -> Tuple[Dict, float]:
    """Run a single training experiment."""

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 70}")
    print(f"  iou_clip:  {iou_clip_start} → {iou_clip_end}")
    print(f"  dfl_clip:  {dfl_clip_start} → {dfl_clip_end}")
    print(f"  (Phase A/B: DISABLED, Phase D: DEFAULT)")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    model = YOLO(MODEL_WEIGHTS)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    print("  [OK] Epoch sync callback registered")

    pa = PHASE_A_DISABLED
    pb = PHASE_B_DISABLED

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
        # Phase C parameters (ACTIVE - what we're testing)
        iou_clip_start=iou_clip_start,
        iou_clip_end=iou_clip_end,
        dfl_clip_start=dfl_clip_start,
        dfl_clip_end=dfl_clip_end,
        # Phase D parameters (FIXED - default TAL)
        tal_topk=TAL_TOPK,
        tal_alpha=TAL_ALPHA,
        tal_beta=TAL_BETA,
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
# PHASE C1: IOU CLIPPING GRID SEARCH (6×6)
# =============================================================================

def run_phase_C1(results: PhaseCResults):
    """
    Phase C1: Grid Search over iou_clip_start × iou_clip_end

    6×6 Grid with DFL clipping disabled (neutral values 100.0 → 100.0)
    Only valid combinations where start >= end are tested.
    """

    print("\n" + "#" * 80)
    print("# PHASE C1: IoU Clipping Grid Search (6×6)")
    print(f"# DFL clipping: DISABLED (100.0 → 100.0)")
    print("# Phase A/B: DISABLED, Phase D: DEFAULT")
    print("#" * 80)

    results.print_c1_grid_status()

    experiments = []
    exp_num = 1
    skipped_count = 0

    for iou_start in IOU_CLIP_START_VALUES:
        for iou_end in IOU_CLIP_END_VALUES:
            # Skip invalid combinations (start < end)
            if not is_valid_clip_pair(iou_start, iou_end):
                continue

            is_tested, prev_name = results.is_iou_config_tested(iou_start, iou_end)

            if is_tested:
                print(f"  Skipping iou_start={iou_start}, iou_end={iou_end} (tested as {prev_name})")
                skipped_count += 1
                continue

            experiments.append({
                "name": f"C1_{exp_num:02d}_iou_s{iou_start}_e{iou_end}",
                "iou_clip_start": iou_start,
                "iou_clip_end": iou_end,
                "dfl_clip_start": DEFAULT_DFL_CLIP_START,
                "dfl_clip_end": DEFAULT_DFL_CLIP_END,
            })
            exp_num += 1

    print(f"\n{len(experiments)} new experiments to run ({skipped_count} skipped)")

    if not experiments:
        print("All C1 experiments already completed!")
        results.analyze_results(phase="C1", top_n=10)
        return

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        if results.is_completed(name):
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (completed)")
            continue

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")

        config = {
            "iou_clip_start": exp["iou_clip_start"],
            "iou_clip_end": exp["iou_clip_end"],
            "dfl_clip_start": exp["dfl_clip_start"],
            "dfl_clip_end": exp["dfl_clip_end"],
            "phase_ab_disabled": True,
            "phase_d_default": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            iou_clip_start=exp["iou_clip_start"],
            iou_clip_end=exp["iou_clip_end"],
            dfl_clip_start=exp["dfl_clip_start"],
            dfl_clip_end=exp["dfl_clip_end"],
        )

        results.add_experiment(name, "C1", config, metrics, time_hours)

        if i % 6 == 0:
            results.print_c1_grid_status()

    print("\n" + "=" * 80)
    print("PHASE C1 COMPLETE")
    print("=" * 80)
    results.print_c1_grid_status()
    results.analyze_results(phase="C1", top_n=10)


# =============================================================================
# PHASE C2: DFL CLIPPING GRID SEARCH (6×6)
# =============================================================================

def run_phase_C2(results: PhaseCResults):
    """
    Phase C2: Grid Search over dfl_clip_start × dfl_clip_end

    6×6 Grid with IoU clipping disabled (neutral values 100.0 → 100.0)
    Only valid combinations where start >= end are tested.
    """

    print("\n" + "#" * 80)
    print("# PHASE C2: DFL Clipping Grid Search (6×6)")
    print(f"# IoU clipping: DISABLED (100.0 → 100.0)")
    print("# Phase A/B: DISABLED, Phase D: DEFAULT")
    print("#" * 80)

    results.print_c2_grid_status()

    experiments = []
    exp_num = 1
    skipped_count = 0

    for dfl_start in DFL_CLIP_START_VALUES:
        for dfl_end in DFL_CLIP_END_VALUES:
            # Skip invalid combinations (start < end)
            if not is_valid_clip_pair(dfl_start, dfl_end):
                continue

            is_tested, prev_name = results.is_dfl_config_tested(dfl_start, dfl_end)

            if is_tested:
                print(f"  Skipping dfl_start={dfl_start}, dfl_end={dfl_end} (tested as {prev_name})")
                skipped_count += 1
                continue

            experiments.append({
                "name": f"C2_{exp_num:02d}_dfl_s{dfl_start}_e{dfl_end}",
                "iou_clip_start": DEFAULT_IOU_CLIP_START,  # DISABLED
                "iou_clip_end": DEFAULT_IOU_CLIP_END,      # DISABLED
                "dfl_clip_start": dfl_start,
                "dfl_clip_end": dfl_end,
            })
            exp_num += 1

    print(f"\n{len(experiments)} new experiments to run ({skipped_count} skipped)")

    if not experiments:
        print("All C2 experiments already completed!")
        results.analyze_results(phase="C2", top_n=10)
        return

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        if results.is_completed(name):
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (completed)")
            continue

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")

        config = {
            "iou_clip_start": exp["iou_clip_start"],
            "iou_clip_end": exp["iou_clip_end"],
            "dfl_clip_start": exp["dfl_clip_start"],
            "dfl_clip_end": exp["dfl_clip_end"],
            "phase_ab_disabled": True,
            "phase_d_default": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            iou_clip_start=exp["iou_clip_start"],
            iou_clip_end=exp["iou_clip_end"],
            dfl_clip_start=exp["dfl_clip_start"],
            dfl_clip_end=exp["dfl_clip_end"],
        )

        results.add_experiment(name, "C2", config, metrics, time_hours)

        if i % 6 == 0:
            results.print_c2_grid_status()

    print("\n" + "=" * 80)
    print("PHASE C2 COMPLETE")
    print("=" * 80)
    results.print_c2_grid_status()
    results.analyze_results(phase="C2", top_n=10)


# =============================================================================
# PHASE C3: VALIDATION (6 runs)
# =============================================================================

def run_phase_C3(results: PhaseCResults):
    """
    Phase C3: Validation combining best IoU and DFL configs with variations.

    Experiments:
    1. Best IoU + Best DFL (final optimal)
    2. Best config with tighter clipping (×0.8)
    3. Best config with looser clipping (×1.3)
    4. IoU clipping only (DFL disabled)
    5. DFL clipping only (IoU disabled)
    6. No clipping baseline (both disabled)
    """

    # Get best configs from C1 and C2
    best_iou_start, best_iou_end = results.get_c1_best()
    best_dfl_start, best_dfl_end = results.get_c2_best()

    if best_iou_start is None:
        print("\nWarning: C1 not complete. Using default IoU config (10.0 → 5.0).")
        best_iou_start, best_iou_end = 10.0, 5.0

    if best_dfl_start is None:
        print("\nWarning: C2 not complete. Using default DFL config (8.0 → 4.0).")
        best_dfl_start, best_dfl_end = 8.0, 4.0

    print("\n" + "#" * 80)
    print("# PHASE C3: Validation (6 runs)")
    print("# Combining best IoU from C1 + best DFL from C2")
    print("# Phase A/B: DISABLED, Phase D: DEFAULT")
    print("#" * 80)
    print(f"\nBest IoU config (from C1): start={best_iou_start}, end={best_iou_end}")
    print(f"Best DFL config (from C2): start={best_dfl_start}, end={best_dfl_end}")

    # Build experiment list
    experiments = [
        {
            "name": "C3_01_best_combined",
            "desc": "Best IoU + Best DFL (optimal)",
            "iou_clip_start": best_iou_start,
            "iou_clip_end": best_iou_end,
            "dfl_clip_start": best_dfl_start,
            "dfl_clip_end": best_dfl_end,
        },
        {
            "name": "C3_02_tighter",
            "desc": "Tighter clipping (×0.8)",
            "iou_clip_start": best_iou_start * 0.8,
            "iou_clip_end": best_iou_end * 0.8,
            "dfl_clip_start": best_dfl_start * 0.8,
            "dfl_clip_end": best_dfl_end * 0.8,
        },
        {
            "name": "C3_03_looser",
            "desc": "Looser clipping (×1.3)",
            "iou_clip_start": best_iou_start * 1.3,
            "iou_clip_end": best_iou_end * 1.3,
            "dfl_clip_start": best_dfl_start * 1.3,
            "dfl_clip_end": best_dfl_end * 1.3,
        },
        {
            "name": "C3_04_iou_only",
            "desc": "IoU clipping only (DFL disabled)",
            "iou_clip_start": best_iou_start,
            "iou_clip_end": best_iou_end,
            "dfl_clip_start": 100.0,
            "dfl_clip_end": 100.0,
        },
        {
            "name": "C3_05_dfl_only",
            "desc": "DFL clipping only (IoU disabled)",
            "iou_clip_start": 100.0,
            "iou_clip_end": 100.0,
            "dfl_clip_start": best_dfl_start,
            "dfl_clip_end": best_dfl_end,
        },
        {
            "name": "C3_06_no_clipping",
            "desc": "No clipping baseline (both disabled)",
            "iou_clip_start": 100.0,
            "iou_clip_end": 100.0,
            "dfl_clip_start": 100.0,
            "dfl_clip_end": 100.0,
        },
    ]

    print(f"\n{len(experiments)} validation experiments:")
    for exp in experiments:
        print(f"  - {exp['name']}: {exp['desc']}")
        print(f"      IoU: {exp['iou_clip_start']} → {exp['iou_clip_end']}")
        print(f"      DFL: {exp['dfl_clip_start']} → {exp['dfl_clip_end']}")

    # Run experiments
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]

        if results.is_completed(name):
            print(f"\n>> [{i}/{len(experiments)}] Skipping {name} (completed)")
            continue

        print(f"\n>> [{i}/{len(experiments)}] Running {name}")
        print(f"   {exp['desc']}")

        config = {
            "iou_clip_start": exp["iou_clip_start"],
            "iou_clip_end": exp["iou_clip_end"],
            "dfl_clip_start": exp["dfl_clip_start"],
            "dfl_clip_end": exp["dfl_clip_end"],
            "phase_ab_disabled": True,
            "phase_d_default": True,
        }

        metrics, time_hours = train_experiment(
            name=name,
            iou_clip_start=exp["iou_clip_start"],
            iou_clip_end=exp["iou_clip_end"],
            dfl_clip_start=exp["dfl_clip_start"],
            dfl_clip_end=exp["dfl_clip_end"],
        )

        results.add_experiment(name, "C3", config, metrics, time_hours)

    print("\n" + "=" * 80)
    print("PHASE C3 COMPLETE")
    print("=" * 80)
    results.analyze_results(phase="C3", top_n=6)


# =============================================================================
# SUMMARY
# =============================================================================

def print_final_summary(results: PhaseCResults):
    """Print final Phase C summary."""

    print("\n" + "=" * 100)
    print("PHASE C GRID SEARCH: FINAL SUMMARY")
    print("=" * 100)
    print("\nNOTE: Phase A/B parameters were DISABLED for this ablation study.")
    print("      Phase D (TAL) parameters were set to DEFAULT values.")
    print("      Only clipping parameters (IoU, DFL) were tested.")

    # Count experiments per phase
    all_exp = results.get_all_experiments()
    c1_count = len([e for e in all_exp if e.get("phase") == "C1"])
    c2_count = len([e for e in all_exp if e.get("phase") == "C2"])
    c3_count = len([e for e in all_exp if e.get("phase") == "C3"])

    # Calculate valid combinations (start >= end)
    c1_valid = sum(1 for s in IOU_CLIP_START_VALUES for e in IOU_CLIP_END_VALUES if s >= e)
    c2_valid = sum(1 for s in DFL_CLIP_START_VALUES for e in DFL_CLIP_END_VALUES if s >= e)

    print(f"\nCompleted experiments:")
    print(f"  C1 (IoU grid, DFL disabled):    {c1_count}/{c1_valid}")
    print(f"  C2 (DFL grid, IoU disabled):    {c2_count}/{c2_valid}")
    print(f"  C3 (validation, best combined): {c3_count}/6")

    # Best from each phase
    if results.data.get("c1_best_config"):
        bc = results.data["c1_best_config"]
        bm = results.data["c1_best_metrics"]
        print(f"\nBest C1 (IoU clipping only):")
        print(f"  iou_clip: {bc.get('iou_clip_start')} → {bc.get('iou_clip_end')}")
        print(f"  mAP_small={bm.get('mAP_small', 0):.4f}")

    if results.data.get("c2_best_config"):
        bc = results.data["c2_best_config"]
        bm = results.data["c2_best_metrics"]
        print(f"\nBest C2 (DFL clipping only):")
        print(f"  dfl_clip: {bc.get('dfl_clip_start')} → {bc.get('dfl_clip_end')}")
        print(f"  mAP_small={bm.get('mAP_small', 0):.4f}")

    # Final best
    if results.data.get("final_best_config"):
        bc = results.data["final_best_config"]
        bm = results.data["final_best_metrics"]

        print(f"""
+------------------------------------------------------------------------------+
|                         FINAL BEST CONFIGURATION                             |
+------------------------------------------------------------------------------+
|  iou_clip_start: {str(bc.get('iou_clip_start', 'N/A')):<60}|
|  iou_clip_end:   {str(bc.get('iou_clip_end', 'N/A')):<60}|
|  dfl_clip_start: {str(bc.get('dfl_clip_start', 'N/A')):<60}|
|  dfl_clip_end:   {str(bc.get('dfl_clip_end', 'N/A')):<60}|
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
    """Run Phase C grid search."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Phase C: Adaptive Loss Clipping Grid Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase_c_clipping_grid.py              # Run full Phase C
  python phase_c_clipping_grid.py --phase C1   # Run only C1 (IoU grid)
  python phase_c_clipping_grid.py --phase C2   # Run only C2 (DFL grid)
  python phase_c_clipping_grid.py --phase C3   # Run only C3 (validation)
  python phase_c_clipping_grid.py --status     # Show current progress
  python phase_c_clipping_grid.py --results    # Show results analysis
        """
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "C1", "C2", "C3"],
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
    results = PhaseCResults(PROJECT_DIR)

    print(f"\nPhase C: Adaptive Loss Clipping Grid Search (ISOLATED)")
    print(f"Project: {PROJECT_DIR}")
    print(f"Previous results: {PREVIOUS_RESULTS_FILE}")

    # Show disabled parameters
    print(f"\n{'=' * 80}")
    print("FIXED HYPERPARAMETERS")
    print(f"{'=' * 80}")
    print(f"  Phase A: DISABLED (alpha=1.0, boost=1.0)")
    print(f"  Phase B: DISABLED (center_loss=0.0)")
    print(f"  Phase D: DEFAULT (tal_topk={TAL_TOPK}, tal_alpha={TAL_ALPHA}, tal_beta={TAL_BETA})")

    # Calculate valid combinations (start >= end)
    c1_valid = sum(1 for s in IOU_CLIP_START_VALUES for e in IOU_CLIP_END_VALUES if s >= e)
    c2_valid = sum(1 for s in DFL_CLIP_START_VALUES for e in DFL_CLIP_END_VALUES if s >= e)

    # Show grid configuration
    print(f"\n{'=' * 80}")
    print("PHASE C GRID CONFIGURATION (ACTIVE)")
    print(f"{'=' * 80}")
    print(f"  C1: IoU Clipping Grid ({c1_valid} valid combinations)")
    print(f"      iou_clip_start: {IOU_CLIP_START_VALUES}")
    print(f"      iou_clip_end:   {IOU_CLIP_END_VALUES}")
    print(f"      DFL clipping: DISABLED (100.0 → 100.0)")
    print(f"")
    print(f"  C2: DFL Clipping Grid ({c2_valid} valid combinations)")
    print(f"      dfl_clip_start: {DFL_CLIP_START_VALUES}")
    print(f"      dfl_clip_end:   {DFL_CLIP_END_VALUES}")
    print(f"      IoU clipping: DISABLED (100.0 → 100.0)")
    print(f"")
    print(f"  C3: Validation (6 experiments)")
    print(f"      Combines best IoU (C1) + best DFL (C2) + variations")
    print(f"")
    print(f"  Total: {c1_valid + c2_valid + 6} experiments (~{(c1_valid + c2_valid + 6) * 1.3:.0f} hours)")

    if args.status:
        results.print_c1_grid_status()
        results.print_c2_grid_status()
        return

    if args.results:
        results.analyze_results(top_n=20)
        return

    # Run phases
    if args.phase == "all":
        run_phase_C1(results)
        run_phase_C2(results)
        run_phase_C3(results)
    elif args.phase == "C1":
        run_phase_C1(results)
    elif args.phase == "C2":
        run_phase_C2(results)
    elif args.phase == "C3":
        run_phase_C3(results)

    print_final_summary(results)


if __name__ == "__main__":
    main()
