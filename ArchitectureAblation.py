#!/usr/bin/env python3
"""
Phase C Ablation: Architecture Search for Small Object Detection
=================================================================

Overview
--------
Phase C performs a systematic architecture ablation study over 20 candidate
architectures (plus 1 baseline) to identify the optimal network topology
for small object detection. Each architecture is trained with identical
hyperparameters; only the model structure varies.

NOTE: All loss/training hyperparameters use the best combined configuration
      from Phase B (or neutral defaults if Phase B is not yet complete).

Training Configuration
----------------------
    Epochs:     70
    Image Size: 640
    Batch Size: 64
    Model:      YOLOv12n (scaled per architecture)

COCO Evaluation Ranges (Standard)
---------------------------------
    Small:  area < 32² pixels   (< 1024 px²)
    Medium: 32² ≤ area < 96²    (1024 - 9216 px²)
    Large:  area ≥ 96² pixels   (≥ 9216 px²)

Architecture Categories
-----------------------
    Lightweight / Efficiency:
        Arch-1:  P2 Head Light (reduced backbone repeats)

    Neck Variants:
        Arch-2:  BiFPN-style Neck (C3k2 in all fusion stages)
        Arch-9:  Auxiliary + BiFPN Neck

    Extra Detection Heads:
        Arch-3:  P6 Extra Head (stride-64, 5 heads)

    Deeper / Wider Backbones:
        Arch-4:  Deeper Backbone (increased repeats throughout)
        Arch-12: Deeper P2 Branch (3x C3k2 True at P2)
        Arch-13: Wider P2 Branch (192 channels at P2)
        Arch-14: Deeper P3 Branch (3x A2C2f at P3)

    Baseline P2 Head:
        Arch-5:  Standard P2 Head (baseline)

    P2 Auxiliary Variants:
        Arch-6:  P2 Auxiliary ★ BEST (dedicated aux branch)
        Arch-7:  P2 Deeper Auxiliary (2x C3k2 in aux)
        Arch-8:  Auxiliary at P3 (aux shifted to stride-8)
        Arch-10: Auxiliary + Residual P2 (1x1 alignment)
        Arch-11: Auxiliary + MultiScale P2 (2x C3k2 before split)
        Arch-16: Balanced Auxiliary (A2C2f in bottom-up)
        Arch-17: Deeper Auxiliary Branch (3x C3k2 in aux)
        Arch-18: Wider Auxiliary Branch (192ch aux)

    Attention at P2:
        Arch-15: A2C2f Attention at P2
        Arch-19: Wide P2 A2C2f (256ch attention)

    Skip Connections:
        Arch-20: P2 Skip Connection (backbone-to-P2 skip)

Step C1: Full Architecture Sweep (20 runs)
-------------------------------------------
Trains all 20 architectures with identical settings.

    Arch-1  through Arch-20 (sequential)
    Fixed hyperparameters from Phase B best (or defaults)

    Total: 20 experiments
    Estimated time: ~26 hours (at ~1.3 hours per 70-epoch run)

Step C2: Top-5 Extended Training (5 runs)
------------------------------------------
Takes the top-5 architectures from C1 (ranked by mAP_small)
and retrains them for 150 epochs to confirm rankings hold
with longer training.

    Epochs: 150
    Total: 5 experiments
    Estimated time: ~14 hours (at ~2.8 hours per 150-epoch run)

Step C3: Sensitivity Analysis (10 runs)
----------------------------------------
For the top-2 architectures from C2, tests sensitivity to:
    - Image size: 640 vs 896 vs 1280
    - Batch size: 32 vs 64
    - Scale variant: n vs s

    Total: up to 10 experiments
    Estimated time: ~20 hours (variable per config)

Total Phase C Statistics
-------------------------
    Step C1: 20 runs (full architecture sweep)
    Step C2:  5 runs (top-5 extended training)
    Step C3: 10 runs (sensitivity analysis)
    ---------------------------------
    Total:   35 runs
    Estimated: ~60 hours (~2.5 days)

Overall Ablation Structure
--------------------------
    Phase A:  Isolated hyperparameter ablation
      ├── A1: Size-Aware Weighting    (phase_a1_ablation.py)
      ├── A2: Center Loss             (phase_a2_ablation.py)
      ├── A3: Adaptive Clipping       (phase_a3_ablation.py)
      └── A4: TAL Parameters          (phase_a4_ablation.py)
    Phase B:  Combined configuration   (phase_b_ablation.py)
    Phase C:  Architecture search      (this script)

Usage
-----
    python phase_c_ablation.py                        # Run full Phase C
    python phase_c_ablation.py --step C1              # Run only C1 (sweep)
    python phase_c_ablation.py --step C2              # Run only C2 (extended)
    python phase_c_ablation.py --step C3              # Run only C3 (sensitivity)
    python phase_c_ablation.py --arch Arch-6          # Run single architecture
    python phase_c_ablation.py --status               # Show current progress
    python phase_c_ablation.py --results              # Show results analysis
    python phase_c_ablation.py --results --sort small # Sort by mAP_small

Author: Constantin
Project: YOLOv12 Small Object Detection Optimization
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# ALL 20 ARCHITECTURES
# ============================================================

ARCH_1 = """# Arch-1: P2 Head Light — 4 det heads, fewer backbone repeats
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 1, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 3, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 1, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 1, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 1, C3k2, [128, False, 0.25]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 1, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 1, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[17, 20, 23, 26], 1, Detect, [nc]]
"""

ARCH_2 = """# Arch-2: BiFPN-style Neck — 4 det heads, C3k2 neck blocks
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False, 0.25]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False, 0.25]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[17, 20, 23, 26], 1, Detect, [nc]]
"""

ARCH_3 = """# Arch-3: P6 Extra Head — 5 det heads including stride-64
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]

  - [[17, 20, 23, 26, 28], 1, Detect, [nc]]
"""

ARCH_4 = """# Arch-4: Deeper Backbone — 4 det heads, increased repeats throughout
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 3, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 5, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 7, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 3, C3k2, [128, False, 0.25]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 3, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 4, C3k2, [1024, True]]

  - [[17, 20, 23, 26], 1, Detect, [nc]]
"""

ARCH_5 = """# Arch-5: Standard P2 Head — 4 det heads, baseline design
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[17, 20, 23, 26], 1, Detect, [nc]]
"""

ARCH_6 = """# Arch-6: P2 Auxiliary ★ BEST — 5 det heads, dedicated aux at P2
nc: 4
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, C3k2, [128, True]]
  - [17, 1, Conv, [128, 3, 1]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[18, 19, 22, 25, 28], 1, Detect, [nc]]
"""

ARCH_7 = """# Arch-7: P2 Deeper Auxiliary — 5 det heads, 2x C3k2 in aux branch
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 2, C3k2, [128, True]]
  - [17, 1, Conv, [128, 3, 1]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[18, 19, 22, 25, 28], 1, Detect, [nc]]
"""

ARCH_8 = """# Arch-8: Auxiliary at P3 — 5 det heads, aux branch at stride-8
nc: 4
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, C3k2, [256, True]]
  - [14, 1, Conv, [256, 3, 1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[15, 19, 22, 25, 28], 1, Detect, [nc]]
"""

ARCH_9 = """# Arch-9: Auxiliary + BiFPN Neck — 5 det heads, C3k2 throughout neck
nc: 4
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False, 0.25]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, C3k2, [128, True]]
  - [17, 1, Conv, [128, 3, 1]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False, 0.25]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[18, 19, 22, 25, 28], 1, Detect, [nc]]
"""

ARCH_10 = """# Arch-10: Auxiliary + Residual P2 — 5 det heads, 1x1 alignment before aux
nc: 4
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]
  - [-1, 1, Conv, [128, 1, 1]]

  - [-1, 1, C3k2, [128, True]]
  - [17, 1, Conv, [128, 3, 1]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[19, 20, 23, 26, 29], 1, Detect, [nc]]
"""

ARCH_11 = """# Arch-11: Auxiliary + MultiScale P2 — 5 det heads, 2x C3k2 before aux
nc: 4
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 1, C3k2, [128, False, 0.25]]
  - [-1, 1, C3k2, [128, False, 0.25]]

  - [-1, 1, C3k2, [128, True]]
  - [18, 1, Conv, [128, 3, 1]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[19, 20, 23, 26, 29], 1, Detect, [nc]]
"""

ARCH_12 = """# Arch-12: Deeper P2 Branch — 4 det heads, 3x C3k2(True) at P2
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 3, C3k2, [128, True]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[17, 20, 23, 26], 1, Detect, [nc]]
"""

ARCH_13 = """# Arch-13: Wider P2 Branch — 4 det heads, 192 channels at P2
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [192, False, 0.25]]

  - [-1, 1, Conv, [192, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[17, 20, 23, 26], 1, Detect, [nc]]
"""

ARCH_14 = """# Arch-14: Deeper P3 Branch — 4 det heads, 3x A2C2f at P3 stages
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 3, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[17, 20, 23, 26], 1, Detect, [nc]]
"""

ARCH_15 = """# Arch-15: A2C2f at P2 — 4 det heads, attention-based P2 processing
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, A2C2f, [128, False, -1]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[17, 20, 23, 26], 1, Detect, [nc]]
"""

ARCH_16 = """# Arch-16: Balanced Auxiliary — 5 det heads, A2C2f in bottom-up path
nc: 4
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, C3k2, [128, True]]
  - [17, 1, Conv, [128, 3, 1]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[18, 19, 22, 25, 28], 1, Detect, [nc]]
"""

ARCH_17 = """# Arch-17: Deeper Aux Branch — 5 det heads, 3x C3k2 in auxiliary
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 3, C3k2, [128, True]]
  - [17, 1, Conv, [128, 3, 1]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[18, 19, 22, 25, 28], 1, Detect, [nc]]
"""

ARCH_18 = """# Arch-18: Wider Aux Branch — 5 det heads, 192-channel auxiliary
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 2, C3k2, [192, True]]
  - [17, 1, Conv, [192, 3, 1]]

  - [-1, 1, Conv, [192, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[18, 19, 22, 25, 28], 1, Detect, [nc]]
"""

ARCH_19 = """# Arch-19: Wide P2 A2C2f — 4 det heads, 256ch attention-based P2
nc: 4
scales:
  n: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 1, Conv, [256, 3, 1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[18, 21, 24, 27], 1, Detect, [nc]]
"""

ARCH_20 = """# Arch-20: P2 Skip Connection — 4 det heads, backbone-to-P2 skip
nc: 4
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2, 1, 4]]
  - [-1, 3, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 5, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2, 0], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False, 0.25]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 3, C3k2, [1024, True]]

  - [[17, 20, 23, 26], 1, Detect, [nc]]
"""

# ============================================================
# ARCHITECTURE LOOKUP TABLE
# ============================================================
ARCHITECTURE_TABLE = {
    "Arch-1":  {"yaml": ARCH_1,  "category": "Lightweight",      "heads": 4, "description": "Lightweight 4-head with reduced backbone repeats for faster inference"},
    "Arch-2":  {"yaml": ARCH_2,  "category": "Neck Variant",     "heads": 4, "description": "BiFPN-style neck using C3k2 blocks instead of A2C2f in all fusion stages"},
    "Arch-3":  {"yaml": ARCH_3,  "category": "Extra Head",       "heads": 5, "description": "5-head design with extra P6 stride-64 head for large-object detection"},
    "Arch-4":  {"yaml": ARCH_4,  "category": "Deeper Backbone",  "heads": 4, "description": "Deeper backbone and neck with increased repeats at every stage"},
    "Arch-5":  {"yaml": ARCH_5,  "category": "Baseline",         "heads": 4, "description": "Standard 4-head P2 baseline with A2C2f backbone and C3k2 at P2"},
    "Arch-6":  {"yaml": ARCH_6,  "category": "P2 Auxiliary",     "heads": 5, "description": "★ BEST — 5-head with dedicated P2 auxiliary branch (C3k2 + Conv)"},
    "Arch-7":  {"yaml": ARCH_7,  "category": "P2 Auxiliary",     "heads": 5, "description": "5-head with deeper P2 auxiliary branch (2x C3k2 repeats in aux)"},
    "Arch-8":  {"yaml": ARCH_8,  "category": "P3 Auxiliary",     "heads": 5, "description": "5-head with auxiliary branch shifted to P3 instead of P2"},
    "Arch-9":  {"yaml": ARCH_9,  "category": "P2 Aux + BiFPN",   "heads": 5, "description": "5-head combining P2 auxiliary with BiFPN-style C3k2 neck"},
    "Arch-10": {"yaml": ARCH_10, "category": "P2 Auxiliary",     "heads": 5, "description": "5-head with 1x1 Conv channel alignment enabling residual flow at P2"},
    "Arch-11": {"yaml": ARCH_11, "category": "P2 Auxiliary",     "heads": 5, "description": "5-head with two sequential C3k2 blocks before aux split (multi-scale P2)"},
    "Arch-12": {"yaml": ARCH_12, "category": "Deeper P2",        "heads": 4, "description": "4-head with 3x C3k2(True) at P2 for deep small-object refinement"},
    "Arch-13": {"yaml": ARCH_13, "category": "Wider P2",         "heads": 4, "description": "4-head with wider P2 channels (192 instead of 128)"},
    "Arch-14": {"yaml": ARCH_14, "category": "Deeper P3",        "heads": 4, "description": "4-head with deeper P3 stages (3x A2C2f in both top-down and bottom-up)"},
    "Arch-15": {"yaml": ARCH_15, "category": "P2 Attention",     "heads": 4, "description": "4-head with A2C2f attention replacing C3k2 at P2"},
    "Arch-16": {"yaml": ARCH_16, "category": "P2 Auxiliary",     "heads": 5, "description": "5-head auxiliary with A2C2f in bottom-up path for balanced attention"},
    "Arch-17": {"yaml": ARCH_17, "category": "P2 Auxiliary",     "heads": 5, "description": "5-head with 3x C3k2 in auxiliary branch for maximum aux capacity"},
    "Arch-18": {"yaml": ARCH_18, "category": "P2 Auxiliary",     "heads": 5, "description": "5-head with wider 192-channel auxiliary branch"},
    "Arch-19": {"yaml": ARCH_19, "category": "P2 Attention",     "heads": 4, "description": "4-head with 256ch A2C2f attention at P2 and Conv projection after concat"},
    "Arch-20": {"yaml": ARCH_20, "category": "Skip Connection",  "heads": 4, "description": "4-head with direct skip connection from first backbone Conv to P2"},
}


# ============================================================
# CONFIGURATION
# ============================================================

# Training defaults
DEFAULT_EPOCHS = 70
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 64
DEFAULT_DEVICE = 0

# Dataset — UPDATE THIS PATH
DATASET_YAML = "data/your_dataset.yaml"

# Output directories
BASE_OUTPUT_DIR = Path("runs/phase_c_architecture")
YAML_DIR = BASE_OUTPUT_DIR / "yaml_configs"
RESULTS_FILE = BASE_OUTPUT_DIR / "phase_c_results.json"
PROGRESS_FILE = BASE_OUTPUT_DIR / "phase_c_progress.json"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def setup_dirs():
    """Create output directories."""
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    YAML_DIR.mkdir(parents=True, exist_ok=True)


def write_yaml(arch_name: str, yaml_content: str) -> Path:
    """Write architecture YAML to disk and return the path."""
    yaml_path = YAML_DIR / f"{arch_name.lower().replace('-', '_')}.yaml"
    yaml_path.write_text(yaml_content.strip() + "\n")
    return yaml_path


def load_progress() -> dict:
    """Load progress tracking file."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": {}, "failed": {}, "start_time": None}


def save_progress(progress: dict):
    """Save progress tracking file."""
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def load_results() -> dict:
    """Load results file."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict):
    """Save results file."""
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def extract_metrics(results_dir: Path) -> dict:
    """
    Extract COCO metrics from a completed training run.

    Looks for results.csv or results.json in the training output directory
    and extracts mAP values at different object sizes.

    Returns dict with keys:
        mAP50, mAP50_95, mAP_small, mAP_medium, mAP_large,
        precision, recall, params, flops, train_time
    """
    metrics = {
        "mAP50": None,
        "mAP50_95": None,
        "mAP_small": None,
        "mAP_medium": None,
        "mAP_large": None,
        "precision": None,
        "recall": None,
        "params_M": None,
        "flops_G": None,
        "train_time_hours": None,
    }

    # Try results CSV (standard ultralytics output)
    csv_path = results_dir / "results.csv"
    if csv_path.exists():
        try:
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    # Column names vary by ultralytics version
                    for key in last:
                        k = key.strip()
                        if "metrics/mAP50(B)" in k:
                            metrics["mAP50"] = float(last[key])
                        elif "metrics/mAP50-95(B)" in k:
                            metrics["mAP50_95"] = float(last[key])
                        elif "metrics/precision(B)" in k:
                            metrics["precision"] = float(last[key])
                        elif "metrics/recall(B)" in k:
                            metrics["recall"] = float(last[key])
        except Exception as e:
            print(f"  Warning: Could not parse CSV: {e}")

    # Try COCO eval JSON for size-specific metrics
    coco_json = results_dir / "coco_eval.json"
    if coco_json.exists():
        try:
            coco_data = json.loads(coco_json.read_text())
            if isinstance(coco_data, dict):
                metrics["mAP_small"] = coco_data.get("mAP_small")
                metrics["mAP_medium"] = coco_data.get("mAP_medium")
                metrics["mAP_large"] = coco_data.get("mAP_large")
        except Exception as e:
            print(f"  Warning: Could not parse COCO JSON: {e}")

    return metrics


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def print_table(headers: list, rows: list, col_widths: list = None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                      for i, h in enumerate(headers)]

    header_line = "│".join(str(h).center(w) for h, w in zip(headers, col_widths))
    separator = "┼".join("─" * w for w in col_widths)

    print(f"┌{'┬'.join('─' * w for w in col_widths)}┐")
    print(f"│{header_line}│")
    print(f"├{separator}┤")
    for row in rows:
        row_line = "│".join(str(v).center(w) for v, w in zip(row, col_widths))
        print(f"│{row_line}│")
    print(f"└{'┴'.join('─' * w for w in col_widths)}┘")


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_architecture(
    arch_name: str,
    yaml_path: Path,
    experiment_name: str,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    device: int = DEFAULT_DEVICE,
) -> dict:
    """
    Train a single architecture and return metrics.

    Parameters
    ----------
    arch_name : str
        Architecture identifier (e.g., 'Arch-6')
    yaml_path : Path
        Path to the model YAML configuration file
    experiment_name : str
        Name for this experiment run
    epochs : int
        Number of training epochs
    imgsz : int
        Input image size
    batch : int
        Batch size
    device : int
        GPU device ID

    Returns
    -------
    dict
        Training metrics
    """
    from ultralytics import YOLO

    project_dir = BASE_OUTPUT_DIR / "trains"
    run_name = experiment_name

    print(f"\n{'─' * 60}")
    print(f"  Training: {arch_name} ({experiment_name})")
    print(f"  Config:   epochs={epochs}, imgsz={imgsz}, batch={batch}")
    print(f"  YAML:     {yaml_path}")
    print(f"  Output:   {project_dir / run_name}")
    print(f"{'─' * 60}")

    start_time = time.time()

    try:
        model = YOLO(str(yaml_path))

        results = model.train(
            data=DATASET_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            close_mosaic=15,
            patience=20,
            verbose=True
        )

        elapsed = (time.time() - start_time) / 3600
        results_dir = project_dir / run_name
        metrics = extract_metrics(results_dir)
        metrics["train_time_hours"] = round(elapsed, 2)
        metrics["status"] = "completed"
        metrics["arch_name"] = arch_name
        metrics["experiment"] = experiment_name

        # Try to get params from model
        try:
            total_params = sum(p.numel() for p in model.model.parameters())
            metrics["params_M"] = round(total_params / 1e6, 2)
        except Exception:
            pass

        print(f"\n  ✓ Completed {arch_name} in {elapsed:.1f}h")
        if metrics["mAP50_95"]:
            print(f"    mAP50-95: {metrics['mAP50_95']:.4f}")
        if metrics["mAP_small"]:
            print(f"    mAP_small: {metrics['mAP_small']:.4f}")

        return metrics

    except Exception as e:
        elapsed = (time.time() - start_time) / 3600
        print(f"\n  ✗ FAILED {arch_name} after {elapsed:.1f}h: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "arch_name": arch_name,
            "experiment": experiment_name,
            "train_time_hours": round(elapsed, 2),
        }


# ============================================================
# STEP C1: FULL ARCHITECTURE SWEEP
# ============================================================

def run_step_c1(device: int = DEFAULT_DEVICE, skip_completed: bool = True):
    """
    Step C1: Train all 20 architectures with identical settings.

    Each architecture is trained for 70 epochs at 640px.
    Results are compared on mAP_small, mAP50-95, and efficiency.
    """
    print_header("STEP C1: Full Architecture Sweep (20 runs)")

    setup_dirs()
    progress = load_progress()
    results = load_results()

    if progress.get("start_time") is None:
        progress["start_time"] = datetime.now().isoformat()

    arch_names = sorted(ARCHITECTURE_TABLE.keys(), key=lambda x: int(x.split("-")[1]))

    total = len(arch_names)
    completed_count = 0
    skipped_count = 0

    for i, arch_name in enumerate(arch_names, 1):
        experiment_name = f"C1_{arch_name.replace('-', '_')}"

        # Skip if already completed
        if skip_completed and experiment_name in progress.get("completed", {}):
            skipped_count += 1
            print(f"  [{i}/{total}] Skipping {arch_name} (already completed)")
            continue

        arch_info = ARCHITECTURE_TABLE[arch_name]
        print(f"\n  [{i}/{total}] {arch_name}: {arch_info['description']}")
        print(f"           Category: {arch_info['category']} | Heads: {arch_info['heads']}")

        # Write YAML
        yaml_path = write_yaml(arch_name, arch_info["yaml"])

        # Train
        metrics = train_architecture(
            arch_name=arch_name,
            yaml_path=yaml_path,
            experiment_name=experiment_name,
            epochs=DEFAULT_EPOCHS,
            imgsz=DEFAULT_IMGSZ,
            batch=DEFAULT_BATCH,
            device=device,
        )

        # Save progress
        if metrics["status"] == "completed":
            progress.setdefault("completed", {})[experiment_name] = {
                "arch_name": arch_name,
                "timestamp": datetime.now().isoformat(),
            }
            results[experiment_name] = metrics
            completed_count += 1
        else:
            progress.setdefault("failed", {})[experiment_name] = {
                "arch_name": arch_name,
                "error": metrics.get("error", "unknown"),
                "timestamp": datetime.now().isoformat(),
            }

        save_progress(progress)
        save_results(results)

    print_header("STEP C1 SUMMARY", "─")
    print(f"  Total architectures: {total}")
    print(f"  Completed this run:  {completed_count}")
    print(f"  Skipped (existing):  {skipped_count}")
    print(f"  Failed:              {total - completed_count - skipped_count}")

    return results


# ============================================================
# STEP C2: TOP-5 EXTENDED TRAINING
# ============================================================

def get_top_k_architectures(results: dict, k: int = 5, metric: str = "mAP_small") -> list:
    """
    Rank architectures by a metric and return top-k.

    Falls back to mAP50_95 if the primary metric is unavailable.
    """
    scored = []
    for exp_name, metrics in results.items():
        if not exp_name.startswith("C1_"):
            continue
        if metrics.get("status") != "completed":
            continue

        score = metrics.get(metric)
        if score is None:
            score = metrics.get("mAP50_95", 0)

        scored.append((exp_name, metrics.get("arch_name", exp_name), score, metrics))

    scored.sort(key=lambda x: x[2] if x[2] is not None else 0, reverse=True)

    print(f"\n  Architecture ranking by {metric}:")
    for rank, (exp, arch, score, _) in enumerate(scored, 1):
        marker = " ★" if rank <= k else ""
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"    #{rank:2d} {arch:10s} {metric}={score_str}{marker}")

    return scored[:k]


def run_step_c2(device: int = DEFAULT_DEVICE, skip_completed: bool = True):
    """
    Step C2: Retrain top-5 architectures for 150 epochs.

    Confirms that rankings from C1 hold with longer training.
    """
    print_header("STEP C2: Top-5 Extended Training (150 epochs)")

    results = load_results()
    progress = load_progress()

    # Get top-5 from C1
    top5 = get_top_k_architectures(results, k=5, metric="mAP_small")

    if not top5:
        print("  ERROR: No C1 results found. Run Step C1 first.")
        return results

    extended_epochs = 150
    completed_count = 0

    for rank, (c1_exp, arch_name, c1_score, c1_metrics) in enumerate(top5, 1):
        experiment_name = f"C2_{arch_name.replace('-', '_')}_150ep"

        if skip_completed and experiment_name in progress.get("completed", {}):
            print(f"  [#{rank}] Skipping {arch_name} (already completed)")
            continue

        arch_info = ARCHITECTURE_TABLE.get(arch_name)
        if arch_info is None:
            print(f"  [#{rank}] ERROR: {arch_name} not found in ARCHITECTURE_TABLE")
            continue

        c1_score_str = f"{c1_score:.4f}" if c1_score is not None else "N/A"
        print(f"\n  [#{rank}] {arch_name} (C1 mAP_small={c1_score_str})")
        print(f"          Training for {extended_epochs} epochs...")

        yaml_path = write_yaml(arch_name, arch_info["yaml"])

        metrics = train_architecture(
            arch_name=arch_name,
            yaml_path=yaml_path,
            experiment_name=experiment_name,
            epochs=extended_epochs,
            imgsz=DEFAULT_IMGSZ,
            batch=DEFAULT_BATCH,
            device=device,
        )

        if metrics["status"] == "completed":
            progress.setdefault("completed", {})[experiment_name] = {
                "arch_name": arch_name,
                "timestamp": datetime.now().isoformat(),
            }
            results[experiment_name] = metrics
            completed_count += 1
        else:
            progress.setdefault("failed", {})[experiment_name] = {
                "arch_name": arch_name,
                "error": metrics.get("error", "unknown"),
                "timestamp": datetime.now().isoformat(),
            }

        save_progress(progress)
        save_results(results)

    print_header("STEP C2 SUMMARY", "─")
    print(f"  Extended training completed: {completed_count}/{len(top5)}")

    return results


# ============================================================
# STEP C3: SENSITIVITY ANALYSIS
# ============================================================

def run_step_c3(device: int = DEFAULT_DEVICE, skip_completed: bool = True):
    """
    Step C3: Sensitivity analysis on top-2 architectures.

    Tests:
      - Image sizes: 640, 896, 1280
      - Batch sizes: 32, 64
      - Scale variants: n, s (if applicable)
    """
    print_header("STEP C3: Sensitivity Analysis (Top-2 Architectures)")

    results = load_results()
    progress = load_progress()

    # Get top-2 from C2 results, fall back to C1
    c2_results = {k: v for k, v in results.items() if k.startswith("C2_")}
    if c2_results:
        top2 = get_top_k_architectures(results, k=2, metric="mAP_small")
        # Filter to only C2 entries for ranking
        c2_top = get_top_k_architectures(
            {k: v for k, v in results.items() if k.startswith("C2_")},
            k=2, metric="mAP_small"
        )
        if c2_top:
            top2 = c2_top
    else:
        top2 = get_top_k_architectures(results, k=2, metric="mAP_small")

    if not top2:
        print("  ERROR: No results found. Run Step C1/C2 first.")
        return results

    # Sensitivity configurations
    sensitivity_configs = [
        {"label": "imgsz_896",  "epochs": DEFAULT_EPOCHS, "imgsz": 896,  "batch": 32},
        {"label": "imgsz_1280", "epochs": DEFAULT_EPOCHS, "imgsz": 1280, "batch": 16},
        {"label": "batch_32",   "epochs": DEFAULT_EPOCHS, "imgsz": 640,  "batch": 32},
        {"label": "epochs_100", "epochs": 100,            "imgsz": 640,  "batch": DEFAULT_BATCH},
    ]

    completed_count = 0

    for rank, (_, arch_name, _, _) in enumerate(top2, 1):
        arch_info = ARCHITECTURE_TABLE.get(arch_name)
        if arch_info is None:
            continue

        for config in sensitivity_configs:
            experiment_name = f"C3_{arch_name.replace('-', '_')}_{config['label']}"

            if skip_completed and experiment_name in progress.get("completed", {}):
                print(f"  Skipping {experiment_name} (already completed)")
                continue

            print(f"\n  [#{rank}] {arch_name} — {config['label']}")
            print(f"          epochs={config['epochs']}, imgsz={config['imgsz']}, batch={config['batch']}")

            yaml_path = write_yaml(arch_name, arch_info["yaml"])

            metrics = train_architecture(
                arch_name=arch_name,
                yaml_path=yaml_path,
                experiment_name=experiment_name,
                epochs=config["epochs"],
                imgsz=config["imgsz"],
                batch=config["batch"],
                device=device,
            )

            if metrics["status"] == "completed":
                progress.setdefault("completed", {})[experiment_name] = {
                    "arch_name": arch_name,
                    "config": config["label"],
                    "timestamp": datetime.now().isoformat(),
                }
                results[experiment_name] = metrics
                completed_count += 1
            else:
                progress.setdefault("failed", {})[experiment_name] = {
                    "arch_name": arch_name,
                    "error": metrics.get("error", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                }

            save_progress(progress)
            save_results(results)

    print_header("STEP C3 SUMMARY", "─")
    print(f"  Sensitivity experiments completed: {completed_count}")

    return results


# ============================================================
# STATUS & RESULTS DISPLAY
# ============================================================

def show_status():
    """Display current progress across all steps."""
    print_header("PHASE C STATUS")

    progress = load_progress()
    results = load_results()

    completed = progress.get("completed", {})
    failed = progress.get("failed", {})

    # Count by step
    c1_done = sum(1 for k in completed if k.startswith("C1_"))
    c2_done = sum(1 for k in completed if k.startswith("C2_"))
    c3_done = sum(1 for k in completed if k.startswith("C3_"))

    c1_fail = sum(1 for k in failed if k.startswith("C1_"))
    c2_fail = sum(1 for k in failed if k.startswith("C2_"))
    c3_fail = sum(1 for k in failed if k.startswith("C3_"))

    print(f"\n  Step C1 (Architecture Sweep):    {c1_done:2d}/20 done, {c1_fail} failed")
    print(f"  Step C2 (Extended Training):     {c2_done:2d}/5  done, {c2_fail} failed")
    print(f"  Step C3 (Sensitivity Analysis):  {c3_done:2d}/~10 done, {c3_fail} failed")
    print(f"  {'─' * 50}")
    print(f"  Total completed: {len(completed)}")
    print(f"  Total failed:    {len(failed)}")

    if progress.get("start_time"):
        start = datetime.fromisoformat(progress["start_time"])
        elapsed = datetime.now() - start
        print(f"\n  Started:  {start.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Elapsed:  {elapsed}")

    # Show failed experiments
    if failed:
        print(f"\n  Failed experiments:")
        for exp_name, info in failed.items():
            print(f"    ✗ {exp_name}: {info.get('error', 'unknown')[:60]}")


def show_results(sort_by: str = "mAP_small"):
    """Display results analysis with rankings."""
    print_header("PHASE C RESULTS ANALYSIS")

    results = load_results()
    if not results:
        print("  No results found. Run experiments first.")
        return

    # Separate by step
    for step_prefix, step_name in [("C1_", "Step C1: Architecture Sweep"),
                                    ("C2_", "Step C2: Extended Training"),
                                    ("C3_", "Step C3: Sensitivity Analysis")]:
        step_results = {k: v for k, v in results.items()
                       if k.startswith(step_prefix) and v.get("status") == "completed"}

        if not step_results:
            continue

        print(f"\n  {step_name}")
        print(f"  {'─' * 60}")

        # Build rows
        rows = []
        for exp_name, metrics in step_results.items():
            arch = metrics.get("arch_name", "?")
            cat = ARCHITECTURE_TABLE.get(arch, {}).get("category", "?")
            m50 = metrics.get("mAP50")
            m50_95 = metrics.get("mAP50_95")
            m_small = metrics.get("mAP_small")
            m_med = metrics.get("mAP_medium")
            m_large = metrics.get("mAP_large")
            params = metrics.get("params_M")
            t = metrics.get("train_time_hours")

            rows.append({
                "arch": arch,
                "category": cat,
                "mAP50": m50,
                "mAP50_95": m50_95,
                "mAP_small": m_small,
                "mAP_medium": m_med,
                "mAP_large": m_large,
                "params_M": params,
                "time_h": t,
            })

        # Sort
        def sort_key(r):
            v = r.get(sort_by)
            return v if v is not None else -1

        rows.sort(key=sort_key, reverse=True)

        # Print
        fmt = "    {rank:>3s}  {arch:<10s} {cat:<16s} {m50:>8s} {m50_95:>8s} {ms:>8s} {mm:>8s} {ml:>8s} {p:>8s} {t:>6s}"
        print(fmt.format(
            rank="#", arch="Arch", cat="Category",
            m50="mAP50", m50_95="mAP5095", ms="Small",
            mm="Medium", ml="Large", p="Params", t="Time"
        ))
        print(f"    {'─' * 95}")

        for i, r in enumerate(rows, 1):
            marker = " ★" if i <= 3 else ""
            print(fmt.format(
                rank=f"{i}",
                arch=r["arch"],
                cat=r["category"][:16],
                m50=f"{r['mAP50']:.4f}" if r["mAP50"] is not None else "—",
                m50_95=f"{r['mAP50_95']:.4f}" if r["mAP50_95"] is not None else "—",
                ms=f"{r['mAP_small']:.4f}" if r["mAP_small"] is not None else "—",
                mm=f"{r['mAP_medium']:.4f}" if r["mAP_medium"] is not None else "—",
                ml=f"{r['mAP_large']:.4f}" if r["mAP_large"] is not None else "—",
                p=f"{r['params_M']:.1f}M" if r["params_M"] is not None else "—",
                t=f"{r['time_h']:.1f}h" if r["time_h"] is not None else "—",
            ) + marker)

    # Category analysis
    print(f"\n\n  Category Performance Summary")
    print(f"  {'─' * 60}")
    c1_results = {k: v for k, v in results.items()
                 if k.startswith("C1_") and v.get("status") == "completed"}
    if c1_results:
        categories = {}
        for metrics in c1_results.values():
            arch = metrics.get("arch_name", "?")
            cat = ARCHITECTURE_TABLE.get(arch, {}).get("category", "?")
            score = metrics.get("mAP_small") or metrics.get("mAP50_95", 0)
            if score:
                categories.setdefault(cat, []).append(score)

        for cat in sorted(categories, key=lambda c: max(categories[c]), reverse=True):
            scores = categories[cat]
            best = max(scores)
            avg = sum(scores) / len(scores)
            print(f"    {cat:<20s}  best={best:.4f}  avg={avg:.4f}  count={len(scores)}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase C Ablation: Architecture Search for Small Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--step",
        choices=["C1", "C2", "C3"],
        default=None,
        help="Run a specific step (default: run all sequentially)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Train a single architecture (e.g., --arch Arch-6)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current progress",
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help="Show results analysis",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="mAP_small",
        choices=["mAP_small", "mAP50_95", "mAP50", "mAP_medium", "mAP_large", "params_M"],
        help="Sort results by metric (default: mAP_small)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_DEVICE,
        help=f"GPU device ID (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-run completed experiments",
    )

    args = parser.parse_args()

    # Status display
    if args.status:
        show_status()
        return

    # Results display
    if args.results:
        show_results(sort_by=args.sort)
        return

    # Single architecture
    if args.arch:
        arch_name = args.arch
        if arch_name not in ARCHITECTURE_TABLE:
            print(f"ERROR: Unknown architecture '{arch_name}'")
            print(f"Valid: {', '.join(sorted(ARCHITECTURE_TABLE.keys()))}")
            sys.exit(1)

        setup_dirs()
        arch_info = ARCHITECTURE_TABLE[arch_name]
        yaml_path = write_yaml(arch_name, arch_info["yaml"])

        experiment_name = f"C1_{arch_name.replace('-', '_')}"
        metrics = train_architecture(
            arch_name=arch_name,
            yaml_path=yaml_path,
            experiment_name=experiment_name,
            device=args.device,
        )

        results = load_results()
        results[experiment_name] = metrics
        save_results(results)

        progress = load_progress()
        if metrics["status"] == "completed":
            progress.setdefault("completed", {})[experiment_name] = {
                "arch_name": arch_name,
                "timestamp": datetime.now().isoformat(),
            }
        save_progress(progress)
        return

    skip = not args.no_skip

    # Run specific step or all
    print_header("PHASE C: ARCHITECTURE ABLATION STUDY")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device:     GPU {args.device}")
    print(f"  Dataset:    {DATASET_YAML}")
    print(f"  Output:     {BASE_OUTPUT_DIR}")

    if args.step == "C1" or args.step is None:
        run_step_c1(device=args.device, skip_completed=skip)

    if args.step == "C2" or args.step is None:
        run_step_c2(device=args.device, skip_completed=skip)

    if args.step == "C3" or args.step is None:
        run_step_c3(device=args.device, skip_completed=skip)

    # Final summary
    print_header("PHASE C COMPLETE")
    show_results(sort_by="mAP_small")


if __name__ == "__main__":
    main()
