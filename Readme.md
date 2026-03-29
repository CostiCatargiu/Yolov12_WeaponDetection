<h1 align="center">🔫 Small-Object Weapon Detection with Custom YOLOv12s & YOLO26s</h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/0754c712-7237-44ff-b93b-e7b061b34bcd" alt="test1gun" width="30%">
  <img src="https://github.com/user-attachments/assets/07c743cf-aff7-4231-9f3a-88f1612b5ee9" alt="test2gun" width="30%">
  <img src="https://github.com/user-attachments/assets/919c529b-797b-4124-9ffd-931b765fd53a" alt="test3gun" width="30%">
</p>

<p align="center">
  <a href="https://universe.roboflow.com/gundetectiondataset/nogun/dataset/2">
    <img src="https://img.shields.io/badge/NoGun_Dataset-Roboflow-6706CE?style=for-the-badge&logo=roboflow&logoColor=white" alt="NoGun Dataset">
  </a>
  <a href="https://app.roboflow.com/gundetectiondataset/weapondataset-oi2g3/8">
    <img src="https://img.shields.io/badge/WeaponDataset_v8-Roboflow-6706CE?style=for-the-badge&logo=roboflow&logoColor=white" alt="WeaponDataset v8">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv12s_Custom-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-YOLO26s-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Focus-Small_Object_Detection-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Public-brightgreen?style=flat-square" />
</p>

<div align="center">

<div align="center">

## 🏆 Research Highlights

<table>
  <tr>
    <th align="center">🧪 Total Experiments</th>
    <th align="center">⏱️ Total Training Time</th>
  </tr>
  <tr>
    <td align="center"><code>225</code></td>
    <td align="center"><code>~302 hours (~12.6 days)</code></td>
  </tr>
</table>

<table>
  <tr>
    <th align="center" colspan="4">📈 Performance Improvements</th>
  </tr>
  <tr>
    <th align="center">Metric</th>
    <th align="center">🔷 YOLOv12s<br><sub>Baseline → Custom</sub></th>
    <th align="center">🔶 YOLO11s<br><sub>Baseline → Custom</sub></th>
    <th align="center">⚔️ Custom vs Custom<br><sub>YOLO11s → YOLOv12s</sub></th>
  </tr>
  <tr>
    <td align="left"><b>mAP50</b></td>
    <td align="center">0.816 → <b>0.857</b> <sub>(+5.04%)</sub></td>
    <td align="center">0.782 → <b>0.828</b> <sub>(+5.88%)</sub></td>
    <td align="center">0.828 → <b>0.857</b> <sub>(+3.43%)</sub></td>
  </tr>
  <tr>
    <td align="left"><b>mAP50-95</b></td>
    <td align="center">0.525 → <b>0.574</b> <sub>(+9.32%)</sub></td>
    <td align="center">0.502 → <b>0.555</b> <sub>(+10.53%)</sub></td>
    <td align="center">0.555 → <b>0.574</b> <sub>(+3.29%)</sub></td>
  </tr>
  <tr>
    <td align="left"><b>Precision</b></td>
    <td align="center">0.831 → <b>0.889</b> <sub>(+6.94%)</sub></td>
    <td align="center">0.792 → <b>0.863</b> <sub>(+8.94%)</sub></td>
    <td align="center">0.863 → <b>0.889</b> <sub>(+2.98%)</sub></td>
  </tr>
  <tr>
    <td align="left"><b>Recall</b></td>
    <td align="center">0.746 → <b>0.811</b> <sub>(+8.79%)</sub></td>
    <td align="center">0.698 → <b>0.776</b> <sub>(+11.14%)</sub></td>
    <td align="center">0.776 → <b>0.811</b> <sub>(+4.59%)</sub></td>
  </tr>
  <tr>
    <td align="left"><b>F1 Score</b></td>
    <td align="center">0.786 → <b>0.848</b> <sub>(+7.91%)</sub></td>
    <td align="center">0.742 → <b>0.817</b> <sub>(+10.10%)</sub></td>
    <td align="center">0.817 → <b>0.848</b> <sub>(+3.82%)</sub></td>
  </tr>
  <tr>
    <td colspan="4" align="center"><b>🔍 Size-Specific mAP50</b></td>
  </tr>
  <tr>
    <td align="left">🔍 <b>Small</b></td>
    <td align="center">0.530 → <b>0.574</b> <sub>(+8.22%)</sub></td>
    <td align="center">0.498 → <b>0.542</b> <sub>(+8.98%)</sub></td>
    <td align="center">0.542 → <b>0.574</b> <sub>(+5.76%)</sub></td>
  </tr>
  <tr>
    <td align="left">📦 <b>Medium</b></td>
    <td align="center">0.750 → <b>0.796</b> <sub>(+6.11%)</sub></td>
    <td align="center">0.718 → <b>0.765</b> <sub>(+6.51%)</sub></td>
    <td align="center">0.765 → <b>0.796</b> <sub>(+4.06%)</sub></td>
  </tr>
  <tr>
    <td align="left">🟫 <b>Large</b></td>
    <td align="center">0.828 → <b>0.871</b> <sub>(+5.24%)</sub></td>
    <td align="center">0.796 → <b>0.842</b> <sub>(+5.76%)</sub></td>
    <td align="center">0.842 → <b>0.871</b> <sub>(+3.54%)</sub></td>
  </tr>
  <tr>
    <td colspan="4" align="center"><b>🔍 Size-Specific mAP50-95</b></td>
  </tr>
  <tr>
    <td align="left">🔍 <b>Small</b></td>
    <td align="center">0.297 → <b>0.336</b> <sub>(+13.10%)</sub></td>
    <td align="center">0.276 → <b>0.316</b> <sub>(+14.55%)</sub></td>
    <td align="center">0.316 → <b>0.336</b> <sub>(+6.32%)</sub></td>
  </tr>
  <tr>
    <td align="left">📦 <b>Medium</b></td>
    <td align="center">0.424 → <b>0.467</b> <sub>(+10.23%)</sub></td>
    <td align="center">0.402 → <b>0.448</b> <sub>(+11.30%)</sub></td>
    <td align="center">0.448 → <b>0.467</b> <sub>(+4.24%)</sub></td>
  </tr>
  <tr>
    <td align="left">🟫 <b>Large</b></td>
    <td align="center">0.550 → <b>0.604</b> <sub>(+9.74%)</sub></td>
    <td align="center">0.524 → <b>0.584</b> <sub>(+11.59%)</sub></td>
    <td align="center">0.584 → <b>0.604</b> <sub>(+3.31%)</sub></td>
  </tr>
</table>

<sub>🔍 <b>Key Findings:</b> Custom loss achieves up to <b>+14.55%</b> improvement on small objects | YOLOv12s outperforms YOLO11s by <b>+6.32%</b> on small objects (mAP50-95)</sub>

</div>

---



---

## 📖 Overview

This repository accompanies our **research paper** on **small-object weapon detection**. We present:

- 📦 A **custom dataset** built from **~1,200 YouTube videos** with **59,305 images** and **76,705 annotated instances**
- 🏗️ A **modified YOLOv12s architecture** with **P2–P5 detection heads** optimized for **small object detection**
- 📉 A **custom loss function** with **size-aware weighting** and **tuned TaskAligned assigner**
- 🔍 A **comprehensive ablation study** to identify optimal hyperparameters for loss and architecture
- 📊 **Transfer validation** to **YOLO26s** using the best-performing configurations from YOLOv12s

### 📑 What You'll Find Below

| Section | Description |
|---------|-------------|
| 📊 **Dataset Statistics** | Class distribution, size analysis, split percentages |
| 🖼️ **Dataset Examples** | Sample images with annotations across different scenarios |
| 🏗️ **Architecture Details** | P2–P5 head modifications and comparisons |
| 📉 **Loss Function Ablation** | Custom loss components and tuning results |
| 🧪 **Experiment Results** | Performance metrics, confusion matrices, training curves |
| 🔍 **Prediction Examples** | Side-by-side comparisons (Original vs Custom model) |
| ⬇️ **Model Weights** | Pre-trained weights for YOLOv12s and YOLO26s |

---

## 💡 Applications

| Domain | Use Cases |
|--------|-----------|
| 📹 **Surveillance** | CCTV monitoring, real-time threat detection, smart city integration |
| 🛡️ **Public Safety** | Transportation hubs, stadiums, schools, public gatherings |
| 🚪 **Access Control** | Entry point screening, secure facilities, building protection |
| 🚔 **Law Enforcement** | Real-time threat assessment, evidence analysis, situational awareness |
| 🤖 **Research & AI** | Benchmark dataset, model training, small-object detection research |

---

## 🖥️ Hardware & Software Configuration

<details>
<summary><b>⚙️ Click to expand System Specifications</b></summary>

<br>

| Component | Specification |
|-----------|---------------|
| 💻 **Operating System** | Ubuntu 22.04.3 LTS |
| 🎮 **GPU** | NVIDIA RTX 4090 24GB |
| 🧠 **CPU** | Intel Core i9-13900KF (5.8 GHz) |
| 🗄️ **RAM** | DDR5 64GB (6000MHz) |
| 💾 **Storage** | SSD 2TB |
| 🐍 **Python** | 3.10.2 |
| 🔥 **PyTorch** | 2.1.2 |
| ⚡ **CUDA** | 12.1 |

</details>

---

## ⚡ Dataset Summary

<table>
  <tr>
    <th align="left" width="200">📋 Property</th>
    <th align="left">📊 Details</th>
  </tr>
  <tr>
    <td>🖼️ <b>Total Images</b></td>
    <td><code>59,305</code></td>
  </tr>
  <tr>
    <td>🔢 <b>Total Instances</b></td>
    <td><code>76,705</code> (0 empty labels)</td>
  </tr>
  <tr>
    <td>🏷️ <b>Classes</b></td>
    <td>
      <img src="https://img.shields.io/badge/knife-E74C3C?style=flat-square" />
      <img src="https://img.shields.io/badge/long__gun-3498DB?style=flat-square" />
      <img src="https://img.shields.io/badge/no__weapon-95A5A6?style=flat-square" />
      <img src="https://img.shields.io/badge/pistol-9B59B6?style=flat-square" />
    </td>
  </tr>
  <tr>
    <td>🧰 <b>Format</b></td>
    <td><code>YOLO</code> — <code>class x_center y_center width height</code> (normalized)</td>
  </tr>
  <tr>
    <td>📜 <b>License</b></td>
    <td><img src="https://img.shields.io/badge/MIT-green?style=flat-square" /></td>
  </tr>
  <tr>
    <td>☁️ <b>Hosting</b></td>
    <td>
      <a href="https://universe.roboflow.com/gundetectiondataset/nogun/dataset/2"><img src="https://img.shields.io/badge/Roboflow-NoGun_Dataset-6706CE?style=flat-square&logo=roboflow&logoColor=white" /></a>
      <a href="https://app.roboflow.com/gundetectiondataset/weapondataset-oi2g3/8"><img src="https://img.shields.io/badge/Roboflow-WeaponDataset_v8-6706CE?style=flat-square&logo=roboflow&logoColor=white" /></a>
    </td>
  </tr>
  <tr>
    <td>📦 <b>Training Results</b></td>
    <td>
      <a href="https://drive.google.com/drive/folders/1TECu5MI4lv36sJH50WSmS4iBd8SuhYgF?usp=sharing"><img src="https://img.shields.io/badge/Google_Drive-Original_Model-4285F4?style=flat-square&logo=googledrive&logoColor=white" /></a>
      <a href="https://drive.google.com/drive/folders/12aaS7CwZfGqb7__BK1UX54j1gQS_DoPi?usp=sharing"><img src="https://img.shields.io/badge/Google_Drive-Custom_Model-4285F4?style=flat-square&logo=googledrive&logoColor=white" /></a>
    </td>
  </tr>
</table>

---

<details>
<summary><b>✨ Key Features of the Dataset</b></summary>

<br>

The **NewWeaponDataset** is specifically designed for **small-object weapon detection** with the following characteristics:

- 🎯 **Multi-class Detection** — Covers 4 classes: `knife`, `pistol`, `long_gun`, and `no_weapon`
- 🎬 **Diverse Sources** — Extracted from **~1,200 YouTube videos** with varied content and scenarios
- 📐 **Resolution Variety** — Includes multiple resolutions and aspect ratios for robust training
- 🌓 **Scene Diversity** — Contains day/night footage, CCTV recordings, and handheld camera shots
- 💨 **Real-world Challenges** — Features occlusions, motion blur, and cluttered backgrounds

</details>

---

<details>
<summary><b>🏷️ Class Descriptions</b></summary>

<br>

- 🗡️ **`knife`** — Bladed weapons including knives and similar sharp objects
- 🔫 **`pistol`** — Handguns and short firearms
- 🎯 **`long_gun`** — Rifles, shotguns, and other long-barreled firearms
- 🚫 **`no_weapon`** — Hard negatives such as phones, tools, umbrellas, and camera equipment

### ❓ Why Include `no_weapon`?

The `no_weapon` class serves as **hard negatives** — visually similar objects that are frequently misclassified as weapons. Including these examples:

- ✅ **Reduces false positives** in production environments
- ✅ **Improves precision** in crowded and complex scenes
- ✅ **Teaches the model to distinguish** weapons from everyday objects like 📱 phones, 🔧 tools, ☂️ umbrellas, and 📷 camera equipment

</details>

---

## 📊 Dataset Details

<details>
<summary><b>📁 Full Dataset (100%)</b></summary>

<br>

### 🗂️ Folder Structure

<pre>
NewWeaponDataset/
├── 📂 train/                   <b>82.76%</b>  │  49,079 images  │  63,452 instances
│   ├── 🖼️ images/
│   └── 🏷️ labels/
├── 📂 valid/                   <b>12.73%</b>  │  7,552 images   │  9,730 instances
│   ├── 🖼️ images/
│   └── 🏷️ labels/
├── 📂 test/                    <b>4.51%</b>   │  2,674 images   │  3,523 instances
│   ├── 🖼️ images/
│   └── 🏷️ labels/
└── ⚙️ data.yaml
────────────────────────────────────────────────────────────────────
<b>Total:</b>                          <b>100%</b>    │  <b>59,305 images</b>  │  <b>76,705 instances</b>
</pre>

---

### 📊 Class Distribution per Split

| Split | Images | % | Instances | 🗡️ knife | 🎯 long_gun | 🚫 no_weapon | 🔫 pistol |
|-------|-------:|--:|----------:|---------:|------------:|-------------:|----------:|
| Train | 49,079 | 82.76% | 63,452 | 10,511 **(16.6%)** | 19,273 **(30.4%)** | 10,161 **(16.0%)** | 23,507 **(37.1%)** |
| Valid | 7,552 | 12.73% | 9,730 | 1,813 **(18.6%)** | 2,750 **(28.3%)** | 1,324 **(13.6%)** | 3,843 **(39.5%)** |
| Test | 2,674 | 4.51% | 3,523 | 686 **(19.5%)** | 941 **(26.7%)** | 656 **(18.6%)** | 1,240 **(35.2%)** |
| **Total** | **59,305** | **100%** | **76,705** | **13,010** | **22,964** | **12,141** | **28,590** |

---

### 📐 Size Distribution (Full Dataset)

Objects categorized by **bounding box dimensions** in pixels:

| Size | Threshold | Description |
|------|-----------|-------------|
| 🔍 **Small** | < 32px | Tiny objects, hardest to detect |
| 📦 **Medium** | 32px – 96px | Standard-sized objects |
| 🟫 **Large** | > 96px | Large, easier to detect |

| Class | Total | 🔍 Small | 📦 Medium | 🟫 Large |
|-------|------:|----------|-----------|----------|
| 🗡️ **knife** | 13,009 | 1,271 (9.8%) | 1,776 (13.7%) | 9,962 (76.6%) |
| 🎯 **long_gun** | 22,960 | 1,494 (6.5%) | 2,556 (11.1%) | 18,910 (82.4%) |
| 🚫 **other** | 12,141 | 1,220 (10.0%) | 1,852 (15.3%) | 9,069 (74.7%) |
| 🔫 **pistol** | 28,590 | 6,163 (21.6%) | 4,862 (17.0%) | 17,565 (61.4%) |
| **TOTAL** | **76,700** | **10,148 (13.2%)** | **11,046 (14.4%)** | **55,506 (72.4%)** |

> 📌 **Note:** Pistols have the highest proportion of small objects (**21.6%**), making them the most challenging class for small-object detection.

</details>

---

<details>
<summary><b>📁 Ablation Dataset (17% Subset)</b></summary>

<br>

A **17% stratified subset** of the full dataset was used for **grid search experiments** to enable faster iteration while maintaining representative class distributions.

### 📊 Ablation Dataset Split

| Split | Images | Instances | 🗡️ knife | 🎯 long_gun | 🚫 other | 🔫 pistol |
|-------|--------|-----------|----------|-------------|----------|-----------|
| **Train** | 8,343 (82.8%) | 10,927 (83.0%) | 1,779 (16.3%) | 3,315 (30.3%) | 1,815 (16.6%) | 4,018 (36.8%) |
| **Valid** | 1,283 (12.7%) | 1,643 (12.5%) | 307 (18.7%) | 493 (30.0%) | 206 (12.5%) | 637 (38.8%) |
| **Test** | 454 (4.5%) | 598 (4.5%) | 121 (20.2%) | 156 (26.1%) | 114 (19.1%) | 207 (34.6%) |
| **TOTAL** | **10,080** | **13,168** | 2,207 (16.8%) | 3,964 (30.1%) | 2,135 (16.2%) | 4,862 (36.9%) |

---

### 📐 Size Distribution (Ablation Dataset)

| Class | Total | 🔍 Small | 📦 Medium | 🟫 Large |
|-------|------:|----------|-----------|----------|
| 🗡️ **knife** | 2,207 | 239 (10.8%) | 284 (12.9%) | 1,684 (76.3%) |
| 🎯 **long_gun** | 3,964 | 264 (6.7%) | 412 (10.4%) | 3,288 (82.9%) |
| 🚫 **other** | 2,135 | 235 (11.0%) | 359 (16.8%) | 1,541 (72.2%) |
| 🔫 **pistol** | 4,862 | 1,053 (21.7%) | 826 (17.0%) | 2,983 (61.4%) |
| **TOTAL** | **13,168** | **1,791 (13.6%)** | **1,881 (14.3%)** | **9,496 (72.1%)** |

> 📌 Class proportions are **consistent with the full dataset**, ensuring ablation results transfer effectively.

</details>

---

<details>
<summary><b>🖼️ Dataset Examples</b></summary>

<br>

### 📸 Sample Annotations

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://github.com/user-attachments/assets/af5b7f23-f809-404d-8da0-17ddc436f894" alt="Example 1" width="100%" />
      <br><sub>Example 1</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://github.com/user-attachments/assets/b0805ccc-09e6-4167-abcb-35a583532762" alt="Example 2" width="100%" />
      <br><sub>Example 2</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://github.com/user-attachments/assets/dbab2e81-8b7b-45b4-be47-50eb64689d9d" alt="Example 3" width="100%" />
      <br><sub>Example 3</sub>
    </td>
  </tr>
</table>

---

### 🔳 Size Comparison — Small, Medium, Large

<p align="center">
  <img src="https://github.com/user-attachments/assets/2f4dd189-1ba0-4340-a101-2c55e42b27e5" alt="mosaic_small" width="30%">
  <img src="https://github.com/user-attachments/assets/727e3de5-209c-4698-a6f0-bf717d19d22a" alt="mosaic_medium" width="30%">
  <img src="https://github.com/user-attachments/assets/8f45f604-baac-4d59-bfcf-20da3475eed3" alt="mosaic_large" width="30%">
</p>
<p align="center"><sub>🔍 Small Objects | 📦 Medium Objects | 🟫 Large Objects</sub></p>

---

### 📊 Class Distribution Visualization

<p align="center">
  <img src="https://github.com/user-attachments/assets/5b18d380-a33e-47e3-aadd-eb3769f93447" alt="class_distribution" width="80%">
</p>

</details>

---

## 🔬 Ablation Study Results

<details>
<summary><b>📊 Click to expand Hyperparameter Search Results</b></summary>

<br>

### ⚙️ Experimental Setup

| Setting | Value |
|---------|-------|
| 📊 **Dataset Size** | **17%** of the full dataset (**10,080 images** / **13,168 instances**) |
| 🔄 **Epochs per Run** | **70 epochs** |
| 📦 **Batch Size** | **64** |
| 🖼️ **Image Size** | **640×640** |
| ⏱️ **Time per Run (YOLOv12s)** | ~1.2 hours |
| ⏱️ **Time per Run (YOLO11s)** | ~1.0 hours |
| 🔬 **Methodology** | Grid search with **isolated phases** |

---

### 🧪 YOLOv12s Grid Search Experiments

We conducted a **comprehensive ablation study** across **180 valid experiments** to identify optimal hyperparameters for small-object weapon detection.

<table>
  <tr>
    <th align="left">🧪 Experiment</th>
    <th align="left">⚙️ Parameter</th>
    <th align="left">🔢 Values Tested</th>
    <th align="center">📈 Combinations<br><sub>(Valid / Invalid / Total)</sub></th>
    <th align="center">⏱️ Time<br><sub>(Per Run / Total)</sub></th>
    <th align="left">✅ Optimal</th>
  </tr>
  <tr>
    <td rowspan="3"><b>1. Alpha Scheduling</b></td>
    <td><code>α_start</code></td>
    <td><sub>0.5, 0.6, 0.7, 0.8, 0.9, 1.0</sub></td>
    <td rowspan="3" align="center"><b>30</b> / 6 / 36<br><sub>❌ Invalid: end &lt; start</sub></td>
    <td rowspan="3" align="center"><sub>~1.2h / ~36h</sub></td>
    <td><code>0.9</code></td>
  </tr>
  <tr>
    <td><code>α_end</code></td>
    <td><sub>0.3, 0.4, 0.5, 0.6, 0.7, 0.8</sub></td>
    <td><code>0.4</code></td>
  </tr>
  <tr>
    <td><code>small_obj_px</code></td>
    <td><sub>32</sub></td>
    <td><code>32</code></td>
  </tr>
  <tr>
    <td rowspan="2"><b>2. Center Loss Weight</b></td>
    <td><code>Loss_min</code></td>
    <td><sub>0.00, 0.005, 0.010, 0.015, 0.020, 0.025</sub></td>
    <td rowspan="2" align="center"><b>32</b> / 4 / 36<br><sub>❌ Invalid: init &lt; min</sub></td>
    <td rowspan="2" align="center"><sub>~1.2h / ~38h</sub></td>
    <td><code>0.01</code></td>
  </tr>
  <tr>
    <td><code>Loss_init</code></td>
    <td><sub>0.01, 0.02, 0.03, 0.05, 0.07, 0.10</sub></td>
    <td><code>0.05</code></td>
  </tr>
  <tr>
    <td rowspan="2"><b>3. IoU Clipping</b></td>
    <td><code>IoU_start</code></td>
    <td><sub>6, 8, 10, 12, 15, 20</sub></td>
    <td rowspan="2" align="center"><b>35</b> / 1 / 36<br><sub>❌ Invalid: end &lt; start</sub></td>
    <td rowspan="2" align="center"><sub>~1.2h / ~42h</sub></td>
    <td><code>6</code></td>
  </tr>
  <tr>
    <td><code>IoU_end</code></td>
    <td><sub>2, 3, 4, 5, 6, 8</sub></td>
    <td><code>2</code></td>
  </tr>
  <tr>
    <td rowspan="2"><b>4. DFL Clipping</b></td>
    <td><code>DFL_start</code></td>
    <td><sub>6, 8, 10, 12, 15, 20</sub></td>
    <td rowspan="2" align="center"><b>35</b> / 1 / 36<br><sub>❌ Invalid: end &lt; start</sub></td>
    <td rowspan="2" align="center"><sub>~1.2h / ~42h</sub></td>
    <td><code>8</code></td>
  </tr>
  <tr>
    <td><code>DFL_end</code></td>
    <td><sub>2, 3, 4, 5, 6, 8</sub></td>
    <td><code>5</code></td>
  </tr>
  <tr>
    <td rowspan="3"><b>5. TAL Alpha-Beta</b></td>
    <td><code>Alpha (α)</code></td>
    <td><sub>0.25, 0.4, 0.5, 0.6, 0.75, 1.0</sub></td>
    <td rowspan="3" align="center"><b>48</b> / 0 / 48<br><sub>✅ All valid</sub></td>
    <td rowspan="3" align="center"><sub>~1.2h / ~58h</sub></td>
    <td><code>1</code></td>
  </tr>
  <tr>
    <td><code>Beta</code></td>
    <td><sub>4, 5, 6, 7, 8, 10</sub></td>
    <td><code>7</code></td>
  </tr>
  <tr>
    <td><code>Topk</code></td>
    <td><sub>4, 5, 6, 8, 10, 12, 15, 20, 25</sub></td>
    <td><code>20</code></td>
  </tr>
  <tr>
    <td><b>📊 Grid Search Total</b></td>
    <td align="center">—</td>
    <td align="center">—</td>
    <td align="center"><b>180 / 12 / 192</b></td>
    <td align="center"><b>~1.2h / ~216h</b><br><sub>(~9 days)</sub></td>
    <td align="center">—</td>
  </tr>
</table>

---

### 🔗 YOLOv12s Phase Combination Experiments

After identifying the **optimal parameters** for each individual phase, we conducted **26 additional experiments** to find the **best combination**:

| Phases | Combinations Tested |
|--------|---------------------|
| **2 phases** | A+B, A+C1, A+C2, A+D, B+C1, B+C2, B+D, C1+C2, C1+D, C2+D |
| **3 phases** | A+B+C1, A+B+C2, A+B+D, A+C1+C2, A+C1+D, A+C2+D, B+C1+C2, B+C1+D, B+C2+D, C1+C2+D |
| **4 phases** | A+B+C1+C2, **A+B+C1+D 🏆**, A+B+C2+D, A+C1+C2+D, B+C1+C2+D |
| **5 phases** | A+B+C1+C2+D |

<sub>**Legend:** A = Alpha Scheduling, B = Center Loss, C1 = IoU Clipping, C2 = DFL Clipping, D = TAL Alpha-Beta</sub>

> 🏆 **Winner:** Combination **A + B + C1 + D** achieved the best overall performance on YOLOv12s.

<details>
<summary><b>📊 Click to view YOLOv12s Combination Results</b></summary>

<br>

### 📈 All Combinations Performance Comparison

<p align="center">
  <img src="https://github.com/user-attachments/assets/e19006d3-229f-4b72-b2c5-5c89ab1abbc8" alt="Combination Results" width="100%" />
</p>

---

### 🏆 Winner (A + B + C1 + D) — Full Dataset Training Results

After identifying the best combination, we trained on the **full dataset (100%)** for **~11 hours**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/6c82aa4b-3533-41a9-918a-e4f48cbcd7eb" alt="Winner Full Dataset Results" width="100%" />
</p>

</details>

---

### 🚀 YOLO11s Transfer Experiments

After obtaining the optimal configuration on **YOLOv12s**, we transferred the best-performing hyperparameters to **YOLO11s** to validate cross-architecture generalization.

> ⚠️ **Note:** Phase C2 (DFL Clipping) is **not applicable** to YOLO11s architecture and was excluded from all experiments.

### 🧪 YOLO11s Individual Phase Validation

First, we validated each phase **individually** on YOLO11s using the optimal parameters from YOLOv12s:

| # | Phase | Description | Experiments |
|---|-------|-------------|-------------|
| 1 | A | Alpha Scheduling only | 1 |
| 2 | B | Center Loss only | 1 |
| 3 | C1 | IoU Clipping only | 1 |
| 4 | D | TAL Alpha-Beta only | 1 |

### 🔗 YOLO11s Combination Experiments

After validating individual phases, we tested the following **11 combinations**:

| Phases | Combinations Tested |
|--------|---------------------|
| **2 phases** | A+B, A+C1, A+D, B+C1, B+D, C1+D |
| **3 phases** | **A+B+C1 🏆**, A+B+D, A+C1+D, B+C1+D |
| **4 phases** | A+B+C1+D |

<sub>**Legend:** A = Alpha Scheduling, B = Center Loss, C1 = IoU Clipping, D = TAL Alpha-Beta</sub>

> 🏆 **Winner:** Combination **A + B + C1** achieved the best performance on YOLO11s, confirming successful cross-architecture transfer.

<details>
<summary><b>📊 Click to view YOLO11s Combination Results</b></summary>

<br>

### 📈 All Combinations Performance Comparison

<p align="center">
  <img src="https://github.com/user-attachments/assets/2d6045d6-c412-4a11-bd78-8f41b95fb52e" alt="YOLO11s Combination Results" width="100%" />
</p>

---

### 🏆 Winner (A + B + C1) — Full Dataset Training Results

After identifying the best combination, we trained on the **full dataset (100%)** for **~9 hours**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/83772459-f208-4c6e-99a0-da55c9d1beef" alt="YOLO11s Winner Full Dataset Results" width="100%" />
</p>

</details>

---

### ✅ Final Optimal Configurations

<table>
  <tr>
    <th align="center">🔷 YOLOv12s Best Config</th>
    <th align="center">🔶 YOLO11s Best Config</th>
  </tr>
  <tr>
    <td><b>A + B + C1 + D</b></td>
    <td><b>A + B + C1</b></td>
  </tr>
  <tr>
    <td>Alpha + Center Loss + IoU Clipping + TAL</td>
    <td>Alpha + Center Loss + IoU Clipping</td>
  </tr>
</table>

<pre>
# ═══════════════════════════════════════════════════════════════
# 🏆 Best Configuration: A + B + C1 (+ D for YOLOv12s)
# ═══════════════════════════════════════════════════════════════

# Phase A: Alpha Scheduling ✅
alpha_start: 0.9
alpha_end: 0.4
small_obj_px: 32

# Phase B: Center Loss ✅
center_loss_weight_init: 0.05
center_loss_weight_min: 0.01

# Phase C1: IoU Clipping ✅
iou_clip_start: 6
iou_clip_end: 2

# Phase C2: DFL Clipping ❌ (Disabled / Not applicable to YOLO11s)
dfl_clip_start: 100.0
dfl_clip_end: 100.0

# Phase D: TAL Alpha-Beta ✅ (YOLOv12s) / Default (YOLO11s)
tal_topk: 20        # YOLOv12s: 20, YOLO11s: 10 (default)
tal_alpha: 1        # YOLOv12s: 1, YOLO11s: 0.5 (default)
tal_beta: 7         # YOLOv12s: 7, YOLO11s: 6.0 (default)
</pre>

---

### 📊 Complete Training Summary

| Model | Phase | Dataset | Experiments | Time per Run | Total Time |
|-------|-------|---------|-------------|--------------|------------|
| **YOLOv12s** | Baseline | **100%** | 1 | — | **~11h** |
| **YOLOv12s** | Grid Search | 17% | 180 | ~1.2h | ~216h |
| **YOLOv12s** | Combinations | 17% | 26 | ~1.2h | ~31h |
| **YOLOv12s** | 🏆 Final Training | **100%** | 1 | — | **~11h** |
| **YOLO11s** | Baseline | **100%** | 1 | — | **~9h** |
| **YOLO11s** | Individual Phases | 17% | 4 | ~1.0h | ~4h |
| **YOLO11s** | Combinations | 17% | 11 | ~1.0h | ~11h |
| **YOLO11s** | 🏆 Final Training | **100%** | 1 | — | **~9h** |

---

### 📈 Total Training Investment

| Metric | Value |
|--------|-------|
| 🧪 **Total Experiments** | **225** (2 baselines + 180 grid + 26 combos + 4 phases + 11 combos + 2 final) |
| ⏱️ **Baseline Training** | ~20 hours (11h + 9h) |
| ⏱️ **Ablation Time (17% dataset)** | ~262 hours (~10.9 days) |
| ⏱️ **Full Dataset Training (Custom)** | ~20 hours (11h + 9h) |
| ⏱️ **Grand Total** | **~302 hours (~12.6 days)** |

---

### 📌 Key Takeaways

- ⏱️ **Total Training Time:** ~302 hours (**~12.6 days**) across **225 experiments**
  - 🔷 **YOLOv12s:** ~269 hours (1 baseline + 180 grid search + 26 combinations + 1 final)
  - 🔶 **YOLO11s:** ~33 hours (1 baseline + 4 individual phases + 11 combinations + 1 final)
- ✅ **Valid Configurations:** 180 out of 192 planned (**93.75% valid rate**)
- ❌ **Skipped Configurations:** 12 invalid combinations where `end < start` or `init < min`
- 🏆 **YOLOv12s Best:** **A + B + C1 + D** (Alpha + Center Loss + IoU Clipping + TAL)
- 🏆 **YOLO11s Best:** **A + B + C1** (Alpha + Center Loss + IoU Clipping)
- 🔄 **Cross-Architecture Transfer:** Optimal config from YOLOv12s **successfully transferred** to YOLO11s
- ⚡ **YOLO11s Efficiency:** ~17% faster per experiment compared to YOLOv12s (~1.0h vs ~1.2h)
- 🚫 **DFL Clipping (C2):** Not applicable to YOLO11s, excluded from transfer experiments
- 🎯 **Most Impactful:** TAL Alpha-Beta tuning showed significant impact on small-object recall (YOLOv12s)
- 📉 **Alpha Scheduling:** Annealing from `α_start=0.9` to `α_end=0.4` prioritizes small objects early
- 🔧 **Loss Clipping:** IoU clipping stabilizes training (DFL clipping provided diminishing returns)
- 📈 **Best Improvements:** Up to **+14.55%** on small objects (mAP50-95) for YOLO11s, **+13.10%** for YOLOv12s

---

### 📄 Reproducibility Config

To reproduce the ablation experiments, use the following default configuration:

<pre>
# ═══════════════════════════════════════════════════════════════
# 🔬 Ablation Study - Default Configuration
# ═══════════════════════════════════════════════════════════════

# Training Settings
epochs: 70
batch: 64
imgsz: 640
dataset: "weapon_dataset_17pct"  # 17% subset for grid search

# ───────────────────────────────────────────────────────────────
# Phase A: Alpha Scheduling (DISABLED)
# ───────────────────────────────────────────────────────────────
alpha_start: 1.0          # No area weighting
alpha_end: 1.0            # No area weighting
small_obj_boost: 1.0      # No boost (multiplier = 1)
small_obj_px: 32          # Default small object threshold

# ───────────────────────────────────────────────────────────────
# Phase B: Center Loss (DISABLED)
# ───────────────────────────────────────────────────────────────
center_loss_weight_init: 0.0    # No center loss
center_loss_weight_min: 0.0     # No center loss
center_loss_decay_epochs: 1     # Irrelevant when weight=0

# ───────────────────────────────────────────────────────────────
# Phase C: Adaptive Clipping (DISABLED)
# ───────────────────────────────────────────────────────────────
iou_clip_start: 100.0     # No clipping (very high)
iou_clip_end: 100.0       # No clipping (very high)
dfl_clip_start: 100.0     # No clipping (very high) — N/A for YOLO11s
dfl_clip_end: 100.0       # No clipping (very high) — N/A for YOLO11s

# ───────────────────────────────────────────────────────────────
# Phase D: TAL Alpha-Beta (DEFAULT)
# ───────────────────────────────────────────────────────────────
tal_topk: 10
tal_alpha: 0.5
tal_beta: 6.0
</pre>

> 💡 **Note:** Enable one phase at a time while keeping others at their disabled/default values to isolate the effect of each hyperparameter.

> ⚠️ **Important:** Parameters from **Phase A** (Alpha Scheduling), **Phase B** (Center Loss), and **Phase C** (Adaptive Clipping) are **only available in our custom loss function implementation**. Phase D (TAL Alpha-Beta) uses the standard Ultralytics parameters.

</details>

