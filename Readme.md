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

## 🏆 Research Highlights

<table>
  <tr>
    <th align="center">🧪 Total Experiments</th>
    <th align="center">⏱️ Total Training Time</th>
  </tr>
  <tr>
    <td align="center"><code>247</code></td>
    <td align="center"><code>~406 hours (~16.9 days)</code></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center" width="45%">
      <img width="100%" alt="Custom YOLOv12s Architecture (Arch-6)" src="https://github.com/user-attachments/assets/ba924428-fd9d-4064-9026-5842fad717d6" />
      <br><sub>🏗️ Custom YOLOv12s Architecture (Arch-6 ★)<br>5-head detection with auxiliary P2 branch at stride 4</sub>
    </td>
    <td align="center" width="55%">
      <img width="100%" alt="YOLOv12s Ablation Study Training Metrics" src="https://github.com/user-attachments/assets/0bf3f8f9-70cc-4d75-8f33-11362a91d23d" />
      <br><sub>📈 Training metrics — Baseline vs Custom Loss + Arch</sub>
    </td>
  </tr>
</table>



<table>
  <tr>
    <th align="center" colspan="5">📈 Performance Improvements</th>
  </tr>
  <tr>
    <th align="center">Metric</th>
    <th align="center">🔷 YOLOv12s<br><sub>Baseline → Custom Loss</sub></th>
    <th align="center">🧪 YOLOv12s<br><sub>Baseline → Custom Loss+Arch</sub></th>
    <th align="center">🔶 YOLO26s<br><sub>Baseline → Custom Loss</sub></th>
    <th align="center">⚔️ Custom Loss<br><sub>YOLO26s → YOLOv12s</sub></th>
  </tr>

  <tr>
    <td align="left"><b>mAP50</b></td>
    <td align="center">0.816 → <b>0.857</b> <sub>(+5.04%)</sub></td>
    <td align="center">0.816 → <b>0.865</b> <sub>(+6.00%)</sub></td>
    <td align="center">0.782 → <b>0.828</b> <sub>(+5.88%)</sub></td>
    <td align="center">0.828 → <b>0.857</b> <sub>(+3.43%)</sub></td>
  </tr>

  <tr>
    <td align="left"><b>mAP50-95</b></td>
    <td align="center">0.525 → <b>0.574</b> <sub>(+9.32%)</sub></td>
    <td align="center">0.525 → <b>0.585</b> <sub>(+11.43%)</sub></td>
    <td align="center">0.502 → <b>0.555</b> <sub>(+10.53%)</sub></td>
    <td align="center">0.555 → <b>0.574</b> <sub>(+3.29%)</sub></td>
  </tr>

  <tr>
    <td align="left"><b>Precision</b></td>
    <td align="center">0.831 → <b>0.889</b> <sub>(+6.94%)</sub></td>
    <td align="center">0.831 → <b>0.902</b> <sub>(+8.54%)</sub></td>
    <td align="center">0.792 → <b>0.863</b> <sub>(+8.94%)</sub></td>
    <td align="center">0.863 → <b>0.889</b> <sub>(+2.98%)</sub></td>
  </tr>

  <tr>
    <td align="left"><b>Recall</b></td>
    <td align="center">0.746 → <b>0.811</b> <sub>(+8.79%)</sub></td>
    <td align="center">0.746 → <b>0.825</b> <sub>(+10.59%)</sub></td>
    <td align="center">0.698 → <b>0.776</b> <sub>(+11.14%)</sub></td>
    <td align="center">0.776 → <b>0.811</b> <sub>(+4.59%)</sub></td>
  </tr>

  <tr>
    <td align="left"><b>F1 Score</b></td>
    <td align="center">0.786 → <b>0.848</b> <sub>(+7.91%)</sub></td>
    <td align="center">0.786 → <b>0.862</b> <sub>(+9.67%)</sub></td>
    <td align="center">0.742 → <b>0.817</b> <sub>(+10.10%)</sub></td>
    <td align="center">0.817 → <b>0.848</b> <sub>(+3.82%)</sub></td>
  </tr>

  <tr>
    <td colspan="5" align="center"><b>🔍 Size-Specific mAP50</b></td>
  </tr>

  <tr>
    <td align="left">🔍 <b>Small</b></td>
    <td align="center">0.530 → <b>0.574</b> <sub>(+8.22%)</sub></td>
    <td align="center">0.530 → <b>0.590</b> <sub>(+11.32%)</sub></td>
    <td align="center">0.498 → <b>0.542</b> <sub>(+8.98%)</sub></td>
    <td align="center">0.542 → <b>0.574</b> <sub>(+5.76%)</sub></td>
  </tr>

  <tr>
    <td align="left">📦 <b>Medium</b></td>
    <td align="center">0.750 → <b>0.796</b> <sub>(+6.11%)</sub></td>
    <td align="center">0.750 → <b>0.812</b> <sub>(+8.27%)</sub></td>
    <td align="center">0.718 → <b>0.765</b> <sub>(+6.51%)</sub></td>
    <td align="center">0.765 → <b>0.796</b> <sub>(+4.06%)</sub></td>
  </tr>

  <tr>
    <td align="left">🟫 <b>Large</b></td>
    <td align="center">0.828 → <b>0.871</b> <sub>(+5.24%)</sub></td>
    <td align="center">0.828 → <b>0.885</b> <sub>(+6.88%)</sub></td>
    <td align="center">0.796 → <b>0.842</b> <sub>(+5.76%)</sub></td>
    <td align="center">0.842 → <b>0.871</b> <sub>(+3.54%)</sub></td>
  </tr>

  <tr>
    <td colspan="5" align="center"><b>🔍 Size-Specific mAP50-95</b></td>
  </tr>

  <tr>
    <td align="left">🔍 <b>Small</b></td>
    <td align="center">0.297 → <b>0.336</b> <sub>(+13.10%)</sub></td>
    <td align="center">0.297 → <b>0.355</b> <sub>(+19.53%)</sub></td>
    <td align="center">0.276 → <b>0.316</b> <sub>(+14.55%)</sub></td>
    <td align="center">0.316 → <b>0.336</b> <sub>(+6.32%)</sub></td>
  </tr>

  <tr>
    <td align="left">📦 <b>Medium</b></td>
    <td align="center">0.424 → <b>0.467</b> <sub>(+10.23%)</sub></td>
    <td align="center">0.424 → <b>0.485</b> <sub>(+14.38%)</sub></td>
    <td align="center">0.402 → <b>0.448</b> <sub>(+11.30%)</sub></td>
    <td align="center">0.448 → <b>0.467</b> <sub>(+4.24%)</sub></td>
  </tr>

  <tr>
    <td align="left">🟫 <b>Large</b></td>
    <td align="center">0.550 → <b>0.604</b> <sub>(+9.74%)</sub></td>
    <td align="center">0.550 → <b>0.628</b> <sub>(+14.18%)</sub></td>
    <td align="center">0.524 → <b>0.584</b> <sub>(+11.59%)</sub></td>
    <td align="center">0.584 → <b>0.604</b> <sub>(+3.31%)</sub></td>
  </tr>

</table>

<sub>🔍 <b>Key Findings:</b> Custom loss achieves up to <b>+14.55%</b> improvement on small objects | YOLOv12s outperforms YOLO26s by <b>+6.32%</b> on small objects (mAP50-95)</sub>

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

## 🔬 Loss Function Ablation Study Results (YOLOv12s/YOLO26s)

<details>
<summary><b>⚙️ 1. Experimental Setup</b></summary>

<br>

| Setting | Value |
|---------|-------|
| 📊 **Dataset Size** | **17%** of the full dataset (**10,080 images** / **13,168 instances**) |
| 🔄 **Epochs per Run** | **70 epochs** |
| 📦 **Batch Size** | **64** |
| 🖼️ **Image Size** | **640×640** |
| ⏱️ **Time per Run (YOLOv12s)** | ~1.2 hours |
| ⏱️ **Time per Run (YOLO26s)** | ~1.0 hours |
| 🔬 **Methodology** | Grid search with **isolated phases** |

</details>

---

<details>
<summary><b>🧪 2. YOLOv12s Grid Search Experiments</b></summary>

<br>

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
    <td rowspan="3"><b>A1. Alpha Scheduling</b></td>
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
    <td rowspan="2"><b>A2. Center Loss Weight</b></td>
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
    <td rowspan="2"><b>A3.1. IoU Clipping</b></td>
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
    <td rowspan="2"><b>A3.2. DFL Clipping</b></td>
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
    <td rowspan="3"><b>A4. TAL Alpha-Beta</b></td>
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

</details>

---

<details>
<summary><b>🔗 3. YOLOv12s Phase Combination Experiments</b></summary>

<br>

After identifying the **optimal parameters** for each individual phase, we conducted **26 additional experiments** to find the **best combination**:

| Phases | Combinations Tested |
|--------|---------------------|
| **2 phases** | A1+A2, A1+A3.1, A1+A3.2, A1+A4, A2+A3.1, A2+A3.2, A2+A4, A3.1+A3.2, A3.1+A4, A3.2+A4 |
| **3 phases** | A1+A2+A3.1, A1+A2+A3.2, A1+A2+A4, A1+A3.1+A3.2, A1+A3.1+A4, A1+A3.2+A4, A2+A3.1+A3.2, A2+A3.1+A4, A2+A3.2+A4, A3.1+A3.2+A4 |
| **4 phases** | A1+A2+A3.1+A3.2, **A1+A2+A3.1+A4 🏆**, A1+A2+A3.2+A4, A1+A3.1+A3.2+A4, A2+A3.1+A3.2+A4 |
| **5 phases** | A1+A2+A3.1+A3.2+A4 |

<sub>**Legend:** A1 = Alpha Scheduling, A2 = Center Loss, A3.1 = IoU Clipping, A3.2 = DFL Clipping, A4 = TAL Alpha-Beta</sub>

> 🏆 **Winner:** Combination **A1 + A2 + A3.1 + A4** achieved the best overall performance on YOLOv12s.

<details>
<summary><b>📊 Click to view YOLOv12s Combination Results</b></summary>

<br>

### 📈 All Combinations Performance Comparison

<p align="center">
<img width="1426" height="1073" alt="image" src="https://github.com/user-attachments/assets/e73654bd-d49d-4c58-96c2-12e2e3024641" />
</p>

---

### 🏆 Winner (A1 + A2 + A3.1 + A4) — Full Dataset Training Results

After identifying the best combination, we trained on the **full dataset (100%)** for **~11 hours**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/6c82aa4b-3533-41a9-918a-e4f48cbcd7eb" alt="Winner Full Dataset Results" width="100%" />
</p>

Training results for **100** epochs on the full dataset.

<p align="center">
    <img width="2000" height="977" alt="image" src="https://github.com/user-attachments/assets/fcaeafe8-87e6-4576-85ea-e0f1efe5e136" />
</p>

</details>

</details>

---

<details>
<summary><b>🚀 4. YOLO26s Transfer Experiments</b></summary>

<br>

After obtaining the optimal configuration on **YOLOv12s**, we transferred the best-performing hyperparameters to **YOLO26s** to validate cross-architecture generalization.

> ⚠️ **Note:** Phase A3.2 (DFL Clipping) is **not applicable** to YOLO26s architecture and was excluded from all experiments.

### 🧪 YOLO26s Individual Phase Validation

First, we validated each phase **individually** on YOLO26s using the optimal parameters from YOLOv12s:

| # | Phase | Description | Experiments |
|---|-------|-------------|-------------|
| 1 | A1 | Alpha Scheduling only | 1 |
| 2 | A2 | Center Loss only | 1 |
| 3 | A3.1 | IoU Clipping only | 1 |
| 4 | A4 | TAL Alpha-Beta only | 1 |

### 🔗 YOLO26s Combination Experiments

After validating individual phases, we tested the following **11 combinations**:

| Phases | Combinations Tested |
|--------|---------------------|
| **2 phases** | A1+A2, A1+A3.1, A1+A4, A2+A3.1, A2+A4, A3.1+A4 |
| **3 phases** | **A1+A2+A3.1 🏆**, A1+A2+A4, A1+A3.1+A4, A2+A3.1+A4 |
| **4 phases** | A1+A2+A3.1+A4 |

<sub>**Legend:** A1 = Alpha Scheduling, A2 = Center Loss, A3.1 = IoU Clipping, A4 = TAL Alpha-Beta</sub>

> 🏆 **Winner:** Combination **A1 + A2 + A3.1** achieved the best performance on YOLO26s, confirming successful cross-architecture transfer.

<details>
<summary><b>📊 Click to view YOLO26s Combination Results</b></summary>

<br>

### 📈 All Combinations Performance Comparison

<p align="center">
<img width="1426" height="1073" alt="image" src="https://github.com/user-attachments/assets/5d1e0b60-5dea-404d-b532-0c35585e2b07" />
</p>

---

### 🏆 Winner (A1 + A2 + A3.1) — Full Dataset Training Results

After identifying the best combination, we trained on the **full dataset (100%)** for **~9 hours**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/83772459-f208-4c6e-99a0-da55c9d1beef" alt="YOLO26s Winner Full Dataset Results" width="100%" />
</p>

Training results for **100** epochs on the full dataset.

<p align="center">
  <img width="2000" height="977" alt="image" src="https://github.com/user-attachments/assets/63d94a6a-b205-4988-b0c6-7bffa7222de9" />
</p>

</details>

</details>

---

<details>
<summary><b>✅ 5. Final Optimal Configurations</b></summary>

<br>

<table>
  <tr>
    <th align="center">🔷 YOLOv12s Best Config</th>
    <th align="center">🔶 YOLO26s Best Config</th>
  </tr>
  <tr>
    <td><b>A1 + A2 + A3.1 + A4</b></td>
    <td><b>A1 + A2 + A3.1</b></td>
  </tr>
  <tr>
    <td>Alpha + Center Loss + IoU Clipping + TAL</td>
    <td>Alpha + Center Loss + IoU Clipping</td>
  </tr>
</table>

<pre>
# ═══════════════════════════════════════════════════════════════
# 🏆 Best Configuration: A1 + A2 + A3.1 (+ A4 for YOLOv12s)
# ═══════════════════════════════════════════════════════════════

# Phase A1: Alpha Scheduling ✅
alpha_start: 0.9
alpha_end: 0.4
small_obj_px: 32

# Phase A2: Center Loss ✅
center_loss_weight_init: 0.05
center_loss_weight_min: 0.01

# Phase A3.1: IoU Clipping ✅
iou_clip_start: 6
iou_clip_end: 2

# Phase A3.2: DFL Clipping ❌ (Disabled / Not applicable to YOLO26s)
dfl_clip_start: 100.0
dfl_clip_end: 100.0

# Phase A4: TAL Alpha-Beta ✅ (YOLOv12s) / Default (YOLO26s)
tal_topk: 20        # YOLOv12s: 20, YOLO26s: 10 (default)
tal_alpha: 1        # YOLOv12s: 1, YOLO26s: 0.5 (default)
tal_beta: 7         # YOLOv12s: 7, YOLO26s: 6.0 (default)
</pre>

</details>

---

<details>
<summary><b>📊 6. Complete Training Summary</b></summary>

<br>

| Model | Phase | Dataset | Experiments | Time per Run | Total Time |
|-------|-------|---------|-------------|--------------|------------|
| **YOLOv12s** | Baseline | **100%** | 1 | — | **~11h** |
| **YOLOv12s** | Grid Search | 17% | 180 | ~1.2h | ~216h |
| **YOLOv12s** | Combinations | 17% | 26 | ~1.2h | ~31h |
| **YOLOv12s** | 🏆 Final Training | **100%** | 1 | — | **~11h** |
| **YOLO26s** | Baseline | **100%** | 1 | — | **~9h** |
| **YOLO26s** | Individual Phases | 17% | 4 | ~1.0h | ~4h |
| **YOLO26s** | Combinations | 17% | 11 | ~1.0h | ~11h |
| **YOLO26s** | 🏆 Final Training | **100%** | 1 | — | **~9h** |

---

### 📈 Total Training Investment

| Metric | Value |
|--------|-------|
| 🧪 **Total Experiments** | **225** (2 baselines + 180 grid + 26 combos + 4 phases + 11 combos + 2 final) |
| ⏱️ **Baseline Training** | ~20 hours (11h + 9h) |
| ⏱️ **Ablation Time (17% dataset)** | ~262 hours (~10.9 days) |
| ⏱️ **Full Dataset Training (Custom)** | ~20 hours (11h + 9h) |
| ⏱️ **Grand Total** | **~302 hours (~12.6 days)** |

</details>

---

<details>
<summary><b>📌 7. Key Takeaways</b></summary>

<br>

- ⏱️ **Total Training Time:** ~302 hours (**~12.6 days**) across **225 experiments**
  - 🔷 **YOLOv12s:** ~269 hours (1 baseline + 180 grid search + 26 combinations + 1 final)
  - 🔶 **YOLO26s:** ~33 hours (1 baseline + 4 individual phases + 11 combinations + 1 final)
- ✅ **Valid Configurations:** 180 out of 192 planned (**93.75% valid rate**)
- ❌ **Skipped Configurations:** 12 invalid combinations where `end < start` or `init < min`
- 🏆 **YOLOv12s Best:** **A1 + A2 + A3.1 + A4** (Alpha + Center Loss + IoU Clipping + TAL)
- 🏆 **YOLO26s Best:** **A1 + A2 + A3.1** (Alpha + Center Loss + IoU Clipping)
- 🔄 **Cross-Architecture Transfer:** Optimal config from YOLOv12s **successfully transferred** to YOLO26s
- ⚡ **YOLO26s Efficiency:** ~17% faster per experiment compared to YOLOv12s (~1.0h vs ~1.2h)
- 🚫 **DFL Clipping (A3.2):** Not applicable to YOLO26s, excluded from transfer experiments
- 🎯 **Most Impactful:** TAL Alpha-Beta tuning showed significant impact on small-object recall (YOLOv12s)
- 📉 **Alpha Scheduling:** Annealing from `α_start=0.9` to `α_end=0.4` prioritizes small objects early
- 🔧 **Loss Clipping:** IoU clipping stabilizes training (DFL clipping provided diminishing returns)
- 📈 **Best Improvements:** Up to **+14.55%** on small objects (mAP50-95) for YOLO26s, **+13.10%** for YOLOv12s

</details>

---

<details>
<summary><b>📄 8. Training Config (Reproducibility)</b></summary>

<br>

### 🔬 Full Training Configuration

The complete YAML configuration used for all experiments, including default Ultralytics parameters and our custom loss additions.

<details>
<summary><b>📋 Click to expand full default YAML (all parameters)</b></summary>

<br>

<pre>
# ═══════════════════════════════════════════════════════════════
# 📋 Complete Training Configuration
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# Core Settings
# ───────────────────────────────────────────────────────────────
task: detect
mode: train
model: yolov12.yaml                    # Architecture definition
data: data.yaml                        # Dataset configuration
pretrained: yolov12s.pt                # Pre-trained weights

# ───────────────────────────────────────────────────────────────
# Training Schedule
# ───────────────────────────────────────────────────────────────
epochs: 120
time: null
patience: 20
batch: 44
imgsz: 640
close_mosaic: 15

# ───────────────────────────────────────────────────────────────
# Optimizer
# ───────────────────────────────────────────────────────────────
optimizer: auto
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.0
cos_lr: false
nbs: 64

# ───────────────────────────────────────────────────────────────
# Loss Weights (Ultralytics Defaults)
# ───────────────────────────────────────────────────────────────
box: 7.5
cls: 0.5
dfl: 1.5
pose: 12.0
kobj: 1.0

# ───────────────────────────────────────────────────────────────
# Data Augmentation
# ───────────────────────────────────────────────────────────────
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
copy_paste: 0.1
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
crop_fraction: 1.0

# ───────────────────────────────────────────────────────────────
# Validation & Inference
# ───────────────────────────────────────────────────────────────
val: true
split: val
conf: null
iou: 0.7
max_det: 300
half: false
agnostic_nms: false

# ───────────────────────────────────────────────────────────────
# General Settings
# ───────────────────────────────────────────────────────────────
save: true
save_period: -1
cache: false
device: null
workers: 8
project: null
name: null
exist_ok: false
verbose: true
seed: 0
deterministic: true
single_cls: false
rect: false
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: false
overlap_mask: true
mask_ratio: 4
dropout: 0.0

# ───────────────────────────────────────────────────────────────
# Output & Visualization
# ───────────────────────────────────────────────────────────────
plots: true
save_json: false
save_hybrid: false
save_txt: false
save_conf: false
save_crop: false
save_frames: false
show: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
visualize: false

# ───────────────────────────────────────────────────────────────
# Export Settings
# ───────────────────────────────────────────────────────────────
format: torchscript
keras: false
optimize: false
int8: false
dynamic: false
simplify: true
opset: null
workspace: null
nms: false

# ───────────────────────────────────────────────────────────────
# Misc
# ───────────────────────────────────────────────────────────────
source: null
vid_stride: 1
stream_buffer: false
augment: false
classes: null
retina_masks: false
embed: null
cfg: null
tracker: botsort.yaml
</pre>

</details>

---

### 🚫 Ablation Baseline — Custom Loss Phases Disabled

For grid search experiments, we used a **17% subset** with **70 epochs** and all custom loss phases **disabled**:

<pre>
# ═══════════════════════════════════════════════════════════════
# 🔬 Ablation Study - Custom Loss Defaults (All Phases Disabled)
# ═══════════════════════════════════════════════════════════════
# 📌 Uses all Ultralytics defaults above, with these overrides:

# Training Overrides (Ablation)
epochs: 70                         # Reduced for faster iteration
batch: 64                         # Increased for ablation
dataset: "weapon_dataset_17pct"   # 17% stratified subset

# ───────────────────────────────────────────────────────────────
# Phase A1: Alpha Scheduling (DISABLED)
# ───────────────────────────────────────────────────────────────
alpha_start: 1.0          # No area weighting
alpha_end: 1.0            # No area weighting
small_obj_boost: 1.0      # No boost (multiplier = 1)
small_obj_px: 32          # Default small object threshold

# ───────────────────────────────────────────────────────────────
# Phase A2: Center Loss (DISABLED)
# ───────────────────────────────────────────────────────────────
center_loss_weight_init: 0.0    # No center loss
center_loss_weight_min: 0.0     # No center loss
center_loss_decay_epochs: 1     # Irrelevant when weight=0

# ───────────────────────────────────────────────────────────────
# Phase A3: Adaptive Clipping (DISABLED)
# ───────────────────────────────────────────────────────────────
iou_clip_start: 100.0     # No clipping (very high)
iou_clip_end: 100.0       # No clipping (very high)
dfl_clip_start: 100.0     # No clipping (very high) — N/A for YOLO26s
dfl_clip_end: 100.0       # No clipping (very high) — N/A for YOLO26s

# ───────────────────────────────────────────────────────────────
# Phase A4: TAL Alpha-Beta (DEFAULT)
# ───────────────────────────────────────────────────────────────
tal_topk: 10
tal_alpha: 0.5
tal_beta: 6.0
</pre>

> 💡 **Note:** Enable one phase at a time while keeping others at their disabled/default values to isolate the effect of each hyperparameter.

---

### 🏆 Best Configuration — YOLOv12s (A1 + A2 + A3.1 + A4)

<pre>
# ═══════════════════════════════════════════════════════════════
# 🏆 YOLOv12s — Best Configuration (Final Training)
# ═══════════════════════════════════════════════════════════════
# 📌 Uses all Ultralytics defaults above, with these overrides:

# Training Overrides (Final)
epochs: 100                        # Full training
dataset: "weapon_dataset_full"     # 100% dataset

# ───────────────────────────────────────────────────────────────
# Phase A1: Alpha Scheduling ✅ ENABLED
# ───────────────────────────────────────────────────────────────
alpha_start: 0.9          # Start with strong area weighting
alpha_end: 0.4            # Anneal to moderate weighting
small_obj_boost: 1.0      # No additional boost
small_obj_px: 32          # Small object threshold (pixels)

# ───────────────────────────────────────────────────────────────
# Phase A2: Center Loss ✅ ENABLED
# ───────────────────────────────────────────────────────────────
center_loss_weight_init: 0.05   # Initial center loss weight
center_loss_weight_min: 0.01    # Minimum (decayed) weight
center_loss_decay_epochs: 1     # Decay per epoch

# ───────────────────────────────────────────────────────────────
# Phase A3.1: IoU Clipping ✅ ENABLED
# ───────────────────────────────────────────────────────────────
iou_clip_start: 6         # Initial IoU loss clip threshold
iou_clip_end: 2           # Final IoU loss clip threshold

# ───────────────────────────────────────────────────────────────
# Phase A3.2: DFL Clipping ❌ DISABLED
# ───────────────────────────────────────────────────────────────
dfl_clip_start: 100.0     # No clipping
dfl_clip_end: 100.0       # No clipping

# ───────────────────────────────────────────────────────────────
# Phase A4: TAL Alpha-Beta ✅ ENABLED
# ───────────────────────────────────────────────────────────────
tal_topk: 20              # Increased from default 10
tal_alpha: 1              # Increased from default 0.5
tal_beta: 7               # Increased from default 6.0
</pre>

---

### 🏆 Best Configuration — YOLO26s (A1 + A2 + A3.1)

<pre>
# ═══════════════════════════════════════════════════════════════
# 🏆 YOLO26s — Best Configuration (Final Training)
# ═══════════════════════════════════════════════════════════════
# 📌 Uses all Ultralytics defaults above, with these overrides:

# Training Overrides (Final)
epochs: 100                        # Full training
dataset: "weapon_dataset_full"     # 100% dataset

# ───────────────────────────────────────────────────────────────
# Phase A1: Alpha Scheduling ✅ ENABLED
# ───────────────────────────────────────────────────────────────
alpha_start: 0.9          # Start with strong area weighting
alpha_end: 0.4            # Anneal to moderate weighting
small_obj_boost: 1.0      # No additional boost
small_obj_px: 32          # Small object threshold (pixels)

# ───────────────────────────────────────────────────────────────
# Phase A2: Center Loss ✅ ENABLED
# ───────────────────────────────────────────────────────────────
center_loss_weight_init: 0.05   # Initial center loss weight
center_loss_weight_min: 0.01    # Minimum (decayed) weight
center_loss_decay_epochs: 1     # Decay per epoch

# ───────────────────────────────────────────────────────────────
# Phase A3.1: IoU Clipping ✅ ENABLED
# ───────────────────────────────────────────────────────────────
iou_clip_start: 6         # Initial IoU loss clip threshold
iou_clip_end: 2           # Final IoU loss clip threshold

# ───────────────────────────────────────────────────────────────
# Phase A3.2: DFL Clipping ❌ NOT APPLICABLE
# ───────────────────────────────────────────────────────────────
# YOLO26s does not use DFL — this phase is skipped entirely

# ───────────────────────────────────────────────────────────────
# Phase A4: TAL Alpha-Beta ⬚ DEFAULT (not tuned)
# ───────────────────────────────────────────────────────────────
tal_topk: 10              # Default value
tal_alpha: 0.5            # Default value
tal_beta: 6.0             # Default value
</pre>

---

### 📋 Quick Comparison Table

<table>
  <tr>
    <th align="left">Parameter</th>
    <th align="center">🚫 Default<br><sub>(Disabled)</sub></th>
    <th align="center">🔷 YOLOv12s<br><sub>(Best)</sub></th>
    <th align="center">🔶 YOLO26s<br><sub>(Best)</sub></th>
  </tr>
  <tr><td colspan="4" align="center"><b>Training Settings</b></td></tr>
  <tr>
    <td><code>epochs</code></td>
    <td align="center">70 <sub>(ablation)</sub></td>
    <td align="center"><b>100</b></td>
    <td align="center"><b>100</b></td>
  </tr>
  <tr>
    <td><code>batch</code></td>
    <td align="center">64 <sub>(ablation)</sub></td>
    <td align="center">44</td>
    <td align="center">44</td>
  </tr>
  <tr>
    <td><code>dataset</code></td>
    <td align="center">17% subset</td>
    <td align="center"><b>100% full</b></td>
    <td align="center"><b>100% full</b></td>
  </tr>
  <tr><td colspan="4" align="center"><b>Ultralytics Loss Weights</b></td></tr>
  <tr>
    <td><code>box</code></td>
    <td align="center">7.5</td>
    <td align="center">7.5</td>
    <td align="center">7.5</td>
  </tr>
  <tr>
    <td><code>cls</code></td>
    <td align="center">0.5</td>
    <td align="center">0.5</td>
    <td align="center">0.5</td>
  </tr>
  <tr>
    <td><code>dfl</code></td>
    <td align="center">1.5</td>
    <td align="center">1.5</td>
    <td align="center"><sub>N/A</sub></td>
  </tr>
  <tr><td colspan="4" align="center"><b>Phase A1: Alpha Scheduling</b></td></tr>
  <tr>
    <td><code>alpha_start</code></td>
    <td align="center">1.0</td>
    <td align="center"><b>0.9</b></td>
    <td align="center"><b>0.9</b></td>
  </tr>
  <tr>
    <td><code>alpha_end</code></td>
    <td align="center">1.0</td>
    <td align="center"><b>0.4</b></td>
    <td align="center"><b>0.4</b></td>
  </tr>
  <tr>
    <td><code>small_obj_px</code></td>
    <td align="center">32</td>
    <td align="center">32</td>
    <td align="center">32</td>
  </tr>
  <tr><td colspan="4" align="center"><b>Phase A2: Center Loss</b></td></tr>
  <tr>
    <td><code>center_loss_weight_init</code></td>
    <td align="center">0.0</td>
    <td align="center"><b>0.05</b></td>
    <td align="center"><b>0.05</b></td>
  </tr>
  <tr>
    <td><code>center_loss_weight_min</code></td>
    <td align="center">0.0</td>
    <td align="center"><b>0.01</b></td>
    <td align="center"><b>0.01</b></td>
  </tr>
  <tr><td colspan="4" align="center"><b>Phase A3.1: IoU Clipping</b></td></tr>
  <tr>
    <td><code>iou_clip_start</code></td>
    <td align="center">100.0</td>
    <td align="center"><b>6</b></td>
    <td align="center"><b>6</b></td>
  </tr>
  <tr>
    <td><code>iou_clip_end</code></td>
    <td align="center">100.0</td>
    <td align="center"><b>2</b></td>
    <td align="center"><b>2</b></td>
  </tr>
  <tr><td colspan="4" align="center"><b>Phase A3.2: DFL Clipping</b></td></tr>
  <tr>
    <td><code>dfl_clip_start</code></td>
    <td align="center">100.0</td>
    <td align="center">100.0 (off)</td>
    <td align="center"><sub>N/A</sub></td>
  </tr>
  <tr>
    <td><code>dfl_clip_end</code></td>
    <td align="center">100.0</td>
    <td align="center">100.0 (off)</td>
    <td align="center"><sub>N/A</sub></td>
  </tr>
  <tr><td colspan="4" align="center"><b>Phase A4: TAL Alpha-Beta</b></td></tr>
  <tr>
    <td><code>tal_topk</code></td>
    <td align="center">10</td>
    <td align="center"><b>20</b></td>
    <td align="center">10 (default)</td>
  </tr>
  <tr>
    <td><code>tal_alpha</code></td>
    <td align="center">0.5</td>
    <td align="center"><b>1</b></td>
    <td align="center">0.5 (default)</td>
  </tr>
  <tr>
    <td><code>tal_beta</code></td>
    <td align="center">6.0</td>
    <td align="center"><b>7</b></td>
    <td align="center">6.0 (default)</td>
  </tr>
  <tr><td colspan="4" align="center"><b>Key Ultralytics Defaults (unchanged)</b></td></tr>
  <tr>
    <td><code>optimizer</code></td>
    <td align="center" colspan="3">auto</td>
  </tr>
  <tr>
    <td><code>lr0</code></td>
    <td align="center" colspan="3">0.01</td>
  </tr>
  <tr>
    <td><code>lrf</code></td>
    <td align="center" colspan="3">0.01</td>
  </tr>
  <tr>
    <td><code>momentum</code></td>
    <td align="center" colspan="3">0.937</td>
  </tr>
  <tr>
    <td><code>weight_decay</code></td>
    <td align="center" colspan="3">0.0005</td>
  </tr>
  <tr>
    <td><code>warmup_epochs</code></td>
    <td align="center" colspan="3">3.0</td>
  </tr>
  <tr>
    <td><code>patience</code></td>
    <td align="center" colspan="3">20</td>
  </tr>
  <tr>
    <td><code>close_mosaic</code></td>
    <td align="center" colspan="3">15</td>
  </tr>
  <tr>
    <td><code>copy_paste</code></td>
    <td align="center" colspan="3">0.1</td>
  </tr>
  <tr>
    <td><code>mosaic</code></td>
    <td align="center" colspan="3">1.0</td>
  </tr>
  <tr>
    <td><code>erasing</code></td>
    <td align="center" colspan="3">0.4</td>
  </tr>
  <tr>
    <td><code>iou</code> <sub>(NMS)</sub></td>
    <td align="center" colspan="3">0.7</td>
  </tr>
  <tr>
    <td><code>imgsz</code></td>
    <td align="center" colspan="3">640</td>
  </tr>
  <tr>
    <td><code>seed</code></td>
    <td align="center" colspan="3">0</td>
  </tr>
  <tr>
    <td><code>deterministic</code></td>
    <td align="center" colspan="3">true</td>
  </tr>
  <tr>
    <td><code>amp</code></td>
    <td align="center" colspan="3">true</td>
  </tr>
</table>

> ⚠️ **Important:** Parameters from **Phase A1** (Alpha Scheduling), **Phase A2** (Center Loss), and **Phase A3** (Adaptive Clipping) are **only available in our custom loss function implementation**. Phase A4 (TAL Alpha-Beta) uses the standard Ultralytics parameters. All other parameters follow Ultralytics defaults — see the expandable full YAML above for the complete list.

</details>

---

## 🏗️ Architecture Ablation Study Results (YOLOv12s)

<details>
<summary><b>⚙️ 1. Experimental Setup</b></summary>

<br>

| Setting | Value |
|---------|-------|
| 📊 **Dataset Size** | **17%** of the full dataset (**10,080 images** / **13,168 instances**) |
| 🔄 **Epochs per Run** | **130 epochs** |
| 📦 **Batch Size** | **48–72** <sub>(varies by architecture size)</sub> |
| 🖼️ **Image Size** | **640×640** |
| ⏱️ **Avg. Time per Run** | ~3.2 hours |
| ⏱️ **Total Search Time** | ~64 hours <sub>(20 architectures × ~3.2h)</sub> |
| 🔬 **Methodology** | Grid search across **20 architectural variants** |
| 🏛️ **Base Architecture** | YOLOv12s (P3–P5 default, 3 detection heads) |
| 🎯 **Goal** | Identify the best detection head configuration for small-object weapon detection |

> 📌 All architecture experiments use the **default loss function** (no custom loss phases enabled) to isolate the effect of architectural changes.

> ⚠️ **Batch size note:** Larger architectures (e.g., wider channels, more heads, deeper backbones) require more GPU memory. Batch size was adjusted per architecture to maximize GPU utilization while fitting within **24GB VRAM** (RTX 4090): smaller architectures used up to **72**, while the largest variants ran at **48**.

</details>

---

<details>
<summary><b>🧬 2. Architecture Variants (Arch-1 to Arch-20)</b></summary>

<br>

We designed **20 architectural variants** exploring different strategies for improving small-object detection:

### 🎯 Design Strategies Explored

| Strategy | Architectures | Description |
|----------|---------------|-------------|
| 🔍 **P2 Head Addition** | Arch-1 to Arch-5, Arch-12 to Arch-15, Arch-19, Arch-20 | Adding a high-resolution P2 detection head (stride 4) |
| 🔀 **Auxiliary Branch (P2×2)** | Arch-6 ★, Arch-7, Arch-9 to Arch-11, Arch-16 to Arch-18 | Dual P2 outputs via auxiliary split branch |
| 📐 **P3 Auxiliary Branch** | Arch-8 | Auxiliary branch at P3 level instead of P2 |
| 🏔️ **P6 Head Addition** | Arch-3 | Extra P6 head at stride 64 for very large objects |
| 🔧 **BiFPN-style Neck** | Arch-2, Arch-9 | Replacing A2C2f with C3k2 throughout the neck |
| 📏 **Width Scaling** | Arch-6 ★, Arch-8, Arch-9 to Arch-11, Arch-16, Arch-19 | Width scale 0.50 (vs default 0.25) |
| 🏋️ **Deeper Backbone** | Arch-4 | Significantly increased backbone depth (3/5/6/7 repeats) |

---

### 📋 Full Architecture Comparison

<table>
  <tr>
    <th align="center">Arch</th>
    <th align="center">Heads</th>
    <th align="left">Key Differences vs. Original YOLOv12s</th>
  </tr>
  <tr>
    <td align="center"><b>Arch-1</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head (upsample→concat with layer 2→C3k2 128ch). Reduced backbone repeats (1/2/3/3 vs 2/2/4/4). Reduced neck repeats (1 vs 2). Fewer P5 C3k2 repeats (2 vs 2).</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-2</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head (C3k2 128ch). All neck A2C2f replaced with C3k2 (BiFPN-style). Backbone P5 A2C2f increased to 5 repeats. Backbone P3 C3k2 increased to 3 repeats.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-3</b></td>
    <td align="center">5 (P2–P6)</td>
    <td>+ P2 head (C3k2 128ch). + P6 head (Conv stride-2→C3k2 1024ch at stride 64). Backbone P5 A2C2f increased to 5 repeats. Backbone P3 C3k2 increased to 3 repeats.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-4</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head (C3k2 128ch). Much deeper backbone (3/5/6/7 repeats vs 2/2/4/4). Deeper neck (3 repeats vs 2). P5 C3k2 increased to 4 repeats.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-5</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head (upsample→concat with layer 2→C3k2 128ch). Backbone P5 A2C2f increased to 5 repeats. Backbone P3 C3k2 increased to 3 repeats. Otherwise matches the original neck structure.</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>Arch-6 ★</b></td>
    <td align="center">5 (P2×2+P3–P5)</td>
    <td><b>+ P2 head with auxiliary branch</b> (C3k2 True + Conv split → 2 separate P2 outputs to Detect). Width scale 0.50 (vs 0.25). Backbone P5 increased to 5 repeats, P3 to 3 repeats. Bottom-up P3/P4 use C3k2 instead of A2C2f.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-7</b></td>
    <td align="center">5 (P2×2+P3–P5)</td>
    <td>Same as Arch-6 but auxiliary C3k2 uses 2 repeats instead of 1. Width scale stays 0.25.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-8</b></td>
    <td align="center">5 (P2+P3×2+P4+P5)</td>
    <td>Auxiliary branch moved to P3 instead of P2 (C3k2 True + Conv at P3 level). P2 uses standard C3k2 128ch. Width scale 0.50. Backbone P5 to 5 repeats, P3 to 3 repeats.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-9</b></td>
    <td align="center">5 (P2×2+P3–P5)</td>
    <td>Arch-6 auxiliary structure + all neck blocks replaced with C3k2 (BiFPN-style). No A2C2f in any neck stage. Width scale 0.50.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-10</b></td>
    <td align="center">5 (P2×2+P3–P5)</td>
    <td>Arch-6 auxiliary structure + 1×1 Conv channel alignment inserted before the aux split at P2. Adds a residual-style projection layer. Width scale 0.50.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-11</b></td>
    <td align="center">5 (P2×2+P3–P5)</td>
    <td>Arch-6 auxiliary structure + 2 sequential C3k2 blocks at P2 before aux split (multi-scale receptive field). Width scale 0.50.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-12</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head with 3× C3k2(True) at P2 (deep small-object refinement, full kernel). Backbone P5 to 5, P3 to 3 repeats.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-13</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head with 192 channels instead of 128 (wider P2). Backbone P5 to 5, P3 to 3 repeats.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-14</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head (C3k2 128ch). P3 stages deepened to 3× A2C2f in both top-down and bottom-up paths. Backbone P5 to 5, P3 to 3 repeats.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-15</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head with A2C2f replacing C3k2 at P2 (attention at highest resolution). Backbone P5 to 5, P3 to 3 repeats.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-16</b></td>
    <td align="center">5 (P2×2+P3–P5)</td>
    <td>Arch-6 auxiliary structure but bottom-up P3/P4 use A2C2f instead of C3k2 for uniform attention across all scales. Width scale 0.50.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-17</b></td>
    <td align="center">5 (P2×2+P3–P5)</td>
    <td>Arch-7 auxiliary structure with 3× C3k2 in the aux branch (maximum aux depth). Bottom-up P3 uses C3k2.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-18</b></td>
    <td align="center">5 (P2×2+P3–P5)</td>
    <td>Arch-7 auxiliary structure with 192-channel aux branch instead of 128. Bottom-up P3 uses C3k2.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-19</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head with 256ch A2C2f attention (widest P2). Conv 256 3×3 projection after P2 concat. Backbone P5 reduced to 4 repeats, P3 to 2. Width scale 0.50.</td>
  </tr>
  <tr>
    <td align="center"><b>Arch-20</b></td>
    <td align="center">4 (P2–P5)</td>
    <td>+ P2 head with direct skip from backbone layer 0 (first Conv stride-2) concatenated into P2 fusion (3-way concat). Backbone P5 to 5, P3 to 3 repeats.</td>
  </tr>
</table>

</details>

---

<details>
<summary><b>📊 3. Grid Search Results</b></summary>

<br>

### 📈 All Architectures Performance Comparison

<p align="center">
<img width="1774" height="876" alt="image" src="https://github.com/user-attachments/assets/fd4ea33d-cc23-4bbd-8116-264fb739be78" />
</p>

---

### 🏆 Arch-6 vs Baseline — Improvement Summary

<table>
  <tr>
    <th align="left">Metric</th>
    <th align="center">Baseline</th>
    <th align="center">Arch-6 ★</th>
    <th align="center">Δ Improvement</th>
  </tr>
  <tr>
    <td><b>mAP50</b></td>
    <td>0.816</td>
    <td><b>0.865</b></td>
    <td align="center"><b>+6.00%</b></td>
  </tr>
  <tr>
    <td><b>mAP50-95</b></td>
    <td>0.525</td>
    <td><b>0.585</b></td>
    <td align="center"><b>+11.43%</b></td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.831</td>
    <td><b>0.902</b></td>
    <td align="center"><b>+8.54%</b></td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.746</td>
    <td><b>0.825</b></td>
    <td align="center"><b>+10.59%</b></td>
  </tr>
  <tr>
    <td><b>F1 Score</b></td>
    <td>0.786</td>
    <td><b>0.862</b></td>
    <td align="center"><b>+9.67%</b></td>
  </tr>
  <tr>
    <td colspan="4" align="center"><b>🔍 Size-Specific mAP50</b></td>
  </tr>
  <tr>
    <td>🔍 <b>Small</b></td>
    <td>0.530</td>
    <td><b>0.590</b></td>
    <td align="center"><b>+11.32%</b></td>
  </tr>
  <tr>
    <td>📦 <b>Medium</b></td>
    <td>0.750</td>
    <td><b>0.812</b></td>
    <td align="center"><b>+8.27%</b></td>
  </tr>
  <tr>
    <td>🟫 <b>Large</b></td>
    <td>0.828</td>
    <td><b>0.885</b></td>
    <td align="center"><b>+6.88%</b></td>
  </tr>
  <tr>
    <td colspan="4" align="center"><b>🔍 Size-Specific mAP50-95</b></td>
  </tr>
  <tr>
    <td>🔍 <b>Small</b></td>
    <td>0.297</td>
    <td><b>0.355</b></td>
    <td align="center"><b>+19.53%</b></td>
  </tr>
  <tr>
    <td>📦 <b>Medium</b></td>
    <td>0.424</td>
    <td><b>0.485</b></td>
    <td align="center"><b>+14.38%</b></td>
  </tr>
  <tr>
    <td>🟫 <b>Large</b></td>
    <td>0.550</td>
    <td><b>0.628</b></td>
    <td align="center"><b>+14.18%</b></td>
  </tr>
</table>

> 📌 **Full results for all 20 architectures** are shown in the image above. The table highlights the **Baseline** and the **winning architecture (Arch-6)**.

</details>

---

<details>
<summary><b>🏆 4. Winner: Arch-6 — Architecture Deep Dive</b></summary>

<br>

### 🧬 Architecture Diagram

<p align="center">
<img width="940" height="620" alt="image" src="https://github.com/user-attachments/assets/4af41ad9-9a55-4e45-ac2a-de177ccc89de" />
</p>

<p align="center"><sub>Arch-6 (P2 Auxiliary ★): 5-head detection architecture with auxiliary P2 branch at stride 4.<br>🟦 New P2 layers | 🟧 Auxiliary split branch | 🔶 Modified layers vs original YOLOv12s.</sub></p>

---

### 🎯 Design Highlights

<table>
  <tr>
    <th align="left">Feature</th>
    <th align="left">Description</th>
    <th align="left">Impact</th>
  </tr>
  <tr>
    <td><b>🔍 P2 Detection Head</b></td>
    <td>Added high-resolution head at stride 4 (160×160)</td>
    <td>Captures fine-grained details for small objects</td>
  </tr>
  <tr>
    <td><b>🔀 Auxiliary P2 Branch</b></td>
    <td>C3k2(True) + Conv split → 2 separate P2 outputs to Detect</td>
    <td>Extra gradient supervision forces stronger P2 features</td>
  </tr>
  <tr>
    <td><b>📏 Width Scale 0.50</b></td>
    <td>Doubled channel width (vs default 0.25)</td>
    <td>2× more representational capacity everywhere</td>
  </tr>
  <tr>
    <td><b>🏋️ Deeper Backbone</b></td>
    <td>P3: ×2→×3 repeats, P5: ×4→×5 repeats</td>
    <td>Richer feature extraction at critical scales</td>
  </tr>
  <tr>
    <td><b>🔧 C3k2 in Bottom-up</b></td>
    <td>P3 bottom-up uses C3k2 instead of A2C2f</td>
    <td>Saves compute at mid-resolution where attention is less critical</td>
  </tr>
  <tr>
    <td><b>🎯 5 Detection Heads</b></td>
    <td>P2×2 (aux+main) + P3 + P4 + P5</td>
    <td>+2 heads at stride 4 dramatically improve small-object recall</td>
  </tr>
</table>

---

<details>
<summary><b>📐 Layer-by-Layer Comparison (Original vs Arch-6)</b></summary>

<br>

#### 🏛️ Backbone (L0–L8)

<table>
  <tr>
    <th align="center">Layer</th>
    <th align="left">Original YOLOv12s</th>
    <th align="left">Arch-6 (P2 Auxiliary ★)</th>
    <th align="center">Change</th>
    <th align="left">Rationale</th>
  </tr>
  <tr>
    <td align="center"><b>L0</b></td>
    <td>Conv [64, 3, 2]</td>
    <td>Conv [64, 3, 2]</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L1</b></td>
    <td>Conv [128, 3, 2, 1, 2]</td>
    <td>Conv [128, 3, 2, 1, 2]</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L2</b></td>
    <td>C3k2 [256, False, 0.25] ×2</td>
    <td>C3k2 [256, False, 0.25] ×2</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L3</b></td>
    <td>Conv [256, 3, 2, 1, 4]</td>
    <td>Conv [256, 3, 2, 1, 4]</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L4</b></td>
    <td>C3k2 [512, False, 0.25] ×2</td>
    <td>C3k2 [512, False, 0.25] <b>×3</b></td>
    <td align="center">🔶 ×2→×3</td>
    <td>Extra repeat enriches P3-level features, providing stronger mid-scale representations before entering the attention stages.</td>
  </tr>
  <tr>
    <td align="center"><b>L5</b></td>
    <td>Conv [512, 3, 2]</td>
    <td>Conv [512, 3, 2]</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L6</b></td>
    <td>A2C2f [512, True, 4] ×4</td>
    <td>A2C2f [512, True, 4] ×4</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L7</b></td>
    <td>Conv [1024, 3, 2]</td>
    <td>Conv [1024, 3, 2]</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L8</b></td>
    <td>A2C2f [1024, True, 1] ×4</td>
    <td>A2C2f [1024, True, 1] <b>×5</b></td>
    <td align="center">🔶 ×4→×5</td>
    <td>Extra repeat at the deepest stage captures richer semantic features before entering the top-down path.</td>
  </tr>
</table>

---

#### 🔽 Neck — Top-Down FPN (L9–L19)

<table>
  <tr>
    <th align="center">Layer</th>
    <th align="left">Original YOLOv12s</th>
    <th align="left">Arch-6 (P2 Auxiliary ★)</th>
    <th align="center">Change</th>
    <th align="left">Rationale</th>
  </tr>
  <tr>
    <td align="center"><b>L9</b></td>
    <td>Upsample ×2</td>
    <td>Upsample ×2</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L10</b></td>
    <td>Concat (L8_up + L6_bb)</td>
    <td>Concat (L8_up + L6_bb)</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L11</b></td>
    <td>A2C2f [512, False, -1] ×2</td>
    <td>A2C2f [512, False, -1] ×2</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L12</b></td>
    <td>Upsample ×2</td>
    <td>Upsample ×2</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L13</b></td>
    <td>Concat (L11_up + L4_bb)</td>
    <td>Concat (L11_up + L4_bb)</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr>
    <td align="center"><b>L14</b></td>
    <td>A2C2f [256, False, -1] ×2</td>
    <td>A2C2f [256, False, -1] ×2</td>
    <td align="center">✅ Same</td>
    <td>—</td>
  </tr>
  <tr style="background-color: #d4edda;">
    <td align="center"><b>L15</b></td>
    <td>Conv [256, 3, 2] <sub>(start bottom-up)</sub></td>
    <td>🟦 Upsample ×2 <sub>(NEW — extends to P2)</sub></td>
    <td align="center">🆕 Added</td>
    <td>Extends the top-down path one scale further to reach stride-4 resolution for small-object detection.</td>
  </tr>
  <tr style="background-color: #d4edda;">
    <td align="center"><b>L16</b></td>
    <td>—</td>
    <td>🟦 Concat (L15_up + L2_bb)</td>
    <td align="center">🆕 Added</td>
    <td>Fuses upsampled P3 features with high-resolution P2 backbone features, combining semantic depth with spatial detail.</td>
  </tr>
  <tr style="background-color: #d4edda;">
    <td align="center"><b>L17</b></td>
    <td>—</td>
    <td>🟦 C3k2 [128, False, 0.25] ×2</td>
    <td align="center">🆕 Added</td>
    <td>Processes fused P2 features with a lightweight block. Uses C3k2 instead of A2C2f to keep compute manageable at the highest resolution (160×160).</td>
  </tr>
  <tr style="background-color: #fce4d6;">
    <td align="center"><b>L18</b></td>
    <td>—</td>
    <td>🟧 C3k2 [128, True] ×1 <sub>(AUX)</sub></td>
    <td align="center">🆕 Added</td>
    <td>Auxiliary branch with full 3×3 kernels. Dead-end path directly to Detect. Provides extra gradient supervision at P2, forcing the network to learn strong high-resolution features.</td>
  </tr>
  <tr style="background-color: #fce4d6;">
    <td align="center"><b>L19</b></td>
    <td>—</td>
    <td>🟧 Conv [128, 3, 1] <sub>(MAIN P2)</sub></td>
    <td align="center">🆕 Added</td>
    <td>Main P2 branch with stride-1 Conv (no downsampling). Projects channels before continuing into the bottom-up PAN path. Also feeds into Detect.</td>
  </tr>
</table>

---

#### 🔼 Head — Bottom-Up PAN (L20–L28 + Detect)

<table>
  <tr>
    <th align="center">Layer</th>
    <th align="left">Original YOLOv12s</th>
    <th align="left">Arch-6 (P2 Auxiliary ★)</th>
    <th align="center">Change</th>
    <th align="left">Rationale</th>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L20</b><br><sub>(was L15)</sub></td>
    <td>Conv [256, 3, 2]</td>
    <td>Conv [<b>128</b>, 3, 2]</td>
    <td align="center">🔶 256→128ch</td>
    <td>Starts bottom-up from P2 (128ch) instead of P3 (256ch). Lower channel count matches the finer P2 scale.</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L21</b><br><sub>(was L16)</sub></td>
    <td>Concat (L15 + L11)</td>
    <td>Concat (L20 + <b>L14</b>)</td>
    <td align="center">🔶 Sources shifted</td>
    <td>Concatenates P2 downsampled with P3 from neck (L14) instead of original P3 with P4. One extra fusion step in the path.</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L22</b><br><sub>(was L17)</sub></td>
    <td>A2C2f [512, False, -1] ×2</td>
    <td><b>C3k2</b> [<b>256</b>, False, 0.25] ×2</td>
    <td align="center">🔶 A2C2f→C3k2<br>512→256ch</td>
    <td>Replaced attention block with lightweight C3k2 at P3 scale. Saves compute at mid-resolution where attention is less critical.</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L23</b><br><sub>(was L18)</sub></td>
    <td>Conv [512, 3, 2]</td>
    <td>Conv [<b>256</b>, 3, 2]</td>
    <td align="center">🔶 512→256ch</td>
    <td>Channel count adjusted to match the new P3→P4 transition (256→512 happens at concat).</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L24</b><br><sub>(was L19)</sub></td>
    <td>Concat (L18 + L8)</td>
    <td>Concat (L23 + <b>L11</b>)</td>
    <td align="center">🔶 Sources shifted</td>
    <td>Fuses with P4 from neck (L11) instead of P5 from backbone. Extended path adds one more fusion opportunity.</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L25</b></td>
    <td>—</td>
    <td>A2C2f [512, False, -1] ×2</td>
    <td align="center">🔶 New position</td>
    <td>A2C2f at P4 in the bottom-up path. Attention applied here where feature maps are smaller (40×40) and attention cost is reasonable.</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L26</b></td>
    <td>—</td>
    <td>Conv [512, 3, 2]</td>
    <td align="center">🔶 New position</td>
    <td>Downsamples P4 to P5 resolution for the final concat.</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L27</b></td>
    <td>—</td>
    <td>Concat (L26 + L8_bb)</td>
    <td align="center">🔶 New position</td>
    <td>Final fusion: bottom-up P4 features + P5 backbone features.</td>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>L28</b><br><sub>(was L20)</sub></td>
    <td>C3k2 [1024, True] ×2</td>
    <td>C3k2 [1024, True] <b>×3</b></td>
    <td align="center">🔶 ×2→×3</td>
    <td>Extra repeat at P5 for stronger final feature refinement before detection.</td>
  </tr>
</table>

---

#### 🎯 Detection Head

<table>
  <tr>
    <th align="center">Component</th>
    <th align="left">Original YOLOv12s</th>
    <th align="left">Arch-6 (P2 Auxiliary ★)</th>
    <th align="center">Change</th>
  </tr>
  <tr style="background-color: #fff3cd;">
    <td align="center"><b>Detect</b></td>
    <td>Detect [L14, L17, L20] — <b>3 heads</b></td>
    <td>Detect [L18, L19, L22, L25, L28] — <b>5 heads</b></td>
    <td align="center">🔶 3→5 heads</td>
  </tr>
</table>

<sub>Two additional detection heads at stride 4 (aux + main P2) dramatically improve small-object recall.</sub>

</details>

---

<details>
<summary><b>📊 Summary of Changes</b></summary>

<br>

<table>
  <tr>
    <th align="center">Change Type</th>
    <th align="center">Count</th>
    <th align="left">Details</th>
  </tr>
  <tr>
    <td align="center">🆕 <b>New layers added</b></td>
    <td align="center"><b>5</b></td>
    <td>L15 (Upsample), L16 (Concat), L17 (C3k2), L18 (C3k2 aux), L19 (Conv main)</td>
  </tr>
  <tr>
    <td align="center">🔶 <b>Backbone repeats increased</b></td>
    <td align="center"><b>2</b></td>
    <td>L4: ×2→×3, L8: ×4→×5</td>
  </tr>
  <tr>
    <td align="center">🔶 <b>Head repeats increased</b></td>
    <td align="center"><b>1</b></td>
    <td>L28: ×2→×3</td>
  </tr>
  <tr>
    <td align="center">🔶 <b>Block type changed</b></td>
    <td align="center"><b>1</b></td>
    <td>L22: A2C2f→C3k2 (lighter at P3 scale)</td>
  </tr>
  <tr>
    <td align="center">🔶 <b>Channel widths adjusted</b></td>
    <td align="center"><b>2</b></td>
    <td>L20: 256→128, L23: 512→256 (to accommodate P2 scale)</td>
  </tr>
  <tr>
    <td align="center">🔶 <b>Detection heads added</b></td>
    <td align="center"><b>2</b></td>
    <td>Aux P2 (L18) + Main P2 (L19) at stride 4</td>
  </tr>
  <tr>
    <td align="center">🔶 <b>Width scale changed</b></td>
    <td align="center"><b>1</b></td>
    <td>0.25→0.50 (doubles effective channel widths across entire network)</td>
  </tr>
</table>

</details>

---

<details>
<summary><b>⚡ Impact Comparison: Original vs Arch-6</b></summary>

<br>

<table>
  <tr>
    <th align="left">Metric</th>
    <th align="center">Original YOLOv12s</th>
    <th align="center">Arch-6 ★</th>
    <th align="left">Impact</th>
  </tr>
  <tr>
    <td><b>Detection heads</b></td>
    <td align="center">3 (P3, P4, P5)</td>
    <td align="center"><b>5 (P2×2, P3, P4, P5)</b></td>
    <td>+2 heads at stride 4 for small-object coverage</td>
  </tr>
  <tr>
    <td><b>Smallest stride</b></td>
    <td align="center">8 (P3)</td>
    <td align="center"><b>4 (P2)</b></td>
    <td>2× higher resolution detection — catches objects missed at stride 8</td>
  </tr>
  <tr>
    <td><b>Feature map at finest head</b></td>
    <td align="center">80×80</td>
    <td align="center"><b>160×160</b></td>
    <td>4× more spatial anchors at the finest scale</td>
  </tr>
  <tr>
    <td><b>Backbone depth</b></td>
    <td align="center">×2/×2/×4/×4</td>
    <td align="center"><b>×2/×3/×4/×5</b></td>
    <td>Deeper P3 and P5 for richer features</td>
  </tr>
  <tr>
    <td><b>Auxiliary supervision</b></td>
    <td align="center">None</td>
    <td align="center"><b>L18 dead-end to Detect</b></td>
    <td>Extra gradient signal forces stronger P2 features during training</td>
  </tr>
  <tr>
    <td><b>Width multiplier</b></td>
    <td align="center">0.25</td>
    <td align="center"><b>0.50</b></td>
    <td>2× wider channels — more representational capacity everywhere</td>
  </tr>
  <tr>
    <td><b>Neck fusion depth</b></td>
    <td align="center">P5→P3 (2 upsample steps)</td>
    <td align="center"><b>P5→P2 (3 upsample steps)</b></td>
    <td>One extra fusion level brings semantic info to highest resolution</td>
  </tr>
  <tr>
    <td><b>Compute cost</b></td>
    <td align="center">Lower</td>
    <td align="center">Higher</td>
    <td>Trade-off: more compute for significantly better small-object detection</td>
  </tr>
</table>

</details>

---

<details>
<summary><b>🔍 Why Arch-6 Won</b></summary>

<br>

| Aspect | Explanation |
|--------|-------------|
| 🎯 **Small Object Focus** | The dual P2 output provides two complementary views of high-resolution features, boosting small object detection by **+19.53%** (mAP50-95) |
| 📐 **Width Scaling** | The 0.50 width scale gives the model enough capacity to learn discriminative features without excessive compute |
| ⚖️ **Balanced Design** | Unlike deeper variants (Arch-4) or wider P2 variants (Arch-19), Arch-6 balances depth, width, and auxiliary branching |
| 🏗️ **C3k2 in Bottom-up** | Using C3k2 instead of A2C2f in the bottom-up path reduces attention overhead at higher resolution while maintaining performance |
| 🔀 **Auxiliary Supervision** | The dead-end L18 branch forces the network to develop strong P2 features during training, acting as a regularizer |
| 📏 **Resolution Coverage** | Stride-4 detection at 160×160 provides 4× more spatial positions than P3 (80×80), crucial for detecting small weapons |

</details>

</details>

---

<details>
<summary><b>📖 5. Architecture Legend & Terminology</b></summary>

<br>

<details>
<summary><b>🔧 Operations</b></summary>

<br>

| Item | Meaning |
|------|---------|
| **Conv** | Standard Convolution + BatchNorm + SiLU activation. Used for downsampling (stride 2) or channel projection (stride 1). Learnable filter that extracts local features. |
| **C3k2** | Cross-Stage Partial Bottleneck with 2 Convolutions. Lightweight block that splits input channels, processes one half through bottleneck convolutions, then concatenates both halves. Efficient and fast. |
| **A2C2f** | Area-Attention Cross-stage 2-conv Fusion. Advanced block with built-in multi-head area attention that captures long-range spatial dependencies. Heavier than C3k2 but more powerful. |
| **Upsample** | Nearest Neighbor Upsampling. Doubles spatial resolution (H×2, W×2) without learnable parameters. Used in top-down path to match feature map sizes. |
| **Concat [1]** | Channel-wise Concatenation along dim=1. Joins two or more feature maps by stacking their channels. Spatial dimensions must match. No learnable parameters. |
| **Detect** | Final Detection Head. Takes multi-scale feature maps and outputs bounding box coordinates + class probability scores. |

</details>

<details>
<summary><b>📐 Parameters</b></summary>

<br>

| Item | Meaning |
|------|---------|
| **[channels]** | Output channel count after the operation. Example: `[512]` → 512 output channels. |
| **[ch, kernel, stride]** | Conv specification. Example: `[128, 3, 2]` → 128 channels, 3×3 kernel, stride 2 (halves resolution). |
| **[ch, k, s, pad, groups]** | Extended Conv specification with padding and grouped convolution. Example: `[128, 3, 2, 1, 2]` → 2-group conv. |
| **True / False** | Kernel size toggle. `True` = full 3×3 kernels (stronger, heavier). `False` = compact 1×1 mix (lighter, faster). |
| **0.25** | Bottleneck width ratio (C3k2 only). 0.25 = only 25% of channels pass through the bottleneck path. |
| **4 / 1** | Number of area-attention heads (A2C2f only). `4` = 4-head attention at P4. `1` = single-head at P5. |
| **-1** | Attention disabled (A2C2f only). Used in neck/head for lightweight fusion mode without attention overhead. |
| **×N** | Number of sequential block repeats. Example: `×3` = the block is stacked 3 times in series. |
| **[None, 2, "nearest"]** | Upsample config: scale factor 2, nearest interpolation (no smoothing). |
| **nc** | Number of object classes. `nc=4` → 4 detection categories. |

</details>

<details>
<summary><b>🏛️ Architecture Sections</b></summary>

<br>

| Section | Layers | Description |
|---------|--------|-------------|
| **BACKBONE** | L0–L8 | Feature Extraction. Progressively downsamples the input image across 5 scales (P1→P5). Uses Conv for downsampling, C3k2 at early stages, A2C2f with attention at deep stages. |
| **NECK** | L9–L19 | Top-Down Fusion (FPN). Upsamples deep features and fuses them with backbone skip connections, propagating rich semantic information from P5 down to P2. Contains the auxiliary split at L18/L19. |
| **HEAD** | L20–L28 + Detect | Bottom-Up Fusion (PAN) + Detection. Downsamples fused features back from P2 to P5, re-fusing with neck outputs. Feeds 5 multi-scale outputs into the Detect layer. |

</details>

<details>
<summary><b>🟦🟧 New & Modified Layers</b></summary>

<br>

| Category | Item | Meaning |
|----------|------|---------|
| 🟦 **P2 Head** | L15, L16, L17 | Added to original YOLOv12s. Upsample + Concat + C3k2 that extends detection down to stride 4 for small-object coverage. Present in all P2 architectures. |
| 🟧 **Auxiliary Split** | L18, L19 | Exclusive to Arch-6. L17 forks into two parallel branches: L18 (C3k2 True → Detect directly) provides auxiliary supervision, L19 (Conv stride 1) continues into bottom-up PAN path. |

</details>

<details>
<summary><b>🔗 Concat Labels</b></summary>

<br>

| Label | Layer | Description |
|-------|-------|-------------|
| (P5 upsampled + P4 backbone) | L10 | Upsampled P5 features concatenated with P4 skip from backbone (L6). |
| (P4 upsampled + P3 backbone) | L13 | Upsampled P4 features concatenated with P3 skip from backbone (L4). |
| (P3 upsampled + P2 backbone) | L16 | Upsampled P3 features concatenated with P2 skip from backbone (L2). |
| (P2 downsampled + P3 neck) | L21 | Main P2 branch downsampled and concatenated with P3 from neck (L14). |
| (P3 downsampled + P4 neck) | L24 | P3 downsampled and concatenated with P4 from neck (L11). |
| (P4 downsampled + P5 backbone) | L27 | P4 downsampled and concatenated with P5 from backbone (L8). |

</details>

<details>
<summary><b>📏 Stride / Resolution Map</b></summary>

<br>

| Stride | Scale | Resolution <sub>(640×640 input)</sub> | Usage |
|--------|-------|---------------------------------------|-------|
| 2 | P1 | 320 × 320 | Initial downsampled features |
| **4** | **P2** | **160 × 160** | **Highest-resolution detection (very small objects)** ★ |
| 8 | P3 | 80 × 80 | Small object detection |
| 16 | P4 | 40 × 40 | Medium object detection |
| 32 | P5 | 20 × 20 | Large object detection |

</details>

<details>
<summary><b>🎯 Detect Outputs</b></summary>

<br>

| Output | Layer | Stride | Description |
|--------|-------|--------|-------------|
| 🟧 **Aux P2** | L18 | 4 | Auxiliary head at stride 4. Dead-end branch providing extra supervision at highest resolution. |
| 🟧 **Main P2** | L19 | 4 | Main P2 head at stride 4. Also feeds into bottom-up path before reaching Detect. |
| 🔷 **P3** | L22 | 8 | P3 head at stride 8. |
| 🔷 **P4** | L25 | 16 | P4 head at stride 16. |
| 🔷 **P5** | L28 | 32 | P5 head at stride 32. |

</details>

</details>

---

<details>
<summary><b>📌 6. Key Takeaways</b></summary>

<br>

- 🏆 **Winner:** **Arch-6** with auxiliary P2 branch (P2×2 + P3–P5, width scale 0.50)
- 📈 **Best Improvement:** **+19.53%** on small objects (mAP50-95), **+11.43%** on overall mAP50-95
- 🔍 **P2 Head is Critical:** All top-performing architectures include a P2 detection head
- 🔀 **Auxiliary Branches Help:** Dual P2 outputs (Arch-6, 7, 9–11, 16–18) consistently outperform single P2 heads
- 📏 **Width Scale Matters:** 0.50 width scale outperforms 0.25 in architectures with P2 heads
- 🏔️ **P6 Head Not Beneficial:** Arch-3's P6 head at stride 64 did not improve small-object detection
- 🏋️ **Diminishing Returns on Depth:** Arch-4's much deeper backbone (3/5/6/7) did not proportionally improve results
- 🔧 **BiFPN-style Neck:** Mixed results — beneficial when combined with auxiliary branches (Arch-9) but not alone (Arch-2)
- ⚡ **A2C2f at P2:** Using attention at the highest resolution (Arch-15) adds compute without significant gains over C3k2
- 🧪 **20 experiments** × ~3.2h = **~64 hours** of architecture search

</details>

## 🏋️ Full Dataset Training — Custom Architecture & Custom Loss (YOLOv12s)

After identifying the best architecture (**Arch-6 ★**) and best loss configuration (**A1 + A2 + A3.1 + A4**) through separate ablation studies, we conducted **2 final training runs** on the **full dataset** to measure their individual and combined impact.

<details>
<summary><b>⚙️ 1. Training Configuration</b></summary>

<br>

| Setting | Value |
|---------|-------|
| 📊 **Dataset** | **100%** full dataset (**59,305 images** / **76,705 instances**) |
| 🔄 **Epochs** | **130** |
| 📦 **Batch Size** | **48** |
| 🖼️ **Image Size** | **640×640** |
| ⏱️ **Training Time** | ~20 hours per run |
| 🏛️ **Architecture** | YOLOv12s with Arch-6 ★ (P2×2 + P3–P5, width scale 0.50) |
| 🎮 **GPU** | NVIDIA RTX 4090 24GB |

---

### 🧪 Training Runs

| Run | Architecture | Loss Function | Description |
|-----|-------------|---------------|-------------|
| **Run 1** | 🏗️ Arch-6 ★ (Custom) | 🚫 Default (Ultralytics) | Isolates the effect of the custom architecture alone |
| **Run 2** | 🏗️ Arch-6 ★ (Custom) | ✅ Custom (A1+A2+A3.1+A4) | Combines both custom architecture and custom loss |

</details>

---

<details>
<summary><b>📊 2. Results — Custom Architecture Only (Run 1)</b></summary>

<br>

### 🏗️ Arch-6 + Default Loss — Full Dataset (130 epochs)

Original YOLOv12s (baseline) vs Custom Architecture (Arch-6) with default loss:

<p align="center">
<img width="2000" height="977" alt="image" src="https://github.com/user-attachments/assets/46ee119c-24e1-421f-bc80-b54f81d9cc39" />
</p>

<table>
  <tr>
    <th align="center" rowspan="2">Metric</th>
    <th align="center" colspan="5">Original → Custom Arch (Δ Improvement)</th>
  </tr>
  <tr>
    <th align="center">OVERALL</th>
    <th align="center">🗡️ Knife</th>
    <th align="center">🎯 Long Gun</th>
    <th align="center">🚫 Other</th>
    <th align="center">🔫 Pistol</th>
  </tr>
  <tr>
    <td><b>mAP50</b></td>
    <td>0.816 → <b>0.840</b> <sub>(+2.94%)</sub></td>
    <td>0.875 → <b>0.897</b> <sub>(+2.61%)</sub></td>
    <td>0.840 → <b>0.890</b> <sub>(+5.90%)</sub></td>
    <td>0.637 → <b>0.662</b> <sub>(+3.99%)</sub></td>
    <td>0.911 → <b>0.909</b> <sub>(-0.22%)</sub></td>
  </tr>
  <tr>
    <td><b>mAP50-95</b></td>
    <td>0.525 → <b>0.562</b> <sub>(+7.13%)</sub></td>
    <td>0.614 → <b>0.665</b> <sub>(+8.45%)</sub></td>
    <td>0.535 → <b>0.587</b> <sub>(+9.79%)</sub></td>
    <td>0.346 → <b>0.371</b> <sub>(+7.42%)</sub></td>
    <td>0.605 → <b>0.624</b> <sub>(+3.28%)</sub></td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.831 → <b>0.871</b> <sub>(+4.80%)</sub></td>
    <td>0.847 → <b>0.853</b> <sub>(+0.70%)</sub></td>
    <td>0.827 → <b>0.885</b> <sub>(+7.09%)</sub></td>
    <td>0.800 → <b>0.841</b> <sub>(+5.08%)</sub></td>
    <td>0.851 → <b>0.906</b> <sub>(+6.39%)</sub></td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.746 → <b>0.795</b> <sub>(+6.62%)</sub></td>
    <td>0.845 → <b>0.883</b> <sub>(+4.58%)</sub></td>
    <td>0.769 → <b>0.828</b> <sub>(+7.64%)</sub></td>
    <td>0.522 → <b>0.574</b> <sub>(+9.88%)</sub></td>
    <td>0.847 → <b>0.895</b> <sub>(+5.72%)</sub></td>
  </tr>
  <tr>
    <td><b>F1 Score</b></td>
    <td>0.786 → <b>0.831</b> <sub>(+5.75%)</sub></td>
    <td>0.846 → <b>0.868</b> <sub>(+2.60%)</sub></td>
    <td>0.797 → <b>0.856</b> <sub>(+7.38%)</sub></td>
    <td>0.632 → <b>0.682</b> <sub>(+7.93%)</sub></td>
    <td>0.849 → <b>0.900</b> <sub>(+6.05%)</sub></td>
  </tr>
  <tr>
    <td colspan="6" align="center"><b>🔍 Size-Specific mAP50</b></td>
  </tr>
  <tr>
    <td>🔍 <b>Small</b></td>
    <td>0.530 → <b>0.562</b> <sub>(+6.06%)</sub></td>
    <td>0.679 → <b>0.729</b> <sub>(+7.32%)</sub></td>
    <td>0.426 → <b>0.469</b> <sub>(+10.25%)</sub></td>
    <td>0.230 → <b>0.244</b> <sub>(+6.05%)</sub></td>
    <td>0.779 → <b>0.803</b> <sub>(+3.03%)</sub></td>
  </tr>
  <tr>
    <td>📦 <b>Medium</b></td>
    <td>0.750 → <b>0.780</b> <sub>(+3.99%)</sub></td>
    <td>0.819 → <b>0.842</b> <sub>(+2.79%)</sub></td>
    <td>0.782 → <b>0.829</b> <sub>(+5.93%)</sub></td>
    <td>0.542 → <b>0.574</b> <sub>(+5.83%)</sub></td>
    <td>0.849 → <b>0.864</b> <sub>(+1.73%)</sub></td>
  </tr>
  <tr>
    <td>🟫 <b>Large</b></td>
    <td>0.828 → <b>0.854</b> <sub>(+3.13%)</sub></td>
    <td>0.879 → <b>0.910</b> <sub>(+3.58%)</sub></td>
    <td>0.859 → <b>0.903</b> <sub>(+5.11%)</sub></td>
    <td>0.642 → <b>0.669</b> <sub>(+4.10%)</sub></td>
    <td>0.922 → <b>0.924</b> <sub>(+0.13%)</sub></td>
  </tr>
  <tr>
    <td colspan="6" align="center"><b>🔍 Size-Specific mAP50-95</b></td>
  </tr>
  <tr>
    <td>🔍 <b>Small</b></td>
    <td>0.297 → <b>0.329</b> <sub>(+10.84%)</sub></td>
    <td>0.492 → <b>0.550</b> <sub>(+11.71%)</sub></td>
    <td>0.229 → <b>0.269</b> <sub>(+17.55%)</sub></td>
    <td>0.106 → <b>0.117</b> <sub>(+10.26%)</sub></td>
    <td>0.351 → <b>0.378</b> <sub>(+7.61%)</sub></td>
  </tr>
  <tr>
    <td>📦 <b>Medium</b></td>
    <td>0.424 → <b>0.458</b> <sub>(+8.03%)</sub></td>
    <td>0.551 → <b>0.610</b> <sub>(+10.64%)</sub></td>
    <td>0.416 → <b>0.456</b> <sub>(+9.79%)</sub></td>
    <td>0.237 → <b>0.257</b> <sub>(+8.58%)</sub></td>
    <td>0.484 → <b>0.502</b> <sub>(+3.86%)</sub></td>
  </tr>
  <tr>
    <td>🟫 <b>Large</b></td>
    <td>0.550 → <b>0.591</b> <sub>(+7.55%)</sub></td>
    <td>0.602 → <b>0.665</b> <sub>(+10.46%)</sub></td>
    <td>0.542 → <b>0.602</b> <sub>(+11.05%)</sub></td>
    <td>0.372 → <b>0.400</b> <sub>(+7.36%)</sub></td>
    <td>0.682 → <b>0.694</b> <sub>(+1.67%)</sub></td>
  </tr>
</table>

</details>

---

<details>
<summary><b>📊 3. Results — Custom Architecture + Custom Loss (Run 2)</b></summary>

<br>

### 🏗️ Arch-6 + Custom Loss (A1+A2+A3.1+A4) — Full Dataset (130 epochs)

Original YOLOv12s (baseline) vs Custom Architecture + Custom Loss:

<p align="center">
<img width="2000" height="977" alt="image" src="https://github.com/user-attachments/assets/f6b87e8d-f3c2-440a-8072-b33e794c2e37" />
</p>

<table>
  <tr>
    <th align="center" rowspan="2">Metric</th>
    <th align="center" colspan="5">Original → Custom Arch + Loss (Δ Improvement)</th>
  </tr>
  <tr>
    <th align="center">OVERALL</th>
    <th align="center">🗡️ Knife</th>
    <th align="center">🎯 Long Gun</th>
    <th align="center">🚫 Other</th>
    <th align="center">🔫 Pistol</th>
  </tr>
  <tr>
    <td><b>mAP50</b></td>
    <td>0.816 → <b>0.865</b> <sub>(+6.09%)</sub></td>
    <td>0.875 → <b>0.925</b> <sub>(+5.75%)</sub></td>
    <td>0.840 → <b>0.917</b> <sub>(+9.15%)</sub></td>
    <td>0.637 → <b>0.682</b> <sub>(+7.17%)</sub></td>
    <td>0.911 → <b>0.937</b> <sub>(+2.84%)</sub></td>
  </tr>
  <tr>
    <td><b>mAP50-95</b></td>
    <td>0.525 → <b>0.579</b> <sub>(+10.41%)</sub></td>
    <td>0.614 → <b>0.686</b> <sub>(+11.77%)</sub></td>
    <td>0.535 → <b>0.605</b> <sub>(+13.15%)</sub></td>
    <td>0.346 → <b>0.383</b> <sub>(+10.71%)</sub></td>
    <td>0.605 → <b>0.644</b> <sub>(+6.44%)</sub></td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td>0.831 → <b>0.898</b> <sub>(+8.01%)</sub></td>
    <td>0.847 → <b>0.879</b> <sub>(+3.78%)</sub></td>
    <td>0.827 → <b>0.913</b> <sub>(+10.37%)</sub></td>
    <td>0.800 → <b>0.867</b> <sub>(+8.29%)</sub></td>
    <td>0.851 → <b>0.933</b> <sub>(+9.65%)</sub></td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td>0.746 → <b>0.819</b> <sub>(+9.88%)</sub></td>
    <td>0.845 → <b>0.910</b> <sub>(+7.78%)</sub></td>
    <td>0.769 → <b>0.853</b> <sub>(+10.94%)</sub></td>
    <td>0.522 → <b>0.592</b> <sub>(+13.25%)</sub></td>
    <td>0.847 → <b>0.923</b> <sub>(+8.95%)</sub></td>
  </tr>
  <tr>
    <td><b>F1 Score</b></td>
    <td>0.786 → <b>0.857</b> <sub>(+8.99%)</sub></td>
    <td>0.846 → <b>0.894</b> <sub>(+5.74%)</sub></td>
    <td>0.797 → <b>0.882</b> <sub>(+10.66%)</sub></td>
    <td>0.632 → <b>0.703</b> <sub>(+11.24%)</sub></td>
    <td>0.849 → <b>0.928</b> <sub>(+9.30%)</sub></td>
  </tr>
  <tr>
    <td colspan="6" align="center"><b>🔍 Size-Specific mAP50</b></td>
  </tr>
  <tr>
    <td>🔍 <b>Small</b></td>
    <td>0.530 → <b>0.579</b> <sub>(+9.31%)</sub></td>
    <td>0.679 → <b>0.751</b> <sub>(+10.60%)</sub></td>
    <td>0.426 → <b>0.484</b> <sub>(+13.63%)</sub></td>
    <td>0.230 → <b>0.251</b> <sub>(+9.30%)</sub></td>
    <td>0.779 → <b>0.827</b> <sub>(+6.19%)</sub></td>
  </tr>
  <tr>
    <td>📦 <b>Medium</b></td>
    <td>0.750 → <b>0.804</b> <sub>(+7.17%)</sub></td>
    <td>0.819 → <b>0.868</b> <sub>(+5.93%)</sub></td>
    <td>0.782 → <b>0.854</b> <sub>(+9.18%)</sub></td>
    <td>0.542 → <b>0.592</b> <sub>(+9.07%)</sub></td>
    <td>0.849 → <b>0.890</b> <sub>(+4.85%)</sub></td>
  </tr>
  <tr>
    <td>🟫 <b>Large</b></td>
    <td>0.828 → <b>0.880</b> <sub>(+6.29%)</sub></td>
    <td>0.879 → <b>0.938</b> <sub>(+6.75%)</sub></td>
    <td>0.859 → <b>0.930</b> <sub>(+8.33%)</sub></td>
    <td>0.642 → <b>0.689</b> <sub>(+7.29%)</sub></td>
    <td>0.922 → <b>0.952</b> <sub>(+3.19%)</sub></td>
  </tr>
  <tr>
    <td colspan="6" align="center"><b>🔍 Size-Specific mAP50-95</b></td>
  </tr>
  <tr>
    <td>🔍 <b>Small</b></td>
    <td>0.297 → <b>0.339</b> <sub>(+14.23%)</sub></td>
    <td>0.492 → <b>0.567</b> <sub>(+15.13%)</sub></td>
    <td>0.229 → <b>0.277</b> <sub>(+21.15%)</sub></td>
    <td>0.106 → <b>0.120</b> <sub>(+13.64%)</sub></td>
    <td>0.351 → <b>0.390</b> <sub>(+10.90%)</sub></td>
  </tr>
  <tr>
    <td>📦 <b>Medium</b></td>
    <td>0.424 → <b>0.472</b> <sub>(+11.34%)</sub></td>
    <td>0.551 → <b>0.629</b> <sub>(+14.03%)</sub></td>
    <td>0.416 → <b>0.470</b> <sub>(+13.15%)</sub></td>
    <td>0.237 → <b>0.265</b> <sub>(+11.90%)</sub></td>
    <td>0.484 → <b>0.518</b> <sub>(+7.04%)</sub></td>
  </tr>
  <tr>
    <td>🟫 <b>Large</b></td>
    <td>0.550 → <b>0.610</b> <sub>(+10.84%)</sub></td>
    <td>0.602 → <b>0.686</b> <sub>(+13.84%)</sub></td>
    <td>0.542 → <b>0.621</b> <sub>(+14.45%)</sub></td>
    <td>0.372 → <b>0.412</b> <sub>(+10.64%)</sub></td>
    <td>0.682 → <b>0.715</b> <sub>(+4.78%)</sub></td>
  </tr>
</table>

</details>

---

<details>
<summary><b>📈 4. Progressive Improvement — All YOLOv12s Configurations</b></summary>

<br>

### 📉 Training Metrics Comparison

<p align="center">
<img width="1477" height="1125" alt="image" src="https://github.com/user-attachments/assets/c942bae4-310a-4e1c-ac0e-d9454872bcfb" />
</p>

<p align="center"><sub>Training curves for all 4 YOLOv12s configurations across 130 epochs. Solid lines = EMA smoothed, dotted lines = raw values, ★ = best value per model.</sub></p>

**Key observations from the training curves:**

- 🏆 **V12s-Both** (Arch-6 + Custom Loss) consistently achieves the **highest values** across all 4 metrics
- 📉 **V12s-Loss** (Custom Loss only) shows **faster early convergence** than V12s-Arch, especially in Precision and Recall
- 🏗️ **V12s-Arch** (Custom Arch only) demonstrates **steadier improvement** in mAP50-95, reflecting the architecture's strength in spatial localization
- 📊 **Convergence patterns differ:** Custom Loss improves classification confidence early, while the P2 architecture provides gradual localization gains
- ⭐ All custom variants significantly outperform the **V12s-Orig** baseline throughout training
- 📈 **mAP50-95 shows the largest spread** between configurations, confirming this metric best captures the differences between approaches

---

### 🏆 Four-Way Comparison: Baseline → Custom Loss → Custom Arch → Custom Arch + Loss

<table>
  <tr>
    <th align="center" rowspan="2">Metric</th>
    <th align="center">🔷 Baseline</th>
    <th align="center">📉 Custom Loss</th>
    <th align="center">🏗️ Custom Arch</th>
    <th align="center">🏆 Custom Arch+Loss</th>
  </tr>
  <tr>
    <th align="center"><sub>Default Arch<br>Default Loss</sub></th>
    <th align="center"><sub>Default Arch<br>A1+A2+A3.1+A4</sub></th>
    <th align="center"><sub>Arch-6<br>Default Loss</sub></th>
    <th align="center"><sub>Arch-6<br>A1+A2+A3.1+A4</sub></th>
  </tr>
  <tr>
    <td><b>mAP50</b></td>
    <td align="center">0.816</td>
    <td align="center">0.857 <sub>(+5.04%)</sub></td>
    <td align="center">0.840 <sub>(+2.94%)</sub></td>
    <td align="center"><b>0.865</b> <sub>(+6.09%)</sub></td>
  </tr>
  <tr>
    <td><b>mAP50-95</b></td>
    <td align="center">0.525</td>
    <td align="center">0.574 <sub>(+9.32%)</sub></td>
    <td align="center">0.562 <sub>(+7.13%)</sub></td>
    <td align="center"><b>0.579</b> <sub>(+10.41%)</sub></td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td align="center">0.831</td>
    <td align="center">0.889 <sub>(+6.94%)</sub></td>
    <td align="center">0.871 <sub>(+4.80%)</sub></td>
    <td align="center"><b>0.898</b> <sub>(+8.01%)</sub></td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td align="center">0.746</td>
    <td align="center">0.811 <sub>(+8.79%)</sub></td>
    <td align="center">0.795 <sub>(+6.62%)</sub></td>
    <td align="center"><b>0.819</b> <sub>(+9.88%)</sub></td>
  </tr>
  <tr>
    <td><b>F1 Score</b></td>
    <td align="center">0.786</td>
    <td align="center">0.848 <sub>(+7.91%)</sub></td>
    <td align="center">0.831 <sub>(+5.75%)</sub></td>
    <td align="center"><b>0.857</b> <sub>(+8.99%)</sub></td>
  </tr>
  <tr>
    <td colspan="5" align="center"><b>🔍 Size-Specific mAP50</b></td>
  </tr>
  <tr>
    <td>🔍 <b>Small</b></td>
    <td align="center">0.530</td>
    <td align="center">0.574 <sub>(+8.22%)</sub></td>
    <td align="center">0.562 <sub>(+6.06%)</sub></td>
    <td align="center"><b>0.579</b> <sub>(+9.31%)</sub></td>
  </tr>
  <tr>
    <td>📦 <b>Medium</b></td>
    <td align="center">0.750</td>
    <td align="center">0.796 <sub>(+6.11%)</sub></td>
    <td align="center">0.780 <sub>(+3.99%)</sub></td>
    <td align="center"><b>0.804</b> <sub>(+7.17%)</sub></td>
  </tr>
  <tr>
    <td>🟫 <b>Large</b></td>
    <td align="center">0.828</td>
    <td align="center">0.871 <sub>(+5.24%)</sub></td>
    <td align="center">0.854 <sub>(+3.13%)</sub></td>
    <td align="center"><b>0.880</b> <sub>(+6.29%)</sub></td>
  </tr>
  <tr>
    <td colspan="5" align="center"><b>🔍 Size-Specific mAP50-95</b></td>
  </tr>
  <tr>
    <td>🔍 <b>Small</b></td>
    <td align="center">0.297</td>
    <td align="center">0.336 <sub>(+13.10%)</sub></td>
    <td align="center">0.329 <sub>(+10.84%)</sub></td>
    <td align="center"><b>0.339</b> <sub>(+14.23%)</sub></td>
  </tr>
  <tr>
    <td>📦 <b>Medium</b></td>
    <td align="center">0.424</td>
    <td align="center">0.467 <sub>(+10.23%)</sub></td>
    <td align="center">0.458 <sub>(+8.03%)</sub></td>
    <td align="center"><b>0.472</b> <sub>(+11.34%)</sub></td>
  </tr>
  <tr>
    <td>🟫 <b>Large</b></td>
    <td align="center">0.550</td>
    <td align="center">0.604 <sub>(+9.74%)</sub></td>
    <td align="center">0.591 <sub>(+7.55%)</sub></td>
    <td align="center"><b>0.610</b> <sub>(+10.84%)</sub></td>
  </tr>
</table>

---

### 📊 Per-Class Best Results (Custom Arch + Loss)

<table>
  <tr>
    <th align="center">Metric</th>
    <th align="center">🗡️ Knife</th>
    <th align="center">🎯 Long Gun</th>
    <th align="center">🚫 Other</th>
    <th align="center">🔫 Pistol</th>
  </tr>
  <tr>
    <td><b>mAP50</b></td>
    <td><b>0.925</b> <sub>(+5.75%)</sub></td>
    <td><b>0.917</b> <sub>(+9.15%)</sub></td>
    <td><b>0.682</b> <sub>(+7.17%)</sub></td>
    <td><b>0.937</b> <sub>(+2.84%)</sub></td>
  </tr>
  <tr>
    <td><b>mAP50-95</b></td>
    <td><b>0.686</b> <sub>(+11.77%)</sub></td>
    <td><b>0.605</b> <sub>(+13.15%)</sub></td>
    <td><b>0.383</b> <sub>(+10.71%)</sub></td>
    <td><b>0.644</b> <sub>(+6.44%)</sub></td>
  </tr>
  <tr>
    <td><b>F1 Score</b></td>
    <td><b>0.894</b> <sub>(+5.74%)</sub></td>
    <td><b>0.882</b> <sub>(+10.66%)</sub></td>
    <td><b>0.703</b> <sub>(+11.24%)</sub></td>
    <td><b>0.928</b> <sub>(+9.30%)</sub></td>
  </tr>
  <tr>
    <td>🔍 <b>Small mAP50-95</b></td>
    <td><b>0.567</b> <sub>(+15.13%)</sub></td>
    <td><b>0.277</b> <sub>(+21.15%)</sub></td>
    <td><b>0.120</b> <sub>(+13.64%)</sub></td>
    <td><b>0.390</b> <sub>(+10.90%)</sub></td>
  </tr>
</table>

> 🎯 **Biggest per-class gain:** Long Gun small-object mAP50-95 improved by **+21.15%** — the largest single improvement in the entire study.

</details>


---

<details>
<summary><b>📌 5. Key Findings</b></summary>

<br>

### 🔬 Architecture vs Loss — Contribution Analysis

<table>
  <tr>
    <th align="left">Metric</th>
    <th align="center">Custom Loss Only<br><sub>(vs Baseline)</sub></th>
    <th align="center">Custom Arch Only<br><sub>(vs Baseline)</sub></th>
    <th align="center">Combined<br><sub>(vs Baseline)</sub></th>
    <th align="center">Synergy<br><sub>(Combined vs Sum)</sub></th>
  </tr>
  <tr>
    <td><b>mAP50</b></td>
    <td align="center">+5.04%</td>
    <td align="center">+2.94%</td>
    <td align="center"><b>+6.09%</b></td>
    <td align="center"><sub>Near-additive</sub></td>
  </tr>
  <tr>
    <td><b>mAP50-95</b></td>
    <td align="center">+9.32%</td>
    <td align="center">+7.13%</td>
    <td align="center"><b>+10.41%</b></td>
    <td align="center"><sub>Partial overlap</sub></td>
  </tr>
  <tr>
    <td><b>Precision</b></td>
    <td align="center">+6.94%</td>
    <td align="center">+4.80%</td>
    <td align="center"><b>+8.01%</b></td>
    <td align="center"><sub>Partial overlap</sub></td>
  </tr>
  <tr>
    <td><b>Recall</b></td>
    <td align="center">+8.79%</td>
    <td align="center">+6.62%</td>
    <td align="center"><b>+9.88%</b></td>
    <td align="center"><sub>Partial overlap</sub></td>
  </tr>
  <tr>
    <td>🔍 <b>Small mAP50-95</b></td>
    <td align="center">+13.10%</td>
    <td align="center">+10.84%</td>
    <td align="center"><b>+14.23%</b></td>
    <td align="center"><sub>Partial overlap</sub></td>
  </tr>
</table>

---

### 💡 Observations

- 🏆 **Best Overall:** **Arch-6 + Custom Loss** achieves the highest performance across **all metrics** and **all object sizes**
- 📉 **Custom Loss > Custom Arch alone:** The custom loss provides larger overall improvements (+5.04% mAP50) than the custom architecture alone (+2.94% mAP50), but the architecture shines on localization metrics (mAP50-95)
- 🔄 **Complementary but overlapping:** Combining both yields the best results, but gains are **partially overlapping** — both modifications target small-object detection through different mechanisms
- 🏗️ **Architecture strength — Localization:** The P2 detection head primarily improves **spatial localization** (mAP50-95: +7.13%) by detecting at higher resolution
- 📉 **Loss strength — Classification:** The custom loss primarily improves **classification confidence** and **recall** through size-aware weighting and TAL tuning
- 🎯 **Long Gun benefits most:** Long gun detection improved by **+21.15%** (small mAP50-95) — likely because long guns frequently appear small in CCTV footage
- 🔫 **Pistol saturating:** Pistol already had the highest baseline mAP50 (0.911) and shows the smallest relative gains, suggesting near-ceiling performance
- 🚫 **Other (hard negatives) hardest:** The `no_weapon` class remains the most challenging but still improved by **+10.71%** (mAP50-95)

---

### 📋 Complete Training Investment (Updated)

<table>
  <tr>
    <th align="left">Phase</th>
    <th align="center">Experiments</th>
    <th align="center">Time</th>
  </tr>
  <tr>
    <td>🔷 YOLOv12s Baseline (full dataset, 100 epochs)</td>
    <td align="center">1</td>
    <td align="center">~11h</td>
  </tr>
  <tr>
    <td>🔬 Loss Ablation — Grid Search (17% dataset, 70 epochs)</td>
    <td align="center">180</td>
    <td align="center">~216h</td>
  </tr>
  <tr>
    <td>🔬 Loss Ablation — Combinations (17% dataset, 70 epochs)</td>
    <td align="center">26</td>
    <td align="center">~31h</td>
  </tr>
  <tr>
    <td>🔷 YOLOv12s Custom Loss (full dataset, 100 epochs)</td>
    <td align="center">1</td>
    <td align="center">~11h</td>
  </tr>
  <tr>
    <td>🏗️ Architecture Ablation (17% dataset, 130 epochs)</td>
    <td align="center">20</td>
    <td align="center">~64h</td>
  </tr>
  <tr>
    <td>🏗️ Arch-6 + Default Loss (full dataset, 130 epochs)</td>
    <td align="center">1</td>
    <td align="center">~20h</td>
  </tr>
  <tr>
    <td>🏆 Arch-6 + Custom Loss (full dataset, 130 epochs)</td>
    <td align="center">1</td>
    <td align="center">~20h</td>
  </tr>
  <tr>
    <td>🔶 YOLO26s Baseline (full dataset, 100 epochs)</td>
    <td align="center">1</td>
    <td align="center">~9h</td>
  </tr>
  <tr>
    <td>🔶 YOLO26s Individual Phases (17% dataset, 70 epochs)</td>
    <td align="center">4</td>
    <td align="center">~4h</td>
  </tr>
  <tr>
    <td>🔶 YOLO26s Combinations (17% dataset, 70 epochs)</td>
    <td align="center">11</td>
    <td align="center">~11h</td>
  </tr>
  <tr>
    <td>🔶 YOLO26s Custom Loss (full dataset, 100 epochs)</td>
    <td align="center">1</td>
    <td align="center">~9h</td>
  </tr>
  <tr style="background-color: #d4edda;">
    <td><b>📊 Grand Total</b></td>
    <td align="center"><b>247</b></td>
    <td align="center"><b>~406 hours (~16.9 days)</b></td>
  </tr>
</table>

</details>


</details>
