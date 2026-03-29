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
  <img src="https://img.shields.io/badge/Model-YOLOv11s-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Focus-Small_Object_Detection-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Public-brightgreen?style=flat-square" />
</p>

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

## 🔬 Ablation Study Results

<details>
<summary><b>📊 Click to expand Hyperparameter Search Results</b></summary>

<br>

### 🧪 Experiment Summary

We conducted a **comprehensive ablation study** across **195 experiments** over approximately **9.1 days** of training time to identify optimal hyperparameters for small-object weapon detection.

<table>
  <tr>
    <th align="left">🧪 Experiment</th>
    <th align="left">⚙️ Parameter</th>
    <th align="left">🔢 Values Tested</th>
    <th align="center">📈 Combinations<br><sub>(Valid / Invalid / Total)</sub></th>
    <th align="center">⏱️ Time<br><sub>(Per Run / Total)</sub></th>
    <th align="left">✅ Optimal Values</th>
  </tr>
  <tr>
    <td rowspan="3"><b>1. Alpha Scheduling</b></td>
    <td><code>α_start</code></td>
    <td>0.3, 0.4, 0.5, 0.6, 0.7, 0.8</td>
    <td rowspan="3" align="center"><b>30</b> / 6 / 36</td>
    <td rowspan="3" align="center">~1.2h / ~36h</td>
    <td><code>α_start = 0.9</code></td>
  </tr>
  <tr>
    <td><code>α_end</code></td>
    <td>0.5, 0.6, 0.7, 0.8, 0.9, 1.0</td>
    <td><code>α_end = 0.4</code></td>
  </tr>
  <tr>
    <td><code>small_obj_ps</code></td>
    <td>0, 5</td>
    <td><code>small_obj_ps = 32</code></td>
  </tr>
  <tr>
    <td rowspan="2"><b>2. Center Loss Weight</b></td>
    <td><code>Loss_min</code></td>
    <td>0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10</td>
    <td rowspan="2" align="center"><b>32</b> / 4 / 36</td>
    <td rowspan="2" align="center">~1.2h / ~39h</td>
    <td><code>Loss_min = 0.01</code></td>
  </tr>
  <tr>
    <td><code>Loss_max</code></td>
    <td>0.01, 0.02, 0.03, 0.05, 0.07, 0.10</td>
    <td><code>Loss_max = 0.05</code></td>
  </tr>
  <tr>
    <td rowspan="2"><b>3. IoU Clipping</b></td>
    <td><code>IoU_start</code></td>
    <td>2, 3, 4, 5, 6, 8</td>
    <td rowspan="2" align="center"><b>35</b> / 1 / 36</td>
    <td rowspan="2" align="center">~1.2h / ~42h</td>
    <td><code>IoU_start = 6</code></td>
  </tr>
  <tr>
    <td><code>IoU_end</code></td>
    <td>6, 8, 10, 12, 15, 20</td>
    <td><code>IoU_end = 2</code></td>
  </tr>
  <tr>
    <td rowspan="2"><b>4. DFL Clipping</b></td>
    <td><code>DFL_start</code></td>
    <td>2, 3, 4, 5, 6, 8</td>
    <td rowspan="2" align="center"><b>35</b> / 1 / 36</td>
    <td rowspan="2" align="center">~1.2h / ~42h</td>
    <td><code>DFL_start = 8</code></td>
  </tr>
  <tr>
    <td><code>DFL_end</code></td>
    <td>6, 8, 10, 12, 15, 20</td>
    <td><code>DFL_end = 5</code></td>
  </tr>
  <tr>
    <td rowspan="3"><b>5. TAL Alpha-Beta</b></td>
    <td><code>Alpha (α)</code></td>
    <td>0.25, 0.5, 0.75, 1.0</td>
    <td rowspan="3" align="center"><b>51</b> / 0 / 51</td>
    <td rowspan="3" align="center">~1.2h / ~60h</td>
    <td><code>Alpha = 1</code></td>
  </tr>
  <tr>
    <td><code>Beta</code></td>
    <td>—</td>
    <td><code>Beta = 7</code></td>
  </tr>
  <tr>
    <td><code>Topk</code></td>
    <td>4, 5, 6, 8, 10, 12, 15, 20, 25</td>
    <td><code>Topk = 20</code></td>
  </tr>
  <tr style="background-color: #f0f0f0;">
    <td><b>📊 Overall Summary</b></td>
    <td>—</td>
    <td>—</td>
    <td align="center"><b>183 / 12 / 195</b></td>
    <td align="center"><b>~1.2h / ~219h</b><br><sub>(~9.1 days)</sub></td>
    <td>—</td>
  </tr>
</table>

### 📌 Key Takeaways

- ⏱️ **Total Training Time:** ~219 hours (**~9.1 days**) across **195 experiments**
- ✅ **Valid Configurations:** 183 out of 195 (**93.8% success rate**)
- 🎯 **Most Impactful:** TAL Alpha-Beta tuning with **51 experiments** showed significant impact on small-object recall
- 📉 **Alpha Scheduling:** Annealing from `α_start=0.9` to `α_end=0.4` prioritizes small objects early in training
- 🔧 **Loss Clipping:** IoU and DFL clipping stabilizes training on dense small-object scenes

</details>

### 🔬 Research Contributions

| Contribution | Description |
|--------------|-------------|
| 🏗️ **P2–P5 Architecture** | Added high-resolution **P2 head (stride 1/4)** for better small object feature extraction |
| 📉 **Custom Loss Function** | Size-aware box weighting, auxiliary center L1 loss, epoch-scheduled clipping |
| 🎯 **Tuned Assigner** | `topk=25`, `β=4.0` for increased positives and softer gating |
| 🔍 **Ablation Study** | Extensive search across loss weights, architecture mods, and training configs |
| 📊 **Cross-Model Transfer** | Applied best YOLOv12s configs to YOLOv11s to measure generalization |

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

<details>
<summary><b>📁 Dataset Structure & Distribution</b></summary>

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

</details>

---

<details>
<summary><b>📐 Annotation Size Analysis</b></summary>

<br>

To better understand the dataset composition, we analyzed the **normalized bounding box areas** (`width × height` in YOLO format) and categorized objects into three size groups:

- 🔍 **Small** — area ≤ `0.02` (tiny objects, hardest to detect)
- 📦 **Medium** — `0.02 < area ≤ 0.20` (standard-sized objects)
- 🟫 **Large** — area > `0.20` (large, easier to detect)

### 📈 Key Findings

**Medium-sized objects dominate** the dataset, making up approximately **52%** of all annotations across all splits. This provides a solid foundation for general object detection.

**Small objects represent ~25%** of the dataset (**19,335 instances**), ensuring sufficient representation for small-object detection — the primary focus of this research. The **Train split** contains **15,548 small objects (24.5%)**, while **Valid** and **Test** splits have slightly higher proportions (**29.0%** and **27.4%** respectively).

**Large objects are less frequent** at around **22%** (**17,334 instances**), but remain important for maintaining scale robustness and preventing the model from overfitting to small targets only.

### ⚖️ Balance & Consistency

The proportions are **remarkably consistent across train/valid/test splits**, indicating a well-balanced dataset that should generalize effectively. The slightly higher percentage of small objects in the validation and test sets provides a more challenging evaluation scenario, which is ideal for assessing real-world performance.

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

