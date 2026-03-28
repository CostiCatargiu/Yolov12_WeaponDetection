# 🔫 Small-Object Weapon Detection with Custom YOLOv12s & YOLOv11s

**Custom architecture and loss modifications for enhanced small-object weapon detection**

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
- 📊 **Transfer validation** to **YOLOv11s** using the best-performing configurations from YOLOv12s

### 🔬 Research Contributions

| Contribution | Description |
|--------------|-------------|
| 🏗️ **P2–P5 Architecture** | Added high-resolution **P2 head (stride 1/4)** for better small object feature extraction |
| 📉 **Custom Loss Function** | Size-aware box weighting, auxiliary center L1 loss, epoch-scheduled clipping |
| 🎯 **Tuned Assigner** | `topk=25`, `β=4.0` for increased positives and softer gating |
| 🔍 **Ablation Study** | Extensive search across loss weights, architecture mods, and training configs |
| 📊 **Cross-Model Transfer** | Applied best YOLOv12s configs to YOLOv11s to measure generalization |

---

## ⚡ At a Glance

<table>
  <tr>
    <td><b>🖼️ Images</b></td>
    <td><b>59,305</b></td>
  </tr>
  <tr>
    <td><b>🔢 Instances</b></td>
    <td><b>76,705</b> (0 empty labels)</td>
  </tr>
  <tr>
    <td><b>🏷️ Classes</b></td>
    <td><code>knife</code>, <code>long_gun</code>, <code>no_weapon</code>, <code>pistol</code></td>
  </tr>
  <tr>
    <td><b>🧰 Format</b></td>
    <td>YOLO: <code>class x_center y_center width height</code> (normalized)</td>
  </tr>
  <tr>
    <td><b>📜 License</b></td>
    <td>MIT</td>
  </tr>
  <tr>
    <td><b>☁️ Hosting</b></td>
    <td>
      <a href="https://universe.roboflow.com/gundetectiondataset/nogun/dataset/2">Roboflow Universe: NoGun</a> &nbsp;•&nbsp;
      <a href="https://app.roboflow.com/gundetectiondataset/weapondataset-oi2g3/8">Roboflow App: WeaponDataset v8</a>
    </td>
  </tr>
  <tr>
    <td><b>📦 Training Results</b></td>
    <td>
      <a href="https://drive.google.com/drive/folders/1TECu5MI4lv36sJH50WSmS4iBd8SuhYgF?usp=sharing">Google Drive – Original Model</a> &nbsp;•&nbsp;
      <a href="https://drive.google.com/drive/folders/12aaS7CwZfGqb7__BK1UX54j1gQS_DoPi?usp=sharing">Google Drive – Custom Model</a>
    </td>
  </tr>
</table>

---

## 🏷️ Classes

| Class | Description |
|-------|-------------|
| 🗡️ `knife` | Bladed weapons including knives and similar objects |
| 🔫 `pistol` | Handguns and short firearms |
| 🎯 `long_gun` | Rifles, shotguns, and other long-barreled firearms |
| 🚫 `no_weapon` | Hard negatives (phones, tools, umbrellas, etc.) |

---

## ❓ Why Include `no_weapon`?

The `no_weapon` class serves as **hard negatives** — visually similar objects frequently misclassified as weapons:

- ✅ **Reduces false positives** in production
- ✅ **Improves precision** in crowded scenes
- ✅ **Distinguishes weapons** from everyday objects (📱 phones, 🔧 tools, ☂️ umbrellas, 📷 camera equipment)

---

## 📊 Dataset Summary

| Split | Images | % of Images | Instances | knife | long_gun | no_weapon | pistol |
|-------|-------:|------------:|----------:|------:|---------:|----------:|-------:|
| Train | 49,079 | 82.76% | 63,452 | 10,511 **(16.57%)** | 19,273 **(30.37%)** | 10,161 **(16.01%)** | 23,507 **(37.05%)** |
| Valid | 7,552 | 12.73% | 9,730 | 1,813 **(18.63%)** | 2,750 **(28.26%)** | 1,324 **(13.61%)** | 3,843 **(39.50%)** |
| Test | 2,674 | 4.51% | 3,523 | 686 **(19.47%)** | 941 **(26.71%)** | 656 **(18.62%)** | 1,240 **(35.20%)** |
| **Total** | **59,305** | **100%** | **76,705** | **13,010 (16.96%)** | **22,964 (29.94%)** | **12,141 (15.83%)** | **28,590 (37.27%)** |

---

## 📐 Annotation Size Analysis

Objects categorized by **normalized bounding box area** (`w × h`):

| Size Category | Area Threshold | Train | Valid | Test | Total | % |
|---------------|----------------|------:|------:|-----:|------:|--:|
| 🔍 **Small** | ≤ 0.02 | 15,548 | 2,821 | 966 | 19,335 | **25.2%** |
| 📦 **Medium** | 0.02 – 0.20 | 33,032 | 5,135 | 1,869 | 40,036 | **52.2%** |
| 🟫 **Large** | > 0.20 | 14,872 | 1,774 | 688 | 17,334 | **22.6%** |

> 📌 **~25% small objects** ensures sufficient representation for small-object detection training.

---

## 🔳 Size Previews

<p align="center">
  <img src="https://github.com/user-attachments/assets/2f4dd189-1ba0-4340-a101-2c55e42b27e5" alt="mosaic_small" width="30%">
  <img src="https://github.com/user-attachments/assets/727e3de5-209c-4698-a6f0-bf717d19d22a" alt="mosaic_medium" width="30%">
  <img src="https://github.com/user-attachments/assets/8f45f604-baac-4d59-bfcf-20da3475eed3" alt="mosaic_large" width="30%">
</p>
<p align="center"><sub>Small | Medium | Large</sub></p>

---

## 🧪 Split Distribution

<p align="left">
  <img src="https://img.shields.io/badge/Train-82.76%25-228be6?style=flat-square&labelColor=111827" />
  <img src="https://img.shields.io/badge/Valid-12.73%25-845ef7?style=flat-square&labelColor=111827" />
  <img src="https://img.shields.io/badge/Test-4.51%25-15aabf?style=flat-square&labelColor=111827" />
</p>
