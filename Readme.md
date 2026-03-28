
---

## 🏗️ Architecture: YOLOv12s P2–P5 vs Baseline P3–P5

| Aspect | P2–P5 (Custom) | P3–P5 (Baseline) | Why It Helps |
|--------|----------------|------------------|--------------|
| **Output Strides** | P2 (1/4), P3 (1/8), P4 (1/16), P5 (1/32) | P3–P5 only | High-res P2 grid covers very small boxes |
| **P2 Pathway** | Upsample + concat with early C3k2 | — | Preserves fine edges/texture |
| **Head Width** | 256-ch Conv/A2C2f | 192–256 | Extra capacity for minute cues |
| **# Detect Heads** | 4 (P2–P5) | 3 (P3–P5) | More scale-specialization |
| **AP_S (tiny)** | ↑ Higher | Lower | Denser supervision at small strides |
| **Best For** | Small, dense, crowded datasets | Medium/large objects | Matches receptive field to object size |

---

## 📉 Loss Function Modifications

| Aspect | Modified (Custom) | Baseline | Effect on Tiny Objects |
|--------|-------------------|----------|------------------------|
| **Assigner** | `topk=25`, `β=4.0` | `topk=10`, `β=6.0` | More positives & softer gate → higher recall |
| **Box Weighting** | Inverse-area × α + score × (1-α), α anneals | Score-only | Prioritizes small boxes early |
| **Center Aux Loss** | L1 on centers for small GT (decay) | — | Faster center alignment for tiny `W×H` |
| **Loss Clipping** | Epoch-scheduled caps for IoU/DFL | — | Tames spikes in crowded scenes |
| **DFL** | Size-aware weighting | Standard | Stabilizes edge bins for small boxes |
| **CLS Loss (Detect)** | BCE (option: Focal) | BCE | Use Focal if small-class recall is low |
| **CLS Loss (OBB)** | Focal | BCE | Better imbalance handling |

---

## 📊 Results: Custom vs Original YOLOv12s

### 🎯 Per-Class Detection Performance

| Class | Original (Detections / Avg Conf) | Custom (Detections / Avg Conf) | Winner |
|-------|----------------------------------|--------------------------------|--------|
| 🗡️ Knife | 525 / 0.746 | **593 / 0.786** | ✅ Custom |
| 🎯 Long Gun | 698 / 0.746 | **814 / 0.781** | ✅ Custom |
| 🚫 No Weapon | 245 / 0.731 | **386 / 0.740** | ✅ Custom |
| 🔫 Pistol | 936 / 0.761 | **1111 / 0.787** | ✅ Custom |

### 📈 Detection Accuracy (TP/FP/FN, P/R, F1)

| Class | Original F1 | Custom F1 | Δ |
|-------|-------------|-----------|---|
| 🗡️ Knife | 0.805 | **0.846** | +0.041 |
| 🎯 Long Gun | 0.792 | **0.850** | +0.058 |
| 🚫 No Weapon | 0.463 | **0.579** | +0.116 |
| 🔫 Pistol | 0.809 | **0.867** | +0.058 |

### 📊 Overall Metrics

| Metric | Original | Custom | Δ |
|--------|----------|--------|---|
| **Precision** | 0.925 | 0.897 | -0.028 |
| **Recall** | 0.631 | **0.740** | **+0.109** |
| **F1 Score** | 0.750 | **0.811** | **+0.061** |
| **False Negatives** | 1,299 | **917** | **-382** |
| **Avg Confidence** | 0.750 | **0.779** | +0.029 |

---

## ✅ Key Findings

| Finding | Impact |
|---------|--------|
| 📈 **Recall +11 pts** | Catches significantly more small objects |
| 🎯 **F1 +6 pts** | Better overall balance |
| 🚫 **No Weapon +8 pts** | Biggest improvement on hard negatives |
| ⚠️ **Precision -2.8 pts** | Acceptable trade-off for recall gains |
| 🔢 **FNs -382** | Fewer missed detections |

---

## 🔍 Detection Examples

<table>
<tr>
<td align="center" width="33%"><img src="https://github.com/user-attachments/assets/89bc6fd9-a52f-4620-8611-9f63bd392599" alt="7_compare" width="100%"><br><sub>Example 1</sub></td>
<td align="center" width="33%"><img src="https://github.com/user-attachments/assets/c50d8762-7e5f-40c3-9e4e-ad8754f5a5dd" alt="13_compare" width="100%"><br><sub>Example 2</sub></td>
<td align="center" width="33%"><img src="https://github.com/user-attachments/assets/d7f883cb-bb85-4432-ad97-285c2e27ddb8" alt="36_compare" width="100%"><br><sub>Example 3</sub></td>
</tr>
<tr>
<td align="center" width="33%"><img src="https://github.com/user-attachments/assets/e2167e20-c0a4-4f4b-8bf9-a8ad4ec9c6fb" alt="68_compare" width="100%"><br><sub>Example 4</sub></td>
<td align="center" width="33%"><img src="https://github.com/user-attachments/assets/a215bab3-9899-4ae2-a6e6-cb5a4e9641a3" alt="127_compare" width="100%"><br><sub>Example 5</sub></td>
<td align="center" width="33%"><img src="https://github.com/user-attachments/assets/cb07b5fc-f636-4d43-b737-d0ea129045cd" alt="128_compare" width="100%"><br><sub>Example 6</sub></td>
</tr>
<tr>
<td align="center" width="33%"><img src="https://github.com/user-attachments/assets/24eb0e7d-72d3-4af7-b3fd-9597424a804d" alt="132_compare" width="100%"><br><sub>Example 7</sub></td>
<td align="center" width="33%"><img src="https://github.com/user-attachments/assets/29c1e79b-d06c-427b-b96e-e531b64c3282" alt="151_compare" width="100%"><br><sub>Example 8</sub></td>
<td align="center" width="33%"><img src="https://github.com/user-attachments/assets/743218e8-fbd7-4b1b-8574-8a68bdc7baa1" alt="200_compare" width="100%"><br><sub>Example 9</sub></td>
</tr>
</table>

---

## 📈 Training Comparison

<table>
<tr>
<td align="center" width="50%">
<img src="https://github.com/user-attachments/assets/09326857-5418-4a62-836e-f9ec443327ff" alt="confusion_matrix_custom" width="100%" />
<br><sub>Confusion Matrix — Custom Model</sub>
</td>
<td align="center" width="50%">
<img src="https://github.com/user-attachments/assets/6faf2f7b-6679-4270-a389-0ac0c3e96a78" alt="confusion_matrix_original" width="100%" />
<br><sub>Confusion Matrix — Original Model</sub>
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="https://github.com/user-attachments/assets/74b5d9eb-638f-4360-a07d-d66cab2dfbbb" alt="training_custom" width="100%" />
<br><sub>Training Results — Custom Model</sub>
</td>
<td align="center" width="50%">
<img src="https://github.com/user-attachments/assets/d064675b-c23a-4aea-a71a-dfcadb355f8f" alt="training_original" width="100%" />
<br><sub>Training Results — Original Model</sub>
</td>
</tr>
</table>

---

## 🚀 Quick Start

### 📦 Installation

pip install roboflow ultralytics

### ⬇️ Download Dataset

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("gundetectiondataset").project("weapondataset-oi2g3")
dataset = project.version(8).download("yolov8")

### 🏋️ Train Custom YOLOv12s

yolo detect train data=path/to/data.yaml model=yolov12s-p2p5.yaml epochs=100 imgsz=640

### 🏋️ Train YOLOv11s with Best Config

yolo detect train data=path/to/data.yaml model=yolo11s-custom.yaml epochs=100 imgsz=640

---

## 💡 Applications

- 📹 **Surveillance** and security systems
- 🛡️ **Public safety** monitoring
- 🚪 **Access control** systems
- ⚠️ **Threat detection** in crowded environments

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 📚 Citation

If you use this dataset or methodology in your research, please cite:

@misc{weapondetection2024,
  title={Small-Object Weapon Detection with Custom YOLOv12s and YOLOv11s},
  author={Authors},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-repo}
}

---

## 🙏 Acknowledgments

- 🌐 Dataset hosted on [Roboflow Universe](https://universe.roboflow.com/)
- 🎥 Video sources from YouTube for diverse real-world scenarios
- 🧠 Built upon [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework

---

<p align="center">
  <b>🛡️ Advancing public safety through optimized small-object detection 🛡️</b>
</p>
