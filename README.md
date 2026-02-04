# Ultralytics YOLO - Custom Training Repository

This repository is forked from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/tree/74b2b91e5c4420af47875801b8e13333ab7026ac) for custom object detection training.

---

## 📦 Local Pretrained Models

| Model File | Size | Type | Description |
|-----------|------|------|-------------|
| `pt/yolo26n.pt` | 3.2 MB | Base | YOLO26 Nano |
| `pt/yolov8n.pt` | 3.6 MB | Base | YOLOv8 Nano |
| `pt/yolo11n.pt` | 5.4 MB | Base | YOLO11 Nano |
| `pt/yolo11s.pt` | 18 MB | Base | YOLO11 Small |
| `pt/yolo11m.pt` | 39 MB | Base | YOLO11 Medium |
| **`pt/yolo11s-0.8-1.pt`** | **18 MB** | **Custom** | **Fine-tuned for seed_leaf & true_leaf detection** ⭐ |

---

## 🔖 Version Features

### **2025-02-04** - YOLO11s Custom Training

**Model:** `yolo11s-0.8-1.pt`

**Task:** Seed leaf and true leaf detection

**Dataset:** 
- Training: 156 images
- Validation: 15 images
- Classes: `seed_leaf`, `true_leaf`

**Performance Metrics (Validation Set):**

| Metric | Value | Note |
|--------|-------|------|
| **mAP@0.3** | **85.34%** 🎯 | **Target achieved!** |
| **mAP@0.4** | **83.54%** | +4.54% vs mAP@0.5 |
| **mAP@0.5** | **79.86%** | Standard YOLO metric |
| **mAP@0.6** | **74.08%** | -7.30% vs mAP@0.5 |
| **mAP@0.75** | **45.20%** | Strict standard |
| **Precision** | **76.49%** | Detection accuracy |
| **Recall** | **75.48%** | Coverage rate |

**Per-class AP@0.5:**
- `seed_leaf`: 76.99%
- `true_leaf`: 82.73%

**Model Generalization:**
- Validation mAP@0.5: 79.86%
- Training mAP@0.5: 78.21%
- Overfitting check: ✅ Pass (差异 < 2%)

---

**Last Updated:** 2025-02-04
