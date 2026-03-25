# YOLOv8 Traffic Vehicle Detection & Tracking

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于 Ultralytics YOLOv8 的交通路口车辆检测与跟踪系统，使用 UA-DETRAC 数据集训练，支持四种改进架构的对比实验。

Based on Ultralytics YOLOv8 for traffic intersection vehicle detection and tracking, trained on UA-DETRAC dataset with 4 architecture variants.

## 📸 Demo

![Detection Result](picture/mAP_Overall_Comparison.png)
![Tracking Result](picture/Tracking_MOTA_Comparison.png)

## ✨ Features

- **🚗 4-Class Vehicle Detection**: Car, Bus, Van, Truck
- **📊 4 Architecture Variants**: Baseline, CBAM, BiFPN, Combined
- **🎯 Multi-Object Tracking**: ByteTrack integration
- **📈 Complete Visualization**: Training curves, confusion matrix, comparison charts

## 🏗️ Architecture Comparison

| Experiment | Config | Improvement | Layers | mAP50 | mAP50-95 |
|------------|--------|-------------|--------|-------|----------|
| A | `expA_baseline.yaml` | YOLOv8n (PANet) | 23 | - | - |
| B | `expB_cbam.yaml` | + CBAM Attention | 26 | - | - |
| C | `expC_bifpn.yaml` | + BiFPN | 24 | - | - |
| D | `expD_combined.yaml` | + CBAM + BiFPN | 27 | - | - |

*Replace `-` with your actual results after training*

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with ≥6GB VRAM (RTX 3060 or better)
- **Python**: 3.9 or 3.10
- **PyTorch**: ≥2.0.0 with CUDA 11.8+

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/yolo-vehicle-tracking.git
cd yolo-vehicle-tracking

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. Download UA-DETRAC dataset from [official website](http://detrac-db.rit.albany.edu/)
2. Convert to YOLO format
3. Place in `data/UA-DETRAC-G2/`

```
data/UA-DETRAC-G2/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
└── labels/
    ├── train/          # YOLO format labels
    └── val/            # YOLO format labels
```

### Training

```bash
# Experiment A: Baseline
python scripts/train.py --config configs/expA_baseline.yaml --name expA_baseline

# Experiment B: CBAM Attention
python scripts/train.py --config configs/expB_cbam.yaml --name expB_cbam

# Experiment C: BiFPN
python scripts/train.py --config configs/expC_bifpn.yaml --name expC_bifpn

# Experiment D: Combined
python scripts/train.py --config configs/expD_combined.yaml --name expD_combined
```

### Detection

```bash
# Detect on image
python scripts/detect.py --source path/to/image.jpg --model runs/train/expA_baseline/weights/best.pt

# Detect on video
python scripts/detect.py --source video.mp4 --model runs/train/expA_baseline/weights/best.pt --save-vid

# Real-time with camera
python scripts/detect.py --source 0 --model runs/train/expA_baseline/weights/best.pt --show
```

### Tracking

```bash
# Track video with ByteTrack
python scripts/track.py --source video.mp4 --model runs/train/expA_baseline/weights/best.pt \
    --tracker configs/track.yaml --save-vid --save-txt

# Real-time tracking
python scripts/track.py --source 0 --model runs/train/expA_baseline/weights/best.pt \
    --tracker configs/track.yaml --show --show-history
```

## 📁 Project Structure

```
yolo-vehicle-tracking/
├── configs/                    # Configuration files
│   ├── UA-DETRAC.yaml         # Dataset config
│   ├── train.yaml             # Training parameters
│   ├── track.yaml             # ByteTrack config
│   ├── expA_baseline.yaml     # Experiment A: Baseline
│   ├── expB_cbam.yaml         # Experiment B: CBAM
│   ├── expC_bifpn.yaml        # Experiment C: BiFPN
│   └── expD_combined.yaml     # Experiment D: Combined
│
├── scripts/                    # Core scripts
│   ├── train.py               # Training script
│   ├── detect.py              # Detection script
│   ├── track.py               # Tracking script
│   ├── utils.py               # Utility functions
│   ├── extract_metrics.py     # Metrics extraction
│   └── convert_to_mot.py      # YOLO to MOT format
│
├── data/                       # Dataset directory (not included)
│   └── UA-DETRAC-G2/
│
├── runs/                       # Training outputs (auto-generated)
│   └── train/
│
├── metrics/                    # Evaluation metrics
├── picture/                    # Result visualizations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📊 Results Analysis

```bash
# Extract metrics after training
python scripts/extract_metrics.py --experiment A
python scripts/extract_metrics.py --experiment B
python scripts/extract_metrics.py --experiment C
python scripts/extract_metrics.py --experiment D

# Generate comparison charts
python plot_results.py
```

Generated charts:
- `results_comparison.png` - mAP50 vs mAP50-95 comparison
- `results_comparison_classes.png` - Per-class AP comparison

## 🔬 Technical Details

### CBAM (Convolutional Block Attention Module)
- **Channel Attention**: Learns channel importance via global pooling
- **Spatial Attention**: Learns spatial focus regions
- **Benefit**: Enhances vehicle feature expression, suppresses background

### BiFPN (Bidirectional Feature Pyramid Network)
- **Top-down**: High-level features pass semantic info to low levels
- **Bottom-up**: Low-level features pass location info to high levels
- **Benefit**: Improves multi-scale detection (small/large vehicles)

### ByteTrack
- Kalman filter + Hungarian matching for detection association
- Uses both high and low score boxes
- Output format: `frame_id, track_id, x, y, w, h, score, class`

## 📝 Class Definitions

| ID | English | Chinese |
|----|---------|---------|
| 0 | car | 小汽车 |
| 1 | bus | 巴士 |
| 2 | van | 货车 |
| 3 | truck | 卡车 |

## ⚠️ Important Notes

- **GPU Required**: Training on CPU is ~20-50x slower
- Use `--device 0` for GPU, reduce `--batch` if OOM
- All scripts support relative paths and Chinese paths
- Run from project root directory for best results

## 📄 License

This project is for educational and research purposes only.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [UA-DETRAC Dataset](http://detrac-db.rit.albany.edu/)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [CBAM Paper](https://arxiv.org/abs/1807.06521)
- [BiFPN Paper](https://arxiv.org/abs/1911.09070)
