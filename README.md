# Unified RGB-UAV Pipeline for Citrus Tree Detection, Geotagging, and HLB Health Assessment

[![Paper](https://img.shields.io/badge/Paper-IGARSS%202026-blue)](PAPER_LINK)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)

Official implementation of our IGARSS 2026 paper: **"Unified RGB-UAV Pipeline for Citrus Tree Detection, Geotagging, and HLB Health Assessment"**

---

## Overview

A complete pipeline for automated citrus orchard monitoring using consumer-grade UAV RGB imagery:

1. **Tree Detection** — YOLOv11-Large ensemble (RGB + Brightness-Normalized + Greyscale)
2. **GPS Geotagging** — Direct coordinate projection using gimbal data
3. **Health Classification** — Three-specialist Swin Transformer ensemble (Poor/Moderate/Good)

---

## Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| **Detection** | Precision | 95.1% |
| | Recall | 93.3% |
| | mAP@50 | 93.1% |
| **GPS Geotagging** | Mean Error | 3.2 m |
| **Health Classification** | Accuracy | 81.0% |
| | Within ±1 Class | 100% |

---

## Installation
```bash
pip install ultralytics torch torchvision timm opencv-python numpy scipy scikit-learn pillow matplotlib pyyaml
```

---

## Download Models & Data

Download models and datasets from Google Drive:

📁 **[Google Drive Link](DRIVE_LINK_PLACEHOLDER)**

After downloading, your folder structure should look like:
```
citrus-uav-pipeline/
│
├── Object Detection Pipeline/
│   ├── Models/
│   │   ├── orange_tree_model_350images/
│   │   └── orange_tree_model_350images_greyscale/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── data.yaml
│   └── Object_detection_pipeline.ipynb
│
├── HLB Health Classification/
│   ├── Tree Rows 5 to 16-Processed/
│   ├── Tree Rows 17 to 28-Processed/
│   ├── swin_3specialists_FINAL.pth
│   └── Health_Monitoring.ipynb
│
├── Geolocation/
│   ├── Pairs + EXIF Data/
│   └── Geolocation_pipeline.ipynb
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Usage

### 1. Tree Detection

Open `Object Detection Pipeline/Object_detection_pipeline.ipynb`

Set your base directory:
```python
# ==================== CONFIGURATION ====================
BASE_DIR = "Object Detection Pipeline"
```

**Available scripts:**
- Training
- RGB Model Evaluation
- Greyscale Model Evaluation  
- Brightness Normalized Evaluation
- Ensemble Evaluation
- Inference Examples

---

### 2. GPS Geotagging

Open `Geolocation/Geolocation_pipeline.ipynb`

Set your base directory:
```python
# ==================== CONFIGURATION ====================
BASE_DIR = "Geolocation"
```

**Features:**
- Gimbal-based north alignment
- GPS coordinate projection
- Validation visualization

---

### 3. Health Classification

Open `HLB Health Classification/Health_Monitoring.ipynb`

Set your base directory:
```python
# ==================== CONFIGURATION ====================
BASE_DIR = "HLB Health Classification"
```

**Available scripts:**
- Data extraction from USDA dataset
- Test set evaluation
- Inference examples

---

## Model Weights

| Model | File | Size |
|-------|------|------|
| RGB Detection | `orange_tree_model_350images/tree_training/weights/best.pt` | ~90 MB |
| Greyscale Detection | `orange_tree_model_350images_greyscale/tree_training/weights/best.pt` | ~90 MB |
| Health Classifier | `swin_3specialists_FINAL.pth` | ~90 MB |

---

## Datasets

| Dataset | Source | Use |
|---------|--------|-----|
| MAPIR Citrus | [MAPIR Open Dataset](https://www.mapir.camera/pages/open-dataset) | Tree Detection |
| USDA Florida Rootstock | [USDA Ag Data Commons](https://doi.org/10.15482/USDA.ADC/26946823) | Health Classification |
| AUB Validation | Collected | GPS Geotagging |

---

## Citation
```bibtex
@inproceedings{elbsat2026citrus,
  title={Unified RGB-UAV Pipeline for Citrus Tree Detection, Geotagging, and HLB Health Assessment},
  author={El Bsat, Afif and Mohanna, Ammar and Kaddouh, Bilal},
  booktitle={IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year={2026},
  organization={IEEE}
}
```

---

## Authors

- **Afif El Bsat** — American University of Beirut
- **Dr. Ammar Mohanna** — American University of Beirut  
- **Dr. Bilal Kaddouh** — American University of Beirut

---

## License

MIT License — see [LICENSE](LICENSE) for details.
