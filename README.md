# Unified RGB-UAV Pipeline for Citrus Tree Detection, Geotagging, and HLB Health Assessment

[![Paper](https://img.shields.io/badge/Paper-IGARSS%202026-blue)](PAPER_LINK)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of our IGARSS 2026 paper: **"Unified RGB-UAV Pipeline for Citrus Tree Detection, Geotagging, and HLB Health Assessment"**

---

## Overview

A complete pipeline for automated citrus orchard monitoring using consumer-grade UAV RGB imagery. The system performs:

1. **Tree Detection** — YOLOv11-Large ensemble (RGB + Brightness-Normalized + Greyscale)
2. **GPS Geotagging** — Direct coordinate projection without orthomosaic generation
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

### 1. Install Libraries

```python
pip install ultralytics torch torchvision timm opencv-python numpy pandas scipy scikit-learn matplotlib seaborn tqdm pillow exifread
```

### 2. Download Models and Datasets

Download all models and datasets from Google Drive:

📁 **[Download Link](DRIVE_LINK_PLACEHOLDER)**

After downloading, your folder structure should look like:

```
citrus-uav-pipeline/
├── models/
│   ├── detection/
│   │   ├── rgb_model.pt
│   │   └── greyscale_model.pt
│   └── health/
│       ├── poor_specialist.pth
│       ├── moderate_specialist.pth
│       └── good_specialist.pth
├── data/
│   ├── MAPIR_detection/
│   ├── USDA_health/
│   └── AUB_gps_validation/
└── notebooks/
    └── ...
```

---

## Project Structure

```
citrus-uav-pipeline/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── notebooks/
│   ├── 1_detection.ipynb           # Tree detection pipeline
│   ├── 2_gps_geotagging.ipynb      # GPS coordinate transformation
│   ├── 3_health_classification.ipynb   # Health assessment
│   └── 4_full_pipeline.ipynb       # End-to-end demo
│
├── models/                          # Download from Google Drive
│   ├── detection/
│   └── health/
│
├── data/                            # Download from Google Drive
│   ├── MAPIR_detection/
│   ├── USDA_health/
│   └── AUB_gps_validation/
│
├── figures/
│   └── pipeline.png
│
└── outputs/                         # Generated results
```

---

## Usage

### 1. Tree Detection

Open `notebooks/1_detection.ipynb`

Set your paths in the configuration cell:

```python
# === CONFIGURATION ===
RGB_MODEL_PATH = "models/detection/rgb_model.pt"
GREY_MODEL_PATH = "models/detection/greyscale_model.pt"
IMAGE_PATH = "data/MAPIR_detection/test/your_image.jpg"
OUTPUT_DIR = "outputs/detection/"
```

Run all cells. Outputs:
- Annotated images with bounding boxes
- Detection coordinates (CSV)

---

### 2. GPS Geotagging

Open `notebooks/2_gps_geotagging.ipynb`

Set your paths:

```python
# === CONFIGURATION ===
IMAGE_PATH = "data/AUB_gps_validation/pair1/image.jpg"
DETECTIONS_PATH = "outputs/detection/detections.csv"
OUTPUT_PATH = "outputs/gps/tree_coordinates.csv"

# Camera parameters (DJI Mini 4 Pro defaults)
FOV_DEGREES = 82.1
ALTITUDE_M = 100
```

Run all cells. Outputs:
- GPS coordinates for each detected tree (CSV)
- Validation error metrics

---

### 3. Health Classification

Open `notebooks/3_health_classification.ipynb`

Set your paths:

```python
# === CONFIGURATION ===
MODEL_DIR = "models/health/"
TREE_CROPS_DIR = "data/USDA_health/test/"
OUTPUT_PATH = "outputs/health/predictions.csv"
```

Run all cells. Outputs:
- Health predictions for each tree (Poor/Moderate/Good)
- Confidence scores
- Confusion matrix

---

### 4. Full Pipeline (End-to-End)

Open `notebooks/4_full_pipeline.ipynb`

Set your paths:

```python
# === CONFIGURATION ===
# Detection
RGB_MODEL_PATH = "models/detection/rgb_model.pt"
GREY_MODEL_PATH = "models/detection/greyscale_model.pt"

# Health
HEALTH_MODEL_DIR = "models/health/"

# Input/Output
INPUT_IMAGE = "data/your_orchard_image.jpg"
OUTPUT_DIR = "outputs/full_pipeline/"

# Camera parameters
FOV_DEGREES = 82.1
ALTITUDE_M = 100
```

Run all cells. Outputs:
- Detected trees with GPS coordinates and health scores
- Orchard health map visualization

---

## Training

### Train Detection Model

Open `notebooks/train_detection.ipynb`

```python
# === CONFIGURATION ===
TRAIN_DATA = "data/MAPIR_detection/train/"
VAL_DATA = "data/MAPIR_detection/val/"
OUTPUT_DIR = "models/detection/"

# Training parameters
EPOCHS = 200
BATCH_SIZE = 8
IMG_SIZE = 640
PATIENCE = 50  # Early stopping
```

### Train Health Classifier

Open `notebooks/train_health.ipynb`

```python
# === CONFIGURATION ===
TRAIN_DATA = "data/USDA_health/train/"
VAL_DATA = "data/USDA_health/val/"
OUTPUT_DIR = "models/health/"

# Training parameters
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 25  # Early stopping
```

---

## Datasets

### 1. MAPIR Detection Dataset

- **Source:** [MAPIR Open Dataset](https://www.mapir.camera/pages/open-dataset)
- **Contents:** 403 RGB images of citrus orchard at 120m AGL
- **Split:** 344 train/val, 59 test
- **Annotations:** YOLO format bounding boxes

### 2. USDA Health Dataset

- **Source:** [USDA Ag Data Commons](https://doi.org/10.15482/USDA.ADC/26946823)
- **Contents:** ~1,500 citrus trees with expert health ratings (1-5 scale)
- **Location:** Fort Pierce, FL (HLB-endemic region)
- **Classes:** Poor (1-2), Moderate (3), Good (4-5)

### 3. AUB GPS Validation Dataset

- **Contents:** 4 image pairs for geotagging validation
- **Equipment:** DJI Mini 4 Pro (82.1° FOV)
- **Altitude:** 100m AGL

---

## Model Weights

| Model | Architecture | Size | Download |
|-------|--------------|------|----------|
| RGB Detection | YOLOv11-Large | ~90 MB | [Google Drive](DRIVE_LINK_PLACEHOLDER) |
| Greyscale Detection | YOLOv11-Large | ~90 MB | [Google Drive](DRIVE_LINK_PLACEHOLDER) |
| Poor Specialist | Swin-Tiny | ~30 MB | [Google Drive](DRIVE_LINK_PLACEHOLDER) |
| Moderate Specialist | Swin-Tiny | ~30 MB | [Google Drive](DRIVE_LINK_PLACEHOLDER) |
| Good Specialist | Swin-Tiny | ~30 MB | [Google Drive](DRIVE_LINK_PLACEHOLDER) |

---

## Citation

If you use this code in your research, please cite:

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

- **Afif El Bsat** — American University of Beirut — ame127@mail.aub.edu
- **Dr. Ammar Mohanna** — American University of Beirut
- **Dr. Bilal Kaddouh** — American University of Beirut

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- MAPIR for the open citrus orchard dataset
- USDA Ag Data Commons for the Florida rootstock trials dataset
- Ultralytics for YOLOv11 implementation
- timm library for Swin Transformer implementation
