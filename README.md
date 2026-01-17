# Unified RGB-UAV Pipeline for Citrus Tree Detection, Geotagging, and HLB Health Assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation for the **2026 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)** paper.

> **Disclaimer:** This is experimental research code provided "as is" without any warranties. Users assume full responsibility for its use. Not intended for production or critical agricultural decisions without independent validation.

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
| **Detection- Greyscale Model** | Precision | 95.1% |
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

Download the complete pipeline from Google Drive:

📁 **[Google Drive Link](https://drive.google.com/drive/folders/1LJl63Lm9rNaNczlUzEI_x7zvZsnFcp8V?usp=drive_link)**

## Folder Structure
```
citrus-uav-pipeline/
│
├── Object Detection Pipeline/
│   ├── Models/
│   │   ├── orange_tree_model_350images/
│   │   │   └── tree_training/weights/best.pt
│   │   └── orange_tree_model_350images_greyscale/
│   │       └── tree_training/weights/best.pt
│   ├── train/                              # 310 images (640x640) + YOLO labels
│   ├── valid/                              # 34 images + YOLO labels
│   ├── test/                               # 59 images + YOLO labels
│   ├── data.yaml
│   ├── Object_detection_pipeline.ipynb
│   ├── yolo11l-seg.pt                      # Pretrained weights
│   ├── yolo11l.pt
│   └── yolo11n.pt
│
├── HLB Health Classification/
│   ├── Tree Rows 5 to 16-Processed/
│   │   ├── Rows R5-R16_Valencia trial_labeled.png
│   │   ├── Rows R5-R16_Valencia trial_labeled.txt
│   │   ├── Rows R5-R16_Valencia trial_labeled_mapping.txt
│   │   └── _annotations.coco.json
│   ├── Tree Rows 17 to 28-Processed/
│   │   ├── train/
│   │   ├── data.yaml
│   │   ├── canopy_index_from_coco17-28.csv
│   │   ├── Rows-R17-R28_Valencia-trial_labeled.jpg
│   │   └── _annotations.coco.json
│   ├── USDA Florida orthomosaics+CSV mapping (unprocessed)/
│   ├── Health_Monitoring.ipynb
│   ├── swin_3specialists_FINAL.pth
│   ├── yolo11l-seg.pt
│   ├── yolo11l.pt
│   └── yolo11n.pt
│
├── Geolocation/
│   ├── Pairs + EXIF Data/
│   │   ├── pair1/
│   │   │   ├── DJI_..._other.JPG
│   │   │   ├── DJI_..._other.txt
│   │   │   ├── DJI_..._ref.JPG
│   │   │   └── DJI_..._ref.txt
│   │   ├── pair2/
│   │   ├── pair3/
│   │   ├── pair4/
│   │   └── GIMBAL_SUMMARY_20251002_212839.txt
│   └── Geolocation_pipeline.ipynb
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Usage

> ⚠️ **Important:** Each notebook contains multiple code cells. You must set the `BASE_DIR` variable in **every code cell you run** to match your local folder path.

### 1. Tree Detection

Open `Object Detection Pipeline/Object_detection_pipeline.ipynb`
```python
# ==================== CONFIGURATION ====================
BASE_DIR = "Object Detection Pipeline"  # Update this path in every cell
```

**Included scripts:**
- Training
- RGB Model Evaluation
- Greyscale Model Evaluation  
- Brightness Normalized Evaluation
- Ensemble Evaluation
- Inference Examples (RGB, Greyscale, Brightness, Ensemble)

---

### 2. GPS Geotagging

Open `Geolocation/Geolocation_pipeline.ipynb`
```python
# ==================== CONFIGURATION ====================
BASE_DIR = "Geolocation"  # Update this path in every cell
```

**Features:**
- Gimbal-based north alignment
- GPS coordinate projection from pixel coordinates
- Validation visualization with error metrics

---

### 3. Health Classification

Open `HLB Health Classification/Health_Monitoring.ipynb`
```python
# ==================== CONFIGURATION ====================
BASE_DIR = "HLB Health Classification"  # Update this path in every cell
```

**Included scripts:**
- Data extraction from USDA orthomosaics
- Test set evaluation with confusion matrix
- Inference examples

---

## Data Description

### Object Detection Dataset

| Split | Images | Resolution | Labels |
|-------|--------|------------|--------|
| Train | 310 | 640×640 | YOLO format |
| Valid | 34 | 640×640 | YOLO format |
| Test | 59 | 640×640 | YOLO format |

For the complete MAPIR dataset (including OCN and RGN bands): [Orange Orchard - MAPIR CAMERA](https://www.mapir.camera/pages/orange-orchard)

---

### Health Classification Dataset

**Processed Data (included):**

| Folder | Contents |
|--------|----------|
| `Tree Rows 5 to 16-Processed/` | Orthomosaic + YOLO labels + mapping file |
| `Tree Rows 17 to 28-Processed/` | Orthomosaic + YOLO labels + canopy index CSV |

**Label Format:** `R##_T##_H#`

| Code | Meaning | Range |
|------|---------|-------|
| R## | Row number | 5-28 |
| T## | Tree number in row | 1 (bottom) to 55 (top) |
| H# | Health score | 0-5 (0 = not a tree) |

**Unprocessed Data (included):**
- `USDA Florida orthomosaics+CSV mapping (unprocessed)/` — Original orthomosaic images and health score CSVs

**Original Source:**

> R. P. Niedz and K. D. Bowman, "Image dataset: UAV images and ground data of one 'Bingo' mandarin and two 'Valencia' orange rootstock trials conducted in Florida," *Data in Brief*, vol. 58, p. 111206, 2025.

For flyover closeup images, visit the original dataset.

---

### GPS Geotagging Dataset

**Included in `Pairs + EXIF Data/`:**

Each pair folder contains:
| File | Description |
|------|-------------|
| `DJI_..._ref.JPG` | Reference image (tree centered) |
| `DJI_..._ref.txt` | YOLO label for reference image |
| `DJI_..._other.JPG` | Other image (tree off-center) |
| `DJI_..._other.txt` | YOLO label for other image |

**EXIF Data:**
- `GIMBAL_SUMMARY_20251002_212839.txt` — Contains gimbal yaw angles for north alignment

**Camera:** DJI Mini 4 Pro | **FOV:** 82.1° | **Altitude:** 100m AGL

---

## Model Weights

| Model | Location | Size |
|-------|----------|------|
| RGB Detection | `Object Detection Pipeline/Models/orange_tree_model_350images/tree_training/weights/best.pt` | ~90 MB |
| Greyscale Detection | `Object Detection Pipeline/Models/orange_tree_model_350images_greyscale/tree_training/weights/best.pt` | ~90 MB |
| Health Classifier | `HLB Health Classification/swin_3specialists_FINAL.pth` | ~90 MB |

**Pretrained YOLO weights (for training from scratch):**
- `yolo11l-seg.pt` — YOLOv11-Large segmentation
- `yolo11l.pt` — YOLOv11-Large detection
- `yolo11n.pt` — YOLOv11-Nano

**Health Classifier Training Notes:**
- Architecture: 3× Swin-Tiny binary specialists
- BCE pos_weight values (tuned via grid search):
  - Poor specialist: 3.0
  - Moderate specialist: 0.5
  - Good specialist: 5.3
- Inference uses argmax (no additional weighting)

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

- **Afif El Bsat** — American University of Beirut
- **Dr. Ammar Mohanna** — American University of Beirut  
- **Dr. Bilal Kaddouh** — American University of Beirut

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [MAPIR](https://www.mapir.camera/pages/orange-orchard) for the open citrus orchard dataset
- R. P. Niedz and K. D. Bowman for the [USDA Florida rootstock trials dataset](https://doi.org/10.1016/j.dib.2024.111206)
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11 implementation
- [timm](https://github.com/huggingface/pytorch-image-models) for Swin Transformer implementation
