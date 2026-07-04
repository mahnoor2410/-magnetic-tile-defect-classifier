# Magnetic Tile Surface Defect Classifier

Computer vision prototype classifying magnetic tile images into 6 classes: `Blowhole`, `Break`, `Crack`, `Fray`, `Uneven` (defects) + `Free`. Built for the AI/ML Engineer Technical Skills Assessment (CV Focus).

---

## Overview

- **Dataset:** [Magnetic Tile Surface Defects](https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects) (Kaggle), ~1,344 images, imbalanced classes.
- **Split:** Stratified 70/15/15 train/val/test.
- **Imbalance handling:** class-weighted loss (not oversampling, to avoid memorizing duplicated minority images).
- **Metrics:** macro-F1 + per-class precision/recall (accuracy alone is misleading given the imbalance).

---

## Models

| Model | Role | Epochs |
|---|---|---|
| Baseline CNN | Scratch-trained reference point | 25 |
| **ResNet18** | Primary model — staged fine-tune (5 epochs frozen backbone, then `layer4`+head unfrozen) | 15 |

ResNet18 chosen over deeper nets for fast CPU inference and lower overfitting risk on this small dataset.

---

## Repository layout

| Path | Contains |
|---|---|
| `src/data/` | Dataset scanning, transforms, EDA |
| `src/models/` | Baseline CNN + ResNet18 architectures |
| `src/training/` | Training loop, checkpointing |
| `src/evaluation/` | Metrics, confusion matrix, error analysis |
| `src/inference/` | `DefectPredictor` script + FastAPI (`api.py`) |
| `config/config.yaml` | All paths/hyperparameters |

---

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place dataset under `data/raw/` (e.g. `data/raw/MT_Blowhole/Imgs/*.jpg`, similarly for other classes).

---

## Running

```bash
python -m src.data.analysis                          # EDA figures
python -m src.training.train --model baseline
python -m src.training.train --model resnet
python -m src.evaluation.evaluate --model resnet
python -m src.evaluation.error_analysis --model resnet
pytest tests/ -v
```

**API:**
```bash
uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/predict" -F "file=@data/raw/MT_Crack/Imgs/exp1_num_10.jpg;type=image/jpeg"
```

---

## Results

| | Accuracy | Macro-F1 |
|---|---|---|
| Baseline CNN | 47.0% | 0.173 |
| **ResNet18** | **85.6%** | **0.768** |

**Main error mode:** `Free ↔ Break` confusion (18 of 29 total errors) — likely subtle break-like textures on defect-free tiles. Next step would be higher-resolution crops or hard-negative mining on this pair.

**With more compute:** ResNet50/EfficientNet-B0, proper LR grid search, synthetic minority-class augmentation, k-fold CV.
