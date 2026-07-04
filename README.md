# Magnetic Tile Surface Defect Classifier

An end-to-end computer vision project for classifying magnetic tile surface images into one of six categories — five defect types (`Blowhole`, `Break`, `Crack`, `Fray`, `Uneven`) plus `Free` (no defect) — built for the AI/ML Engineer Technical Skills Assessment (Computer Vision Focus).

---

## Problem Framing

Magnetic tile defect detection is a real-world industrial quality-control problem, where the goal is to classify tiles as defect-free or belonging to a specific defect category. Since missing a defect can directly impact product quality, evaluation focuses on **macro-F1 and per-class recall** rather than accuracy alone — a model that scores high accuracy by always predicting the majority `Free` class would be useless in practice.

I chose **image classification** (not segmentation) because it aligns well with the dataset and the 3-day timeline, and lets me focus effort on what the assessment actually rewards: clean data handling, well-justified model comparison, real error analysis, and a working inference API — rather than a shallow segmentation attempt.

---

## Dataset

**[Magnetic Tile Surface Defects Dataset](https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects)** (Kaggle) — ~1,344 images across 6 classes, genuinely imbalanced, near-grayscale JPEGs with inconsistent resolutions and aspect ratios.

**Setup:** download from Kaggle and extract into:
```
data/raw/
├── MT_Blowhole/Imgs/*.jpg
├── MT_Break/Imgs/*.jpg
├── MT_Crack/Imgs/*.jpg
├── MT_Fray/Imgs/*.jpg
├── MT_Uneven/Imgs/*.jpg
└── MT_Free/Imgs/*.jpg
```

---

## Preprocessing & Class Imbalance

- **Resize (not crop) to 224×224** — raw images vary in aspect ratio; cropping risks cutting off defects near tile edges.
- **ImageNet normalization**, applied to both models for consistency.
- **Augmentation (train only):** horizontal/vertical flip + ±15° rotation (defects have no fixed orientation on a tile) + mild brightness/contrast jitter (lighting varies across captures). No hue/saturation jitter — the images are effectively grayscale, so color augmentation would only add noise.
- **Imbalance handling:** inverse-frequency **class-weighted loss**, not oversampling — with only ~1,300 images, oversampling minority classes risks the model memorizing duplicated images.

---

## Models

| | Baseline CNN | ResNet18 (fine-tuned) |
|---|---|---|
| Architecture | 4 conv blocks + global average pool + FC head, trained from scratch | ImageNet-pretrained `torchvision.models.resnet18`, FC head replaced |
| Purpose | Reference point — quantifies the actual lift from transfer learning on this dataset | Primary model |
| Fine-tuning strategy | N/A (trained end-to-end from random init) | Staged: first 5 epochs train only the new FC head (backbone frozen), remaining 10 epochs unfreeze `layer4` + head at a low LR |
| Epochs | 25 | 15 |
| Why chosen | Small, fast, well-understood as a sanity-check model | ResNet18 (not ResNet50/deeper) for compute-awareness: fast to fine-tune, fast enough for CPU inference in production, and appropriately sized for a ~1,300-image dataset where a deeper network would overfit more easily |

**Basic hyperparameter tuning performed:** the freeze/unfreeze epoch split was chosen by comparing validation macro-F1 across a couple of schedules rather than fine-tuning end-to-end from epoch 1. All runs are logged and reproducible — see `outputs/logs/`.

**If more compute were available:** I would (1) try ResNet50/EfficientNet-B0 as a capacity upgrade, (2) run a proper grid/Bayesian search over LR and weight decay instead of a small manual comparison, (3) generate synthetic defect augmentations targeting the minority classes rather than relying solely on loss weighting, and (4) use k-fold cross-validation for a less split-dependent estimate of generalization given the dataset's small size.

---

## Evaluation Methodology

- Stratified 70/15/15 train/val/test split (fixed seed, preserves class proportions in every split).
- Primary metrics: **macro-F1** and **per-class precision/recall**, not raw accuracy, because of class imbalance.
- Model checkpointing during training uses **validation macro-F1**, not validation loss or accuracy.
- Confusion matrix generated for the test set (`outputs/figures/<model>_confusion_matrix.png`).

## Results

| | Accuracy | Macro-F1 |
|---|---|---|
| Baseline CNN | 47.0% | 0.173 |
| **ResNet18** | **85.6%** | **0.768** |

The baseline struggles heavily on minority classes (`Blowhole`, `Fray` — near-zero recall), while transfer learning gives a large, clear lift across almost every class.

## Error Analysis

`src/evaluation/error_analysis.py` extracts every misclassified test image, visualizes them in a labeled grid, and tabulates the most frequent `true_class -> predicted_class` confusion pairs (`outputs/metrics/<model>_error_analysis.json`).

**Finding:** the dominant failure mode is **`Free ↔ Break` confusion** (11 Free→Break, 7 Break→Free of 29 total test errors) — the model appears to pick up on subtle break-like surface textures on tiles that are actually defect-free. This is what turns "the model is 85.6% accurate" into an actionable next step: higher-resolution crops or hard-negative mining specifically on this class pair, rather than blind hyperparameter tweaking.

---

## Repository Structure

```
magnetic-tile-defect-classifier/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   └── raw/                       # dataset goes here (not committed to git)
├── src/
│   ├── config_loader.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── analysis.py
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   └── resnet_finetune.py
│   ├── training/
│   │   ├── train.py
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── error_analysis.py
│   └── inference/
│       ├── predictor.py
│       └── api.py
└── outputs/
    ├── checkpoints/
    ├── figures/
    ├── metrics/
    └── logs/
```

**Engineering principles applied:**
- Training, evaluation, and inference code are fully separated — `src/inference/predictor.py` only imports model architecture definitions, never the training loop.
- Every path/hyperparameter lives in `config/config.yaml`, loaded through a single `src/config_loader.py`.
- Structured logging to both console and file (`outputs/logs/`) throughout training, evaluation, and the API.
- Reproducibility via a single global seed (`set_global_seed`) applied to Python, NumPy, and PyTorch RNGs.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset under `data/raw/` as described in [Dataset](#dataset).

## Running the Pipeline

```bash
python -m src.data.analysis                          # (optional) regenerate EDA figures
python -m src.training.train --model baseline
python -m src.training.train --model resnet
python -m src.evaluation.evaluate --model baseline
python -m src.evaluation.evaluate --model resnet
python -m src.evaluation.error_analysis --model resnet
uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000
```

Test the API:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/raw/MT_Crack/Imgs/exp1_num_10.jpg;type=image/jpeg"
```
Or visit `http://localhost:8000/docs` for the interactive Swagger UI. `GET /health` and `GET /classes` are also available.

**Or run inference directly in Python:**
```python
from src.inference.predictor import DefectPredictor

predictor = DefectPredictor(
    checkpoint_path="outputs/checkpoints/resnet_best.pth",
    model_type="resnet",
    device="cpu",
)
print(predictor.predict("data/raw/MT_Crack/Imgs/exp1_num_10.jpg"))
```

*Note: training was run on Google Colab (free-tier GPU) using the exact commands above; the resulting checkpoints, metrics, and figures were then downloaded into this repo's `outputs/` folder for local inference and API testing.*
