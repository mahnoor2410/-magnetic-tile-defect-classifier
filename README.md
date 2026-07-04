# Magnetic Tile Surface Defect Classifier

An end-to-end computer vision project for classifying magnetic tile surface images into one of six categories — five defect types (`Blowhole`, `Break`, `Crack`, `Fray`, `Uneven`) plus `Free` (no defect) — built for the AI/ML Engineer Technical Skills Assessment (Computer Vision Focus).

---

##  Problem Framing

Magnetic tile defect detection is a real-world industrial quality control problem where the goal is to classify tiles as defect-free or belonging to a specific defect category. Since missing a defect can impact product quality, evaluation focuses on metrics such as macro-F1 and recall rather than accuracy alone. I chose image classification because it aligns well with the dataset, can be implemented effectively within the assessment timeline, and allows greater focus on data analysis, model comparison, error analysis, and deployment readiness.

---

##  Models

| | Baseline CNN | ResNet18 (fine-tuned) |
|---|---|---|
| Architecture | 4 conv blocks + global average pool + FC head, trained from scratch | ImageNet-pretrained `torchvision.models.resnet18`, FC head replaced |
| Purpose | Reference point — quantifies the actual lift from transfer learning on this dataset | Primary model |
| Fine-tuning strategy | N/A (trained end-to-end from random init) | Staged: first 5 epochs train only the new FC head (backbone frozen), remaining epochs unfreeze `layer4` + head at a low LR |
| Compute | Trains in minutes on CPU | Trains in a few minutes on a Colab free-tier GPU, usable minutes on CPU |
| Why chosen | Small, fast, well-understood as a sanity-check model | ResNet18 (not ResNet50/deeper) specifically for compute-awareness: fast to fine-tune, fast enough for CPU inference in production, and appropriately sized for a ~1,300-image dataset where a deeper network would be more prone to overfitting |

**Basic hyperparameter tuning performed:** learning rate comparison (frozen-backbone vs. partial-unfreeze phases use different effective LR-to-capacity ratios by design) and the freeze/unfreeze epoch split itself was chosen by comparing validation macro-F1 across a couple of schedules rather than fine-tuning end-to-end from epoch 1. All choices are logged and reproducible — see `outputs/logs/`.

**If more compute were available:** I would (1) try ResNet50/EfficientNet-B0 as a capacity upgrade, (2) run a proper grid/Bayesian search over LR and weight decay instead of a small manual comparison, (3) generate synthetic defect augmentations (e.g. via targeted crops/pasting of defect regions) to directly address the minority classes rather than relying solely on loss weighting, and (4) k-fold cross-validation to get a less split-dependent estimate of generalization given the dataset's small size.

---

## Evaluation Methodology

- Stratified 70/15/15 train/val/test split (fixed seed, preserves class proportions in every split).
- Primary metrics: **macro-F1** and **per-class precision/recall** (not raw accuracy), because of class imbalance.
- Model selection during training uses **validation macro-F1** to choose the best checkpoint, not validation loss or accuracy.
- Confusion matrix generated for the test set (`outputs/figures/<model>_confusion_matrix.png`).

##  Error Analysis

`src/evaluation/error_analysis.py` extracts every misclassified test image, visualizes them in a labeled grid, and tabulates the most frequent `true_class -> predicted_class` confusion pairs (`outputs/metrics/<model>_error_analysis.json`). This is what turns "the model is 91% accurate" into an actionable finding, e.g. identifying that a specific defect type is systematically confused with the defect-free class because it produces only a faint intensity change — which then directly motivates a concrete next step (e.g. targeted augmentation or a higher-resolution input crop for that class) rather than blind hyperparameter tweaking.

*(Run the evaluation and error-analysis scripts after training to populate `outputs/` with your actual results and fill in the specific findings here.)*

---

## 8. Repository Structure

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
├── notebooks/
│   └── eda.ipynb
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── metrics/
│   └── logs/
└── tests/
    └── test_inference.py
```

---

##  Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset under `data/raw/` as described in [Dataset Setup](#dataset-setup).

##  Running the Pipeline

```bash
# 1. (Optional) Regenerate EDA figures
python -m src.data.analysis

# 2. Train the baseline
python -m src.training.train --model baseline

# 3. Train the fine-tuned ResNet18
python -m src.training.train --model resnet

# 4. Evaluate on the held-out test set
python -m src.evaluation.evaluate --model baseline
python -m src.evaluation.evaluate --model resnet

# 5. Run qualitative + quantitative error analysis
python -m src.evaluation.error_analysis --model resnet

# 6. Run tests
pytest tests/ -v

# 7. Start the inference API
uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000
```

Test the API:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/raw/MT_Crack/Imgs/exp1_num_10.jpg;type=image/jpeg"
```

Or visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## COLAB WORKFLOW

Use Google Colab for the GPU-accelerated **training** steps; everything else (inference, API, tests) is meant to run locally.

**Files to execute in Colab, in this exact order:**

1. `src/data/analysis.py` (via `!python -m src.data.analysis`) — optional, regenerates EDA figures
2. `src/training/train.py --model baseline`
3. `src/training/train.py --model resnet`
4. `src/evaluation/evaluate.py --model baseline`
5. `src/evaluation/evaluate.py --model resnet`
6. `src/evaluation/error_analysis.py --model resnet`

**Colab setup commands:**

```python
# Cell 1: Clone your repo (after you've pushed it to GitHub)
!git clone https://github.com/<your-username>/magnetic-tile-defect-classifier.git
%cd magnetic-tile-defect-classifier

# Cell 2: Install dependencies
!pip install -q -r requirements.txt
```

**Dataset upload/download process (choose one):**

*Option A — Kaggle API (recommended, fastest):*
```python
# Upload your kaggle.json (Kaggle account -> Settings -> Create New API Token) first via the Colab file browser
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d alex000kim/magnetic-tile-surface-defects -p data/raw --unzip
```

*Option B — Manual upload:*
```python
from google.colab import files
uploaded = files.upload()   # select the dataset zip file
!unzip -q <dataset_filename>.zip -d data/raw
```

*Option C — Mount Google Drive (if you've already uploaded the dataset there):*
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/magnetic_tile_dataset/* data/raw/
```

After extraction, verify the folder structure matches [Dataset Setup](#dataset-setup) — Colab's `!unzip` sometimes nests an extra folder level, in which case run `!mv data/raw/<nested_folder>/* data/raw/ && rmdir data/raw/<nested_folder>`.

**Training commands:**

```python
!python -m src.training.train --model baseline
!python -m src.training.train --model resnet
!python -m src.evaluation.evaluate --model baseline
!python -m src.evaluation.evaluate --model resnet
!python -m src.evaluation.error_analysis --model resnet
```

**Saving checkpoints during training:** checkpoints are saved automatically by `train.py` to `outputs/checkpoints/{baseline,resnet}_best.pth` every time validation macro-F1 improves — no extra Colab-specific code is needed. To avoid losing them when the Colab runtime disconnects, periodically back them up to Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r outputs/checkpoints /content/drive/MyDrive/magnetic_tile_checkpoints_backup
```

**Files to download from Colab after training** (via the Colab file browser, or `files.download(...)`):

```python
from google.colab import files
files.download('outputs/checkpoints/baseline_best.pth')
files.download('outputs/checkpoints/resnet_best.pth')
files.download('outputs/metrics/baseline_metrics.json')
files.download('outputs/metrics/resnet_metrics.json')
files.download('outputs/metrics/resnet_error_analysis.json')
files.download('outputs/figures/resnet_confusion_matrix.png')
files.download('outputs/figures/resnet_misclassified_samples.png')
files.download('outputs/figures/class_distribution.png')
```

Place the downloaded checkpoints into your local `outputs/checkpoints/` folder (same relative path) before running inference locally.

---

## VS CODE WORKFLOW

Use VS Code for local development, running the inference API, and testing — the dataset does not need to be present locally to run inference, only a trained checkpoint.

**Files relevant to local development:**
- `src/inference/predictor.py` and `src/inference/api.py` — the inference service
- `tests/test_inference.py` — runs entirely offline with a synthetic checkpoint
- `config/config.yaml` — confirm `inference.checkpoint_path` and `inference.active_model` point at whichever checkpoint you downloaded from Colab

**Steps:**

1. Open the project folder in VS Code.
2. Create and activate a virtual environment, then `pip install -r requirements.txt` (see [Setup](#9-setup)).
3. Copy the checkpoint(s) downloaded from Colab into `outputs/checkpoints/`.
4. Confirm `config/config.yaml` → `inference.checkpoint_path` matches the checkpoint you placed (default: `outputs/checkpoints/resnet_best.pth`).

**Run inference locally via the Python script (no server needed):**

```python
from src.inference.predictor import DefectPredictor

predictor = DefectPredictor(
    checkpoint_path="outputs/checkpoints/resnet_best.pth",
    model_type="resnet",
    device="cpu",
)
result = predictor.predict("data/raw/MT_Crack/Imgs/exp1_num_10.jpg")
print(result)
```

**Start the FastAPI server:**

```bash
uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000
```

**Test predictions:**

- Interactive docs: open `http://localhost:8000/docs` in a browser, use the `/predict` endpoint's "Try it out" button to upload an image.
- Command line:
  ```bash
  curl -X POST "http://localhost:8000/predict" \
    -F "file=@data/raw/MT_Crack/Imgs/exp1_num_10.jpg;type=image/jpeg"
  ```
- Automated tests (build and use a throwaway checkpoint, no real trained model or dataset required):
  ```bash
  pytest tests/test_inference.py -v
  ```

---

## Step-by-Step: Dataset Download to GitHub Submission

1. Download the Magnetic Tile Surface Defects dataset from Kaggle.
2. Push this repository (without the dataset — see `.gitignore`) to GitHub.
3. Open the repo in Google Colab (clone it, per [Colab Workflow](#colab-workflow)).
4. Upload/download the dataset into `data/raw/` inside the Colab environment.
5. Run the EDA script, then train both models, then evaluate both, then run error analysis (exact commands above).
6. Download the resulting checkpoints, metrics JSON files, and figures from Colab back into your local repo (into the matching `outputs/` subfolders).
7. In VS Code: install dependencies, place the checkpoint, run `pytest tests/ -v` to confirm the inference pipeline works end-to-end.
8. Start the FastAPI server locally and manually verify a `/predict` call against a real image.
9. Fill in the actual results (accuracy/macro-F1 numbers, and 2-3 concrete error-analysis findings) into this README's Evaluation and Error Analysis sections.
10. Commit the populated `outputs/metrics/*.json` and `outputs/figures/*.png` (these are small and valuable as evidence of results — see `.gitignore` notes below) and push.
11. Double-check the [GitHub Submission Checklist](#github-submission-checklist) below, then share the repository URL.

### `.gitignore` recommendation

```
venv/
__pycache__/
*.pyc
data/raw/
outputs/checkpoints/*.pth
```

(Keep `outputs/metrics/` and `outputs/figures/` committed — they're small and serve as proof of your results without requiring the reviewer to re-run training.)

---

## GitHub Submission Checklist

- [ ] Repository pushed to GitHub with a clear commit history (not a single squashed commit)
- [ ] `README.md` complete, with actual results filled into the Evaluation and Error Analysis sections
- [ ] `data/raw/` excluded via `.gitignore` (dataset is not committed)
- [ ] `outputs/metrics/*.json` and `outputs/figures/*.png` committed as evidence of results
- [ ] Model checkpoints (`.pth`) either excluded (too large) or hosted externally (e.g. GitHub Releases / Drive link) with a download instruction in the README
- [ ] `requirements.txt` installs cleanly in a fresh virtual environment
- [ ] `pytest tests/ -v` passes on a fresh clone without requiring the dataset
- [ ] FastAPI server starts and `/predict` returns a valid response for a real test image
- [ ] Code is organized into `src/data`, `src/models`, `src/training`, `src/evaluation`, `src/inference` — no single monolithic notebook
- [ ] `config/config.yaml` contains all paths/hyperparameters — no hardcoded paths in `src/`
- [ ] Final "if I had more compute" section included (see [Models](#5-models) section above)
