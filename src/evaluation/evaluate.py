"""
Test-set evaluation.

Usage (from the project root):

    python -m src.evaluation.evaluate --model resnet
    python -m src.evaluation.evaluate --model baseline

Produces:
    - outputs/metrics/<model>_metrics.json   (accuracy, macro-F1, per-class P/R/F1)
    - outputs/figures/<model>_confusion_matrix.png

We report macro-F1 and per-class precision/recall as the primary metrics
(not just accuracy) because the dataset is imbalanced: a model that always
predicts the majority class can still score high accuracy while being
useless for minority defect classes, which is precisely the failure mode a
defect-detection system cannot afford.
"""

import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config_loader import load_config, set_global_seed
from src.data.dataset import MagneticTileDataset, scan_dataset, stratified_split
from src.data.transforms import get_eval_transforms
from src.models.baseline_cnn import build_baseline_model
from src.models.resnet_finetune import build_resnet_model
from src.training.utils import load_checkpoint, setup_logging

logger = logging.getLogger(__name__)


def get_test_loader(config: dict):
    samples = scan_dataset(
        raw_dir=config["data"]["raw_dir"],
        classes=config["classes"],
        class_folder_prefix=config["class_folder_prefix"],
        image_extension=config["data"]["image_extension"],
    )
    _, _, test_samples = stratified_split(
        samples,
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        seed=config["seed"],
    )
    test_dataset = MagneticTileDataset(test_samples, transform=get_eval_transforms(config["data"]["image_size"]))
    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False,
                              num_workers=config["data"]["num_workers"])
    return test_loader, test_samples


def load_model_for_eval(model_type: str, config: dict, device: torch.device):
    num_classes = len(config["classes"])
    if model_type == "baseline":
        model = build_baseline_model(num_classes, image_size=config["data"]["image_size"])
        checkpoint_name = config["model"]["baseline_checkpoint_name"]
    else:
        model = build_resnet_model(num_classes, pretrained=False)
        checkpoint_name = config["model"]["resnet_checkpoint_name"]

    checkpoint_path = os.path.join(config["model"]["checkpoint_dir"], checkpoint_name)
    load_checkpoint(model, checkpoint_path, device=str(device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference_on_loader(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_labels, all_preds, all_probs


def evaluate_model(model_type: str, config: dict, device: torch.device):
    test_loader, test_samples = get_test_loader(config)
    model = load_model_for_eval(model_type, config, device)

    labels, preds, _ = run_inference_on_loader(model, test_loader, device)
    classes = config["classes"]

    report = classification_report(labels, preds, target_names=classes,
                                     output_dict=True, zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    logger.info("Test accuracy: %.4f", report["accuracy"])
    logger.info("Test macro-F1: %.4f", macro_f1)
    for class_name in classes:
        cls_report = report[class_name]
        logger.info("  %-12s precision=%.3f recall=%.3f f1=%.3f support=%d",
                     class_name, cls_report["precision"], cls_report["recall"],
                     cls_report["f1-score"], cls_report["support"])

    metrics_path = os.path.join(config["evaluation"]["metrics_dir"], f"{model_type}_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": report["accuracy"], "macro_f1": macro_f1,
                    "per_class": {c: report[c] for c in classes}}, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_type}")
    plt.tight_layout()

    fig_path = os.path.join(config["evaluation"]["figures_dir"], f"{model_type}_confusion_matrix.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", fig_path)

    return labels, preds, test_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the held-out test set.")
    parser.add_argument("--model", choices=["baseline", "resnet"], required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config["seed"])
    setup_logging(config["logging"]["log_dir"], f"evaluate_{args.model}.log", config["logging"]["level"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_model(args.model, config, device)


if __name__ == "__main__":
    main()
