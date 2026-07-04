"""
Qualitative error analysis.

Usage (from the project root):

    python -m src.evaluation.error_analysis --model resnet

Produces:
    - outputs/figures/<model>_misclassified_samples.png
    - outputs/metrics/<model>_error_analysis.json  (per-misclassified-sample detail
      plus a summary of the most common true->predicted confusion pairs)

This directly addresses the assessment's "qualitative and quantitative error
analysis" requirement: rather than only reporting a confusion matrix, we
surface the *actual images* the model gets wrong, which is what lets us form
concrete hypotheses about failure modes (e.g. "small/faint blowholes are
confused with the defect-free class").
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config_loader import load_config, set_global_seed
from src.evaluation.evaluate import get_test_loader, load_model_for_eval, run_inference_on_loader
from src.training.utils import setup_logging

logger = logging.getLogger(__name__)


def find_misclassified(labels, preds, test_samples):
    """Return a list of dicts describing every misclassified test sample."""
    misclassified = []
    for (img_path, _), true_label, pred_label in zip(test_samples, labels, preds):
        if true_label != pred_label:
            misclassified.append({
                "image_path": img_path,
                "true_label": true_label,
                "pred_label": pred_label,
            })
    return misclassified


def plot_misclassified_grid(misclassified: list, classes: list, output_path: str,
                             max_samples: int = 16):
    """Save a grid of misclassified images annotated with true vs predicted labels."""
    samples_to_show = misclassified[:max_samples]
    if not samples_to_show:
        logger.info("No misclassified samples to plot — model achieved perfect test accuracy.")
        return None

    cols = 4
    rows = (len(samples_to_show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(samples_to_show):
            sample = samples_to_show[i]
            img = Image.open(sample["image_path"]).convert("RGB")
            ax.imshow(img)
            true_name = classes[sample["true_label"]]
            pred_name = classes[sample["pred_label"]]
            ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=9, color="crimson")

    plt.suptitle("Misclassified Test Samples")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved misclassified sample grid to %s", output_path)
    return output_path


def summarize_confusion_pairs(misclassified: list, classes: list) -> list:
    """Return the most common (true_class -> predicted_class) confusion pairs,
    sorted by frequency, to make failure modes easy to describe in the README."""
    pair_counts = Counter(
        (classes[m["true_label"]], classes[m["pred_label"]]) for m in misclassified
    )
    return [
        {"true_class": pair[0], "predicted_class": pair[1], "count": count}
        for pair, count in pair_counts.most_common()
    ]


def run_error_analysis(model_type: str, config: dict, device: torch.device):
    test_loader, test_samples = get_test_loader(config)
    model = load_model_for_eval(model_type, config, device)
    labels, preds, _ = run_inference_on_loader(model, test_loader, device)

    classes = config["classes"]
    misclassified = find_misclassified(labels, preds, test_samples)
    confusion_pairs = summarize_confusion_pairs(misclassified, classes)

    logger.info("Total misclassified: %d / %d test samples", len(misclassified), len(test_samples))
    for pair in confusion_pairs:
        logger.info("  %s -> %s : %d occurrences", pair["true_class"],
                     pair["predicted_class"], pair["count"])

    fig_path = os.path.join(config["evaluation"]["figures_dir"], f"{model_type}_misclassified_samples.png")
    plot_misclassified_grid(misclassified, classes, fig_path,
                             max_samples=config["evaluation"]["num_error_samples"])

    report = {
        "total_test_samples": len(test_samples),
        "total_misclassified": len(misclassified),
        "confusion_pairs": confusion_pairs,
        "misclassified_samples": [
            {
                "image_path": m["image_path"],
                "true_class": classes[m["true_label"]],
                "predicted_class": classes[m["pred_label"]],
            }
            for m in misclassified
        ],
    }

    report_path = os.path.join(config["evaluation"]["metrics_dir"], f"{model_type}_error_analysis.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved error analysis report to %s", report_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="Run qualitative error analysis on the test set.")
    parser.add_argument("--model", choices=["baseline", "resnet"], required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config["seed"])
    setup_logging(config["logging"]["log_dir"], f"error_analysis_{args.model}.log", config["logging"]["level"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_error_analysis(args.model, config, device)


if __name__ == "__main__":
    main()
