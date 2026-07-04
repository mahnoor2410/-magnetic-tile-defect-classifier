"""
Exploratory data analysis utilities.

These functions are used by notebooks/eda.ipynb and can also be called
standalone (`python -m src.data.analysis`) to regenerate the dataset
analysis figures used in the README.
"""

import logging
import os
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
from PIL import Image

logger = logging.getLogger(__name__)


def compute_class_distribution(samples: List[Tuple[str, int]], classes: List[str]) -> dict:
    """Return {class_name: count} for the given samples."""
    counts = Counter(label for _, label in samples)
    return {classes[idx]: counts.get(idx, 0) for idx in range(len(classes))}


def plot_class_distribution(samples: List[Tuple[str, int]], classes: List[str],
                             output_path: str) -> str:
    """Bar chart of class counts, saved to output_path. Returns the saved path."""
    distribution = compute_class_distribution(samples, classes)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(distribution.keys(), distribution.values(), color="#4C72B0")
    ax.set_title("Class Distribution — Magnetic Tile Surface Defects")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.bar_label(bars)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved class distribution plot to %s", output_path)
    return output_path


def plot_sample_grid(samples: List[Tuple[str, int]], classes: List[str],
                      output_path: str, samples_per_class: int = 3) -> str:
    """Grid of sample images, one row per class, saved to output_path."""
    fig, axes = plt.subplots(len(classes), samples_per_class,
                              figsize=(samples_per_class * 2.2, len(classes) * 2.2))

    for class_idx, class_name in enumerate(classes):
        class_samples = [s for s in samples if s[1] == class_idx][:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[class_idx, col] if len(classes) > 1 else axes[col]
            ax.axis("off")
            if col < len(class_samples):
                img_path, _ = class_samples[col]
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            if col == 0:
                ax.set_ylabel(class_name, rotation=0, labelpad=40, fontsize=10)

    plt.suptitle("Sample Images per Class")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved sample grid to %s", output_path)
    return output_path


def compute_image_size_stats(samples: List[Tuple[str, int]]) -> dict:
    """Return min/max/mean width and height across the dataset (a quick way
    to confirm the resize-not-crop preprocessing decision is justified)."""
    widths, heights = [], []
    for img_path, _ in samples:
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    return {
        "width_min": min(widths), "width_max": max(widths),
        "width_mean": sum(widths) / len(widths),
        "height_min": min(heights), "height_max": max(heights),
        "height_mean": sum(heights) / len(heights),
        "num_images": len(samples),
    }


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.config_loader import load_config
    from src.data.dataset import scan_dataset

    logging.basicConfig(level=logging.INFO)
    config = load_config()

    samples = scan_dataset(
        raw_dir=config["data"]["raw_dir"],
        classes=config["classes"],
        class_folder_prefix=config["class_folder_prefix"],
        image_extension=config["data"]["image_extension"],
    )

    figures_dir = config["evaluation"]["figures_dir"]
    plot_class_distribution(samples, config["classes"],
                             os.path.join(figures_dir, "class_distribution.png"))
    plot_sample_grid(samples, config["classes"],
                      os.path.join(figures_dir, "sample_grid.png"))

    stats = compute_image_size_stats(samples)
    print("Image size statistics:", stats)
