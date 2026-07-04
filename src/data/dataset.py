"""
Dataset handling for the Magnetic Tile Surface Defects dataset.

Expected raw layout (as distributed on Kaggle):

    data/raw/
        MT_Blowhole/Imgs/*.jpg (+ matching *.png masks, unused for classification)
        MT_Break/Imgs/*.jpg
        MT_Crack/Imgs/*.jpg
        MT_Fray/Imgs/*.jpg
        MT_Uneven/Imgs/*.jpg
        MT_Free/Imgs/*.jpg

This module is responsible for:
    1. Scanning the raw directory into a flat list of (image_path, label) pairs.
    2. Producing a reproducible, stratified train/val/test split.
    3. Exposing a torch.utils.data.Dataset that applies transforms on the fly.

We deliberately do not use masks: the assessment task is framed as
classification, not segmentation (see README for the reasoning).
"""

import glob
import logging
import os
from typing import List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def scan_dataset(raw_dir: str, classes: List[str], class_folder_prefix: str,
                  image_extension: str = ".jpg") -> List[Tuple[str, int]]:
    """
    Walk the raw dataset directory and build a list of (absolute_path, label_idx).

    Each class folder is expected at `<raw_dir>/<prefix><ClassName>/Imgs/*.jpg`.
    If the `Imgs` subfolder is not present, we fall back to scanning the class
    folder directly, which makes this robust to minor re-packaging differences
    in how the dataset is distributed/unzipped.
    """
    samples = []
    for label_idx, class_name in enumerate(classes):
        folder_name = f"{class_folder_prefix}{class_name}"
        class_dir = os.path.join(raw_dir, folder_name)

        candidate_dirs = [
            os.path.join(class_dir, "Imgs"),
            class_dir,
        ]

        found_images = []
        for candidate in candidate_dirs:
            if os.path.isdir(candidate):
                found_images = sorted(glob.glob(os.path.join(candidate, f"*{image_extension}")))
                if found_images:
                    break

        if not found_images:
            logger.warning("No images found for class '%s' under '%s'. "
                            "Check that the dataset was downloaded/extracted correctly.",
                            class_name, class_dir)
            continue

        for img_path in found_images:
            samples.append((img_path, label_idx))

    if not samples:
        raise RuntimeError(
            f"No images found anywhere under '{raw_dir}'. "
            "Verify the dataset has been downloaded and extracted into data/raw/ "
            "(see README 'Dataset Setup' section)."
        )

    logger.info("Scanned %d total images across %d classes.", len(samples), len(classes))
    return samples


def stratified_split(samples: List[Tuple[str, int]], train_split: float,
                      val_split: float, test_split: float, seed: int = 42):
    """
    Split (path, label) samples into train/val/test sets, preserving class
    proportions in each split (important given the dataset's class imbalance).
    """
    assert abs((train_split + val_split + test_split) - 1.0) < 1e-6, \
        "train/val/test splits must sum to 1.0"

    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        train_size=train_split,
        stratify=labels,
        random_state=seed,
    )

    # Split the remaining data proportionally into val/test.
    relative_val_size = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        train_size=relative_val_size,
        stratify=temp_labels,
        random_state=seed,
    )

    train_set = list(zip(train_paths, train_labels))
    val_set = list(zip(val_paths, val_labels))
    test_set = list(zip(test_paths, test_labels))

    logger.info("Split sizes -> train: %d, val: %d, test: %d",
                len(train_set), len(val_set), len(test_set))

    return train_set, val_set, test_set


class MagneticTileDataset(Dataset):
    """
    A torch Dataset wrapping a list of (image_path, label) tuples.

    Images are loaded and converted to RGB on the fly (the raw dataset ships
    grayscale JPEGs; converting to 3-channel RGB is required for pretrained
    ImageNet backbones such as ResNet18).
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_labels(self) -> List[int]:
        """Convenience accessor used for computing class weights."""
        return [label for _, label in self.samples]
