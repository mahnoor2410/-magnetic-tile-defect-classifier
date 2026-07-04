"""
Shared training utilities: logging setup, class-weight computation,
checkpoint save/load, and a small running-average helper for loss tracking.
"""

import logging
import os
from typing import List

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def setup_logging(log_dir: str, log_filename: str, level: str = "INFO") -> logging.Logger:
    """Configure logging to both console and a file under `log_dir`."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if setup_logging is called more than once
    # (e.g. once from train.py, once implicitly via module import order).
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.getLogger(__name__).info("Logging initialized. Writing to %s", log_path)
    return logger


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for use in a weighted
    CrossEntropyLoss, to counteract the dataset's class imbalance.

    We use loss re-weighting rather than oversampling because oversampling a
    ~1,300 image dataset risks the model repeatedly memorizing duplicated
    minority-class images (a realistic overfitting concern given the small
    dataset size), whereas weighted loss adjusts the gradient contribution
    without duplicating data.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.array(labels),
    )
    return torch.tensor(weights, dtype=torch.float32)


class AverageMeter:
    """Tracks a running average of a scalar value (e.g. loss) over a training epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def save_checkpoint(model: torch.nn.Module, checkpoint_path: str, metadata: dict = None) -> None:
    """Save model weights plus arbitrary metadata (e.g. class names, epoch, metric)."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if metadata:
        payload.update(metadata)
    torch.save(payload, checkpoint_path)
    logging.getLogger(__name__).info("Saved checkpoint to %s", checkpoint_path)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str = "cpu") -> dict:
    """Load model weights in-place and return the full checkpoint dict
    (so callers can access metadata such as class names)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Train a model first (see README 'Training' section) or check the "
            "config's inference.checkpoint_path setting."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.getLogger(__name__).info("Loaded checkpoint from %s", checkpoint_path)
    return checkpoint
