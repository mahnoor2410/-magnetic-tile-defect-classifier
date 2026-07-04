"""
Training entry point.

Usage (from the project root):

    python -m src.training.train --model baseline
    python -m src.training.train --model resnet

Both the baseline CNN and the ResNet18 fine-tune share this single training
loop; only the model-building, optimizer, and (for ResNet) the
freeze/unfreeze schedule differ, which are branched on `--model`.

The checkpoint with the best validation macro-F1 (not accuracy) is kept,
since macro-F1 is the metric that actually reflects performance under the
dataset's class imbalance (see README 'Evaluation Methodology').
"""

import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config_loader import load_config, set_global_seed
from src.data.dataset import MagneticTileDataset, scan_dataset, stratified_split
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.baseline_cnn import build_baseline_model
from src.models.resnet_finetune import (build_resnet_model, freeze_backbone,
                                         get_trainable_param_count, unfreeze_last_block)
from src.training.utils import (AverageMeter, compute_class_weights,
                                 save_checkpoint, setup_logging)

logger = logging.getLogger(__name__)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run a single training or validation epoch. Returns (avg_loss, macro_f1)."""
    model.train() if train else model.eval()

    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), n=images.size(0))
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return loss_meter.avg, macro_f1


def build_dataloaders(config: dict):
    samples = scan_dataset(
        raw_dir=config["data"]["raw_dir"],
        classes=config["classes"],
        class_folder_prefix=config["class_folder_prefix"],
        image_extension=config["data"]["image_extension"],
    )

    train_samples, val_samples, _ = stratified_split(
        samples,
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        seed=config["seed"],
    )

    image_size = config["data"]["image_size"]
    train_dataset = MagneticTileDataset(train_samples, transform=get_train_transforms(image_size))
    val_dataset = MagneticTileDataset(val_samples, transform=get_eval_transforms(image_size))

    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
    )

    class_weights = compute_class_weights(train_dataset.get_labels(), len(config["classes"]))
    return train_loader, val_loader, class_weights


def train_baseline(config: dict, device: torch.device):
    train_loader, val_loader, class_weights = build_dataloaders(config)
    num_classes = len(config["classes"])

    model = build_baseline_model(num_classes, image_size=config["data"]["image_size"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_cfg = config["training"]["baseline"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])

    checkpoint_path = os.path.join(config["model"]["checkpoint_dir"],
                                    config["model"]["baseline_checkpoint_name"])

    best_f1 = -1.0
    for epoch in range(1, train_cfg["epochs"] + 1):
        start = time.time()
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - start

        logger.info(
            "[Baseline] Epoch %d/%d | train_loss=%.4f train_f1=%.4f | "
            "val_loss=%.4f val_f1=%.4f | %.1fs",
            epoch, train_cfg["epochs"], train_loss, train_f1, val_loss, val_f1, elapsed,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, checkpoint_path, metadata={
                "epoch": epoch, "val_macro_f1": val_f1, "classes": config["classes"],
                "model_type": "baseline", "image_size": config["data"]["image_size"],
            })
            logger.info("New best baseline model (val_macro_f1=%.4f) saved.", best_f1)

    logger.info("Baseline training complete. Best val_macro_f1=%.4f", best_f1)


def train_resnet(config: dict, device: torch.device):
    train_loader, val_loader, class_weights = build_dataloaders(config)
    num_classes = len(config["classes"])

    model = build_resnet_model(num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_cfg = config["training"]["resnet"]
    checkpoint_path = os.path.join(config["model"]["checkpoint_dir"],
                                    config["model"]["resnet_checkpoint_name"])

    best_f1 = -1.0
    for epoch in range(1, train_cfg["epochs"] + 1):
        # Staged fine-tuning schedule: freeze backbone for the first N epochs,
        # then unfreeze the last residual block for the remainder.
        if epoch <= train_cfg["freeze_backbone_epochs"]:
            freeze_backbone(model)
            stage = "frozen-backbone"
        else:
            unfreeze_last_block(model)
            stage = "layer4-unfrozen"

        trainable_params = get_trainable_param_count(model)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
        )

        start = time.time()
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - start

        logger.info(
            "[ResNet18 | %s | trainable_params=%d] Epoch %d/%d | "
            "train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f | %.1fs",
            stage, trainable_params, epoch, train_cfg["epochs"],
            train_loss, train_f1, val_loss, val_f1, elapsed,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, checkpoint_path, metadata={
                "epoch": epoch, "val_macro_f1": val_f1, "classes": config["classes"],
                "model_type": "resnet", "image_size": config["data"]["image_size"],
            })
            logger.info("New best ResNet18 model (val_macro_f1=%.4f) saved.", best_f1)

    logger.info("ResNet18 fine-tuning complete. Best val_macro_f1=%.4f", best_f1)


def main():
    parser = argparse.ArgumentParser(description="Train the magnetic tile defect classifier.")
    parser.add_argument("--model", choices=["baseline", "resnet"], required=True,
                         help="Which model architecture to train.")
    parser.add_argument("--config", default=None, help="Optional path to config.yaml override.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config["seed"])

    setup_logging(config["logging"]["log_dir"], f"train_{args.model}.log", config["logging"]["level"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if args.model == "baseline":
        train_baseline(config, device)
    else:
        train_resnet(config, device)


if __name__ == "__main__":
    main()
