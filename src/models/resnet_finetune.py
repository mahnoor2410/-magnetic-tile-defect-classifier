"""
Fine-tuned ResNet18 (ImageNet-pretrained).

Strategy (documented in README under "Model Selection & Training"):
  - Load torchvision's ImageNet-pretrained ResNet18.
  - Replace the final fully-connected layer to output `num_classes` logits.
  - Stage the fine-tuning: for the first `freeze_backbone_epochs` epochs,
    freeze every layer except the new classifier head (lets the head adapt
    to the new classes without corrupting pretrained features early on).
    After that, unfreeze the last residual block (`layer4`) plus the head
    for the remaining epochs, using a low learning rate. This is a
    compute-aware compromise: full end-to-end fine-tuning on a small
    (~1,300 image) dataset risks overfitting and costs more compute than
    partial fine-tuning, with little accuracy benefit in practice here.

ResNet18 is chosen over deeper variants (ResNet50, etc.) specifically for
compute reasons: it trains in minutes on a Colab free-tier GPU and is still
fast enough for CPU inference in production, which matters for the
deployment-readiness criterion of this assessment.
"""

import torch.nn as nn
from torchvision import models


def build_resnet_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 with its final layer replaced for `num_classes` outputs."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter except the classifier head (`fc`)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_last_block(model: nn.Module) -> None:
    """Unfreeze `layer4` (the last residual block) plus the classifier head,
    keeping earlier layers (which encode generic, low-level features) frozen."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer4.") or name.startswith("fc.")


def get_trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
