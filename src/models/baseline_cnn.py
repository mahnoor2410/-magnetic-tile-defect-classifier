"""
Baseline CNN, trained from scratch.

This model exists to establish a reference point: it lets us quantify how
much lift transfer learning actually provides on this dataset, rather than
assuming pretrained features are necessary without evidence. It is
deliberately small (4 conv blocks) to stay fast on CPU / Colab free-tier GPUs.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 224):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global average pooling makes the classifier head independent of
        # the exact spatial size produced by `features`, so image_size can
        # be changed in config without having to recompute a flatten dimension.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def build_baseline_model(num_classes: int, image_size: int = 224) -> BaselineCNN:
    return BaselineCNN(num_classes=num_classes, image_size=image_size)
