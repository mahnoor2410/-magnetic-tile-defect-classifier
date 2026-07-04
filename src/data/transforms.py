"""
Preprocessing and augmentation pipelines.

Design decisions (documented here and in the README):
  - Resize to a fixed square size: the raw images have inconsistent aspect
    ratios, so a plain resize (not center-crop) is used to avoid cutting off
    defects that sit near the tile edges.
  - Normalize with ImageNet mean/std: required for the pretrained ResNet18
    backbone to receive inputs in the distribution it was trained on. We use
    the same normalization for the baseline CNN for consistency, since it
    does not hurt a from-scratch model.
  - Augmentations are intentionally conservative:
      * Horizontal/vertical flip + small rotation: defects on a magnetic tile
        have no canonical orientation, so these are label-preserving.
      * Mild brightness/contrast jitter: surface lighting varies between
        captures, so this improves robustness.
      * No hue/saturation jitter: the images are effectively grayscale
        (near-zero saturation), so color-space augmentation would inject
        noise rather than useful invariance.
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224):
    """Used for validation, test, and inference: no augmentation, deterministic."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
