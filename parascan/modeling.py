"""DenseNet121 construction, image transforms and checkpoint I/O."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

from parascan.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_densenet121(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier.in_features, num_classes),
    )
    return model


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
        # Stain colour is diagnostic, so keep colour jitter mild.
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def save_checkpoint(path: Path, model: nn.Module, classes: list[str], extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": "densenet121",
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "state_dict": model.state_dict(),
        **extra,
    }, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, list[str], dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_densenet121(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt


def pick_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
