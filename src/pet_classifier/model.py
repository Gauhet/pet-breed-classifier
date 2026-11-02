from pathlib import Path

import torch
from torch import nn
from torchvision import models

# Each architecture with the ImageNet weights torchvision ships for it.
ARCHITECTURES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def build_model(arch: str = "resnet50", num_classes: int = 37, pretrained: bool = True) -> nn.Module:
    """A ResNet with its 1000-class ImageNet head swapped for one sized to our classes."""
    constructor, weights = ARCHITECTURES[arch]
    model = constructor(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the final linear layer, so only the new head learns."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def save_checkpoint(model: nn.Module, path: str | Path, arch: str, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "num_classes": model.fc.out_features, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild the model from a checkpoint and return it with the checkpoint's metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["arch"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
