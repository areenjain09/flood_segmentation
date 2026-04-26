from __future__ import annotations

import torch
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> torch.nn.Module:
    """Build a ResNet50 DeepLabV3 model for one-channel flood masks."""
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, weights_backbone=None, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)

    if model.aux_classifier is not None:
        model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


def forward_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the segmentation logits from torchvision's output dict."""
    output = model(images)
    if isinstance(output, dict):
        return output["out"]
    return output
