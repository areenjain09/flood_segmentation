from __future__ import annotations

import torch


def dice_score(logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """Compute mean Dice overlap for binary segmentation logits."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    masks = (masks >= 0.5).float()

    preds = preds.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)
    intersection = (preds * masks).sum(dim=1)
    denominator = preds.sum(dim=1) + masks.sum(dim=1)
    return ((2 * intersection + eps) / (denominator + eps)).mean()


def iou_score(logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """Compute mean intersection-over-union for binary segmentation logits."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    masks = (masks >= 0.5).float()

    preds = preds.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)
    intersection = (preds * masks).sum(dim=1)
    union = preds.sum(dim=1) + masks.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


class DiceBCELoss(torch.nn.Module):
    """Binary cross-entropy plus soft Dice loss."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        masks = (masks >= 0.5).float()

        flat_probs = probs.flatten(start_dim=1)
        flat_masks = masks.flatten(start_dim=1)
        intersection = (flat_probs * flat_masks).sum(dim=1)
        dice = (2 * intersection + 1e-7) / (flat_probs.sum(dim=1) + flat_masks.sum(dim=1) + 1e-7)
        dice_loss = 1 - dice.mean()
        return self.bce(logits, masks) + dice_loss
