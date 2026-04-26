from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as TF


def save_prediction_panel(
    image: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    output_path: str | Path,
    threshold: float = 0.5,
) -> None:
    """Save image, ground-truth mask, and predicted mask side by side."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted_mask = (torch.sigmoid(logits) >= threshold).float()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(TF.to_pil_image(image.cpu()))
    axes[0].set_title("Image")
    axes[1].imshow(mask.squeeze().cpu(), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(predicted_mask.squeeze().cpu(), cmap="gray")
    axes[2].set_title("Prediction")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
