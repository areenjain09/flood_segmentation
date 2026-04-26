from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from floodseg.model import build_model, forward_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a binary flood mask for one image.")
    parser.add_argument("image", help="Path to an input satellite/aerial image.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", default="outputs/predictions/single_mask.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    original = Image.open(args.image).convert("RGB")
    resized = TF.resize(original, [args.image_size, args.image_size], antialias=True)
    image_tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = forward_logits(model, image_tensor)
        prediction = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255

    output_mask = Image.fromarray(prediction).resize(original.size, resample=Image.Resampling.NEAREST)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask.save(output_path)
    print(f"Saved predicted mask to: {output_path}")


if __name__ == "__main__":
    main()
