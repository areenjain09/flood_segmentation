from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from floodseg.dataset import FloodSegmentationDataset, discover_pairs, split_pairs
from floodseg.metrics import dice_score, iou_score
from floodseg.model import build_model, forward_logits
from floodseg.visualize import save_prediction_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained flood segmentation model.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-csv", default="outputs/test_predictions.csv")
    parser.add_argument("--prediction-dir", default="outputs/predictions")
    parser.add_argument("--max-panels", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    pairs = discover_pairs(args.data_dir)
    _, _, test_pairs = split_pairs(pairs, seed=args.seed)
    loader = DataLoader(
        FloodSegmentationDataset(test_pairs, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_dir = Path(args.prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    all_dice: list[float] = []
    all_iou: list[float] = []
    saved_panels = 0

    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "dice", "iou"])
        writer.writeheader()

        with torch.no_grad():
            for images, masks, filenames in tqdm(loader, desc="Evaluating"):
                images = images.to(device)
                masks = masks.to(device)
                logits = forward_logits(model, images)

                for idx, filename in enumerate(filenames):
                    item_logits = logits[idx : idx + 1]
                    item_mask = masks[idx : idx + 1]
                    dice = dice_score(item_logits, item_mask).item()
                    iou = iou_score(item_logits, item_mask).item()
                    writer.writerow({"filename": filename, "dice": dice, "iou": iou})
                    all_dice.append(dice)
                    all_iou.append(iou)

                    if saved_panels < args.max_panels:
                        save_prediction_panel(
                            images[idx].cpu(),
                            masks[idx].cpu(),
                            logits[idx].cpu(),
                            prediction_dir / f"{Path(filename).stem}_prediction.png",
                        )
                        saved_panels += 1

    mean_dice = sum(all_dice) / max(len(all_dice), 1)
    mean_iou = sum(all_iou) / max(len(all_iou), 1)
    print(f"Test Dice: {mean_dice:.4f}")
    print(f"Test IoU: {mean_iou:.4f}")
    print(f"Saved per-image results to: {output_csv}")
    print(f"Saved prediction panels to: {prediction_dir}")


if __name__ == "__main__":
    main()
