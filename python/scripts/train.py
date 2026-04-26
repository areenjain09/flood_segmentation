from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from floodseg.dataset import FloodSegmentationDataset, discover_pairs, split_pairs
from floodseg.metrics import DiceBCELoss, dice_score, iou_score
from floodseg.model import build_model, forward_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a flood segmentation model.")
    parser.add_argument("--data-dir", default="data/raw", help="Folder containing Kaggle images and masks.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--metrics-csv", default="outputs/metrics.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, criterion: DiceBCELoss) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = forward_logits(model, images)
            total_loss += criterion(logits, masks).item()
            total_dice += dice_score(logits, masks).item()
            total_iou += iou_score(logits, masks).item()
            batches += 1

    return {
        "loss": total_loss / max(batches, 1),
        "dice": total_dice / max(batches, 1),
        "iou": total_iou / max(batches, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pairs = discover_pairs(args.data_dir)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, seed=args.seed)
    print(f"Found {len(pairs)} pairs: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
    print(f"Using device: {device}")

    train_loader = DataLoader(
        FloodSegmentationDataset(train_pairs, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FloodSegmentationDataset(val_pairs, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(pretrained=args.pretrained, freeze_backbone=args.freeze_backbone).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0
    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_batches = 0

            for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = forward_logits(model, images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            val_metrics = evaluate(model, val_loader, device, criterion)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / max(train_batches, 1),
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
            }
            writer.writerow(row)
            file.flush()

            print(
                f"Epoch {epoch}: train_loss={row['train_loss']:.4f}, "
                f"val_dice={row['val_dice']:.4f}, val_iou={row['val_iou']:.4f}"
            )

            if row["val_iou"] > best_iou:
                best_iou = row["val_iou"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "val_iou": best_iou,
                    },
                    checkpoint_dir / "best_model.pt",
                )

    print(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
