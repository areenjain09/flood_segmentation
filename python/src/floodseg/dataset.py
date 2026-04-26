from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SegmentationPair:
    image_path: Path
    mask_path: Path


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()


def _normalize_stem(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_mask", "-mask", " mask", "_label", "-label", " label"):
        stem = stem.replace(token, "")
    return stem


def discover_pairs(data_dir: str | Path) -> list[SegmentationPair]:
    """Discover image-mask pairs from common Kaggle segmentation folder layouts."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    files = [path for path in root.rglob("*") if _is_image_file(path)]
    mask_candidates = [
        path
        for path in files
        if any(part.lower() in {"mask", "masks", "label", "labels"} for part in path.parts)
    ]
    image_candidates = [
        path
        for path in files
        if path not in mask_candidates
        and any(part.lower() in {"image", "images", "img", "imgs"} for part in path.parts)
    ]

    if not image_candidates:
        image_candidates = [
            path
            for path in files
            if path not in mask_candidates and "mask" not in path.stem.lower() and "label" not in path.stem.lower()
        ]

    masks_by_stem = {_normalize_stem(path): path for path in mask_candidates}
    pairs: list[SegmentationPair] = []
    for image_path in image_candidates:
        mask_path = masks_by_stem.get(_normalize_stem(image_path))
        if mask_path is not None:
            pairs.append(SegmentationPair(image_path=image_path, mask_path=mask_path))

    pairs.sort(key=lambda pair: pair.image_path.name)
    if not pairs:
        raise ValueError(
            f"No image-mask pairs found under {root}. Expected folders such as images/ and masks/."
        )
    return pairs


def split_pairs(
    pairs: list[SegmentationPair],
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[list[SegmentationPair], list[SegmentationPair], list[SegmentationPair]]:
    if len(pairs) < 3:
        raise ValueError("Need at least 3 image-mask pairs to create train/val/test splits.")

    train_val, test = train_test_split(pairs, test_size=test_size, random_state=seed)
    adjusted_val_size = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=adjusted_val_size, random_state=seed)
    return list(train), list(val), list(test)


class FloodSegmentationDataset(Dataset):
    def __init__(self, pairs: Iterable[SegmentationPair], image_size: int = 256) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.pairs[index]
        image = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path).convert("L")

        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        mask_array = np.array(mask, dtype=np.float32)
        mask_tensor = torch.from_numpy((mask_array > 127).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor, pair.image_path.name
