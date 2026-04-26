from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Kaggle Flood Area Segmentation dataset.")
    parser.add_argument("--output-dir", default="data/raw", help="Where to copy the downloaded dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(kagglehub.dataset_download("faizalkarim/flood-area-segmentation"))
    for item in downloaded_path.iterdir():
        destination = output_dir / item.name
        if destination.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    print(f"Dataset copied to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
