# Flood Area Segmentation Project

This repository contains two versions of the same flood/water segmentation project:

```text
areen_project/
  python/   # PyTorch DeepLabV3-ResNet50 version
  R/        # R torch U-Net version
```

Both versions use the Kaggle Flood Area Segmentation dataset:

https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation

## Python Version

The Python project is in:

```text
python/
```

It uses a pretrained DeepLabV3 model with a ResNet50 backbone. This is the version you already trained. Your latest Python test results were:

```text
Test Dice: 0.8822
Test IoU: 0.7952
```

Run from the `python/` folder:

```bash
cd python
source ../.venv/bin/activate
python scripts/train.py --data-dir data/raw --epochs 10 --batch-size 4 --image-size 256
python scripts/evaluate.py --data-dir data/raw --checkpoint outputs/checkpoints/best_model.pt
```

## R Version

The R project is in:

```text
R/
```

It uses an R `torch` U-Net style segmentation model. It shares the dataset downloaded by the Python project at `python/data/raw` by default.

Run from the `R/` folder:

```bash
cd R
Rscript install_packages.R
Rscript scripts/train.R --data-dir ../python/data/raw --epochs 10 --batch-size 4 --image-size 256
Rscript scripts/evaluate.R --data-dir ../python/data/raw --checkpoint outputs/checkpoints/best_model.pt
```

Both versions compute Dice and IoU, then save prediction images that compare the original input image, the ground-truth mask, and the predicted flood mask.
