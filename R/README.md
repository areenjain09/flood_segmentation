# R Flood Area Segmentation

This is the R version of the flood/water-covered region segmentation project. It uses the Kaggle Flood Area Segmentation dataset and trains a binary segmentation model with R `torch`.

Dataset:

https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation

## Model

The R version uses a U-Net style convolutional neural network:

- The encoder learns image features from the satellite/aerial image.
- The decoder upsamples those features back into a pixel-level mask.
- The final layer outputs one binary channel: flood/water-covered vs background.

The Python version uses DeepLabV3-ResNet50. The R version is not exactly the same architecture, but it solves the same segmentation task and reports the same metrics.

## Setup

From the `R/` folder:

```bash
Rscript install_packages.R
```

This installs packages into the project-local `packages/` folder, so it does not need write access to the system R library. It installs:

- `torch`
- `magick`
- `png`

## Dataset

By default, the R scripts use the dataset already downloaded by the Python project:

```text
../python/data/raw
```

You can also pass a different dataset folder with `--data-dir`.

## Train

```bash
Rscript scripts/train.R --data-dir ../python/data/raw --epochs 10 --batch-size 1 --image-size 128 --base-channels 8
```

If your terminal prints repeated `NNPACK` hardware warnings from R `torch`, use the quiet wrapper instead:

```bash
bash scripts/train_quiet.sh --data-dir ../python/data/raw --epochs 10 --batch-size 1 --image-size 128 --base-channels 8 --test-filenames ../python/outputs/test_predictions.csv
```

Training saves:

```text
outputs/checkpoints/best_model.pt
outputs/metrics.csv
```

## Evaluate

```bash
Rscript scripts/evaluate.R --data-dir ../python/data/raw --checkpoint outputs/checkpoints/best_model.pt
```

By default, R uses `../python/outputs/test_predictions.csv` so the R test set contains the same image filenames used by the Python evaluation.

Evaluation prints the held-out test Dice and IoU, saves per-image scores to:

```text
outputs/test_predictions.csv
```

and saves prediction panels to:

```text
outputs/predictions/
```

Each prediction panel contains:

```text
Left: original image
Middle: true flood mask
Right: predicted flood mask
```

## Predict One Image

```bash
Rscript scripts/predict.R --image ../python/data/raw/path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pt --output outputs/predictions/single_mask.png
```

## Metrics

Dice and IoU measure overlap between the predicted flood mask and the true flood mask. Higher is better, and `1.0` means a perfect match.

```text
Dice = 2 * overlap / (predicted flood area + true flood area)
IoU = overlap / combined predicted-or-true flood area
```
