# Flood Area Segmentation

This project segments flooded or water-covered regions from satellite/aerial imagery using the Kaggle Flood Area Segmentation dataset:

https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation

The first Python version uses a ResNet50 DeepLabV3 segmentation model from PyTorch/TorchVision. It trains a binary mask predictor and reports Dice overlap and Intersection over Union (IoU).

## Project Structure

```text
areen_project/
  data/
    raw/                 # Put the Kaggle dataset here
  outputs/
    checkpoints/         # Trained model weights
    predictions/         # Saved prediction visualizations
  scripts/
    download_dataset.py
    train.py
    evaluate.py
    predict.py
  src/floodseg/
    dataset.py
    metrics.py
    model.py
    visualize.py
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset

Option 1: Download with the helper script:

```bash
python scripts/download_dataset.py
```

Option 2: Download the dataset manually from Kaggle and place the extracted files in:

```text
data/raw/
```

The loader searches recursively for common segmentation layouts such as `images/` and `masks/`, then pairs files by matching filename stems.

## Train

Run a baseline pretrained ResNet50 DeepLabV3 model:

```bash
python scripts/train.py --data-dir data/raw --epochs 10 --batch-size 4 --image-size 256
```

For a faster first test on a CPU-only machine, use fewer epochs and disable pretrained weights:

```bash
python scripts/train.py --data-dir data/raw --epochs 2 --batch-size 2 --image-size 128 --no-pretrained
```

Training saves:

```text
outputs/checkpoints/best_model.pt
outputs/metrics.csv
```

## Evaluate

After training, evaluate on the held-out test split:

```bash
python scripts/evaluate.py --data-dir data/raw --checkpoint outputs/checkpoints/best_model.pt
```

This prints the average test Dice and IoU, writes per-image scores to:

```text
outputs/test_predictions.csv
```

and saves visual prediction panels to:

```text
outputs/predictions/
```

## Predict One Image

```bash
python scripts/predict.py path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pt --output outputs/predictions/single_mask.png
```

## Project Summary

The goal of this project is to detect flooded or water-covered regions in satellite/aerial imagery. I use the Kaggle Flood Area Segmentation dataset, which contains input images and corresponding binary flood masks. The baseline model is a ResNet50 DeepLabV3 semantic segmentation network. Model performance is evaluated using Dice overlap and Intersection over Union, which are standard metrics for segmentation tasks.

Initial preliminary testing on a small subset produced an average Dice overlap of about 0.68 and an average IoU of about 0.567. The Python pipeline in this repository extends that baseline into a reproducible training and evaluation workflow so the model can be fine-tuned on more data and compared across experiments.
