# LeafSense AI — Training & Research

Vision Transformer (DeiT-Base) training pipeline for plant disease classification across 88 disease categories.

## About

This repository contains the complete training implementation, evaluation metrics, and ablation studies for the LeafSense AI plant disease detection model.

## 🏗️ Technical Architecture

### 1. Vision Transformer Backbone
- **Model:** Data-efficient Image Transformer (DeiT-Base)
- **Parameters:** 86M
- **Input Resolution:** 224x224
- **Regularization:** Stochastic Depth (drop path), Label Smoothing, and Weight Decay (AdamW).

### 2. Data Pipeline & Augmentation
- **Stratified Splitting:** Ensures equal disease representation across Train/Val/Test sets.
- **Class Balancing:** `WeightedRandomSampler` implemented to handle extreme frequency variance.
- **Advanced Augmentation:** 
  - **MixUp:** Linear interpolation of image pairs to improve decision boundary robustness.
  - **CutMix:** Patch-based mixing to improve spatial localization.
  - **Albumentations:** Heavy geometric and color jittering.

### 3. Conditional GAN Pipeline
- **Purpose:** Synthetic oversampling of minority disease classes (e.g., rare rusts/blights).
- **Architecture:** Conditional DCGAN (Generator + Discriminator) with class-embedding layers.


## Project Structure

```
LeafSense-AI-Training/
├── src/                    # Source code
├── outputs/                # Training results & metrics
├── requirements.txt        # Python dependencies
└── README.md
```

## Dataset

The training pipeline expects the dataset in the following structure:

```
data/
└── plant-disease-classification-merged-dataset/
    ├── Apple__Black_rot/
    │   ├── image001.jpg
    │   └── ...
    ├── Tomato__Early_Blight/
    └── ...
```

> **Note:** The dataset is not included in this repository due to size constraints. 

## GAN Training & Synthetic Data Generation

To handle extreme class imbalance in rare plant diseases, this repository includes a Conditional DCGAN implementation.

### Training the GAN
```bash
python src/gan.py --data ./data/plant-disease-classification-merged-dataset --epochs 100
```
This will generate checkpoint files in the `gan_checkpoints/` directory.

### Generating Synthetic Images
Once the GAN is trained, you can generate synthetic images for minority classes:
```bash
python src/generate.py --ckpt gan_checkpoints/gan_epoch_100.pth --num_images 500 --class_idx 12 --out_dir synthetic_data
```

## Evaluation

To generate a full scikit-learn classification report and a seaborn confusion matrix for a trained model:
```bash
python src/eval_only.py --data ./data/plant-disease-classification-merged-dataset --ckpt checkpoints/latest_checkpoint.pth --out_dir eval_results
```
This will output `classification_report.txt` and a high-resolution `confusion_matrix.png`.

## Status

🚧 Under active development
