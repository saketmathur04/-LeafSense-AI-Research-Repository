# LeafSense AI — Training & Research

Vision Transformer (DeiT-Base) training pipeline for plant disease classification across 88 disease categories.

## About

This repository contains the complete training implementation, evaluation metrics, and ablation studies for the LeafSense AI plant disease detection model.

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

## Status

🚧 Under active development
