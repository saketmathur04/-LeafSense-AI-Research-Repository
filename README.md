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

## Status

🚧 Under active development
