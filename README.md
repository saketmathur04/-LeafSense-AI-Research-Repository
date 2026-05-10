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


## 🧪 Training Methodology

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| LR Scheduler | CosineAnnealingLR |
| Batch Size | 32 (with 4x Gradient Accumulation) |
| Loss Function | Focal Loss (gamma=2.0) |
| Mixed Precision | FP16 (torch.amp) |
| Epochs | 100 (Early Stopping patience=7) |

### Optimization Strategy
To achieve state-of-the-art performance on the 88-class LeafSense dataset, we utilized:
1. **Focal Loss:** Specifically chosen to penalize errors on "hard" minority disease classes more than "easy" majority classes.
2. **Mixed Augmentation:** A 50/50 toggle between MixUp and CutMix for every training batch to generalize across both global color distributions and local textures.
3. **AMP & Gradient Accumulation:** Combined to allow for effective batch sizes of 128 even on consumer-grade hardware.



## 🚀 Getting Started

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/saketmathur04/LeafSense-AI-Research-Repository.git
cd LeafSense-AI-Research-Repository

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model
To start training the DeiT-Base model from scratch:
```bash
python src/train.py
```

### 3. Evaluation
Once training is complete, evaluate the model on the test set:
```bash
python src/eval_only.py --data ./data/path --ckpt checkpoints/latest_checkpoint.pth --out_dir results
```


## Project Structure

```
LeafSense-AI-Training/
├── src/                    # Source code
├── outputs/                # Training results & metrics
├── requirements.txt        # Python dependencies
└── README.md
```

## 📊 Dataset & Preprocessing

### Composition
The model is trained on the **LeafSense-AI Dataset**, a large-scale collection containing:
- **Total Classes:** 88 disease categories (including healthy leaves).
- **Total Images:** ~70,000 across all classes.
- **Split Strategy:** Stratified 80% Training, 10% Validation, 10% Testing.

### Preprocessing Pipeline
We use a multi-stage preprocessing pipeline via `src/dataset.py`:
1. **Geometric Transforms:** RandomResizedCrop (224x224), Horizontal/Vertical Flips, and ShiftScaleRotate to handle varying camera angles.
2. **Photometric Transforms:** ColorJitter, RandomBrightnessContrast, and CLAHE to normalize lighting conditions in field-taken images.
3. **Noise Injection:** Gaussian Noise to simulate sensor noise in low-end mobile cameras.
4. **Normalization:** Standard ImageNet mean/std normalization.


## 📈 Results & Performance

The LeafSense-AI model achieved state-of-the-art results on the 88-class plant disease dataset.

### Training Progress
The training process stabilized within 20 epochs thanks to the Cosine Annealing scheduler and heavy regularization. 
![Training Curves](./outputs/training_curves_detailed.png)

### Model Evaluation
We generated a high-resolution confusion matrix to visualize the model's performance across all 88 disease categories.
![Confusion Matrix](./outputs/confusion_matrix_detailed.png)

#### Performance Metrics Summary
| Metric | Score |
|--------|-------|
| **Accuracy** | 96.5% |
| **Precision (Weighted)** | 96.6% |
| **Recall (Weighted)** | 96.5% |
| **F1-Score (Weighted)** | 96.5% |

A detailed per-class classification report can be found in [outputs/classification_report.txt](./outputs/classification_report.txt) and [outputs/model_evaluation_table.pdf](./outputs/model_evaluation_table.pdf).


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

## 🤖 GAN-Based Class Balancing

To address the extreme long-tail distribution of plant diseases (where some common diseases have 5,000 images while rare ones have only 50), we implemented a **Generative Adversarial Network (GAN)** oversampling strategy:

1. **Conditional DCGAN:** A generator learns to synthesize 64x64 leaf images conditioned on a specific disease class index.
2. **Synthetic Injection:** Rare classes are augmented by generating 500-1000 synthetic samples per class.
3. **Hard-Linking:** Our `src/dataset.py` includes a merging utility that hard-links these synthetic images into the training root, allowing the `WeightedRandomSampler` to draw from a much more diverse pool for minority classes.


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

## 🌐 Production Deployment

This repository focuses on the **research and training pipeline**. For the full-stack production application (built with Next.js, FastAPI, and Sanity CMS) that utilizes these models, please visit the main repository:

👉 **[LeafSense-AI Production App](https://github.com/saketmathur04/LeafSense-AI)**

The production app includes:
- **Interactive UI:** For real-time plant disease diagnosis.
- **Scalable Backend:** Serving the DeiT-Base model via optimized inference.
- **CMS Integration:** Managing disease metadata and treatment recommendations.


## Status

🚧 Under active development
