# train.py
"""
Base training script for LeafSense AI models.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model

# Config
DATA_DIR = "./data/plant-disease-classification-merged-dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    num_classes = len(classes)
    
    # Build model
    model = build_model(num_classes=num_classes)
    model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Basic Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*train_correct/train_total:.2f}%"
            })

        epoch_loss = train_loss / train_total
        epoch_acc = train_correct / train_total
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}%")

if __name__ == "__main__":
    main()
