# train.py
"""
Base training script for LeafSense AI models.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import random

from dataset import get_dataloaders
from model import build_model
from utils import FocalLoss, mixup_data, mixup_criterion, cutmix_data

# Config
DATA_DIR = "./data/plant-disease-classification-merged-dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
MIXUP_ALPHA = 0.2

def validate_one_epoch(model, val_loader, criterion, device):
    """Runs one full validation epoch."""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    return val_loss / val_total, val_correct / val_total

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

    # Loss, Optimizer & Scheduler
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/latest_checkpoint.pth"
    best_model_path = "best_vit_model.pth"
    start_epoch = 0
    best_val_loss = float('inf')
    patience = 7
    epochs_no_improve = 0
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = 4

    # Basic Training Loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply MixUp or CutMix
            if random.random() > 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=MIXUP_ALPHA)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += (loss.item() * accumulation_steps) * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*train_correct/train_total:.2f}%"
            })

        epoch_loss = train_loss / train_total
        epoch_acc = train_correct / train_total

        # Validation Phase
        epoch_val_loss, epoch_val_acc = validate_one_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

        scheduler.step()

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, ckpt_path)

        # Early Stopping & Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs.")
                break

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
