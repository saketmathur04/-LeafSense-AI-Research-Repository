# dataset.py
"""
dataset.py
Provides PyTorch DataLoaders with Albumentations-based transforms
for the plant disease classification dataset.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_alb_transforms(img_size=224, train=True):
    """Get Albumentations transform pipeline for train or val/test."""
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.5),
            A.OneOf([
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(),
                A.Sharpen(),
                A.Emboss(),
                A.RandomGamma()
            ], p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


class AlbumentationsDataset(Dataset):
    """Custom dataset that loads images from class subdirectories and applies Albumentations transforms."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = []
        classes = sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            p = Path(root_dir) / c
            for f in p.iterdir():
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((str(f), self.class_to_idx[c]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = np.array(Image.open(p).convert("RGB"))
        if self.transform:
            res = self.transform(image=img)
            img = res["image"]
        return img, label


def get_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=2):
    """
    Simple dataloader factory with random 80/10/10 split.
    Returns train_loader, val_loader, test_loader, class_names.
    """
    full_ds = AlbumentationsDataset(data_dir, transform=None)
    class_names = sorted(full_ds.class_to_idx.keys())

    # 80/10/10 split
    total = len(full_ds)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_ds, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names
