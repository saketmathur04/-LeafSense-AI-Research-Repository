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
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
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


def stratified_split(dataset_root, val_size=0.1, test_size=0.1, seed=42):
    """Perform stratified train/val/test split preserving class distribution."""
    ds = AlbumentationsDataset(dataset_root, transform=None)
    X = [p for p, _ in ds.samples]
    y = [lab for _, lab in ds.samples]
    if len(y) == 0:
        raise RuntimeError(f"No images found under {dataset_root}.")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=seed)
    train_idx, temp_idx = next(sss.split(X, y))
    # split temp into val/test
    rel = test_size / (test_size + val_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel, random_state=seed)
    temp_y = [y[i] for i in temp_idx]
    val_idx_rel, test_idx_rel = next(sss2.split([X[i] for i in temp_idx], temp_y))
    val_idx = [temp_idx[i] for i in val_idx_rel]
    test_idx = [temp_idx[i] for i in test_idx_rel]
    return train_idx, val_idx, test_idx, ds.class_to_idx


class WrappedSubset(Dataset):
    """Wraps a Subset to apply a specific transform, since Subsets don't carry transforms."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            res = self.transform(image=np.array(img))
            img = res["image"]
        else:
            img = transforms.ToTensor()(img)
        return img, label


def make_dataloaders(root_dir, img_size=224, batch_size=32, val_batch=64,
                     val_size=0.1, test_size=0.1, seed=42,
                     use_sampler=True, num_workers=4):
    """
    Primary dataloader factory with stratified split and class-balanced sampling.
    root_dir: dataset root with class subfolders
    """
    train_idx, val_idx, test_idx, class_to_idx = stratified_split(root_dir, val_size=val_size, test_size=test_size, seed=seed)
    ds_full = AlbumentationsDataset(root_dir, transform=None)

    train_ds = Subset(ds_full, train_idx)
    val_ds = Subset(ds_full, val_idx)
    test_ds = Subset(ds_full, test_idx)

    # attach transforms
    train_ds = WrappedSubset(train_ds, get_alb_transforms(img_size, train=True))
    val_ds = WrappedSubset(val_ds, get_alb_transforms(img_size, train=False))
    test_ds = WrappedSubset(test_ds, get_alb_transforms(img_size, train=False))

    # Sampler to balance classes in training
    full_labels = [y for _, y in ds_full.samples]
    train_labels = [full_labels[i] for i in train_idx]
    num_classes = len(class_to_idx)
    class_sample_count = np.array([train_labels.count(t) for t in range(num_classes)])
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    weights = 1.0 / class_sample_count
    samples_weight = [weights[t] for t in train_labels]
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=(sampler if use_sampler else None),
                              shuffle=(not use_sampler), num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=val_batch, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_to_idx


# Compatibility wrapper for older code (your original train.py used get_dataloaders)
def get_dataloaders(data_dir, img_size=256, batch_size=32, num_workers=2):
    """
    Legacy signature compatibility function.
    Calls make_dataloaders with default val/test splits and no synthetic merging.
    """
    train_loader, val_loader, test_loader, classes = make_dataloaders(
        data_dir, img_size=img_size, batch_size=batch_size, val_batch=batch_size*2,
        val_size=0.1, test_size=0.1, seed=42,
        use_sampler=True, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, sorted(list(classes.keys()))
