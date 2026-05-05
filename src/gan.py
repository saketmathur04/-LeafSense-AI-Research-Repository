# gan.py
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

class SimpleImageFolder(Dataset):
    def __init__(self, root, img_size=128):
        self.root = Path(root)
        self.samples = []
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        for idx, cls in enumerate(self.classes):
            for f in (self.root/cls).glob("*"):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((str(f), idx))
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), y

class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3, n_classes=10, embed_dim=50):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        in_dim = nz + embed_dim
        self.fc = nn.Linear(in_dim, ngf * 8 * 4 * 4)
        self.net = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, nc, 3, 1, 1), nn.Tanh()
        )
    def forward(self, z, labels):
        e = self.embed(labels)
        x = torch.cat([z, e], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_classes=10, embed_dim=50):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, 1, 4, 1, 0)
        )
        self.fc_emb = nn.Linear(embed_dim, 1)
    def forward(self, img, labels):
        out = self.conv(img).view(img.size(0), -1)  # (B,1)
        emb = self.embed(labels)
        proj = self.fc_emb(emb).view(-1, 1)
        return out + proj
