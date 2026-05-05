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

def train_gan(data_dir, epochs=100, batch_size=128, lr=0.0002, nz=128, img_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GAN on {device}")

    ds = SimpleImageFolder(data_dir, img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    n_classes = len(ds.classes)

    netG = Generator(nz=nz, n_classes=n_classes).to(device)
    netD = Discriminator(n_classes=n_classes).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    os.makedirs("gan_checkpoints", exist_ok=True)

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for real_imgs, labels in pbar:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            # Train D
            netD.zero_grad()
            real_out = netD(real_imgs, labels)
            errD_real = criterion(real_out, torch.ones_like(real_out))
            
            noise = torch.randn(b_size, nz, device=device)
            fake_imgs = netG(noise, labels)
            fake_out = netD(fake_imgs.detach(), labels)
            errD_fake = criterion(fake_out, torch.zeros_like(fake_out))
            
            errD = errD_real + errD_fake
            errD.backward()
            optD.step()

            # Train G
            netG.zero_grad()
            out = netD(fake_imgs, labels)
            errG = criterion(out, torch.ones_like(out))
            errG.backward()
            optG.step()

            pbar.set_postfix({'D_loss': f"{errD.item():.4f}", 'G_loss': f"{errG.item():.4f}"})

        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
            }, f"gan_checkpoints/gan_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to dataset root")
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    train_gan(args.data, epochs=args.epochs)
