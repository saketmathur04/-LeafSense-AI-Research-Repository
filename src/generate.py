# generate.py
import os
import argparse
import torch
import torchvision.utils as vutils
from gan import Generator

def generate_images(ckpt_path, num_images=100, class_idx=0, out_dir="synthetic_data", nz=128, n_classes=88):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading generator from {ckpt_path} onto {device}")

    netG = Generator(nz=nz, n_classes=n_classes).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(checkpoint['netG'])
    netG.eval()

    class_dir = os.path.join(out_dir, f"class_{class_idx:02d}")
    os.makedirs(class_dir, exist_ok=True)

    print(f"Generating {num_images} images for class {class_idx}...")
    batch_size = 64
    num_batches = (num_images + batch_size - 1) // batch_size
    
    img_count = 0
    with torch.no_grad():
        for i in range(num_batches):
            b_size = min(batch_size, num_images - img_count)
            noise = torch.randn(b_size, nz, device=device)
            labels = torch.full((b_size,), class_idx, dtype=torch.long, device=device)
            
            fakes = netG(noise, labels)
            # Denormalize from [-1, 1] to [0, 1]
            fakes = (fakes + 1) / 2.0
            
            for j in range(b_size):
                vutils.save_image(fakes[j], os.path.join(class_dir, f"fake_{img_count:04d}.png"))
                img_count += 1

    print(f"Done! Saved to {class_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to GAN checkpoint")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--class_idx", type=int, default=0, help="Class index to generate")
    parser.add_argument("--out_dir", type=str, default="synthetic_data", help="Output directory")
    args = parser.parse_args()
    generate_images(args.ckpt, args.num_images, args.class_idx, args.out_dir)
