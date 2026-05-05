# eval_only.py
import argparse
import os
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import get_dataloaders
from model import build_model
from utils import save_classification_report, plot_confusion_matrix

def tta_transform(img_size=224):
    """Test-Time Augmentation base transforms."""
    return [
        A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.Normalize(), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0), A.Normalize(), ToTensorV2()])
    ]

def evaluate(data_dir, ckpt_path, out_dir="eval_results", img_size=224, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    # Note: get_dataloaders uses standard center crop. For advanced TTA, 
    # you would apply the tta_transform dynamically over multiple passes.
    _, _, test_loader, classes = get_dataloaders(data_dir, img_size=img_size, batch_size=batch_size)
    num_classes = len(classes)
    
    model = build_model(num_classes=num_classes)
    
    # Load model weights (handle both dict with 'model_state_dict' or raw state dict)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    save_classification_report(all_targets, all_preds, classes, out_dir)
    plot_confusion_matrix(all_targets, all_preds, classes, out_dir)
    print("Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--out_dir", type=str, default="eval_results", help="Output directory")
    args = parser.parse_args()
    evaluate(args.data, args.ckpt, args.out_dir)
