# utils.py
import os
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

def plot_curves(history, out_dir="."):
    """
    Draws detailed training/validation curves with annotations for each epoch.
    history: dict with keys -> 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    Saves: training_curves_detailed.png into out_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    axes[0].set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy Curve
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ca02c")
    axes[1].plot(epochs, history["val_acc"], "s--", label="Val Accuracy", color="#d62728")
    axes[1].set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Model Training Performance per Epoch", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(out_dir, "training_curves_detailed.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


import numpy as np

def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam


import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def _save_table_pages_pdf(df, pdf_path, rows_per_page=30):
    """Saves a large Pandas DataFrame as a multi-page PDF."""
    with PdfPages(pdf_path) as pdf:
        num_pages = (len(df) + rows_per_page - 1) // rows_per_page
        for page in range(num_pages):
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.axis('tight')
            ax.axis('off')
            subset = df.iloc[page*rows_per_page:(page+1)*rows_per_page]
            table = ax.table(cellText=subset.values, colLabels=subset.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            plt.title(f"Evaluation Metrics - Page {page+1}/{num_pages}", fontsize=14, pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

def save_classification_report(y_true, y_pred, class_names, out_dir="."):
    """Generates and saves a detailed per-class precision/recall/f1-score report."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Save standard TXT report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        
    # Save multi-page PDF
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report_dict).transpose().round(4)
    df.reset_index(inplace=True)
    df.columns = ['Class / Metric', 'Precision', 'Recall', 'F1-Score', 'Support']
    
    pdf_path = os.path.join(out_dir, "model_evaluation_table.pdf")
    _save_table_pages_pdf(df, pdf_path)
    
    print(f"Saved classification report to {os.path.join(out_dir, 'classification_report.txt')} and PDF")

def plot_confusion_matrix(y_true, y_pred, class_names, out_dir="."):
    """Plots and saves a high-resolution heatmap of the confusion matrix."""
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {os.path.join(out_dir, 'confusion_matrix.png')}")
