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


import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def save_classification_report(y_true, y_pred, class_names, out_dir="."):
    """Generates and saves a detailed per-class precision/recall/f1-score report."""
    os.makedirs(out_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report to {os.path.join(out_dir, 'classification_report.txt')}")

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
