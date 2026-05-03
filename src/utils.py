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
