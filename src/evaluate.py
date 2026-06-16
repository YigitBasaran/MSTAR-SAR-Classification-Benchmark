"""Shared evaluation utilities: metrics, confusion-matrix plotting, timing, seeding."""

import os
import time
import json
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)


def seed_everything(seed=42):
    """Seed python, numpy and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred, classes):
    """Return a dict with accuracy, macro P/R/F1, per-class F1, report and CM."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    per_class_f1 = {c: report[c]["f1-score"] for c in classes}
    per_class_precision = {c: report[c]["precision"] for c in classes}
    per_class_recall = {c: report[c]["recall"] for c in classes}
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm, classes, title, path):
    """Save a 10x10 confusion matrix heatmap (counts)."""
    cm = np.asarray(cm)
    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True,
                xticklabels=classes, yticklabels=classes, cbar=True,
                annot_kws={"size": 8})
    plt.title(title, fontsize=14)
    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def count_params(model):
    """Return ``(total_params, trainable_params)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def file_size_mb(path):
    """Size of a file in megabytes (0 if missing)."""
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


@torch.no_grad()
def measure_latency_ms(model, dataset, device, n=500, warmup=10):
    """Average per-image forward latency in milliseconds (batch size 1).

    Tensors are pre-loaded so that only the model forward pass is timed.
    """
    model.eval()
    n = min(n, len(dataset))
    tensors = [dataset[i][0].unsqueeze(0).to(device) for i in range(n)]
    for i in range(min(warmup, n)):
        _ = model(tensors[i])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x in tensors:
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / n * 1000.0


def save_json(obj, path):
    """Write ``obj`` to ``path`` as pretty JSON (creating parent dirs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_training_curves(history, phase_boundary, title, path):
    """Plot train/val loss and accuracy with the phase-1 -> phase-2 boundary."""
    epochs = list(range(1, len(history) + 1))
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    tr_acc = [h["train_acc"] for h in history]
    va_acc = [h["val_acc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, tr_loss, "-o", ms=3, label="Train")
    axes[0].plot(epochs, va_loss, "-o", ms=3, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, tr_acc, "-o", ms=3, label="Train")
    axes[1].plot(epochs, va_acc, "-o", ms=3, label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    for ax in axes:
        if 0 < phase_boundary < len(history):
            ax.axvline(phase_boundary + 0.5, color="gray", ls="--", alpha=0.7)
            ymin = ax.get_ylim()[0]
            ax.text(phase_boundary + 0.5, ymin, " P1→P2", color="gray",
                    fontsize=9, va="bottom")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)
