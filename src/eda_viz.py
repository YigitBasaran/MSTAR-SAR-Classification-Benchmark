"""EDA visualization helpers for the MSTAR SAR dataset (used by 01_eda.ipynb).

All functions take explicit output paths so the EDA notebook can write under
``results/eda/``.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.dataset import CLASSES

IMG_SIZE = 128
SEED = 42
COLORS = list(plt.get_cmap("tab10").colors)


def load_images(samples, size=IMG_SIZE):
    """Load all samples into an ``(N, size, size)`` uint8 array + label array."""
    imgs = np.zeros((len(samples), size, size), dtype=np.uint8)
    labels = np.zeros(len(samples), dtype=np.int64)
    for i, (path, label) in enumerate(tqdm(samples, desc="load-train", disable=None)):
        img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
        imgs[i] = np.asarray(img, dtype=np.uint8)
        labels[i] = label
    return imgs, labels


def plot_class_distribution(train_counts, test_counts, path):
    x = np.arange(len(CLASSES))
    w = 0.4
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w / 2, train_counts, w, label="Train (17°)", color="#4C72B0")
    b2 = ax.bar(x + w / 2, test_counts, w, label="Test (15°)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Number of images", fontsize=12)
    ax.set_title("MSTAR SAR — Class Distribution (Train vs Test)", fontsize=14)
    ax.bar_label(b1, fontsize=8, padding=2)
    ax.bar_label(b2, fontsize=8, padding=2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_images(imgs, labels, path, rng):
    fig, axes = plt.subplots(len(CLASSES), 5, figsize=(10, 20))
    for r, cls in enumerate(CLASSES):
        idxs = np.where(labels == r)[0]
        pick = rng.choice(idxs, size=5, replace=False)
        for j, pi in enumerate(pick):
            ax = axes[r, j]
            ax.imshow(imgs[pi], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(cls, rotation=0, ha="right", va="center",
                              fontsize=11, labelpad=28)
    fig.suptitle("MSTAR SAR Dataset — Sample Images per Class", fontsize=14, y=0.997)
    fig.tight_layout(rect=[0.05, 0, 1, 0.99])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pixel_intensity(imgs, labels, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for r, cls in enumerate(CLASSES):
        px = imgs[labels == r].ravel()
        ax.hist(px, bins=64, range=(0, 255), density=True, alpha=0.5,
                color=COLORS[r], label=cls, histtype="stepfilled")
    ax.set_xlabel("Pixel intensity (0-255)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("MSTAR SAR — Per-class Pixel Intensity Distribution", fontsize=14)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mean_images(imgs, labels, path):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for r, cls in enumerate(CLASSES):
        mean_img = imgs[labels == r].mean(axis=0)
        ax = axes[r // 5, r % 5]
        ax.imshow(mean_img, cmap="gray")
        ax.set_title(cls, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("MSTAR SAR — Per-class Mean Image (Train)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tsne_raw(imgs, labels, path, npy_path=None):
    print("Running PCA-50 -> t-SNE on raw pixels (this can take ~1 min)...")
    X = imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0
    pca = PCA(n_components=50, random_state=SEED).fit_transform(X)
    emb = TSNE(n_components=2, perplexity=30, random_state=SEED,
               init="pca").fit_transform(pca)
    fig, ax = plt.subplots(figsize=(10, 8))
    for r, cls in enumerate(CLASSES):
        m = labels == r
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=COLORS[r], label=cls, alpha=0.7)
    ax.set_title("t-SNE of Raw Pixels (Train, PCA-50 → t-SNE, perplexity=30)",
                 fontsize=14)
    ax.legend(ncol=2, fontsize=9, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if npy_path is not None:
        np.save(npy_path, np.column_stack([emb, labels]))
