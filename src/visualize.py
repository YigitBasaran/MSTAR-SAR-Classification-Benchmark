"""Feature-space and explainability visualizations (t-SNE comparison, Grad-CAM)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import build_transforms

CLASS_COLORS = list(plt.get_cmap("tab10").colors)


# --------------------------------------------------------------------------- #
# t-SNE comparison
# --------------------------------------------------------------------------- #
def run_tsne(X, seed=42, pca_dim=50, perplexity=30):
    """PCA (if needed) -> 2D t-SNE embedding."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                init="pca").fit_transform(X)


def plot_tsne_comparison(panels, classes, path,
                         suptitle="Feature Space Comparison: Raw Pixels vs ResNet18 vs DeiT-Small"):
    """``panels``: list of ``(title, emb2d, labels)`` (one per subplot).

    ``suptitle`` overrides the figure title; the default reproduces the original
    raw-pixel comparison so existing callers are unaffected.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, emb, labels) in zip(axes, panels):
        for r, cls in enumerate(classes):
            m = labels == r
            ax.scatter(emb[m, 0], emb[m, 1], s=6, color=CLASS_COLORS[r],
                       label=cls, alpha=0.7)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=10, fontsize=9, markerscale=2)
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def _deit_reshape_transform(tensor, height=14, width=14):
    """Reshape DeiT token sequence (drop CLS) into a 2D feature map for CAM."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def _cam_for(model, target_layers, pil_img, size, label, device, reshape=None):
    """Return ``(rgb01, grayscale_cam)`` for a single image."""
    eval_tf = build_transforms(size, train=False)
    inp = eval_tf(pil_img).unsqueeze(0).to(device)
    rgb = np.asarray(pil_img.resize((size, size)), dtype=np.float32) / 255.0
    rgb = np.stack([rgb] * 3, axis=-1)
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape)
    grayscale = cam(input_tensor=inp, targets=[ClassifierOutputTarget(int(label))])[0]
    return rgb, grayscale


def gradcam_grid(resnet_model, deit_model, test_samples, cnn_pred, vit_pred,
                 y_true, classes, device, path, seed=42):
    """10x6 grid: [Original | RN CAM | RN Overlay | Original | DeiT CAM | DeiT Overlay]."""
    resnet_model.eval()
    deit_model.eval()
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    cnn_pred = np.asarray(cnn_pred)
    vit_pred = np.asarray(vit_pred)

    rn_targets = [resnet_model.layer4[-1]]
    deit_targets = [deit_model.blocks[-1].norm1]

    n = len(classes)
    fig, axes = plt.subplots(n, 6, figsize=(20, 3.2 * n))
    col_titles = ["Original", "ResNet18 CAM", "ResNet18 Overlay",
                  "Original", "DeiT CAM", "DeiT Overlay"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12)

    for r, cls in enumerate(classes):
        both = np.where((y_true == r) & (cnn_pred == r) & (vit_pred == r))[0]
        pool = both if len(both) else np.where(y_true == r)[0]
        idx = int(rng.choice(pool))
        path_img, _ = test_samples[idx]
        pil = Image.open(path_img).convert("L")

        rgb128, cam_rn = _cam_for(resnet_model, rn_targets, pil, 128, r, device)
        rgb224, cam_dt = _cam_for(deit_model, deit_targets, pil, 224, r, device,
                                  reshape=_deit_reshape_transform)
        ov_rn = show_cam_on_image(rgb128, cam_rn, use_rgb=True)
        ov_dt = show_cam_on_image(rgb224, cam_dt, use_rgb=True)

        panels = [rgb128, cam_rn, ov_rn, rgb224, cam_dt, ov_dt]
        cmaps = ["gray", "jet", None, "gray", "jet", None]
        for j, (img, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[r, j]
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(cls, rotation=0, ha="right", va="center",
                              fontsize=12, labelpad=30)

    fig.suptitle("Grad-CAM: ResNet18 vs DeiT-Small (one correct sample per class)",
                 fontsize=15, y=0.997)
    fig.tight_layout(rect=[0.03, 0, 1, 0.99])
    fig.savefig(path, dpi=130)
    plt.close(fig)
