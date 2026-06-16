"""Cross-model comparison helpers (used by 05_comparison.ipynb).

All plotting functions take an explicit output ``path``; the notebook loads the
per-experiment metrics/NPZ and passes ``results/comparison/`` paths.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from src.dataset import CLASSES

NAMES = ["Handcrafted+SVM", "ResNet18", "DeiT-Small"]
MODEL_COLORS = ["#C44E52", "#4C72B0", "#55A868"]
CLASS_COLORS = list(plt.get_cmap("tab10").colors)


def fmt_time(sec):
    return f"{sec:.0f}s" if sec < 60 else f"{int(sec // 60)}m {int(sec % 60)}s"


def params_str(m):
    if "total_params" in m:
        return f"{m['total_params'] / 1e6:.1f}M"
    return f"{m.get('n_support_vectors', 0):,} SVs"


def params_value(m):
    return m.get("total_params", m.get("n_support_vectors", 0))


def plot_comparison_table(t, c, v, path):
    models = [t, c, v]
    rows = ["Accuracy (%)", "Macro-F1 (%)", "Macro-Precision (%)",
            "Macro-Recall (%)", "Training Time", "Inference (ms/img)",
            "Model Size (MB)", "Parameters", "Feature Dim"]
    cell = []
    cell.append([f"{m['accuracy'] * 100:.2f}" for m in models])
    cell.append([f"{m['macro_f1'] * 100:.2f}" for m in models])
    cell.append([f"{m['macro_precision'] * 100:.2f}" for m in models])
    cell.append([f"{m['macro_recall'] * 100:.2f}" for m in models])
    cell.append([fmt_time(m["train_time_sec"]) for m in models])
    cell.append([f"{m['inference_ms_per_image']:.2f}" for m in models])
    cell.append([f"{m['model_size_mb']:.2f}" for m in models])
    cell.append([params_str(m) for m in models])
    cell.append([f"{m['feature_dim']}" for m in models])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    table = ax.table(cellText=cell, rowLabels=rows, colLabels=NAMES,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.1)
    for j in range(len(NAMES)):
        table[(0, j)].set_facecolor("#40466e")
        table[(0, j)].set_text_props(color="w", fontweight="bold")
    ax.set_title("SAR ATR — Model Comparison", fontsize=15, pad=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cm_comparison(t, c, v, path):
    fig, axes = plt.subplots(1, 3, figsize=(27, 8))
    for ax, m, name in zip(axes, [t, c, v], NAMES):
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=CLASSES, yticklabels=CLASSES, square=True,
                    annot_kws={"size": 7})
        ax.set_title(f"{name} — Acc {m['accuracy'] * 100:.1f}%", fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASSES, rotation=0, fontsize=8)
    fig.suptitle("Confusion Matrices", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(t, c, v, path):
    x = np.arange(len(CLASSES))
    w = 0.27
    fig, ax = plt.subplots(figsize=(14, 6))
    for off, m, name, col in zip([-w, 0, w], [t, c, v], NAMES, MODEL_COLORS):
        vals = [m["per_class_f1"][cl] for cl in CLASSES]
        ax.bar(x + off, vals, w, label=name, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 Score Comparison", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_comparison(c, v, path):
    ch, vh = c["history"], v["history"]
    ce = range(1, len(ch) + 1)
    ve = range(1, len(vh) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(ce, [h["train_loss"] for h in ch], color="#4C72B0", label="ResNet18 Train")
    axes[0].plot(ce, [h["val_loss"] for h in ch], color="#4C72B0", ls="--", label="ResNet18 Val")
    axes[0].plot(ve, [h["train_loss"] for h in vh], color="#55A868", label="DeiT Train")
    axes[0].plot(ve, [h["val_loss"] for h in vh], color="#55A868", ls="--", label="DeiT Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].plot(ce, [h["train_acc"] for h in ch], color="#4C72B0", label="ResNet18 Train")
    axes[1].plot(ce, [h["val_acc"] for h in ch], color="#4C72B0", ls="--", label="ResNet18 Val")
    axes[1].plot(ve, [h["train_acc"] for h in vh], color="#55A868", label="DeiT Train")
    axes[1].plot(ve, [h["val_acc"] for h in vh], color="#55A868", ls="--", label="DeiT Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.axvline(c["phase_boundary"] + 0.5, color="#4C72B0", ls=":", alpha=0.5)
        ax.axvline(v["phase_boundary"] + 0.5, color="#55A868", ls=":", alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Training Curves: ResNet18 vs DeiT-Small (dotted = phase 1→2)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_inference_time(t, c, v, path):
    vals = [m["inference_ms_per_image"] for m in [t, c, v]]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(NAMES, vals, color=MODEL_COLORS)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel("Inference time (ms / image)")
    ax.set_title("Inference Latency per Image", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_complexity(t, c, v, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sizes = [m["model_size_mb"] for m in [t, c, v]]
    b1 = axes[0].bar(NAMES, sizes, color=MODEL_COLORS)
    axes[0].bar_label(b1, fmt="%.1f", padding=3)
    axes[0].set_ylabel("Model size (MB)")
    axes[0].set_title("Model Size")
    axes[0].grid(axis="y", alpha=0.3)

    params = [params_value(m) for m in [t, c, v]]
    b2 = axes[1].bar(NAMES, params, color=MODEL_COLORS)
    axes[1].bar_label(b2, labels=[params_str(m) for m in [t, c, v]], padding=3)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("# Parameters (deep) / # Support Vectors (SVM), log scale")
    axes[1].set_title("Model Complexity")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Model Complexity Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusion_pairs(cm, k=3):
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    return pairs[:k]


def plot_misclassified(models_data, test_samples, path):
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    for mi, (name, yt, yp, cm) in enumerate(models_data):
        pairs = top_confusion_pairs(cm, 3)
        col = 0
        for cnt, i, j in pairs:
            idxs = np.where((yt == i) & (yp == j))[0][:2]
            for k in range(2):
                ax = axes[mi, col]
                if k < len(idxs):
                    img = Image.open(test_samples[idxs[k]][0]).convert("L")
                    ax.imshow(img, cmap="gray")
                    ax.set_title(f"T:{CLASSES[i]}\nP:{CLASSES[j]} (n={cnt})", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                col += 1
        while col < 6:
            axes[mi, col].axis("off")
            col += 1
        axes[mi, 0].set_ylabel(name, rotation=0, ha="right", va="center",
                               fontsize=11, labelpad=45)
    fig.suptitle("Most-confused class pairs — misclassified test examples",
                 fontsize=14)
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(models_data, path):
    n_classes = len(CLASSES)
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, (name, yt, scores) in zip(axes, models_data):
        Y = label_binarize(yt, classes=list(range(n_classes)))
        fpr, tpr, roc_auc = {}, {}, {}
        for k in range(n_classes):
            fpr[k], tpr[k], _ = roc_curve(Y[:, k], scores[:, k])
            roc_auc[k] = auc(fpr[k], tpr[k])
        fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[k] for k in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[k], tpr[k])
        mean_tpr /= n_classes
        roc_auc["macro"] = auc(all_fpr, mean_tpr)

        for k in range(n_classes):
            ax.plot(fpr[k], tpr[k], color=CLASS_COLORS[k], lw=1, alpha=0.6,
                    label=f"{CLASSES[k]} ({roc_auc[k]:.2f})")
        ax.plot(fpr["micro"], tpr["micro"], "k--", lw=2.2,
                label=f"micro ({roc_auc['micro']:.3f})")
        ax.plot(all_fpr, mean_tpr, "k:", lw=2.2,
                label=f"macro ({roc_auc['macro']:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8)
        ax.set_title(f"{name} — ROC (One-vs-Rest)", fontsize=13)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("ROC Curves", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_test_images(samples, size=128):
    arr = np.zeros((len(samples), size, size), dtype=np.float32)
    for i, (p, _) in enumerate(samples):
        img = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
        arr[i] = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(len(samples), -1)


def write_report(t, c, v, path):
    def row(label, key, scale=100, suffix="%"):
        return (f"| {label} | {t[key] * scale:.2f}{suffix} | "
                f"{c[key] * scale:.2f}{suffix} | {v[key] * scale:.2f}{suffix} |")

    worst = sorted(CLASSES, key=lambda cl: t["per_class_f1"][cl])[:3]
    worst_txt = ", ".join(
        f"{cl} (SVM F1={t['per_class_f1'][cl]:.2f} → CNN {c['per_class_f1'][cl]:.2f} "
        f"→ ViT {v['per_class_f1'][cl]:.2f})" for cl in worst)
    md = f"""# SAR Automatic Target Recognition: Handcrafted Features vs Deep Learning vs Vision Transformers

## 1. Problem Statement
Classify military vehicles from Synthetic Aperture Radar (SAR) imagery on the
MSTAR benchmark. SAR images contain speckle noise and look very different from
optical imagery, which makes target recognition challenging.

## 2. Dataset
MSTAR Standard Operating Condition (SOC), 10 classes, single-channel 128×128
images. Train/test follow the standard depression-angle split (train = 17°,
test = 15°): **{t['n_train']} train / {t['n_test']} test** images. An additional
stratified 15% of the training set is held out for validation of the deep models.

## 3. Methods
1. **Handcrafted + SVM** — intensity histogram + HOG + uniform LBP
   ({t['feature_dim']}-d) classified with an RBF-SVM.
2. **ResNet18 (CNN)** — ImageNet-pretrained, two-phase fine-tuning (head, then
   last two residual blocks), 128×128 input.
3. **DeiT-Small (ViT)** — ImageNet-pretrained Vision Transformer, two-phase
   fine-tuning (head warm-up, then full-network fine-tuning), 224×224 input.
   ViTs are data-hungry and transfer poorly with a frozen backbone, so unlike
   the CNN they require deeper fine-tuning to adapt to the SAR domain.

## 4. Results
| Metric | Handcrafted+SVM | ResNet18 | DeiT-Small |
|---|---|---|---|
{row("Accuracy", "accuracy")}
{row("Macro-F1", "macro_f1")}
{row("Macro-Precision", "macro_precision")}
{row("Macro-Recall", "macro_recall")}
| Training time | {fmt_time(t['train_time_sec'])} | {fmt_time(c['train_time_sec'])} | {fmt_time(v['train_time_sec'])} |
| Inference (ms/img) | {t['inference_ms_per_image']:.2f} | {c['inference_ms_per_image']:.2f} | {v['inference_ms_per_image']:.2f} |
| Model size (MB) | {t['model_size_mb']:.2f} | {c['model_size_mb']:.2f} | {v['model_size_mb']:.2f} |
| Parameters | {params_str(t)} | {params_str(c)} | {params_str(v)} |
| Feature dim | {t['feature_dim']} | {c['feature_dim']} | {v['feature_dim']} |

## 5. Key Findings
- **Where the traditional method breaks.** The handcrafted+SVM pipeline is
  weakest on the most visually similar vehicles, e.g. {worst_txt}. HOG/LBP encode
  local edge/texture statistics that cannot separate vehicles whose SAR
  signatures differ only in subtle structural detail.
- **What the CNN adds.** ResNet18 lifts accuracy from
  {t['accuracy'] * 100:.1f}% to {c['accuracy'] * 100:.1f}%
  (+{(c['accuracy'] - t['accuracy']) * 100:.1f} pts) and macro-F1 from
  {t['macro_f1'] * 100:.1f}% to {c['macro_f1'] * 100:.1f}%. Learned hierarchical
  features are far more discriminative than fixed descriptors, and the t-SNE plots
  show much cleaner class clusters than raw pixels.
- **The transformer's extra edge.** DeiT-Small reaches
  {v['accuracy'] * 100:.1f}% accuracy / {v['macro_f1'] * 100:.1f}% macro-F1
  ({'+' if v['accuracy'] >= c['accuracy'] else ''}{(v['accuracy'] - c['accuracy']) * 100:.1f} pts vs ResNet18).
  Global self-attention captures long-range spatial relationships across the target.
- **Trade-off (accuracy vs compute).** Higher accuracy costs compute: DeiT-Small
  is the largest ({params_str(v)}) and slowest ({v['inference_ms_per_image']:.2f} ms/img)
  model, ResNet18 is a strong middle ground ({c['inference_ms_per_image']:.2f} ms/img),
  and the SVM is cheapest to store but bottlenecked by handcrafted-feature extraction
  at inference.

## 6. Conclusion
On MSTAR SOC, moving from handcrafted features to deep learning yields a large,
consistent accuracy gain, and the Vision Transformer matches or exceeds the CNN
while costing the most compute. For SAR ATR, learned representations clearly
dominate fixed descriptors; the CNN is the best accuracy/efficiency compromise,
whereas the ViT is preferable when maximum accuracy justifies the extra cost.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
