"""Handcrafted feature extraction for the traditional pipeline.

Each 128x128 grayscale image is described by the concatenation of:
    A) a 64-bin intensity histogram          (64 dims)
    B) Histogram of Oriented Gradients (HOG)  (8100 dims)
    C) a uniform Local Binary Pattern histogram (26 dims)
=> 8190-dimensional feature vector.
"""

import time

import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern

FEAT_SIZE = 128
HIST_BINS = 64
LBP_P = 24
LBP_R = 3
LBP_BINS = LBP_P + 2          # 26 bins for the 'uniform' method
HOG_KWARGS = dict(orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm="L2-Hys",
                  feature_vector=True)


def load_gray_128(path):
    """Load an image as a 128x128 uint8 grayscale array (0-255)."""
    img = Image.open(path).convert("L").resize((FEAT_SIZE, FEAT_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def extract_features(img):
    """Compute the 8190-d handcrafted feature vector for a uint8 image."""
    img = np.asarray(img)
    # A) Intensity histogram (normalized).
    hist, _ = np.histogram(img.ravel(), bins=HIST_BINS, range=(0, 255), density=True)
    # B) HOG descriptor.
    hog_feat = hog(img, **HOG_KWARGS)
    # C) Uniform LBP histogram.
    lbp = local_binary_pattern(img, P=LBP_P, R=LBP_R, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS,
                               range=(0, LBP_BINS), density=True)
    return np.concatenate([hist, hog_feat, lbp_hist]).astype(np.float32)


def extract_dataset(samples, desc="features"):
    """Extract features for a list of ``(path, label)`` samples.

    Returns ``(X, y, timing)`` where ``timing`` reports the total and
    per-image extraction cost.
    """
    feats, labels = [], []
    t0 = time.perf_counter()
    for path, label in tqdm(samples, desc=desc, disable=None):
        feats.append(extract_features(load_gray_128(path)))
        labels.append(label)
    elapsed = time.perf_counter() - t0
    X = np.vstack(feats)
    y = np.asarray(labels, dtype=np.int64)
    timing = {
        "total_sec": elapsed,
        "per_image_ms": elapsed / len(samples) * 1000.0,
        "feature_dim": int(X.shape[1]),
    }
    return X, y, timing
