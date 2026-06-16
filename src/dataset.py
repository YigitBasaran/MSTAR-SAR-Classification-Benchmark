"""MSTAR SAR dataset utilities.

Provides the PyTorch ``MSTARDataset`` class, transform builders, a stratified
train/validation split and ready-to-use DataLoaders.

Notes
-----
* The dataset is expected at ``data/mstar_raw/soc/{train,test}/<CLASS>/*.jpg``
  (the layout of the ``omae11/MSTAR-dataset`` GitHub repository).
* Grayscale -> 3-channel conversion uses a picklable callable class
  (``RepeatTo3Channels``) instead of ``transforms.Lambda(lambda ...)`` so that
  ``num_workers > 0`` works under the Windows ``spawn`` start method.
"""

import os
import glob
import json
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "mstar_raw", "soc")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
STATS_PATH = os.path.join(PROJECT_ROOT, "results", "eda", "dataset_stats.json")

# 10 MSTAR SOC classes (sorted == torchvision ImageFolder convention).
CLASSES = ["2S1", "BMP2", "BRDM2", "BTR60", "BTR70",
           "D7", "T62", "T72", "ZIL131", "ZSU234"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

RESNET_SIZE = 128       # ResNet18 keeps the native 128x128 resolution.
DEIT_SIZE = 224         # DeiT-Small expects 224x224 (patch16).
RESNET_BATCH = 64
DEIT_BATCH = 32         # Smaller batch keeps DeiT@224 within 8 GB of VRAM.
NUM_WORKERS = 0         # 0 avoids Windows 'spawn' DataLoader hangs (dataset is tiny).
VAL_FRAC = 0.15
SEED = 42

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
class RepeatTo3Channels:
    """Repeat a single grayscale channel into 3 channels (picklable)."""

    def __call__(self, x):
        return x.repeat(3, 1, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def get_dataset_stats():
    """Return ``(mean, std)`` scalars in [0, 1].

    Reads ``results/metrics/dataset_stats.json`` (written by ``eda.py``); if the
    file is missing it falls back to ``(0.5, 0.5)`` so the pipeline still runs.
    """
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return float(stats["pixel_mean"]), float(stats["pixel_std"])
    warnings.warn("dataset_stats.json not found; using mean=0.5, std=0.5. "
                  "Run eda.py first for dataset-specific normalization.")
    return 0.5, 0.5


def build_transforms(size, train, mean=None, std=None):
    """Build a torchvision transform pipeline for the given image ``size``."""
    if mean is None or std is None:
        mean, std = get_dataset_stats()
    ops = [transforms.Resize((size, size))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ]
    ops += [
        transforms.ToTensor(),                       # -> (1, H, W) in [0, 1]
        transforms.Normalize(mean=[mean], std=[std]),
        RepeatTo3Channels(),                         # -> (3, H, W)
    ]
    return transforms.Compose(ops)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def scan_samples(root):
    """Return a sorted list of ``(image_path, label_idx)`` under ``root``."""
    samples = []
    for cls in CLASSES:
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for p in glob.glob(os.path.join(cdir, "*")):
            if p.lower().endswith(IMG_EXTS):
                samples.append((p, CLASS_TO_IDX[cls]))
    return sorted(samples)


class MSTARDataset(Dataset):
    """MSTAR SAR image dataset.

    Either pass ``root_dir`` (folder is scanned) or an explicit ``samples`` list
    (used for the stratified train/val split so both share label indices while
    using different transforms).
    """

    def __init__(self, root_dir=None, transform=None, samples=None, classes=None):
        self.transform = transform
        self.classes = list(classes) if classes is not None else CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        if samples is not None:
            self.samples = samples
        else:
            if root_dir is None:
                raise ValueError("Provide either root_dir or samples.")
            self.samples = scan_samples(root_dir)
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found (root_dir={root_dir}). "
                "Has the dataset clone finished into data/mstar_raw/soc/ ?")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")          # force single-channel
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def stratified_split(samples, val_frac=VAL_FRAC, seed=SEED):
    """Stratified split of ``samples`` into ``(train_samples, val_samples)``."""
    labels = [lbl for _, lbl in samples]
    train_s, val_s = train_test_split(
        samples, test_size=val_frac, stratify=labels, random_state=seed)
    return train_s, val_s


def make_loaders(size, batch_size, num_workers=NUM_WORKERS,
                 mean=None, std=None, seed=SEED):
    """Build ``(train_loader, val_loader, test_loader, datasets)``.

    Train uses augmentation; val/test use the deterministic eval transform.
    """
    if mean is None or std is None:
        mean, std = get_dataset_stats()
    train_tf = build_transforms(size, train=True, mean=mean, std=std)
    eval_tf = build_transforms(size, train=False, mean=mean, std=std)

    full_train = scan_samples(TRAIN_DIR)
    train_s, val_s = stratified_split(full_train)

    train_ds = MSTARDataset(samples=train_s, transform=train_tf, classes=CLASSES)
    val_ds = MSTARDataset(samples=val_s, transform=eval_tf, classes=CLASSES)
    test_ds = MSTARDataset(root_dir=TEST_DIR, transform=eval_tf, classes=CLASSES)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              generator=g, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
