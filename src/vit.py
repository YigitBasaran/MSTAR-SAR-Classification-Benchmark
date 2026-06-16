"""DeiT-Small Vision Transformer pipeline.

Reuses the generic two-phase trainer / inference helpers from ``src.cnn``.
"""

import timm
import torch

# Re-export the shared trainer/inference helpers for convenience.
from src.cnn import train_two_phase, predict, run_epoch  # noqa: F401


def build_deit(num_classes=10):
    """ImageNet-pretrained DeiT-Small (patch16, 224) with a ``num_classes`` head."""
    return timm.create_model("deit_small_patch16_224", pretrained=True,
                             num_classes=num_classes)


def set_phase1_trainable(model):
    """Phase 1: train only the classification head."""
    for name, p in model.named_parameters():
        p.requires_grad = ("head" in name)


def set_phase2_trainable(model):
    """Phase 2: full fine-tuning (unfreeze the entire network).

    ViTs are more data-hungry than CNNs and transfer poorly with a frozen
    backbone on an out-of-domain target such as SAR. Unfreezing the whole DeiT
    lets it adapt to the SAR domain and reach its accuracy potential on MSTAR
    (the shallow last-2-blocks recipe that suffices for ResNet18 leaves the ViT
    stuck on ImageNet-centric low-level features)."""
    for p in model.parameters():
        p.requires_grad = True


@torch.no_grad()
def extract_deit_features(model, loader, device, use_amp=True):
    """Extract 384-d pre-logits CLS features from DeiT (for t-SNE)."""
    from tqdm import tqdm
    model.eval()
    feats, labels = [], []
    for x, y in tqdm(loader, desc="deit-feats", leave=False, disable=None):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            tokens = model.forward_features(x)              # (B, N, C)
            pooled = model.forward_head(tokens, pre_logits=True)  # (B, 384)
        feats.append(pooled.float().cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()
