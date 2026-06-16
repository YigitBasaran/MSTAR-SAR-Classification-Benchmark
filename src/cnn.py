"""ResNet18 CNN pipeline + the generic two-phase fine-tuning trainer.

The trainer (``train_two_phase``), the inference helper (``predict``) and the
per-epoch loop (``run_epoch``) are model-agnostic and are reused by ``vit.py``.
"""

import os

import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_resnet18(num_classes=10):
    """ImageNet-pretrained ResNet18 with a fresh ``num_classes`` head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_phase1_trainable(model):
    """Phase 1: train only the classification head (``fc``)."""
    for name, p in model.named_parameters():
        p.requires_grad = ("fc" in name)


def set_phase2_trainable(model):
    """Phase 2: unfreeze the last two residual blocks + head."""
    for name, p in model.named_parameters():
        p.requires_grad = ("layer3" in name or "layer4" in name or "fc" in name)


# --------------------------------------------------------------------------- #
# Generic training / inference (reused by the ViT pipeline)
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer, scaler, device, train, use_amp,
              desc=""):
    """Run one epoch; return ``(avg_loss, accuracy)``."""
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc=desc, leave=False, disable=None):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train_two_phase(model, train_loader, val_loader, device, ckpt_path,
                    phase1_setup, phase2_setup,
                    epochs_phase1=10, epochs_phase2=15,
                    lr1=1e-3, lr2=1e-4, weight_decay=1e-4, patience=7,
                    use_amp=True, log_fn=print):
    """Two-phase fine-tuning with early stopping and best-val-acc checkpointing.

    Returns ``(history, phase_boundary)`` where ``history`` is a list of per-epoch
    dicts and ``phase_boundary`` is the number of phase-1 epochs actually run
    (used to draw the phase transition line on the training curves).
    """
    criterion = nn.CrossEntropyLoss()
    history = []
    state = {"best_val_acc": -1.0}
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    def run_phase(phase_id, setup_fn, n_epochs, lr, use_scheduler):
        setup_fn(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
                     if use_scheduler else None)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        epochs_no_improve = 0
        for ep in range(1, n_epochs + 1):
            cur_lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                        scaler, device, True, use_amp,
                                        desc=f"P{phase_id} ep{ep} train")
            va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer,
                                        scaler, device, False, use_amp,
                                        desc=f"P{phase_id} ep{ep} val")
            if scheduler is not None:
                scheduler.step()
            history.append({"phase": phase_id, "epoch": ep, "lr": cur_lr,
                            "train_loss": tr_loss, "val_loss": va_loss,
                            "train_acc": tr_acc, "val_acc": va_acc})
            improved = va_acc > state["best_val_acc"]
            if improved:
                state["best_val_acc"] = va_acc
                torch.save(model.state_dict(), ckpt_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            log_fn(f"[P{phase_id} {ep:02d}/{n_epochs}] lr={cur_lr:.2e} "
                   f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                   f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} "
                   f"best={state['best_val_acc']:.4f}" + (" *" if improved else ""))
            if epochs_no_improve >= patience:
                log_fn(f"[P{phase_id}] early stopping at epoch {ep}")
                break

    run_phase(1, phase1_setup, epochs_phase1, lr1, use_scheduler=False)
    phase_boundary = len(history)
    run_phase(2, phase2_setup, epochs_phase2, lr2, use_scheduler=True)
    return history, phase_boundary


@torch.no_grad()
def predict(model, loader, device, use_amp=True):
    """Return ``(y_true, y_pred, probs)`` over a loader."""
    model.eval()
    logits_all, y_all = [], []
    for x, y in tqdm(loader, desc="predict", leave=False, disable=None):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            out = model(x)
        logits_all.append(out.float().cpu())
        y_all.append(y)
    logits = torch.cat(logits_all)
    y_true = torch.cat(y_all).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    y_pred = probs.argmax(1)
    return y_true, y_pred, probs


@torch.no_grad()
def extract_resnet_features(model, loader, device, use_amp=True):
    """Extract 512-d global-average-pool features from a ResNet (for t-SNE)."""
    model.eval()
    store = {}

    def hook(_m, _i, out):
        store["feat"] = out.flatten(1).float().cpu()

    handle = model.avgpool.register_forward_hook(hook)
    feats, labels = [], []
    try:
        for x, y in tqdm(loader, desc="resnet-feats", leave=False, disable=None):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                _ = model(x)
            feats.append(store["feat"])
            labels.append(y)
    finally:
        handle.remove()
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()
