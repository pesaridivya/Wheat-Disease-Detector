"""
Wheat Disease Detection — Hybrid EfficientNet-B3 + ViT Model
Classes: Yellow Rust | Black Rust | Brown Rust | Healthy
Target accuracy: 95%+

Requirements:
    pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm

Your dataset folder structure (matches exactly):
    Wheat_Disease/
        train/
            Black Rust/       (~1000 images)
            Brown Rust/       (~1000 images)
            Healthy Wheat/    (~1000 images)
            Yellow Rust/      (~1000 images)
        validation/
            Black Rust/
            Brown Rust/
            Healthy Wheat/
            Yellow Rust/

No pre-processing needed — run train.py directly.
"""

import os
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.models.vision_transformer import VisionTransformer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Matches your exact folder layout ──────────────────────────────────
    # Wheat_Disease/
    #     train/  Black Rust | Brown Rust | Healthy Wheat | Yellow Rust
    #     validation/  (same sub-folders)
    "data_dir":       "Wheat_Disease",    # root folder — change if path differs
    "val_split":      "validation",       # your val folder is called "validation"
    "img_size":       224,
    "batch_size":     32,
    "num_classes":    4,
    # ImageFolder sorts sub-folders alphabetically → Black Rust=0, Brown Rust=1,
    # Healthy Wheat=2, Yellow Rust=3
    "class_names":    ["Black Rust", "Brown Rust", "Healthy Wheat", "Yellow Rust"],

    # Phase 1 — train head only (frozen backbones)
    "phase1_epochs":  10,
    "phase1_lr":      1e-3,

    # Phase 2 — fine-tune top layers
    "phase2_epochs":  20,
    "phase2_lr":      1e-4,

    "weight_decay":   1e-4,
    "dropout1":       0.40,
    "dropout2":       0.30,
    "label_smoothing": 0.1,
    "num_workers":    4,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "save_path":      "best_wheat_model.pth",
    "results_dir":    "results",
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)
print(f"Using device: {CONFIG['device']}")


# ─── Data transforms ─────────────────────────────────────────────────────────
def get_transforms(split: str, img_size: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def build_dataloaders(cfg):
    loaders = {}
    # Map logical split names → actual folder names on disk
    split_map = {
        "train": "train",
        "val":   cfg.get("val_split", "val"),   # "validation" for this dataset
        "test":  "test",
    }
    for split, folder in split_map.items():
        path = os.path.join(cfg["data_dir"], folder)
        if not os.path.isdir(path):
            print(f"  [warn] '{path}' not found — skipping {split} loader")
            continue
        ds = datasets.ImageFolder(path, transform=get_transforms(split, cfg["img_size"]))
        # Print the class→index mapping once so you can verify the order
        if split == "train":
            print(f"  Class mapping: {ds.class_to_idx}")
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=shuffle,
            num_workers=cfg["num_workers"],
            pin_memory=(cfg["device"] == "cuda"),
        )
        print(f"  {split:5s} ({folder}): {len(ds):5d} images  |  {len(loaders[split]):3d} batches")
    return loaders


# ─── Squeeze-and-Excitation attention ────────────────────────────────────────
class SEBlock(nn.Module):
    """Channel-wise attention applied to concatenated features."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        scale = self.fc(x)          # (B, C)
        return x * scale


# ─── Hybrid model ─────────────────────────────────────────────────────────────
class WheatHybridNet(nn.Module):
    """
    Parallel EfficientNet-B3 + ViT-B/16 trunks.
    Features concatenated → SE attention → 2-layer FC head.
    """
    def __init__(self, num_classes: int = 4, dropout1: float = 0.4,
                 dropout2: float = 0.3):
        super().__init__()

        # ── Trunk 1: EfficientNet-B3 ──────────────────────────────────────
        self.eff = timm.create_model(
            "efficientnet_b3", pretrained=True, num_classes=0, global_pool="avg"
        )
        eff_dim = self.eff.num_features          # 1536

        # ── Trunk 2: ViT-B/16 ────────────────────────────────────────────
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        vit_dim = self.vit.num_features          # 768

        fused_dim = eff_dim + vit_dim            # 2304

        # ── Channel attention ─────────────────────────────────────────────
        self.se = SEBlock(fused_dim, reduction=16)

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(256, num_classes),
        )

    def freeze_backbones(self):
        """Phase 1: only head trains."""
        for p in self.eff.parameters():
            p.requires_grad = False
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self):
        """Phase 2: unfreeze last 2 blocks of EfficientNet + last 4 ViT blocks."""
        # Unfreeze EfficientNet blocks 5-7 (top layers)
        for name, param in self.eff.named_parameters():
            if any(f"blocks.{i}" in name for i in [5, 6, 7]):
                param.requires_grad = True
        # Unfreeze ViT last 4 transformer blocks
        for name, param in self.vit.named_parameters():
            if any(f"blocks.{i}" in name for i in [8, 9, 10, 11]):
                param.requires_grad = True
            if "norm." in name:
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_eff = self.eff(x)                     # (B, 1536)
        f_vit = self.vit(x)                     # (B, 768)
        fused = torch.cat([f_eff, f_vit], dim=1)  # (B, 2304)
        fused = self.se(fused)
        return self.head(fused)


# ─── Training helpers ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device):
            logits = model(imgs)
            loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


def run_phase(model, loaders, cfg, phase: int):
    device = cfg["device"]
    lr     = cfg[f"phase{phase}_lr"]
    epochs = cfg[f"phase{phase}_epochs"]

    # Only optimise params that require grad
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"\n  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"]).to(device)
    scaler    = torch.amp.GradScaler()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        print(f"\nPhase {phase} | Epoch {epoch}/{epochs}")
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device)
        va_loss, va_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"  Loss  train={tr_loss:.4f}  val={va_loss:.4f}")
        print(f"  Acc   train={tr_acc:.4f}  val={va_acc:.4f}  ({'NEW BEST' if va_acc > best_acc else ''})")

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"\n  Phase {phase} best val accuracy: {best_acc:.4f}")
    return history


# ─── Plotting ─────────────────────────────────────────────────────────────────
def plot_history(histories, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = ["#3266ad", "#E85D24"]

    for ax_idx, metric in enumerate(["loss", "acc"]):
        ax = axes[ax_idx]
        offset = 0
        for ph, hist in enumerate(histories, start=1):
            x = list(range(offset + 1, offset + len(hist[f"train_{metric}"]) + 1))
            ax.plot(x, hist[f"train_{metric}"], color=colours[ph - 1],
                    linestyle="--", alpha=0.7, label=f"Phase {ph} train")
            ax.plot(x, hist[f"val_{metric}"],   color=colours[ph - 1],
                    linestyle="-",  label=f"Phase {ph} val")
            offset += len(x)
        ax.set_title(f"{'Loss' if metric == 'loss' else 'Accuracy'} over epochs")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_history.png"), dpi=150)
    plt.close()
    print(f"  Saved training_history.png")


def plot_confusion_matrix(labels, preds, class_names, results_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — test set")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"  Saved confusion_matrix.png")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    cfg    = CONFIG
    device = cfg["device"]

    print("=" * 60)
    print("  Wheat Disease Detection — Hybrid EfficientNet-B3 + ViT")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    loaders = build_dataloaders(cfg)
    assert "train" in loaders and "val" in loaders, \
        "Need at least train/ and val/ folders inside dataset/"

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2] Building hybrid model ...")
    model = WheatHybridNet(
        num_classes=cfg["num_classes"],
        dropout1=cfg["dropout1"],
        dropout2=cfg["dropout2"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # ── Phase 1: frozen backbones ─────────────────────────────────────────────
    print("\n[3] Phase 1 — training head only (backbones frozen) ...")
    model.freeze_backbones()
    history1 = run_phase(model, loaders, cfg, phase=1)

    # ── Phase 2: fine-tune top layers ─────────────────────────────────────────
    print("\n[4] Phase 2 — fine-tuning top backbone layers ...")
    model.unfreeze_top_layers()
    history2 = run_phase(model, loaders, cfg, phase=2)

    # ── Save model ────────────────────────────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "config":      cfg,
    }, cfg["save_path"])
    print(f"\n  Model saved to {cfg['save_path']}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    if "test" in loaders:
        print("\n[5] Evaluating on test set ...")
        criterion = nn.CrossEntropyLoss().to(device)
        _, test_acc, preds, labels = evaluate(model, loaders["test"], criterion, device)
        print(f"\n  Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print("\n  Per-class report:")
        print(classification_report(labels, preds, target_names=cfg["class_names"]))

        plot_confusion_matrix(labels, preds, cfg["class_names"], cfg["results_dir"])

    # ── Plot training curves ───────────────────────────────────────────────────
    print("\n[6] Saving plots ...")
    plot_history([history1, history2], cfg["results_dir"])

    # Save history JSON
    with open(os.path.join(cfg["results_dir"], "history.json"), "w") as f:
        json.dump({"phase1": history1, "phase2": history2}, f, indent=2)

    print("\n  Done. Check the results/ folder for plots.")


if __name__ == "__main__":
    main()