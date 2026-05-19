import os, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CONFIG = {
    "data_dir"        : "Wheat_Small",
    "val_split"       : "val",
    "img_size"        : 224,
    "batch_size"      : 32,
    "num_classes"     : 4,
    "class_names"     : ["Black Rust", "Brown Rust", "Healthy Wheat", "Yellow Rust"],
    "phase1_epochs"   : 5,
    "phase1_lr"       : 1e-3,
    "phase2_epochs"   : 10,
    "phase2_lr"       : 1e-4,
    "weight_decay"    : 1e-4,
    "dropout"         : 0.4,
    "label_smoothing" : 0.1,
    "num_workers"     : 0,
    "device"          : "cpu",
    "save_path"       : "wheat_model.pth",
    "results_dir"     : "results",
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)
print(f"Device  : {CONFIG['device']}")
print(f"PyTorch : {torch.__version__}")

def get_transforms(split, img_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def build_loaders(cfg):
    loaders = {}
    split_map = {"train": "train", "val": cfg["val_split"]}
    for key, folder in split_map.items():
        path = os.path.join(cfg["data_dir"], folder)
        if not os.path.isdir(path):
            print(f"  [warn] '{path}' not found")
            continue
        ds = datasets.ImageFolder(path, transform=get_transforms(key, cfg["img_size"]))
        if key == "train":
            print(f"  Class mapping: {ds.class_to_idx}")
        loaders[key] = DataLoader(ds, batch_size=cfg["batch_size"],
                                  shuffle=(key=="train"),
                                  num_workers=cfg["num_workers"])
        print(f"  {key}: {len(ds)} images | {len(loaders[key])} batches")
    return loaders

class WheatNet(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )
        in_f = base.classifier[0].in_features
        base.classifier = nn.Sequential(
            nn.Linear(in_f, 512), nn.Hardswish(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.Hardswish(), nn.Dropout(dropout * 0.6),
            nn.Linear(256, num_classes),
        )
        self.model = base

    def freeze(self):
        for p in self.model.features.parameters():
            p.requires_grad = False

    def unfreeze_top(self):
        for i in range(13, 17):
            for p in self.model.features[i].parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.model(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc="  train", leave=False,
               bar_format="{l_bar}{bar:25}{r_bar}")
    for imgs, labels in bar:
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
        bar.set_postfix(loss=f"{loss_sum/total:.3f}",
                        acc=f"{correct/total*100:.1f}%")
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  val  ", leave=False,
                              bar_format="{l_bar}{bar:25}{r_bar}"):
        logits = model(imgs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (preds == labels).sum().item()
        total    += imgs.size(0)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
    return loss_sum / total, correct / total, all_preds, all_labels

def run_phase(model, loaders, cfg, phase):
    lr, epochs, device = cfg[f"phase{phase}_lr"], cfg[f"phase{phase}_epochs"], cfg["device"]
    trainable  = [p for p in model.parameters() if p.requires_grad]
    print(f"\n  Trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer  = optim.AdamW(trainable, lr=lr, weight_decay=cfg["weight_decay"])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    criterion  = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    best_acc, best_state, best_preds, best_labels = 0.0, None, [], []

    for epoch in range(1, epochs + 1):
        print(f"\nPhase {phase} | Epoch {epoch}/{epochs}")
        tr_loss, tr_acc = train_epoch(model, loaders["train"], criterion, optimizer, device)
        va_loss, va_acc, preds, labels = eval_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        flag = "  ◀ BEST" if va_acc > best_acc else ""
        print(f"  loss  train={tr_loss:.4f}  val={va_loss:.4f}")
        print(f"  acc   train={tr_acc*100:.2f}%  val={va_acc*100:.2f}%{flag}")
        if va_acc > best_acc:
            best_acc, best_state = va_acc, copy.deepcopy(model.state_dict())
            best_preds, best_labels = preds, labels

    model.load_state_dict(best_state)
    print(f"\n  Phase {phase} best val accuracy: {best_acc*100:.2f}%")
    return best_acc, best_preds, best_labels

def save_plots(preds, labels, class_names, results_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"  Saved → results/confusion_matrix.png")

def main():
    cfg = CONFIG
    print("=" * 50)
    print("  Wheat Disease — MobileNetV3 (CPU)")
    print("=" * 50)

    print("\n[1] Loading data ...")
    loaders = build_loaders(cfg)

    print("\n[2] Building model ...")
    model = WheatNet(cfg["num_classes"], cfg["dropout"])
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    print("\n[3] Phase 1 — head only ...")
    model.freeze()
    acc1, p1, l1 = run_phase(model, loaders, cfg, phase=1)

    print("\n[4] Phase 2 — fine-tuning ...")
    model.unfreeze_top()
    acc2, p2, l2 = run_phase(model, loaders, cfg, phase=2)

    best_preds  = p2  if acc2 >= acc1 else p1
    best_labels = l2  if acc2 >= acc1 else l1
    best_acc    = max(acc1, acc2)

    torch.save({"model_state": model.state_dict(), "config": cfg}, cfg["save_path"])

    print("\n" + "=" * 50)
    print(f"  FINAL ACCURACY : {best_acc*100:.2f}%")
    print("=" * 50)
    print("\nPer-class report:")
    print(classification_report(best_labels, best_preds,
                                target_names=cfg["class_names"]))
    save_plots(best_preds, best_labels, cfg["class_names"], cfg["results_dir"])

if __name__ == "__main__":
    main()
