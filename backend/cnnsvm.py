"""
Wheat Disease Detection — CNN (MobileNetV3) + SVM
Auto-splits Wheat_Disease/train into 80% train / 20% val
Loads fine-tuned CNN weights, extracts features, trains SVM

Run:
    python cnnsvm.py
"""

import os, random, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CONFIG = {
    "src_dir"     : "Wheat_Disease/train",
    "train_dir"   : "cnn_train",
    "val_dir"     : "cnn_val",
    "img_size"    : 224,
    "batch_size"  : 32,
    "num_workers" : 0,
    "class_names" : ["Black Rust", "Brown Rust", "Healthy Wheat", "Yellow Rust"],
    "device"      : "cpu",
    "model_path"  : r"C:\Users\pesar\OneDrive\Desktop\ai\wheat_model_v2.pth",
    "results_dir" : "results_cnn_svm",
    "svm_C"       : 1000,
    "svm_gamma"   : 0.0001,
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)
print(f"Device  : {CONFIG['device']}")
print(f"PyTorch : {torch.__version__}")


def create_split(cfg):
    src     = cfg["src_dir"]
    classes = cfg["class_names"]
    for folder in [cfg["train_dir"], cfg["val_dir"]]:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"  Removed old {folder}/")
    print(f"\n  Splitting {src} into 80% train / 20% val ...")
    for cls in classes:
        cls_path = os.path.join(src, cls)
        imgs     = [f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(imgs)
        n_train = int(len(imgs) * 0.8)
        for i, img in enumerate(imgs):
            split = cfg["train_dir"] if i < n_train else cfg["val_dir"]
            dst   = os.path.join(split, cls)
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(os.path.join(cls_path, img), os.path.join(dst, img))
        print(f"  {cls}: {n_train} train | {len(imgs)-n_train} val")


TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_loaders(cfg):
    loaders = {}
    split_map = {"train": cfg["train_dir"], "val": cfg["val_dir"]}
    for key, folder in split_map.items():
        if not os.path.isdir(folder):
            print(f"  [warn] '{folder}' not found")
            continue
        ds = datasets.ImageFolder(folder, transform=TRANSFORM)
        if key == "train":
            print(f"  Class mapping : {ds.class_to_idx}")
        loaders[key] = DataLoader(ds, batch_size=cfg["batch_size"],
                                  shuffle=False, num_workers=cfg["num_workers"])
        print(f"  {key:5s}: {len(ds):5d} images | {len(loaders[key])} batches")
    return loaders


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(weights=None)
        in_f = base.classifier[0].in_features
        base.classifier = nn.Sequential(
            nn.Linear(in_f, 512), nn.Hardswish(), nn.Dropout(0.4),
            nn.Linear(512, 256),  nn.Hardswish(), nn.Dropout(0.24),
            nn.Linear(256, 4),
        )
        self.full_model = base

    def forward(self, x):
        x = self.full_model.features(x)
        x = self.full_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.full_model.classifier[0](x)
        x = self.full_model.classifier[1](x)
        x = self.full_model.classifier[3](x)
        x = self.full_model.classifier[4](x)
        return x


def load_weights(extractor, model_path, device):
    if not os.path.exists(model_path):
        print(f"  [warn] {model_path} not found — using ImageNet weights")
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        extractor.full_model.features.load_state_dict(base.features.state_dict())
        return extractor
    print(f"  Loading weights from {model_path} ...")
    ckpt      = torch.load(model_path, map_location=device)
    new_state = {k.replace("model.", "full_model."): v
                 for k, v in ckpt["model_state"].items()}
    extractor.load_state_dict(new_state, strict=False)
    print("  Weights loaded!")
    return extractor


@torch.no_grad()
def extract_features(model, loader, device, name):
    model.eval()
    feats, labels = [], []
    for imgs, lbls in tqdm(loader, desc=f"  Extracting {name:5s}"):
        feats.append(model(imgs.to(device)).cpu().numpy())
        labels.extend(lbls.numpy())
    return np.vstack(feats), np.array(labels)


def save_confusion_matrix(labels, preds, class_names, results_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — CNN + SVM")
    plt.tight_layout()
    path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


def main():
    cfg    = CONFIG
    device = cfg["device"]

    print("=" * 55)
    print("  Wheat Disease — CNN (MobileNetV3) + SVM")
    print("=" * 55)

    print("\n[1] Creating train/val split from Wheat_Disease/train ...")
    create_split(cfg)

    print("\n[2] Loading data ...")
    loaders = build_loaders(cfg)
    assert "train" in loaders and "val" in loaders, \
        "Split failed — check Wheat_Disease/train folder!"

    print("\n[3] Building CNN feature extractor ...")
    extractor = CNNFeatureExtractor().to(device)
    extractor = load_weights(extractor, cfg["model_path"], device)
    print(f"  Params: {sum(p.numel() for p in extractor.parameters()):,}")

    print("\n[4] Extracting CNN features (~8-10 min on CPU) ...")
    X_train, y_train = extract_features(extractor, loaders["train"], device, "train")
    X_val,   y_val   = extract_features(extractor, loaders["val"],   device, "val")
    print(f"\n  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")

    np.save(os.path.join(cfg["results_dir"], "X_train.npy"), X_train)
    np.save(os.path.join(cfg["results_dir"], "y_train.npy"), y_train)
    np.save(os.path.join(cfg["results_dir"], "X_val.npy"),   X_val)
    np.save(os.path.join(cfg["results_dir"], "y_val.npy"),   y_val)
    print("  Features saved -> results_cnn_svm/")

    print(f"\n[5] Training SVM (C={cfg['svm_C']}, gamma={cfg['svm_gamma']}) ...")
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=cfg["svm_C"], gamma=cfg["svm_gamma"],
                    probability=True, random_state=SEED)),
    ])
    svm.fit(X_train, y_train)
    print("  SVM training complete!")

    print("\n[6] Evaluating ...")
    y_pred = svm.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)

    print("\n" + "=" * 55)
    print(f"  FINAL ACCURACY : {acc*100:.2f}%")
    print("=" * 55)
    print("\nPer-class report:")
    print(classification_report(y_val, y_pred, target_names=cfg["class_names"]))

    save_confusion_matrix(y_val, y_pred, cfg["class_names"], cfg["results_dir"])

    joblib.dump(svm, os.path.join(cfg["results_dir"], "svm_model.pkl"))
    print("  SVM saved -> results_cnn_svm/svm_model.pkl")

    torch.save(extractor.state_dict(),
               os.path.join(cfg["results_dir"], "cnn_extractor.pth"))
    print("  CNN saved -> results_cnn_svm/cnn_extractor.pth")


if __name__ == "__main__":
    main()