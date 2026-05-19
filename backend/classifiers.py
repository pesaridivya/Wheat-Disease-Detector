"""
Wheat Disease Detection — CNN Features + Multiple Classifiers
Classifiers: Random Forest, KNN, Gradient Boosting, Logistic Regression

Run:
    python classifiers.py
"""

import os, random, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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

# ── Config ────────────────────────────────────────────────────────────────────
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
    "results_dir" : "results_classifiers",
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)
print(f"Device  : {CONFIG['device']}")
print(f"PyTorch : {torch.__version__}")


# ── Split ─────────────────────────────────────────────────────────────────────
def create_split(cfg):
    for folder in [cfg["train_dir"], cfg["val_dir"]]:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    print(f"  Splitting {cfg['src_dir']} into 80/20 ...")
    for cls in cfg["class_names"]:
        cls_path = os.path.join(cfg["src_dir"], cls)
        imgs     = [f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(imgs)
        n_train  = int(len(imgs) * 0.8)
        for i, img in enumerate(imgs):
            split = cfg["train_dir"] if i < n_train else cfg["val_dir"]
            dst   = os.path.join(split, cls)
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(os.path.join(cls_path, img), os.path.join(dst, img))
        print(f"  {cls}: {n_train} train | {len(imgs)-n_train} val")


# ── Transform ─────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── DataLoaders ───────────────────────────────────────────────────────────────
def build_loaders(cfg):
    loaders = {}
    for key, folder in {"train": cfg["train_dir"], "val": cfg["val_dir"]}.items():
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


# ── CNN Feature Extractor ─────────────────────────────────────────────────────
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


# ── Extract features ──────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(model, loader, device, name):
    model.eval()
    feats, labels = [], []
    for imgs, lbls in tqdm(loader, desc=f"  Extracting {name:5s}"):
        feats.append(model(imgs.to(device)).cpu().numpy())
        labels.extend(lbls.numpy())
    return np.vstack(feats), np.array(labels)


# ── Confusion matrix ──────────────────────────────────────────────────────────
def save_confusion_matrix(labels, preds, class_names, results_dir, name):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — CNN + {name}")
    plt.tight_layout()
    path = os.path.join(results_dir, f"confusion_matrix_{name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg    = CONFIG
    device = cfg["device"]

    print("=" * 60)
    print("  Wheat Disease — CNN + Multiple Classifiers")
    print("=" * 60)

    # Step 1: Split
    print("\n[1] Creating train/val split ...")
    create_split(cfg)

    # Step 2: Load data
    print("\n[2] Loading data ...")
    loaders = build_loaders(cfg)
    assert "train" in loaders and "val" in loaders

    # Step 3: CNN feature extractor
    print("\n[3] Building CNN feature extractor ...")
    extractor = CNNFeatureExtractor().to(device)
    extractor = load_weights(extractor, cfg["model_path"], device)

    # Step 4: Extract features
    x_train_path = os.path.join(cfg["results_dir"], "X_train.npy")
    if os.path.exists(x_train_path):
        print("\n[4] Loading saved features ...")
        X_train = np.load(os.path.join(cfg["results_dir"], "X_train.npy"))
        y_train = np.load(os.path.join(cfg["results_dir"], "y_train.npy"))
        X_val   = np.load(os.path.join(cfg["results_dir"], "X_val.npy"))
        y_val   = np.load(os.path.join(cfg["results_dir"], "y_val.npy"))
    else:
        print("\n[4] Extracting CNN features (~8-10 min) ...")
        X_train, y_train = extract_features(extractor, loaders["train"], device, "train")
        X_val,   y_val   = extract_features(extractor, loaders["val"],   device, "val")
        np.save(os.path.join(cfg["results_dir"], "X_train.npy"), X_train)
        np.save(os.path.join(cfg["results_dir"], "y_train.npy"), y_train)
        np.save(os.path.join(cfg["results_dir"], "X_val.npy"),   X_val)
        np.save(os.path.join(cfg["results_dir"], "y_val.npy"),   y_val)

    print(f"  Train : {X_train.shape} | Val : {X_val.shape}")

    # Step 5: Define all classifiers
    classifiers = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators = 300,
                max_depth    = None,
                random_state = SEED,
                n_jobs       = -1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators  = 200,
                learning_rate = 0.1,
                max_depth     = 4,
                random_state  = SEED,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C            = 10,
                max_iter     = 1000,
                random_state = SEED,
                n_jobs       = -1,
            )),
        ]),
    }

    # Step 6: Train and evaluate each
    print("\n[5] Training and evaluating all classifiers ...")
    print("-" * 60)

    results = {}

    for name, pipeline in classifiers.items():
        print(f"\n  {name} ...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        acc    = accuracy_score(y_val, y_pred)
        results[name] = acc

        print(f"  Accuracy : {acc*100:.2f}%")
        print(classification_report(y_val, y_pred,
                                    target_names=cfg["class_names"]))
        save_confusion_matrix(y_val, y_pred, cfg["class_names"],
                               cfg["results_dir"], name)

        joblib.dump(pipeline,
                    os.path.join(cfg["results_dir"],
                                 f"{name.replace(' ','_')}_model.pkl"))
        print(f"  Saved -> results_classifiers/{name.replace(' ','_')}_model.pkl")
        print("-" * 60)

    # Step 7: Summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Classifier':<25} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<25} {acc*100:>9.2f}%")
    print("=" * 60)

    best_name = max(results, key=results.get)
    print(f"\n  Best classifier : {best_name} ({results[best_name]*100:.2f}%)")

    # Step 8: Bar chart comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    names   = list(results.keys())
    accs    = [results[n] * 100 for n in names]
    colors  = ["#3266ad", "#E85D24", "#27ae60", "#8e44ad"]
    bars    = ax.bar(names, accs, color=colors, width=0.5)
    ax.set_ylim(70, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CNN Features + Classifier Comparison")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    chart_path = os.path.join(cfg["results_dir"], "comparison_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n  Chart saved -> {chart_path}")


if __name__ == "__main__":
    main()
