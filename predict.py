"""
predict.py — Run inference on a single image or a folder of images.

Usage:
    # Single image
    python predict.py --model best_wheat_model.pth --image leaf.jpg

    # Entire folder
    python predict.py --model best_wheat_model.pth --folder /path/to/test_images
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import torch.nn as nn


# ── Re-define the same model class ───────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x)


class WheatHybridNet(nn.Module):
    def __init__(self, num_classes=4, dropout1=0.4, dropout2=0.3):
        super().__init__()
        self.eff  = timm.create_model("efficientnet_b3", pretrained=False,
                                      num_classes=0, global_pool="avg")
        self.vit  = timm.create_model("vit_base_patch16_224", pretrained=False,
                                      num_classes=0)
        fused_dim = self.eff.num_features + self.vit.num_features
        self.se   = SEBlock(fused_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(dropout1),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(dropout2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.se(torch.cat([self.eff(x), self.vit(x)], dim=1)))


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CLASS_NAMES = ["yellow_rust", "black_rust", "brown_rust", "healthy"]


def load_model(path: str, device: str):
    ckpt  = torch.load(path, map_location=device)
    cfg   = ckpt.get("config", {})
    model = WheatHybridNet(
        num_classes=cfg.get("num_classes", 4),
        dropout1=cfg.get("dropout1", 0.4),
        dropout2=cfg.get("dropout2", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_image(model, img_path: str, device: str):
    img    = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred   = probs.argmax()
    return CLASS_NAMES[pred], float(probs[pred]), {c: float(p) for c, p in zip(CLASS_NAMES, probs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image",  default=None,  help="Single image path")
    parser.add_argument("--folder", default=None,  help="Folder of images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.model, device)
    print(f"Model loaded from {args.model}\n")

    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    if args.image:
        cls, conf, all_probs = predict_image(model, args.image, device)
        print(f"Image   : {args.image}")
        print(f"Disease : {cls}  ({conf*100:.1f}% confidence)")
        print("All probabilities:")
        for c, p in sorted(all_probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(p * 40)
            print(f"  {c:15s} {p*100:5.1f}%  {bar}")

    elif args.folder:
        files = [f for f in os.listdir(args.folder)
                 if os.path.splitext(f)[1].lower() in IMG_EXT]
        print(f"Predicting {len(files)} images in {args.folder} ...\n")
        counts = {c: 0 for c in CLASS_NAMES}
        for fname in sorted(files):
            path = os.path.join(args.folder, fname)
            cls, conf, _ = predict_image(model, path, device)
            counts[cls] += 1
            print(f"  {fname:40s}  →  {cls:15s}  {conf*100:.1f}%")
        print("\nSummary:")
        for c, n in counts.items():
            print(f"  {c:15s}: {n}")
    else:
        print("Please provide --image or --folder")


if __name__ == "__main__":
    main()
