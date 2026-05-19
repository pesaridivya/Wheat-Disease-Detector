import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy
from train import WheatNet, build_loaders, eval_epoch, train_epoch, save_plots, CONFIG
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load saved model
print("Loading saved model...")
checkpoint = torch.load("wheat_model.pth")
model = WheatNet(4, 0.4)
model.load_state_dict(checkpoint["model_state"])

# Unfreeze more layers this time
for i in range(10, 17):
    for p in model.model.features[i].parameters():
        p.requires_grad = True

# Build data
CONFIG["device"] = "cpu"
loaders = build_loaders(CONFIG)

# Train 5 more epochs at very low LR
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-5,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_acc, best_state = 0.0, None
best_preds, best_labels = [], []

for epoch in range(1, 6):
    print(f"\nEpoch {epoch}/5")
    tr_loss, tr_acc = train_epoch(model, loaders["train"], criterion, optimizer, "cpu")
    va_loss, va_acc, preds, labels = eval_epoch(model, loaders["val"], criterion, "cpu")
    scheduler.step()
    flag = "  <-- BEST" if va_acc > best_acc else ""
    print(f"  loss  train={tr_loss:.4f}  val={va_loss:.4f}")
    print(f"  acc   train={tr_acc*100:.2f}%  val={va_acc*100:.2f}%{flag}")
    if va_acc > best_acc:
        best_acc   = va_acc
        best_state = copy.deepcopy(model.state_dict())
        best_preds, best_labels = preds, labels

model.load_state_dict(best_state)
torch.save({"model_state": model.state_dict(), "config": CONFIG}, "wheat_model_v2.pth")

print(f"\n{'='*50}")
print(f"  FINAL ACCURACY : {best_acc*100:.2f}%")
print(f"{'='*50}")
print(classification_report(best_labels, best_preds,
                             target_names=CONFIG["class_names"]))
save_plots(best_preds, best_labels, CONFIG["class_names"], CONFIG["results_dir"])