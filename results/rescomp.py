import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Confusion matrices
cms = {
    "CNN ": np.array([
        [185,  6,  4,  1],
        [ 16, 179,  3,  2],
        [  6,   1, 193,  0],
        [  1,   7,  1, 191]
    ]),
    "CNN + Random Forest": np.array([
        [178, 13,  8,  1],
        [  7, 192,  0,  1],
        [  3,   0, 198,  0],
        [  1,   0,  0, 200]
    ]),
    "CNN + SVM": np.array([
        [184,  8,  6,  2],
        [  6, 193,  0,  1],
        [  3,   1, 197,  0],
        [  1,   1,  0, 199]
    ]),
    "CNN + Logistic Regression": np.array([
        [185,  9,  6,  0],
        [  5, 193,  1,  1],
        [  4,   0, 197,  0],
        [  0,   2,  0, 199]
    ]),
}

classes = ["Black Rust", "Brown Rust", "Healthy Wheat", "Yellow Rust"]

# Compute accuracies
accuracies = {}
for name, cm in cms.items():
    acc = cm.diagonal().sum() / cm.sum()
    accuracies[name] = round(acc * 100, 2)
    print(f"{name}: {accuracies[name]}%")

# Plot confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, cm) in zip(axes.flatten(), cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"{name}\nAccuracy: {accuracies[name]}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

# Bar chart comparison
plt.figure(figsize=(8, 5))
bars = plt.bar(accuracies.keys(), accuracies.values(),
               color=['#888780', '#639922', '#D4537E', '#378ADD'])
plt.ylim(92, 97.5)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
for bar, val in zip(bars, accuracies.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val}%", ha='center', va='bottom', fontsize=11)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150)
plt.show()