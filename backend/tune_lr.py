# add to classifiers.py or run separately
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Load saved features
X_train = np.load("results_cnn_svm/X_train.npy")
y_train = np.load("results_cnn_svm/y_train.npy")
X_val   = np.load("results_cnn_svm/X_val.npy")
y_val   = np.load("results_cnn_svm/y_val.npy")

# Try stronger Logistic Regression
from sklearn.metrics import accuracy_score

configs = [
    {"C": 10,   "max_iter": 1000},
    {"C": 50,   "max_iter": 2000},
    {"C": 100,  "max_iter": 3000},
    {"C": 500,  "max_iter": 5000},
    {"C": 1000, "max_iter": 5000},
]

for cfg in configs:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C          = cfg["C"],
            max_iter   = cfg["max_iter"],
            random_state = 42,
            n_jobs     = -1,
        )),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_val, pipe.predict(X_val))
    print(f"  C={cfg['C']:5d}  max_iter={cfg['max_iter']:5d}  acc={acc*100:.2f}%")

# Load saved features
X_train = np.load("results_cnn_svm/X_train.npy")
y_train = np.load("results_cnn_svm/y_train.npy")
X_val   = np.load("results_cnn_svm/X_val.npy")
y_val   = np.load("results_cnn_svm/y_val.npy")

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

configs = [
    {"C": 10,   "max_iter": 1000},
    {"C": 50,   "max_iter": 2000},
    {"C": 100,  "max_iter": 3000},
    {"C": 500,  "max_iter": 5000},
    {"C": 1000, "max_iter": 5000},
]

for cfg in configs:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C          = cfg["C"],
            max_iter   = cfg["max_iter"],
            random_state = 42,
            n_jobs     = -1,
        )),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_val, pipe.predict(X_val))
    print(f"  C={cfg['C']:5d}  max_iter={cfg['max_iter']:5d}  acc={acc*100:.2f}%")

# ── ADD THIS BELOW ────────────────────────────────────────────────────────────
print("\nTrying PCA + Logistic Regression ...")
from sklearn.decomposition import PCA

for n in [64, 128, 256]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n)),
        ("clf",    LogisticRegression(C=10, max_iter=1000,
                                      random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_val, pipe.predict(X_val))
    print(f"  PCA={n:3d}  acc={acc*100:.2f}%")