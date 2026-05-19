import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset paths
train_path = "Wheat_Disease/train"
val_path = "Wheat_Disease/validation"

img_size = (224,224)
batch_size = 32

# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

val_data = val_gen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load CNN model (feature extractor)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3),
    pooling="avg"
)

# Freeze CNN layers
for layer in base_model.layers:
    layer.trainable = False

print("Extracting features from CNN...")

# Extract CNN features
train_features = base_model.predict(train_data)
val_features = base_model.predict(val_data)

# Labels
train_labels = train_data.classes
val_labels = val_data.classes

# Train SVM classifier
print("Training SVM...")

svm = SVC(kernel='rbf')
svm.fit(train_features, train_labels)

# Prediction
pred = svm.predict(val_features)

# Accuracy
acc = accuracy_score(val_labels, pred)

print("CNN + SVM Accuracy:", acc)

# Detailed report
print(classification_report(val_labels, pred))