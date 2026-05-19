import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- 1. SETTINGS ---
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
TRAIN_DIR = 'Wheat_Disease/train'
VAL_DIR = 'Wheat_Disease/validation'

# --- 2. DATA LOADING (CRITICAL FIX) ---
# We use the official efficientnet preprocess_input function
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Use shuffle=True for training, but False for the final extraction
train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
val_gen = val_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# --- 3. THE MODEL ---
print("🚀 Building Fine-Tuned EfficientNetB3...")
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3), pooling='avg')

# To get 95%, we MUST train the head for a few epochs first
x = layers.Dense(256, activation='relu')(base_model.output)
x = layers.Dropout(0.5)(x)
output = layers.Dense(4, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print("🧠 Phase 1: Training CNN Head (5 Epochs)...")
model.fit(train_gen, epochs=5, validation_data=val_gen)

# --- 4. HYBRID EXTRACTION ---
print("⚖️ Phase 2: SVM Refinement...")
# We use the layer before the softmax as features
feature_extractor = models.Model(inputs=model.input, outputs=model.layers[-3].output)

# Extract features
X_train = feature_extractor.predict(train_gen)
y_train = train_gen.classes
X_val = feature_extractor.predict(val_gen)
y_val = val_gen.classes

# --- 5. SVM ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

svm = SVC(kernel='rbf', C=10, probability=True)
svm.fit(X_train_scaled, y_train)

preds = svm.predict(X_val_scaled)
print(f"\n🎯 NEW ACCURACY: {accuracy_score(y_val, preds)*100:.2f}%")