import tensorflow as tf
from tensorflow.keras import layers, models
import os

# create model folder if not exists
os.makedirs("model", exist_ok=True)

model = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Conv2D(16,(3,3),activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation="relu"),
    layers.Dense(4,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.save("model/wheat_model.h5")

print("Model created successfully")