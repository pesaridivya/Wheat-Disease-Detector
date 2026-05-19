import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Setup the same Data Generator used during training
val_path = "Wheat_Disease/validation"
img_size = (224, 224)
batch_size = 32

val_gen = ImageDataGenerator(rescale=1./255)
val_data = val_gen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Good practice for evaluation
)

# 2. Load the trained model
model = tf.keras.models.load_model("wheat_disease_model.h5")

# 3. Evaluate and display only accuracy
loss, accuracy = model.evaluate(val_data, verbose=0)

print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")