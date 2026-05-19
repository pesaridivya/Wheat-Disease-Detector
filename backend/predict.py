from fastapi import APIRouter, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

router = APIRouter()

model = tf.keras.models.load_model("model/wheat_model.h5")

classes = [
    "Healthy",
    "Leaf Rust",
    "Stem Rust",
    "Powdery Mildew"
]

@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents))
    image = image.resize((224,224))

    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return {
        "disease": classes[index],
        "confidence": round(confidence*100,2)
    }