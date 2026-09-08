import os
import sys
import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "src")
)

from fingerprint import generate_fingerprint
from cache import get_cached_result, store_result
from label_data import save_label
from monitoring import log_prediction

MODEL_PATH = "models/garbage_mobilenetv2_best.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

app = FastAPI(
    title="Garbage Classification API",
    description="Garbage classification with fingerprinting, caching, data labeling and monitoring",
    version="1.0.0"
)

print("=" * 50)
print("LOADING GARBAGE CLASSIFICATION MODEL")
print("=" * 50)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

class LabelRequest(BaseModel):
    fingerprint: str
    image_path: str
    predicted_label: str
    confidence: float
    actual_label: str

@app.get("/")
def home():
    return {
        "message": "Garbage Classification API is running!"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    
    start_time = time.time()

    

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    temp_path = os.path.join(temp_dir, file.filename)

    contents = await file.read()

    with open(temp_path, "wb") as image_file:
        image_file.write(contents)

    try:

        

        fingerprint = generate_fingerprint(temp_path)

        print("\n" + "=" * 50)
        print("NEW PREDICTION REQUEST")
        print("=" * 50)

        print("Filename:", file.filename)
        print("Fingerprint:", fingerprint)

        

        cached_result = get_cached_result(fingerprint)

        if cached_result is not None:

            print("Cache HIT ✅")
            print("Model was NOT executed.")

            
            processing_time = time.time() - start_time

            
            log_prediction(
                fingerprint=fingerprint,
                prediction=cached_result["prediction"],
                confidence=cached_result["confidence"],
                cache_status="HIT",
                processing_time_seconds=processing_time
            )

            return {
                "prediction": cached_result["prediction"],
                "confidence": cached_result["confidence"],
                "cached": True,
                "fingerprint": fingerprint
            }

        

        print("Cache MISS ❌")
        print("Running model...")

        image = tf.keras.utils.load_img(
            temp_path,
            target_size=IMAGE_SIZE
        )

        image_array = tf.keras.utils.img_to_array(image)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        image_array = preprocess_input(image_array)

        

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = np.argmax(predictions[0])

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(
            predictions[0][predicted_index]
        )

        

        store_result(
            fingerprint,
            predicted_class,
            confidence
        )

        

        processing_time = time.time() - start_time

        log_prediction(
            fingerprint=fingerprint,
            prediction=predicted_class,
            confidence=confidence,
            cache_status="MISS",
            processing_time_seconds=processing_time
        )

        print("Prediction:", predicted_class)
        print("Confidence:", f"{confidence:.2%}")
        print("Result saved to cache.")
        print("Processing time:", f"{processing_time:.4f} seconds")

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "cached": False,
            "fingerprint": fingerprint
        }

    finally:

        

        if os.path.exists(temp_path):
            os.remove(temp_path)




@app.post("/label")
def label_data(request: LabelRequest):

    
    if request.predicted_label not in CLASS_NAMES:
        raise HTTPException(
            status_code=400,
            detail="Invalid predicted label."
        )

    

    if request.actual_label not in CLASS_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid actual label. Choose one of: {CLASS_NAMES}"
        )

    
    try:

        save_label(
            fingerprint=request.fingerprint,
            image_path=request.image_path,
            predicted_label=request.predicted_label,
            confidence=request.confidence,
            actual_label=request.actual_label
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    print("\n" + "=" * 50)
    print("NEW LABELED DATA")
    print("=" * 50)

    print("Fingerprint:", request.fingerprint)
    print("Model prediction:", request.predicted_label)
    print("Actual label:", request.actual_label)
    print("Confidence:", f"{request.confidence:.2%}")

    return {
        "message": "New labeled data saved successfully!",
        "fingerprint": request.fingerprint,
        "predicted_label": request.predicted_label,
        "actual_label": request.actual_label,
        "confidence": request.confidence
    }