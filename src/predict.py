import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from fingerprint import generate_fingerprint
from cache import get_cached_result, store_result



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



print("=" * 50)
print("IMAGE PREDICTION")
print("=" * 50)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")



def predict_image(image_path):



    fingerprint = generate_fingerprint(
        image_path
    )

    print("\nFingerprint:")
    print(fingerprint)



    cached_result = get_cached_result(
        fingerprint
    )

    if cached_result is not None:

        print("\nCache HIT ✅")
        print("Model was NOT executed.")

        return cached_result



    print("\nCache MISS ❌")
    print("Running model...")


    

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )


    

    image_array = tf.keras.utils.img_to_array(
        image
    )


    

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    

    image_array = preprocess_input(
        image_array
    )




    predictions = model.predict(
        image_array,
        verbose=0
    )


    

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[0][predicted_index]
    )


    

    result = {
        "prediction": predicted_class,
        "confidence": confidence
    }


    

    store_result(
        fingerprint,
        predicted_class,
        confidence
    )

    print("\nPrediction:")
    print(predicted_class)

    print("\nConfidence:")
    print(f"{confidence:.2%}")

    print("\nResult saved to cache.")


    return result



if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python src/predict.py <image_path>"
        )

        sys.exit(1)


    image_path = sys.argv[1]


    # Check image exists

    if not os.path.exists(image_path):

        print(
            "\nERROR: Image does not exist!"
        )

        print(
            "Path:",
            image_path
        )

        sys.exit(1)


    result = predict_image(
        image_path
    )


    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)

    print(result)