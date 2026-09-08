import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# ==============================
# MODEL SETTINGS
# ==============================

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 6


def build_model():
    """
    Build and return a MobileNetV2 transfer learning model.
    """

    
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    
    base_model.trainable = False

    
    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    
    x = base_model(
        inputs,
        training=False
    )

    
    x = layers.GlobalAveragePooling2D()(x)

    
    x = layers.Dropout(0.2)(x)

    
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    # Create model
    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    # Compile
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    print("=" * 50)
    print("BUILDING MODEL")
    print("=" * 50)

    model = build_model()

    model.summary()

    print("\nModel created successfully!")