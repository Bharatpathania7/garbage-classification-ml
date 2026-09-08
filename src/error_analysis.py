import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocess import test_generator


MODEL_PATH = "models/garbage_mobilenetv2_best.keras"

NUM_ERRORS_TO_SHOW = 12


print("=" * 50)
print("ERROR ANALYSIS")
print("=" * 50)


print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


print("\nGenerating predictions...")

test_generator.reset()

predictions = model.predict(
    test_generator,
    verbose=1
)


class_names = list(
    test_generator.class_indices.keys()
)

true_classes = test_generator.classes

predicted_classes = np.argmax(
    predictions,
    axis=1
)

confidence_scores = np.max(
    predictions,
    axis=1
)


wrong_indices = np.where(
    true_classes != predicted_classes
)[0]


print("\nTotal test images:", len(true_classes))
print("Correct predictions:",
len(true_classes) - len(wrong_indices))
print("Wrong predictions:",
len(wrong_indices))


error_data = []

for index in wrong_indices:

    filepath = test_generator.filepaths[index]

    actual = class_names[
        true_classes[index]
    ]

    predicted = class_names[
        predicted_classes[index]
    ]

    confidence = confidence_scores[index]

    error_data.append({
        "filepath": filepath,
        "actual": actual,
        "predicted": predicted,
        "confidence": confidence
    })


errors_df = pd.DataFrame(error_data)


errors_df = errors_df.sort_values(
    by="confidence",
    ascending=False
)

print("\n" + "=" * 50)
print("TOP WRONG PREDICTIONS")
print("=" * 50)

print(
    errors_df[
        ["actual", "predicted", "confidence"]
    ].head(20).to_string(index=False)
)


os.makedirs("logs", exist_ok=True)

error_report_path = (
    "logs/error_analysis.csv"
)

errors_df.to_csv(
    error_report_path,
    index=False
)

print("\nError report saved to:")
print(error_report_path)

num_images = min(
    NUM_ERRORS_TO_SHOW,
    len(errors_df)
)

if num_images > 0:

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14, 10)
    )

    axes = axes.flatten()

    for i in range(num_images):

        row = errors_df.iloc[i]

        image = tf.keras.utils.load_img(
            row["filepath"],
            target_size=(224, 224)
        )

        axes[i].imshow(image)

        axes[i].set_title(
            f"Actual: {row['actual']}\n"
            f"Pred: {row['predicted']}\n"
            f"Confidence: {row['confidence']:.2%}"
        )

        axes[i].axis("off")

    # Hide unused plots
    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    output_path = (
        "logs/wrong_predictions.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.show()

    print("\nWrong prediction visualization saved to:")
    print(output_path)


print("\nError analysis completed successfully!")