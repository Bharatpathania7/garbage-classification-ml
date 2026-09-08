import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from preprocess import test_generator


MODEL_PATH = "models/garbage_mobilenetv2_best.keras"


print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)


print("\nLoading best model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

print("\nEvaluating on test dataset...\n")

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)


print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


print("\nGenerating predictions...")

test_generator.reset()

predictions = model.predict(
    test_generator,
    verbose=1
)


predicted_classes = np.argmax(
    predictions,
    axis=1
)


true_classes = test_generator.classes

class_names = list(
    test_generator.class_indices.keys()
)


accuracy = accuracy_score(
    true_classes,
    predicted_classes
)

print("\nPrediction Accuracy:", accuracy)


print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names
)

print(report)

print("=" * 50)
print("CONFUSION MATRIX")
print("=" * 50)

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\nClass order:")
print(class_names)

print("\nConfusion Matrix:")
print(cm)


os.makedirs("logs", exist_ok=True)

report_path = "logs/evaluation_report.txt"

with open(report_path, "w") as file:

    file.write("MODEL EVALUATION REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(
        f"Test Loss: {test_loss}\n"
    )

    file.write(
        f"Test Accuracy: {test_accuracy}\n\n"
    )

    file.write("Classification Report\n")
    file.write("=" * 50 + "\n")
    file.write(report)

    file.write("\n\nConfusion Matrix\n")
    file.write("=" * 50 + "\n")

    file.write(
        str(cm)
    )


print("\nEvaluation report saved to:")
print(report_path)

print("\nEvaluation completed successfully!")