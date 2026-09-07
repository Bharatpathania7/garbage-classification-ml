import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_PATH = "data/Garbage_Dataset_Classification"
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(METADATA_PATH)

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# Class Distribution
# --------------------------------------------------

class_counts = df["label"].value_counts()

print("\nClass Distribution:")
print(class_counts)


# --------------------------------------------------
# Class Distribution Bar Chart
# --------------------------------------------------

plt.figure(figsize=(10, 6))

class_counts.plot(kind="bar")

plt.title("Garbage Dataset - Class Distribution")
plt.xlabel("Garbage Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Sample Images from Each Class
# --------------------------------------------------

IMAGE_PATH = os.path.join(DATASET_PATH, "images")

classes = sorted(df["label"].unique())

fig, axes = plt.subplots(
    nrows=len(classes),
    ncols=3,
    figsize=(10, 18)
)

for row_index, class_name in enumerate(classes):

    class_images = df[df["label"] == class_name].sample(
        n=3,
        random_state=42
    )

    for col_index, (_, row) in enumerate(class_images.iterrows()):

        image_path = os.path.join(
            IMAGE_PATH,
            class_name,
            row["filename"]
        )

        image = Image.open(image_path)

        axes[row_index, col_index].imshow(image)
        axes[row_index, col_index].set_title(class_name)
        axes[row_index, col_index].axis("off")

plt.suptitle("Sample Images from Each Garbage Class")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Image Brightness Analysis
# --------------------------------------------------

print("\nCalculating image brightness...")

brightness_values = []

for _, row in df.iterrows():

    label = row["label"]
    filename = row["filename"]

    image_path = os.path.join(
        IMAGE_PATH,
        label,
        filename
    )

    try:
        image = Image.open(image_path).convert("RGB")

        image_array = np.array(image)

        brightness = image_array.mean()

        brightness_values.append(brightness)

    except Exception:
        pass


print("\nBrightness Statistics:")
print(f"Images analyzed: {len(brightness_values)}")
print(f"Minimum brightness: {min(brightness_values):.2f}")
print(f"Maximum brightness: {max(brightness_values):.2f}")
print(f"Average brightness: {np.mean(brightness_values):.2f}")


# --------------------------------------------------
# Brightness Distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(brightness_values, bins=30)

plt.title("Image Brightness Distribution")
plt.xlabel("Average Pixel Brightness")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.show()
# --------------------------------------------------
# Per-Class Brightness Analysis
# --------------------------------------------------

print("\nCalculating brightness by class...")

class_brightness = {}

for class_name in classes:

    class_images = df[df["label"] == class_name]

    brightness_values = []

    for _, row in class_images.iterrows():

        image_path = os.path.join(
            IMAGE_PATH,
            class_name,
            row["filename"]
        )

        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)

            brightness = image_array.mean()
            brightness_values.append(brightness)

        except Exception:
            pass

    if brightness_values:
        class_brightness[class_name] = np.mean(brightness_values)


print("\nAverage Brightness by Class:")

for class_name, brightness in class_brightness.items():
    print(f"{class_name}: {brightness:.2f}")


# --------------------------------------------------
# Per-Class Brightness Chart
# --------------------------------------------------

brightness_series = pd.Series(class_brightness)

plt.figure(figsize=(10, 6))

brightness_series.plot(kind="bar")

plt.title("Average Image Brightness by Garbage Class")
plt.xlabel("Garbage Class")
plt.ylabel("Average Pixel Brightness")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# RGB Channel Statistics
# --------------------------------------------------

print("\nCalculating RGB channel statistics...")

red_values = []
green_values = []
blue_values = []

for _, row in df.iterrows():

    label = row["label"]
    filename = row["filename"]

    image_path = os.path.join(
        IMAGE_PATH,
        label,
        filename
    )

    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        red_values.append(image_array[:, :, 0].mean())
        green_values.append(image_array[:, :, 1].mean())
        blue_values.append(image_array[:, :, 2].mean())

    except Exception:
        pass


print("\nRGB Channel Statistics:")

print(f"Images analyzed: {len(red_values)}")

print(f"Average Red intensity: {np.mean(red_values):.2f}")
print(f"Average Green intensity: {np.mean(green_values):.2f}")
print(f"Average Blue intensity: {np.mean(blue_values):.2f}")


# --------------------------------------------------
# RGB Channel Comparison
# --------------------------------------------------

channel_means = [
    np.mean(red_values),
    np.mean(green_values),
    np.mean(blue_values)
]

channels = ["Red", "Green", "Blue"]

plt.figure(figsize=(8, 5))

plt.bar(channels, channel_means)

plt.title("Average RGB Channel Intensity")
plt.xlabel("Color Channel")
plt.ylabel("Average Pixel Intensity")

plt.tight_layout()
plt.show()