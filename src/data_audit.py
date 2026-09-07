import pandas as pd
import os
from PIL import Image
import hashlib


DATASET_PATH = "data/Garbage_Dataset_Classification"

METADATA_PATH = f"{DATASET_PATH}/metadata.csv"

df = pd.read_csv(METADATA_PATH)


print("DATASET AUDIT")


print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)
print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nNumber of Classes:")
print(df["label"].nunique())
print("\nMissing Values:")
print(df.isnull().sum())
#check the missing values in the dataset
print("\nChecking image files...")

IMAGE_PATH = f"{DATASET_PATH}/images"

missing_images = []

for _, row in df.iterrows():

    filename = row["filename"]
    label = row["label"]

    image_path = os.path.join(IMAGE_PATH, label, filename)

    if not os.path.exists(image_path):
        missing_images.append(image_path)

print("Total metadata records:", len(df))
print("Missing image files:", len(missing_images))

# see for corrupted imagessss
print("\nChecking for corrupted images...")

corrupted_images = []

for _, row in df.iterrows():

    filename = row["filename"]
    label = row["label"]

    image_path = os.path.join(IMAGE_PATH, label, filename)

    try:
        with Image.open(image_path) as img:
            img.verify()

    except Exception:
        corrupted_images.append(image_path)

print("Total images checked:", len(df))
print("Corrupted images:", len(corrupted_images))

# Check image dimensions
print("\nChecking image dimensions...")

image_dimensions = {}

for _, row in df.iterrows():

    filename = row["filename"]
    label = row["label"]

    image_path = os.path.join(IMAGE_PATH, label, filename)

    try:
        with Image.open(image_path) as img:
            size = img.size

            if size not in image_dimensions:
                image_dimensions[size] = 0

            image_dimensions[size] += 1

    except Exception:
        pass

print("\nImage Dimensions:")
for size, count in image_dimensions.items():
    print(f"{size}: {count} images")

    print("\nChecking image channels...")

image_modes = {}

for _, row in df.iterrows():
    filename = row["filename"]
    label = row["label"]
    image_path = os.path.join(IMAGE_PATH, label, filename)

    try:
        with Image.open(image_path) as img:
            mode = img.mode

            if mode not in image_modes:
                image_modes[mode] = 0

            image_modes[mode] += 1

    except Exception:
        pass

print("\nImage Modes:")
for mode, count in image_modes.items():
    print(f"{mode}: {count} images")

    print("\nChecking for duplicate images...")

image_hashes = {}

duplicate_images = []
cross_class_duplicates = []

for _, row in df.iterrows():
    filename = row["filename"]
    label = row["label"]

    image_path = os.path.join(IMAGE_PATH, label, filename)

    try:
        with open(image_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        if file_hash in image_hashes:
            previous = image_hashes[file_hash]

            duplicate_images.append({
                "current": image_path,
                "duplicate_of": previous["path"]
            })

            if label != previous["label"]:
                cross_class_duplicates.append({
                    "current": image_path,
                    "current_label": label,
                    "duplicate_of": previous["path"],
                    "previous_label": previous["label"]
                })

        else:
            image_hashes[file_hash] = {
                "path": image_path,
                "label": label
            }

    except Exception:
        pass


print("Total images:", len(df))
print("Unique image hashes:", len(image_hashes))
print("Duplicate images:", len(duplicate_images))
print("Cross-class duplicates:", len(cross_class_duplicates))
print("\nCross-class duplicate details:")

for item in cross_class_duplicates:
    print(f"\nCurrent image: {item['current']}")
    print(f"Current label: {item['current_label']}")
    print(f"Duplicate of:  {item['duplicate_of']}")
    print(f"Previous label: {item['previous_label']}")