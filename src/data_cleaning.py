import os
import pandas as pd

DATASET_PATH = "data/Garbage_Dataset_Classification"
IMAGE_PATH = os.path.join(DATASET_PATH, "images")
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")


files_to_remove = [
    os.path.join(IMAGE_PATH, "glass", "glass_02907.jpg"),
    os.path.join(IMAGE_PATH, "metal", "metal_01182.jpg"),
]

print("=" * 50)
print("DATA CLEANING")
print("=" * 50)


for file_path in files_to_remove:

    if os.path.exists(file_path):
        print(f"\nRemoving image: {file_path}")
        os.remove(file_path)
        print("Removed successfully.")

    else:
        print(f"\nImage already removed/not found: {file_path}")




df = pd.read_csv(METADATA_PATH)

filenames_to_remove = {
    "glass_02907.jpg",
    "metal_01182.jpg"
}

before_count = len(df)

df = df[~df["filename"].isin(filenames_to_remove)]

after_count = len(df)

df.to_csv(METADATA_PATH, index=False)

print("\nMetadata cleaning:")
print("Records before:", before_count)
print("Records removed:", before_count - after_count)
print("Records after:", after_count)

print("\nCleaning completed.")