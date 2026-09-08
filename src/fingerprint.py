import os
import hashlib
import pandas as pd


# ==============================
# PATHS
# ==============================

DATASET_PATH = "data/Garbage_Dataset_Classification"
METADATA_PATH = os.path.join(
    DATASET_PATH,
    "metadata.csv"
)

IMAGE_PATH = os.path.join(
    DATASET_PATH,
    "images"
)


# ==============================
# GENERATE FINGERPRINT
# ==============================

def generate_fingerprint(image_path):
    """
    Generate SHA-256 fingerprint for an image file.
    """

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as image_file:

        while True:

            data = image_file.read(8192)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    print("=" * 50)
    print("IMAGE FINGERPRINT")
    print("=" * 50)

    # Load metadata
    df = pd.read_csv(METADATA_PATH)

    # Take the first real image from metadata
    row = df.iloc[0]

    image_path = os.path.join(
        IMAGE_PATH,
        row["label"],
        row["filename"]
    )

    print("\nImage:")
    print(image_path)

    # Check file exists
    if not os.path.exists(image_path):
        print("\nERROR: Image file does not exist!")
        exit()

    # Generate fingerprint
    fingerprint = generate_fingerprint(
        image_path
    )

    print("\nSHA-256 fingerprint:")
    print(fingerprint)

    print("\nFingerprint length:")
    print(len(fingerprint))

    print("\nFingerprint generated successfully!")