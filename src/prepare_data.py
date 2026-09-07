import os
import pandas as pd

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_PATH = "data/Garbage_Dataset_Classification"
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")

OUTPUT_PATH = "data/splits"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("=" * 50)
print("DATA PREPARATION")
print("=" * 50)

df = pd.read_csv(METADATA_PATH)

print("\nTotal images:", len(df))
print("\nOriginal class distribution:")
print(df["label"].value_counts())


# --------------------------------------------------
# First Split
# Train = 70%
# Temporary = 30%
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)


# --------------------------------------------------
# Second Split
# Validation = 15%
# Test = 15%
#
# temp = 30%
# Half of temp = 15%
# --------------------------------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# --------------------------------------------------
# Reset Index
# --------------------------------------------------

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# --------------------------------------------------
# Create Output Directory
# --------------------------------------------------

os.makedirs(OUTPUT_PATH, exist_ok=True)


# --------------------------------------------------
# Save Split Metadata
# --------------------------------------------------

train_df.to_csv(
    os.path.join(OUTPUT_PATH, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(OUTPUT_PATH, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_PATH, "test.csv"),
    index=False
)


# --------------------------------------------------
# Print Results
# --------------------------------------------------

print("\nSplit completed!")

print("\nTrain images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

print("\nTrain class distribution:")
print(train_df["label"].value_counts())

print("\nValidation class distribution:")
print(val_df["label"].value_counts())

print("\nTest class distribution:")
print(test_df["label"].value_counts())

print("\nFiles saved to:", OUTPUT_PATH)