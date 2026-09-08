import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input




DATASET_PATH = "data/Garbage_Dataset_Classification"
IMAGE_PATH = os.path.join(DATASET_PATH, "images")

SPLIT_PATH = "data/splits"

TRAIN_CSV = os.path.join(SPLIT_PATH, "train.csv")
VAL_CSV = os.path.join(SPLIT_PATH, "validation.csv")
TEST_CSV = os.path.join(SPLIT_PATH, "test.csv")




IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


print("=" * 50)
print("IMAGE PREPROCESSING")
print("=" * 50)




train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)


print("\nTrain images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))




train_df["filepath"] = train_df.apply(
    lambda row: os.path.join(
        IMAGE_PATH,
        row["label"],
        row["filename"]
    ),
    axis=1
)

val_df["filepath"] = val_df.apply(
    lambda row: os.path.join(
        IMAGE_PATH,
        row["label"],
        row["filename"]
    ),
    axis=1
)

test_df["filepath"] = test_df.apply(
    lambda row: os.path.join(
        IMAGE_PATH,
        row["label"],
        row["filename"]
    ),
    axis=1
)




train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)




val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)




train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",

    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,

    class_mode="categorical",
    shuffle=True,

    seed=42
)


val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filepath",
    y_col="label",

    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,

    class_mode="categorical",
    shuffle=False
)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="label",

    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,

    class_mode="categorical",
    shuffle=False
)



print("\nClass mapping:")
print(train_generator.class_indices)

print("\nNumber of classes:")
print(len(train_generator.class_indices))

print("\nTrain batches:", len(train_generator))
print("Validation batches:", len(val_generator))
print("Test batches:", len(test_generator))




images, labels = next(train_generator)

print("\nOne batch loaded successfully!")

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("\nPixel range:")
print("Minimum:", images.min())
print("Maximum:", images.max())

print("\nPreprocessing completed successfully!")