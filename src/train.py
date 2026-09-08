import os
import tensorflow as tf

from model import build_model
from preprocess import train_generator, val_generator




EPOCHS = 1

MODEL_DIR = "models"
LOG_DIR = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


print("=" * 50)
print("MODEL TRAINING")
print("=" * 50)



model = build_model()

print("\nModel ready for training.")



checkpoint_path = os.path.join(
    MODEL_DIR,
    "garbage_mobilenetv2_best.keras"
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(LOG_DIR, "training.csv"),
    append=False
)



print("\nStarting training...\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        csv_logger
    ]
)



final_model_path = os.path.join(
    MODEL_DIR,
    "garbage_mobilenetv2_final.keras"
)

model.save(final_model_path)



print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)

print("\nBest model saved at:")
print(checkpoint_path)

print("\nFinal model saved at:")
print(final_model_path)

print("\nTraining log saved at:")
print(os.path.join(LOG_DIR, "training.csv"))