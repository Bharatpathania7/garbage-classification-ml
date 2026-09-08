import os
import pandas as pd
from datetime import datetime

LABEL_DIR = "data/labeled_data"
LABEL_FILE = os.path.join(LABEL_DIR, "new_labels.csv")

os.makedirs(LABEL_DIR, exist_ok=True)

COLUMNS = [
    "timestamp",
    "fingerprint",
    "image_path",
    "predicted_label",
    "confidence",
    "actual_label"
]


def save_label(
    fingerprint,
    image_path,
    predicted_label,
    confidence,
    actual_label
):
    """Save a human-confirmed label for a new image."""

    valid_labels = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash"
    ]

    if actual_label not in valid_labels:
        raise ValueError(
            f"Invalid label. Choose one of: {valid_labels}"
        )

    if os.path.exists(LABEL_FILE):
        df = pd.read_csv(LABEL_FILE)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    # Avoid saving the same image twice
    if fingerprint in df["fingerprint"].astype(str).values:
        print("This image is already labeled.")
        return

    new_record = {
        "timestamp": datetime.now().isoformat(),
        "fingerprint": fingerprint,
        "image_path": image_path,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "actual_label": actual_label
    }

    df = pd.concat(
        [df, pd.DataFrame([new_record])],
        ignore_index=True
    )

    df.to_csv(LABEL_FILE, index=False)

    print("New labeled data saved successfully!")