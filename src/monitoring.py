import os
import csv
from datetime import datetime

LOG_DIR = "logs"
MONITORING_FILE = os.path.join(LOG_DIR, "prediction_monitoring.csv")

os.makedirs(LOG_DIR, exist_ok=True)

COLUMNS = [
    "timestamp",
    "fingerprint",
    "prediction",
    "confidence",
    "cache_status",
    "processing_time_seconds"
]


def log_prediction(
    fingerprint,
    prediction,
    confidence,
    cache_status,
    processing_time_seconds
):
    """Save one prediction request to the monitoring log."""

    file_exists = os.path.exists(MONITORING_FILE)

    with open(
        MONITORING_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "fingerprint": fingerprint,
            "prediction": prediction,
            "confidence": confidence,
            "cache_status": cache_status,
            "processing_time_seconds": processing_time_seconds
        })