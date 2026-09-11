# ♻️ Garbage Classification ML

An end-to-end **Garbage Image Classification** project using **Deep Learning and Transfer Learning** to classify waste images into six categories:

- 📦 Cardboard
- 🪟 Glass
- 🔩 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

The project covers the complete ML workflow — from dataset preparation and auditing to model training, evaluation, prediction, caching, and API integration.

---

## 📌 Project Overview

Garbage classification is an image classification problem where the objective is to automatically identify the type of waste present in an image.

This project uses **MobileNetV2 with Transfer Learning** to classify garbage images into six waste categories.

### Classes

| Class | Description |
|---|---|
| `cardboard` | Cardboard waste |
| `glass` | Glass waste |
| `metal` | Metal waste |
| `paper` | Paper waste |
| `plastic` | Plastic waste |
| `trash` | General/other trash |

---

## 🎯 Objectives

- Build an image classification pipeline for garbage/waste images.
- Audit and clean the dataset before training.
- Perform Exploratory Data Analysis (EDA).
- Create reproducible train, validation, and test splits.
- Apply image preprocessing and augmentation.
- Use MobileNetV2 transfer learning for classification.
- Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
- Save trained models for inference.
- Provide prediction functionality.
- Support API-based inference through FastAPI.
- Maintain prediction caching and monitoring components.

---

## 🏗️ Project Architecture

```text
garbage-classification-ml/
│
├── api/
│   └── main.py
│
├── cache/
│   └── predictions.json
│
├── data/
│   └── Garbage_Dataset_Classification/
│
├── logs/
│
├── models/
│   ├── garbage_mobilenetv2_best.keras
│   └── garbage_mobilenetv2_final.keras
│
├── notebooks/
│
└── src/
    ├── cache.py
    ├── data_audit.py
    ├── data_cleaning.py
    ├── eda.py
    ├── error_analysis.py
    ├── evaluate.py
    ├── fingerprint.py
    ├── label_data.py
    ├── model.py
    ├── monitoring.py
    ├── predict.py
    ├── prepare_data.py
    ├── preprocess.py
    └── train.py


    📊 Dataset

The prepared dataset contains:

13,899 image records

Class Distribution
Class	Images
Trash	2,500
Glass	2,499
Paper	2,315
Plastic	2,288
Cardboard	2,214
Metal	2,083
Total	13,899

The dataset is not perfectly balanced, but the class distribution is reasonably suitable for training.

🔀 Dataset Splitting

The project uses a stratified random train-validation-test split.

Split Ratio
Training     → 70%
Validation   → 15%
Testing      → 15%

Actual split:

Dataset	Images
Train	9,729
Validation	2,085
Test	2,085
Total	13,899
Stratification

The split uses:

stratify=df["label"]

This ensures that the class distribution is approximately preserved across training, validation, and test datasets.

A fixed:

random_state=42

is also used to make the split reproducible.

🔍 Data Pipeline

The project follows the following pipeline:

Raw Dataset
     │
     ▼
Data Audit
     │
     ▼
Data Cleaning
     │
     ▼
EDA
     │
     ▼
Train / Validation / Test Split
     │
     ▼
Image Preprocessing
     │
     ▼
Data Augmentation
     │
     ▼
MobileNetV2
     │
     ▼
Classification Head
     │
     ▼
Model Training
     │
     ▼
Evaluation
     │
     ▼
Prediction / API
🧹 Data Audit & Cleaning

Before training, the dataset is checked for common data-quality problems.

The project includes:

Metadata validation
Missing values checking
Image file existence checking
Corrupted image detection
Duplicate detection using SHA-256 fingerprints
Data cleaning

The audit checks image files against the metadata and helps prevent invalid samples from entering the training pipeline.

Note: SHA-256 detects exact binary duplicates. It does not detect visually similar images that have been resized, recompressed, or slightly modified.

📈 Exploratory Data Analysis

The EDA module analyzes the dataset before model training.

Current analysis includes:

Dataset shape
Class distribution
Image brightness statistics
Average brightness by class
Dataset Shape
13,899 rows × 2 columns

Metadata contains:

filename
label
Brightness Analysis

Across the analyzed images:

Minimum brightness  ≈ 2.32
Maximum brightness  ≈ 253.70
Average brightness  ≈ 161.52

Average brightness by class:

Class	Average Brightness
Cardboard	164.43
Glass	166.62
Metal	166.06
Paper	160.54
Plastic	167.89
Trash	145.15
🖼️ Image Preprocessing

Images are processed before being passed to the neural network.

Input Size
224 × 224 × 3

The model therefore receives:

(batch_size, 224, 224, 3)

Example batch:

Images → (32, 224, 224, 3)
Labels → (32, 6)
Pixel Scaling

The processed pixel values are in approximately:

[-1, 1]

This preprocessing is compatible with the MobileNetV2 input pipeline used in the project.

🧠 Model

The project uses:

MobileNetV2 + Transfer Learning

MobileNetV2 is used as the pretrained feature extractor.

The architecture consists of:

Input
  │
  ▼
MobileNetV2
  │
  ▼
Global Average Pooling
  │
  ▼
Dropout
  │
  ▼
Dense Layer
  │
  ▼
6 Classes
Model Configuration
Component	Details
Input	224 × 224 × 3
Backbone	MobileNetV2
Backbone parameters	2,257,984
Backbone	Non-trainable
Pooling	GlobalAveragePooling2D
Regularization	Dropout
Output layer	Dense
Number of classes	6
Trainable parameters	7,686
Total parameters	2,265,670

The pretrained MobileNetV2 backbone is frozen while the classification head is trained for the six garbage categories.

🏋️ Training

Training is performed using the prepared training and validation datasets.

The project saves:

models/
├── garbage_mobilenetv2_best.keras
└── garbage_mobilenetv2_final.keras

Training logs are stored in:

logs/training.csv

The best checkpoint observed in the provided training run was saved during epoch 1.

📊 Model Evaluation

The model was evaluated on the independent test dataset containing:

2,085 images
Test Results
Metric	Result
Test Accuracy	82.40%
Test Loss	0.4923

The verified test accuracy from the evaluation run is approximately:

82.40%
📋 Classification Report
Class	Precision	Recall	F1-Score
Cardboard	0.82	0.83	0.83
Glass	0.74	0.93	0.83
Metal	0.88	0.80	0.84
Paper	0.80	0.84	0.82
Plastic	0.83	0.69	0.75
Trash	0.91	0.84	0.88
Macro Avg	0.83	0.82	0.82
Weighted Avg	0.83	0.82	0.82
Best Performing Class

trash achieved the highest F1-score:

F1 = 0.88
Main Challenge

plastic had the lowest recall:

Recall = 0.69

This indicates that a significant number of plastic images were classified as other waste categories.

🔥 Confusion Matrix

The test confusion matrix was:

[[277   3   7  33   6   6]
 [  2 348   4   3  14   4]
 [  8  30 250   7  12   6]
 [ 35   7   4 291   5   5]
 [  6  65  10  16 237   9]
 [ 10  15   8  15  12 315]]

Class order:

0 → cardboard
1 → glass
2 → metal
3 → paper
4 → plastic
5 → trash

A major confusion pattern can be observed between:

Plastic ↔ Glass

which is consistent with the lower recall for the plastic class.

🔎 Error Analysis

The project includes an error_analysis.py module for analyzing incorrect predictions.

The evaluation results indicate that the main classification challenges are related to visually similar waste categories.

Examples include:

Plastic → Glass
Glass   → Plastic
Cardboard → Paper
Metal → Glass

Further improvement could focus on these confusing class pairs.

🔮 Prediction

The project includes:

src/predict.py

for model inference.

The prediction pipeline loads the trained model, preprocesses an input image, and produces a classification result across the six garbage categories.

🌐 API

The project contains a FastAPI application:

api/main.py

The API layer is intended to expose the trained classification model for application-level inference.

The exact request/response contract should be kept aligned with the implementation in api/main.py.

⚡ Prediction Cache

The project contains a caching component:

src/cache.py

and prediction cache storage:

cache/predictions.json

Caching can help avoid repeatedly processing the same prediction request when an identical input has already been handled.

📡 Monitoring

The project also contains:

src/monitoring.py

for monitoring-related functionality.

This provides a foundation for tracking prediction/system behavior after deployment.

🧰 Technology Stack
Programming Language
Python
Machine Learning
TensorFlow
Keras
MobileNetV2
Scikit-learn
Data Processing
Pandas
NumPy
Image Processing
PIL / Pillow
API
FastAPI
Data Visualization
Matplotlib
Model Format
.keras
📦 Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd garbage-classification-ml

Create a virtual environment:

Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
▶️ Running the Project
1. Prepare the Dataset
python src/prepare_data.py

This creates:

data/splits/
├── train.csv
├── validation.csv
└── test.csv
2. Run Data Audit
python src/data_audit.py
3. Run Data Cleaning
python src/data_cleaning.py
4. Run EDA
python src/eda.py
5. Train the Model
python src/train.py

The trained models will be stored under:

models/
6. Evaluate the Model
python src/evaluate.py

The evaluation report is saved to:

logs/evaluation_report.txt
7. Run Prediction
python src/predict.py
📁 Important Files
File	Purpose
data_audit.py	Dataset quality and duplicate checks
data_cleaning.py	Dataset cleaning
eda.py	Exploratory data analysis
prepare_data.py	Train/validation/test split
preprocess.py	Image preprocessing and data pipeline
model.py	MobileNetV2 model architecture
train.py	Model training
evaluate.py	Model evaluation
predict.py	Image prediction
error_analysis.py	Prediction error analysis
fingerprint.py	File fingerprinting
cache.py	Prediction caching
monitoring.py	Monitoring functionality
api/main.py	FastAPI application
🔁 Reproducibility

The dataset split uses:

random_state=42

and stratification:

stratify=df["label"]

This makes the train/validation/test split reproducible while maintaining class proportions.

⚠️ Limitations

Current limitations include:

The dataset is not perfectly balanced.
Plastic has comparatively lower recall.
Exact duplicate detection does not identify near-duplicate images.
Visually similar waste categories can be difficult to distinguish.
Further tuning and validation would be required before production deployment.
The provided training run was executed using CPU-based TensorFlow on native Windows.
🚀 Future Improvements

Potential improvements include:

Fine-tuning selected MobileNetV2 layers.
More extensive data augmentation.
Class-weighted training or other imbalance-handling techniques.
Near-duplicate detection using perceptual hashing or embeddings.
Hyperparameter tuning.
Cross-validation experiments.
Better handling of confusing classes such as plastic and glass.
Model explainability using Grad-CAM.
API deployment using Docker.
Production monitoring and logging.
Performance benchmarking on real-world images.
📌 Project Status

Status: Completed ML pipeline with trained model and evaluation.

Current verified test performance:

Accuracy: 82.40%
👨‍💻 Author

Bharat

Garbage Classification ML Project
