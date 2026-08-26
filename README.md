# Chest X-Ray Anomaly Detection

A deep learning project for detecting abnormal findings in chest X-ray images using **EfficientNetB0**, transfer learning, fine-tuning, and medical image preprocessing techniques.

The goal of the project is to classify chest X-rays into two categories:

- **Normal**
- **Abnormal / Finding**

The project was developed using the **VinDr-Chest** dataset.

## Project Highlights

- Binary chest X-ray classification
- EfficientNetB0-based deep learning model
- Transfer Learning and Fine-Tuning
- CLAHE-based image enhancement
- Data augmentation during training
- Experimental SE Attention integration
- Evaluation with Accuracy, Recall, F1-Score, Confusion Matrix, ROC and AUC
- Saved trained Keras model for reuse

## Dataset

The dataset was balanced across all splits.

| Split | Normal | Abnormal |
|---|---:|---:|
| Train | 4,277 | 4,277 |
| Validation | 535 | 535 |
| Test | 535 | 535 |

## Image Preprocessing

Several image enhancement techniques were evaluated, including:

- Histogram Equalization
- CLAHE
- Gaussian Blur
- Median Blur

**CLAHE** was selected because it improved image contrast while preserving anatomical details in the lung regions.

Data augmentation was also applied during training using:

- Rotation
- Horizontal Flip
- Zoom
- Brightness Adjustment

## Model Development

The primary architecture used in this project was **EfficientNetB0**.

The training process included:

1. Transfer Learning
2. Fine-Tuning
3. Experimental SE (Squeeze-and-Excitation) Attention integration

Fine-Tuning produced the best validation performance and was selected for the final model.

## Results

### Final Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | ~86% |
| Abnormal Recall / Sensitivity | 88% |
| F1-Score | ~86–87% |
| ROC-AUC | 0.935 |

### Confusion Matrix

| | Predicted Normal | Predicted Abnormal |
|---|---:|---:|
| Actual Normal | 452 | 83 |
| Actual Abnormal | 63 | 472 |

The model correctly identified **472 abnormal chest X-rays**, while 63 abnormal cases were classified as normal.

### Model Comparison

| Model | Validation Accuracy |
|---|---:|
| EfficientNetB0 + Fine-Tuning | **84.49%** |
| EfficientNetB0 + SE Attention | 80.28% |
| EfficientNetB0 + SE Attention + Fine-Tuning | 79.81% |

The experiments showed that the fine-tuned EfficientNetB0 model performed better than the tested SE Attention variants.

## Project Structure

```text
chest-xray-anomaly-detection/
├── models/
│   └── best_efficientnet_model.keras
├── notebooks/
│   └── ChestXray_Anomaly_Detection.ipynb
├── README.md
└── .gitignore
```

## Repository Contents

### Notebook

`notebooks/ChestXray_Anomaly_Detection.ipynb`

Contains the complete workflow including:

- Data exploration
- Image preprocessing
- Data augmentation
- Model training
- Transfer learning
- Fine-tuning
- Evaluation and performance analysis

### Trained Model

`models/best_efficientnet_model.keras`

Contains the saved trained model selected during the experiments.

## Technologies & Concepts

- Python
- TensorFlow / Keras
- EfficientNetB0
- Transfer Learning
- Fine-Tuning
- Convolutional Neural Networks
- Medical Image Processing
- CLAHE
- Data Augmentation
- SE Attention
- ROC / AUC
- Confusion Matrix

## Academic Context

This project was developed collaboratively as a deep learning study focused on medical image classification and chest X-ray anomaly detection.

## Authors

- Zemzem Ertekin
- Kübra Özatak
- Helen Benzer
- Furkan Yıldız

## Disclaimer

This project was developed for **educational and research purposes only**.

The model is not intended for clinical diagnosis or medical decision-making.
