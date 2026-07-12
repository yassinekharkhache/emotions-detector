# Emotion Detection using VGG16

## Overview

This project implements a real-time facial emotion detection system using a **VGG16-based Convolutional Neural Network (CNN)**. The model classifies facial expressions into seven different emotions from webcam video or images.

The project was developed using **TensorFlow/Keras** for deep learning and **OpenCV** for face detection and real-time video processing.

## Features

* Train an emotion classification model using a VGG16 architecture.
* Detect faces from webcam or recorded videos.
* Predict emotions in real time.
* Evaluate model performance on a test dataset.
* Save the trained model for later inference.
* Training monitored using TensorBoard.
* Learning curves for analyzing overfitting.

## Emotion Classes

The model predicts the following seven emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## Project Structure

```text
project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── results/
│   ├── model/
│   │   ├── final_emotion_model.keras
│   │   ├── final_emotion_model_arch.txt
│   │   ├── learning_curves.png
│   │   └── tensorboard.png
│   └── preprocessing_test/
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── predict_live_stream.py
│   ├── preprocess.py
│   └── validation_loss_accuracy.py
│
├── requirements.txt
└── README.md
```

## Model Architecture

The emotion classifier is based on the **VGG16 architecture**, a deep convolutional neural network well known for image classification tasks.

### Why VGG16?

VGG16 was chosen because it:

* Uses small 3×3 convolution filters.
* Learns rich hierarchical image features.
* Performs well on image classification problems.
* Is simple, reliable, and easy to customize.

To reduce overfitting, the training pipeline includes:

* Data augmentation
* Dropout
* Early Stopping
* Model Checkpoint
* ReduceLROnPlateau

Training progress is monitored with TensorBoard.

## Requirements

* Python 3.x
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Facial Expression Recognition (FER) dataset provided in CSV format.

Place the dataset inside the `data/` directory:

```text
data/
├── train.csv
└── test.csv
```

## How to Run

### 1. Train the model

```bash
python scripts/train.py
```

The trained model will be saved in:

```text
results/model/final_emotion_model.keras
```

---

### 2. Evaluate the model

```bash
python scripts/predict.py
```

Example output:

```text
Accuracy on test set: 65%
```

---

### 3. Real-time emotion detection

```bash
python scripts/predict_live_stream.py
```

Example output:

```text
Reading video stream...

11:21:05 Happy      94%
11:21:06 Happy      91%
11:21:07 Neutral    78%
11:21:08 Surprise   83%
```

## Training Results

During training, the following files are generated:

* `final_emotion_model.keras` – trained model
* `final_emotion_model_arch.txt` – model architecture
* `learning_curves.png` – training/validation curves
* `tensorboard.png` – TensorBoard visualization
