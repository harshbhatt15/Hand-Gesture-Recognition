# 🤟 ASL Hand Gesture Digit Recognition

A real-time **American Sign Language (ASL) digit recognition system** that uses deep learning and computer vision to recognize hand gestures representing the digits **0–9**.

The project uses **EfficientNetB0 Transfer Learning** for image classification and **MediaPipe Hand Landmarker** with OpenCV for real-time webcam interaction.

---

## 🚀 Features

- 🔢 Recognizes ASL digits from **0–9**
- 🧠 EfficientNetB0 transfer learning
- 🎯 Image-based prediction
- 📷 Real-time webcam prediction
- ✋ MediaPipe hand detection
- 📊 Confidence score for predictions
- 🔄 Data augmentation
- 🔥 Two-stage training:
  - Transfer learning
  - Fine-tuning
- 💾 Saves the best-performing Keras model
- 🐍 Built with Python

---

## 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Model development |
| EfficientNetB0 | Transfer learning model |
| OpenCV | Webcam and image processing |
| MediaPipe | Hand detection and tracking |
| NumPy | Numerical operations |

---

## 📁 Project Structure

```text
Hand gesture/
│
├── Dataset/
│   └── American Sign Language Digits Dataset/
│       ├── 0/
│       ├── 1/
│       ├── 2/
│       ├── 3/
│       ├── 4/
│       ├── 5/
│       ├── 6/
│       ├── 7/
│       ├── 8/
│       └── 9/
│
├── models/
│   ├── best_model.keras
│   ├── class_names.json
│   └── hand_landmarker.task
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── webcam.py
│
├── .gitignore
├── requirements.txt
└── README.md

---

## 🧠 Model Architecture

The project uses EfficientNetB0 with ImageNet pretrained weights.

Input Image
     │
     ▼
224 × 224 × 3
     │
     ▼
Data Augmentation
     │
     ▼
EfficientNetB0
     │
     ▼
Global Average Pooling
     │
     ▼
Batch Normalization
     │
     ▼
Dropout
     │
     ▼
Dense Layer (256)
     │
     ▼
Batch Normalization
     │
     ▼
Dropout
     │
     ▼
10-Class Softmax
     │
     ▼
0 - 9 Prediction

---

# ✋ MediaPipe Hand Detection

MediaPipe is used to detect and locate the hand in the webcam feed.

The processing pipeline is:

Webcam
   │
   ▼
OpenCV
   │
   ▼
MediaPipe Hand Detection
   │
   ▼
Hand Region
   │
   ▼
Image Preprocessing
   │
   ▼
EfficientNetB0
   │
   ▼
Digit Prediction

The MediaPipe hand model is stored as:
models/hand_landmarker.task

---

## 👨‍💻 Author

Harsh Bhatt

