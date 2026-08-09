import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ==========================
# Project Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.keras"

# ==========================
# Load Model
# ==========================

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# Class Names
# ==========================

class_names = [
    '0',
    '1',
    '10',
    '11',
    '12',
    '13',
    '14',
    '15',
    '16',
    '17',
    '18',
    '19',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9'
]

# ==========================
# Open Webcam
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press 'q' to quit.")

# ==========================
# Real-Time Prediction
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip frame so it behaves like a mirror
    frame = cv2.flip(frame, 1)

    # Resize frame to model input size
    img = cv2.resize(frame, (224, 224))

    # OpenCV uses BGR, TensorFlow models expect RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float32
    img = img.astype(np.float32)

    # Normalize exactly like training
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction) * 100

    # ==========================
    # Display Result
    # ==========================

    text = f"Gesture: {predicted_class}"
    confidence_text = f"Confidence: {confidence:.2f}%"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        confidence_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Display webcam
    cv2.imshow("Hand Gesture Number Recognition", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# Release Resources
# ==========================

cap.release()
cv2.destroyAllWindows()