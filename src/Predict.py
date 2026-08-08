import numpy as np
import tensorflow as tf
from pathlib import Path

# ==========================
# Project Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.keras"

IMAGE_PATH = (
    BASE_DIR
    / "dataset"
    / "test"
    / "test"
    / "0"
    / "902.jpg"
)

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
# Load Image
# ==========================

image_bytes = tf.io.read_file(str(IMAGE_PATH))

img = tf.io.decode_jpeg(
    image_bytes,
    channels=3
)

# ==========================
# Resize
# ==========================

img = tf.image.resize(
    img,
    [224, 224]
)

# ==========================
# Convert to Float
# ==========================

img = tf.cast(img, tf.float32)

# ==========================
# Normalize
# ==========================

img = img / 255.0

# ==========================
# Add Batch Dimension
# ==========================

img = tf.expand_dims(img, axis=0)

# ==========================
# Prediction
# ==========================

prediction = model.predict(
    img,
    verbose=0
)

predicted_index = np.argmax(prediction)

predicted_class = class_names[predicted_index]

confidence = np.max(prediction) * 100

# ==========================
# Output
# ==========================

print("=" * 40)
print("HAND GESTURE RECOGNITION")
print("=" * 40)

print("Image:", IMAGE_PATH.name)
print("Predicted Gesture:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

print("=" * 40)