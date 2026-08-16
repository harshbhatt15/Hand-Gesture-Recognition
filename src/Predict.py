import json
import numpy as np
import tensorflow as tf
from pathlib import Path


# ============================================================
# Project Paths

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"


# ============================================================
# Image to Predict

# Change this path when testing another image
IMAGE_PATH = (
    BASE_DIR
    / "Dataset"
    / "ASL-HG American Sign Language Hand Gesture Image D"
    / "ASL_HG_36000"
    / "Processed_images"
    / "asl_processed"
    / "test"
    / "0"
    / "P1_0_1.jpg"
)


# ============================================================
# Check Files


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}\n\n"
        "Run train.py first."
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found:\n{CLASS_NAMES_PATH}\n\n"
        "Run train.py first."
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}\n\n"
        "Change IMAGE_PATH to a valid image."
    )


# ============================================================
# Load Model

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# Load Class Names

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Number of classes:", len(class_names))
print("Classes:", class_names)


# ============================================================
# Load Image

image_bytes = tf.io.read_file(
    str(IMAGE_PATH)
)

img = tf.io.decode_image(
    image_bytes,
    channels=3,
    expand_animations=False
)


# ============================================================
# Resize


img = tf.image.resize(
    img,
    [224, 224]
)


# ============================================================
# Normalize

img = tf.cast(
    img,
    tf.float32
)

img = img / 255.0


# ============================================================
# Add Batch Dimension

img = tf.expand_dims(
    img,
    axis=0
)


# ============================================================
# Prediction

prediction = model.predict(
    img,
    verbose=0
)

predicted_index = int(
    np.argmax(prediction[0])
)

predicted_class = class_names[
    predicted_index
]

confidence = float(
    np.max(prediction[0]) * 100
)


# ============================================================
# Display Result

print()
print("=" * 45)
print("       ASL HAND GESTURE RECOGNITION")
print("=" * 45)

print(f"Image       : {IMAGE_PATH.name}")
print(f"Prediction  : {predicted_class}")
print(f"Confidence  : {confidence:.2f}%")

print("=" * 45)