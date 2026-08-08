import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "models/best_model.keras"
CLASS0_PATH = r"D:\Hand gesture\dataset\test\test\0"

# ==========================
# Load Model
# ==========================

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# Get Class 0 Images
# ==========================

files = [
    f for f in os.listdir(CLASS0_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

correct = 0
total = len(files)

# ==========================
# Test Images
# ==========================

for file in files:

    img_path = os.path.join(CLASS0_PATH, file)

    # Load image
    img = tf.keras.utils.load_img(
        img_path,
        target_size=(224, 224)
    )

    # Convert to array
    img = tf.keras.utils.img_to_array(img)

    # Normalize exactly like training
    img = img / 255.0

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)

    # Class 0 corresponds to index 0
    if predicted_index == 0:
        correct += 1

# ==========================
# Results
# ==========================

accuracy = correct / total

print()
print("==============================")
print("CLASS 0 RESULTS")
print("==============================")
print("Total images:", total)
print("Correct:", correct)
print("Wrong:", total - correct)
print(f"Accuracy: {accuracy * 100:.2f}%")