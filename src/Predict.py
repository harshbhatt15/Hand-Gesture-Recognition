import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\Hand gesture\models\best_model.keras"

CLASS_NAMES_PATH = r"D:\Hand gesture\models\class_names.json"

IMAGE_PATH = r"D:\Hand gesture\Dataset\American Sign Language Digits Dataset\0\Input Images - Sign 0\Sign 0 (1).jpeg"


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNetB0 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD CLASSES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r"
) as file:

    class_names = json.load(file)


print(
    "Number of classes:",
    len(class_names)
)

print(
    "Classes:",
    class_names
)


# ============================================================
# LOAD IMAGE
# ============================================================

img = image.load_img(
    IMAGE_PATH,
    target_size=(
        224,
        224
    )
)

img_array = image.img_to_array(
    img
)

# EfficientNetB0 preprocessing is built
# into the Keras EfficientNet model.
# Keep pixels in 0-255 range.

img_array = np.asarray(
    img_array,
    dtype=np.float32
)

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    img_array,
    verbose=1
)

predicted_index = np.argmax(
    prediction[0]
)

confidence = (
    prediction[0][predicted_index]
    * 100
)

predicted_class = class_names[
    predicted_index
]


# ============================================================
# RESULT
# ============================================================

print("\n=============================================")
print("       ASL DIGIT RECOGNITION")
print("=============================================")

print(
    "Image      :",
    os.path.basename(IMAGE_PATH)
)

print(
    "Prediction :",
    predicted_class
)

print(
    f"Confidence : {confidence:.2f}%"
)

print("=============================================")


# ============================================================
# ALL PROBABILITIES
# ============================================================

print("\nPrediction probabilities:")

for i, probability in enumerate(
    prediction[0]
):

    print(
        f"{class_names[i]} : "
        f"{probability * 100:.2f}%"
    )