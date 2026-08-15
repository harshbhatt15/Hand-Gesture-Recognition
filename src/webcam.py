import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Project Paths

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"


# ============================================================
# Check Required Files

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names not found: {CLASS_NAMES_PATH}"
    )

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Hand landmarker model not found: {HAND_MODEL_PATH}"
    )


# ============================================================
# Load MobileNetV2 Model

print("Loading gesture recognition model...")

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
# MediaPipe Hand Landmarker

base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# Open Webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )

print()
print("========================================")
print("ASL HAND GESTURE RECOGNITION")
print("========================================")
print("Show your hand to the camera.")
print("Press Q to quit.")
print("========================================")


# ============================================================
# Webcam Loop


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # --------------------------------------------------------
    # Convert OpenCV BGR → RGB
    

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------------------------
    # Detect Hand


    result = detector.detect(mp_image)

    # --------------------------------------------------------
    # If Hand Detected
    

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        # Get bounding box
        x_coordinates = [
            int(landmark.x * width)
            for landmark in landmarks
        ]

        y_coordinates = [
            int(landmark.y * height)
            for landmark in landmarks
        ]

        x_min = min(x_coordinates)
        x_max = max(x_coordinates)

        y_min = min(y_coordinates)
        y_max = max(y_coordinates)

        # Add padding
        padding = 40

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)

        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Crop Hand
        

        hand = frame[
            y_min:y_max,
            x_min:x_max
        ]

        if hand.size > 0:

            # BGR → RGB
            hand_rgb = cv2.cvtColor(
                hand,
                cv2.COLOR_BGR2RGB
            )

            # Resize
            hand_rgb = cv2.resize(
                hand_rgb,
                (224, 224)
            )

            # Float32
            hand_rgb = hand_rgb.astype(
                np.float32
            )

            # Normalize
            hand_rgb = hand_rgb / 255.0

            # Add batch dimension
            input_image = np.expand_dims(
                hand_rgb,
                axis=0
            )

            # ------------------------------------------------
            # Prediction
        

            prediction = model.predict(
                input_image,
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

            # ------------------------------------------------
            # Display Prediction

            cv2.putText(
                frame,
                f"Gesture: {predicted_class}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}%",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Show Processed Hand
            

            cv2.imshow(
                "Processed Hand",
                cv2.cvtColor(
                    hand_rgb,
                    cv2.COLOR_RGB2BGR
                )
            )

        # ----------------------------------------------------
        # Draw Hand Landmarks
        

        for landmark in landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ========================================================
    # Show Webcam
    

    cv2.imshow(
        "ASL Hand Gesture Recognition",
        frame
    )

    # ========================================================
    # Quit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# Cleanup


cap.release()

detector.close()

cv2.destroyAllWindows()

print("\nWebcam stopped.")