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
# Webcam
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Place your hand inside the box.")
print("Press 'q' to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # ==========================
    # Region of Interest
    # ==========================

    x1 = int(width * 0.05)
    y1 = int(height * 0.15)

    x2 = int(width * 0.45)
    y2 = int(height * 0.85)

    roi = frame[y1:y2, x1:x2]

    # ==========================
    # Convert to HSV
    # ==========================

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(
        hsv,
        lower_skin,
        upper_skin
    )

    # ==========================
    # Clean Mask
    # ==========================

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.GaussianBlur(
        mask,
        (5, 5),
        0
    )

    # ==========================
    # Find Hand Contour
    # ==========================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    hand_mask = np.zeros_like(mask)

    if contours:

        # Largest contour
        contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(contour)

        if area > 3000:

            cv2.drawContours(
                hand_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

            # ==========================
            # Crop Hand
            # ==========================

            hx, hy, hw, hh = cv2.boundingRect(contour)

            hand = hand_mask[
                hy:hy + hh,
                hx:hx + hw
            ]

            if hand.size > 0:

                # Resize to model input
                hand = cv2.resize(
                    hand,
                    (224, 224)
                )

                # Convert grayscale → RGB
                hand = cv2.cvtColor(
                    hand,
                    cv2.COLOR_GRAY2RGB
                )

                # Float32
                hand = hand.astype(
                    np.float32
                )

                # Normalize
                hand = hand / 255.0

                # Batch dimension
                hand = np.expand_dims(
                    hand,
                    axis=0
                )

                # ==========================
                # Prediction
                # ==========================

                prediction = model.predict(
                    hand,
                    verbose=0
                )

                predicted_index = np.argmax(
                    prediction
                )

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = (
                    np.max(prediction) * 100
                )

                # ==========================
                # Display Prediction
                # ==========================

                cv2.putText(
                    frame,
                    f"Gesture: {predicted_class}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.2f}%",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Show processed hand
                cv2.imshow(
                    "Processed Hand",
                    hand[0]
                )

    # ==========================
    # Draw ROI
    # ==========================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Place hand here",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ==========================
    # Display Webcam
    # ==========================

    cv2.imshow(
        "Hand Gesture Number Recognition",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# Cleanup
# ==========================

cap.release()
cv2.destroyAllWindows()