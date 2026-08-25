import os
import json
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\Hand gesture\models\best_model.keras"

HAND_LANDMARKER_PATH = r"D:\Hand gesture\models\hand_landmarker.task"

CLASS_NAMES_PATH = r"D:\Hand gesture\models\class_names.json"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 40.0


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNetB0 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD CLASSES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r"
) as f:

    class_names = json.load(f)


print("Classes:", class_names)


# ============================================================
# MEDIAPIPE
# ============================================================

print("\nLoading MediaPipe...")

base_options = python.BaseOptions(
    model_asset_path=HAND_LANDMARKER_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

detector = vision.HandLandmarker.create_from_options(
    options
)

print("MediaPipe loaded successfully.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    detector.close()

    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# START
# ============================================================

print("\n=============================================")
print("       ASL DIGIT RECOGNITION")
print("       EfficientNetB0")
print("=============================================")
print("Show your hand.")
print("Press Q to quit.")
print("=============================================\n")


try:

    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        ret, frame = cap.read()

        if not ret:

            print("Could not read camera.")

            break


        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )


        height, width, _ = frame.shape


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ====================================================
        # DETECT HAND
        # ====================================================

        result = detector.detect(
            mp_image
        )


        prediction_text = "No Hand"

        confidence = 0.0


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ------------------------------------------------
            # LANDMARK COORDINATES
            # ------------------------------------------------

            x_points = [
                int(point.x * width)
                for point in hand
            ]

            y_points = [
                int(point.y * height)
                for point in hand
            ]


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x_min = min(x_points)

            x_max = max(x_points)

            y_min = min(y_points)

            y_max = max(y_points)


            # ------------------------------------------------
            # PADDING
            # ------------------------------------------------

            padding = 50

            x_min -= padding
            y_min -= padding

            x_max += padding
            y_max += padding


            # Keep inside image

            x_min = max(
                0,
                x_min
            )

            y_min = max(
                0,
                y_min
            )

            x_max = min(
                width,
                x_max
            )

            y_max = min(
                height,
                y_max
            )


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                3
            )


            # =================================================
            # CROP HAND
            # =================================================

            hand_crop = frame[
                y_min:y_max,
                x_min:x_max
            ]


            if hand_crop.size > 0:

                # =============================================
                # RESIZE
                # =============================================

                hand_crop = cv2.resize(
                    hand_crop,
                    (
                        IMG_SIZE,
                        IMG_SIZE
                    ),
                    interpolation=cv2.INTER_AREA
                )


                # =============================================
                # BGR -> RGB
                # =============================================

                hand_crop = cv2.cvtColor(
                    hand_crop,
                    cv2.COLOR_BGR2RGB
                )


                # =============================================
                # FLOAT32
                # =============================================

                hand_crop = hand_crop.astype(
                    np.float32
                )


                # =============================================
                # ADD BATCH
                # =============================================

                input_image = np.expand_dims(
                    hand_crop,
                    axis=0
                )


                # =================================================
                # PREDICT
                # =================================================

                predictions = model.predict(
                    input_image,
                    verbose=0
                )[0]


                predicted_index = int(
                    np.argmax(predictions)
                )


                confidence = float(
                    predictions[
                        predicted_index
                    ] * 100
                )


                predicted_class = class_names[
                    predicted_index
                ]


                # =================================================
                # RESULT
                # =================================================

                if confidence >= CONFIDENCE_THRESHOLD:

                    prediction_text = (
                        f"Digit: {predicted_class}"
                    )

                else:

                    prediction_text = (
                        "Uncertain"
                    )


        # ====================================================
        # DISPLAY
        # ====================================================

        # Background panel

        cv2.rectangle(
            frame,
            (10, 10),
            (450, 130),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        cv2.putText(
            frame,
            prediction_text,
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}%",
            (25, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # INSTRUCTION
        # ====================================================

        cv2.putText(
            frame,
            "Press Q to quit",
            (
                width - 210,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "ASL Digit Recognition - EfficientNetB0",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(
            1
        ) & 0xFF


        if key == ord("q"):

            break


finally:

    cap.release()

    cv2.destroyAllWindows()

    detector.close()

    print("\nWebcam stopped.")