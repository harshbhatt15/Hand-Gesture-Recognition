import os
import json
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

HAND_LANDMARKER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

# How much extra area to include around the hand.
#
# 0.10 = very tight
# 0.15 = recommended
# 0.20 = more space
#
# THIS MUST MATCH WHATEVER PADDING RATIO WAS USED TO BUILD YOUR
# TRAINING DATASET CROPS. Your training script (train.py) uses
# whole images via image_dataset_from_directory with no explicit
# hand-cropping step, so this value only affects how tightly this
# script frames the hand before resizing to 224x224 - it does not
# need to match a "training crop ratio" since none was used at
# training time. Keep this reasonable (default 0.15) and prefer
# tuning it empirically against your webcam's live behavior.
PADDING_RATIO = 0.15

# ------------------------------------------------------------
# PREPROCESSING MODE
# ------------------------------------------------------------
# "efficientnet" -> pass raw [0,255] float32 pixels through unchanged.
#                    This matches train.py exactly: Keras's
#                    EfficientNetB0 (include_top=False) has its own
#                    internal Rescaling + Normalization layers and
#                    expects raw [0,255] input.
#                    tf.keras.applications.efficientnet.preprocess_input
#                    is a documented no-op for EfficientNet, included
#                    here only for symmetry/clarity.
#
# "rescale"       -> image / 255.0. Do NOT use this against the
#                     current train.py - it does not rescale its
#                     inputs, so dividing by 255 here would feed the
#                     model a different distribution than training
#                     and silently break predictions (this was the
#                     cause of the earlier "always predicts one
#                     class at ~50% confidence" bug).
#
# Keep this in sync with whatever train.py actually does. If you
# change train.py's preprocessing, change this to match.
PREPROCESSING_MODE = "efficientnet"

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.60
MIN_PRESENCE_CONFIDENCE = 0.60
MIN_TRACKING_CONFIDENCE = 0.60

# Prediction settings
MIN_PREDICTION_CONFIDENCE = 0.50

# Number of previous predictions used for smoothing
SMOOTHING_FRAMES = 5


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("\nERROR: best_model.keras not found!")
    print("Expected location:")
    print(MODEL_PATH)
    exit()

if not os.path.exists(CLASS_NAMES_PATH):
    print("\nERROR: class_names.json not found!")
    print("Expected location:")
    print(CLASS_NAMES_PATH)
    exit()

if not os.path.exists(HAND_LANDMARKER_PATH):
    print("\nERROR: hand_landmarker.task not found!")
    print("Expected location:")
    print(HAND_LANDMARKER_PATH)
    exit()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("\nClasses:", class_names)

if "models" in class_names or "model" in class_names:
    print(
        "\nWARNING: class_names.json contains a non-gesture class "
        "('models'). This model was likely trained on a contaminated "
        "dataset. Consider retraining with the fixed train.py before "
        "relying on these predictions.\n"
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading EfficientNetB0 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print(f"Preprocessing mode: {PREPROCESSING_MODE}")


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=HAND_LANDMARKER_PATH
)

hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

hand_detector = vision.HandLandmarker.create_from_options(hand_options)


# ============================================================
# MEDIAPIPE HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17)
]


# ============================================================
# DRAW HAND SKELETON
# ============================================================

def draw_hand_skeleton(frame, landmarks):

    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        points.append((x, y))

    for start_index, end_index in HAND_CONNECTIONS:
        if start_index >= len(points) or end_index >= len(points):
            continue
        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

    for x, y in points:
        cv2.circle(frame, (x, y), 8, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1, cv2.LINE_AA)

    return points


# ============================================================
# CREATE TIGHT HAND BOUNDING BOX (square, padded)
# ============================================================

def get_hand_bounding_box(landmarks, frame_width, frame_height):

    x_coordinates = []
    y_coordinates = []

    for landmark in landmarks:
        x_coordinates.append(int(landmark.x * frame_width))
        y_coordinates.append(int(landmark.y * frame_height))

    if not x_coordinates:
        return None

    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)

    hand_width = max_x - min_x
    hand_height = max_y - min_y

    if hand_width <= 0 or hand_height <= 0:
        return None

    padding_x = int(hand_width * PADDING_RATIO)
    padding_y = int(hand_height * PADDING_RATIO)

    min_x -= padding_x
    max_x += padding_x
    min_y -= padding_y
    max_y += padding_y

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(frame_width - 1, max_x)
    max_y = min(frame_height - 1, max_y)

    # Make the box square
    width = max_x - min_x
    height = max_y - min_y
    square_size = max(width, height)

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    min_x = center_x - square_size // 2
    max_x = min_x + square_size
    min_y = center_y - square_size // 2
    max_y = min_y + square_size

    # Correct boundaries if box spills outside frame
    if min_x < 0:
        max_x += abs(min_x)
        min_x = 0

    if min_y < 0:
        max_y += abs(min_y)
        min_y = 0

    if max_x >= frame_width:
        difference = max_x - frame_width + 1
        min_x -= difference
        max_x = frame_width - 1

    if max_y >= frame_height:
        difference = max_y - frame_height + 1
        min_y -= difference
        max_y = frame_height - 1

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(frame_width - 1, max_x)
    max_y = min(frame_height - 1, max_y)

    if max_x <= min_x or max_y <= min_y:
        return None

    return (int(min_x), int(min_y), int(max_x), int(max_y))


# ============================================================
# PREPROCESS HAND IMAGE
# ============================================================

def preprocess_hand(hand_image):

    if hand_image is None or hand_image.size == 0:
        return None

    image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )
    image = image.astype(np.float32)

    if PREPROCESSING_MODE == "rescale":
        image = image / 255.0

    elif PREPROCESSING_MODE == "efficientnet":
        image = tf.keras.applications.efficientnet.preprocess_input(image)

    else:
        raise ValueError(
            f"Unknown PREPROCESSING_MODE: {PREPROCESSING_MODE}. "
            "Use 'rescale' or 'efficientnet'."
        )

    image = np.expand_dims(image, axis=0)
    return image


# ============================================================
# PREDICTION SMOOTHING
# ============================================================

prediction_history = []


def get_stable_prediction(prediction_index):

    prediction_history.append(prediction_index)

    if len(prediction_history) > SMOOTHING_FRAMES:
        prediction_history.pop(0)

    counts = np.bincount(prediction_history, minlength=len(class_names))
    stable_index = int(np.argmax(counts))
    return stable_index


# ============================================================
# OPEN WEBCAM
# ============================================================

print("\nStarting webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("\nERROR: Cannot open webcam.")
    hand_detector.close()
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "ASL Digit Recognition - Hand Tracking"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


# ============================================================
# START MESSAGE
# ============================================================

print()
print("=============================================")
print("       ASL DIGIT WEBCAM RECOGNITION")
print("=============================================")
print()
print("Green dots  = hand landmarks")
print("Green lines = finger connections")
print()
print("Place your complete hand inside the camera.")
print("Press Q to exit.")
print()


# ============================================================
# MAIN WEBCAM LOOP


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Failed to capture webcam frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]

    # Clean copy - model always predicts from this, never from
    # the frame with landmarks/boxes drawn on it.
    clean_frame = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    try:
        detection_result = hand_detector.detect(mp_image)
    except Exception as e:
        print(f"Detection error: {e}")
        detection_result = None

    cv2.putText(
        frame, "Show one hand", (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

    if detection_result and detection_result.hand_landmarks:

        landmarks = detection_result.hand_landmarks[0]

        bbox = get_hand_bounding_box(landmarks, frame_width, frame_height)

        if bbox is not None:

            x1, y1, x2, y2 = bbox

            hand_crop = clean_frame[y1:y2, x1:x2]

            draw_hand_skeleton(frame, landmarks)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            processed_image = preprocess_hand(hand_crop)

            if processed_image is not None:

                try:
                    predictions = model.predict(processed_image, verbose=0)
                except Exception as e:
                    print(f"Inference error: {e}")
                    predictions = None

                if predictions is not None:

                    raw_index = int(np.argmax(predictions[0]))

                    stable_index = get_stable_prediction(raw_index)
                    stable_confidence = float(predictions[0][stable_index])

                    predicted_digit = class_names[stable_index]

                    if stable_confidence >= MIN_PREDICTION_CONFIDENCE:

                        cv2.putText(
                            frame, f"Gesture: {predicted_digit}",
                            (25, 85), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 3, cv2.LINE_AA
                        )

                        cv2.putText(
                            frame,
                            f"Confidence: {stable_confidence * 100:.2f}%",
                            (25, 125), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2, cv2.LINE_AA
                        )

                    else:
                        cv2.putText(
                            frame, "Gesture: Detecting...",
                            (25, 85), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2, cv2.LINE_AA
                        )

                    # Model input preview thumbnail (guarded against
                    # frames too narrow to fit the 180x180 preview box)
                    if hand_crop.size > 0 and frame_width > 200 and frame_height > 200:

                        preview = cv2.resize(
                            hand_crop, (180, 180),
                            interpolation=cv2.INTER_AREA
                        )

                        preview_x = frame_width - 200
                        preview_y = 20

                        frame[
                            preview_y:preview_y + 180,
                            preview_x:preview_x + 180
                        ] = preview

                        cv2.rectangle(
                            frame,
                            (preview_x, preview_y),
                            (preview_x + 180, preview_y + 180),
                            (0, 255, 0), 2
                        )

                        cv2.putText(
                            frame, "MODEL INPUT",
                            (preview_x, preview_y + 205),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 2, cv2.LINE_AA
                        )

    else:
        prediction_history.clear()

        cv2.putText(
            frame, "No hand detected",
            (25, 85), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 0, 255), 2, cv2.LINE_AA
        )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# ============================================================
# CLEANUP


camera.release()
cv2.destroyAllWindows()
hand_detector.close()

print("\nWebcam closed.")