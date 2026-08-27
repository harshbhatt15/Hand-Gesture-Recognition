import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\Hand gesture"

DATASET_DIR = (
    r"D:\Hand gesture\Dataset"
    r"\American Sign Language Digits Dataset"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.20

SEED = 42

INITIAL_EPOCHS = 10

FINE_TUNE_EPOCHS = 15


# ============================================================
# SAFETY CHECK - MODEL_DIR MUST NOT BE INSIDE DATASET_DIR
# ============================================================
#
# This is the exact bug that previously caused a "models" class
# to appear in the trained model's class list: MODEL_DIR ended up
# nested inside (or equal to) DATASET_DIR, so image_dataset_from_directory
# scanned it as if it were a gesture class.
#
# This check makes that failure mode impossible to hit silently again.

_normalized_model_dir = os.path.normcase(os.path.abspath(MODEL_DIR))
_normalized_dataset_dir = os.path.normcase(os.path.abspath(DATASET_DIR))

if (
    _normalized_model_dir == _normalized_dataset_dir
    or _normalized_model_dir.startswith(_normalized_dataset_dir + os.sep)
):
    raise RuntimeError(
        "\n\nMODEL_DIR is inside (or equal to) DATASET_DIR.\n"
        f"MODEL_DIR:   {MODEL_DIR}\n"
        f"DATASET_DIR: {DATASET_DIR}\n\n"
        "This causes the model-output folder to be scanned as a "
        "gesture class during training. Move MODEL_DIR outside of "
        "DATASET_DIR before running this script."
    )


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("\n=============================================")
print("     ASL DIGIT RECOGNITION - EFFICIENTNETB0")
print("=============================================\n")

print("Dataset:")
print(DATASET_DIR)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_DIR}"
    )


# ============================================================
# WARN ABOUT UNEXPECTED SUBFOLDERS BEFORE LOADING
# ============================================================
#
# image_dataset_from_directory treats every subfolder as a class.
# Catching stray folders (like a stray "models" directory) here,
# before any training compute is spent, is much cheaper than
# discovering it after a full training run.

expected_classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

_actual_subfolders = sorted([
    name for name in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, name))
])

_unexpected = [
    name for name in _actual_subfolders
    if name not in expected_classes
]

if _unexpected:
    raise RuntimeError(
        "\n\nUnexpected subfolder(s) found inside DATASET_DIR:\n"
        f"{_unexpected}\n\n"
        f"DATASET_DIR should only contain: {expected_classes}\n"
        f"Found: {_actual_subfolders}\n\n"
        "Remove or relocate the unexpected folder(s) before training."
    )

_missing = [
    name for name in expected_classes
    if name not in _actual_subfolders
]

if _missing:
    raise RuntimeError(
        "\n\nExpected class folder(s) missing from DATASET_DIR:\n"
        f"{_missing}\n\n"
        f"DATASET_DIR should contain: {expected_classes}\n"
        f"Found: {_actual_subfolders}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")


train_ds = tf.keras.utils.image_dataset_from_directory(

    DATASET_DIR,

    labels="inferred",

    label_mode="int",

    validation_split=VALIDATION_SPLIT,

    subset="training",

    seed=SEED,

    image_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    shuffle=True
)


val_ds = tf.keras.utils.image_dataset_from_directory(

    DATASET_DIR,

    labels="inferred",

    label_mode="int",

    validation_split=VALIDATION_SPLIT,

    subset="validation",

    seed=SEED,

    image_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names

num_classes = len(
    class_names
)


print("\n=============================================")

print(
    "Number of classes:",
    num_classes
)

print(
    "Classes:",
    class_names
)

print("=============================================")


# ============================================================
# VERIFY CLASSES - HARD STOP ON MISMATCH
# ============================================================
#
# Previously this only printed a warning and continued training
# anyway, which is exactly how a "models" class made it into a
# real trained model. A mismatch here now stops the run.

if class_names != expected_classes:

    raise RuntimeError(
        "\n\nClass mismatch detected - training aborted.\n"
        f"Expected: {expected_classes}\n"
        f"Found:    {class_names}\n\n"
        "Do not proceed with training until this matches exactly. "
        "Check DATASET_DIR for stray folders, empty class folders, "
        "or unsupported image files."
    )


print("\nClass check passed - training will proceed.\n")


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


print(
    "\nClass names saved:"
)

print(
    CLASS_NAMES_PATH
)


# ============================================================
# PREFETCH
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomRotation(
        0.12
    ),

    layers.RandomZoom(
        height_factor=(-0.15, 0.15),

        width_factor=(-0.15, 0.15)
    ),

    layers.RandomTranslation(
        height_factor=0.10,

        width_factor=0.10
    ),

    layers.RandomContrast(
        0.15
    )

], name="data_augmentation")


# ============================================================
# LOAD EFFICIENTNETB0
# ============================================================

print("\nLoading EfficientNetB0...")


base_model = EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# ============================================================
# FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================
#
# NOTE ON PREPROCESSING:
# Keras's EfficientNetB0 (include_top=False) expects raw pixel
# values in [0, 255] - it has its own internal Rescaling +
# Normalization layers baked in. image_dataset_from_directory
# already returns float32 images in [0, 255] by default, so no
# extra rescale/preprocess_input step is needed here. This must
# match webcam.py's preprocessing mode exactly ("efficientnet"
# mode there = passthrough, matching this).

inputs = layers.Input(

    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


x = data_augmentation(
    inputs
)


x = base_model(
    x,

    training=False
)


x = layers.GlobalAveragePooling2D()(
    x
)


x = layers.BatchNormalization()(
    x
)


x = layers.Dropout(
    0.35
)(
    x
)


x = layers.Dense(
    256,

    activation="relu"
)(
    x
)


x = layers.BatchNormalization()(
    x
)


x = layers.Dropout(
    0.30
)(
    x
)


outputs = layers.Dense(

    num_classes,

    activation="softmax"

)(x)


model = models.Model(

    inputs,

    outputs
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# COMPILE - STAGE 1
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


callbacks = [

    checkpoint,

    early_stopping,

    reduce_lr

]


# ============================================================
# STAGE 1
# ============================================================

print("\n=============================================")
print("       STAGE 1 - TRANSFER LEARNING")
print("=============================================\n")


history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=INITIAL_EPOCHS,

    callbacks=callbacks
)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n=============================================")
print("       STAGE 2 - FINE TUNING")
print("=============================================\n")


base_model.trainable = True


# Freeze all but last 40 layers

for layer in base_model.layers[:-40]:

    layer.trainable = False


# Keep BatchNormalization frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# RECOMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# FINE TUNING
# ============================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n=============================================")
print("          LOADING BEST MODEL")
print("=============================================\n")


best_model = tf.keras.models.load_model(

    MODEL_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n=============================================")
print("        FINAL VALIDATION RESULT")
print("=============================================\n")


loss, accuracy = best_model.evaluate(

    val_ds,

    verbose=1
)


print("\n=============================================")

print(
    f"Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Validation Loss: "
    f"{loss:.4f}"
)

print("=============================================")


# ============================================================
# SAVE MODEL
# ============================================================
#
# NOTE: best_model was just loaded from MODEL_PATH above, so this
# re-save is a no-op in terms of content - kept for clarity/safety
# in case evaluate() or future code between load and here ever
# mutates the model.

best_model.save(

    MODEL_PATH
)


print("\nModel saved to:")

print(
    MODEL_PATH
)


print("\nClasses:")

print(
    class_names
)


print("\n=============================================")
print("           TRAINING COMPLETE")
print("=============================================\n")