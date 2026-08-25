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
# VERIFY CLASSES
# ============================================================

expected_classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]


if class_names != expected_classes:

    print("\nWARNING!")

    print(
        "Expected:",
        expected_classes
    )

    print(
        "Found:",
        class_names
    )


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