import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = r"D:\Hand gesture\Dataset\ASL-HG American Sign Language Hand Gesture Image D\ASL_HG_36000\Processed_images\asl_processed\train"

TEST_DIR = r"D:\Hand gesture\Dataset\ASL-HG American Sign Language Hand Gesture Image D\ASL_HG_36000\Processed_images\asl_processed\test"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Parameters
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


# ============================================================
# Load Training Dataset
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True
)


# ============================================================
# Load Validation Dataset
# ============================================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=True
)


# ============================================================
# Get Class Names BEFORE map()
# ============================================================

class_names = train_ds.class_names

print("\n========================================")
print("CLASS INFORMATION")
print("========================================")

print("Number of classes:", len(class_names))
print("Classes:", class_names)


# Check that we have 36 classes
if len(class_names) != 36:
    raise ValueError(
        f"Expected 36 classes, but found {len(class_names)} classes."
    )


# ============================================================
# Normalize Images
# ============================================================

normalization_layer = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# Improve Dataset Performance
# ============================================================

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# Load MobileNetV2
# ============================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Freeze MobileNetV2
base_model.trainable = False


# ============================================================
# Build Model
# ============================================================

model = models.Sequential([
    
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.2),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ============================================================
# Compile Model
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    
    loss="sparse_categorical_crossentropy",
    
    metrics=["accuracy"]
)


# ============================================================
# Display Model
# ============================================================

model.summary()


# ============================================================
# Model Checkpoint
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_model.keras"
)

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


# ============================================================
# Early Stopping
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# Train Model
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# Save Class Names
# ============================================================

class_names_path = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=4)


# ============================================================
# Load Best Model
# ============================================================

best_model = tf.keras.models.load_model(
    checkpoint_path
)


# ============================================================
# Load Test Dataset
# ============================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_class_names = test_ds.class_names

print("\n========================================")
print("TEST DATASET")
print("========================================")

print("Test classes:", test_class_names)


# ============================================================
# Normalize Test Dataset
# ============================================================

test_ds = test_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# Evaluate Model
# ============================================================

print("\n========================================")
print("TESTING MODEL")
print("========================================")

test_loss, test_accuracy = best_model.evaluate(
    test_ds
)

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print("Model saved to:")
print(checkpoint_path)

print("\nClass names saved to:")
print(class_names_path)