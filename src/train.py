# Importing Dependencies #
 
import tensorflow as tf 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt
import os

# Dataset Paths #

train_path = r"D:\Hand gesture\dataset\train\train"


# Parameters #

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load Dataset #

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split = 0.2,
    subset = "training",
    seed = 42,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split = 0.2,
    subset="training",
    seed = 42,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("classes: ",class_names)
print("Number of classes: ",num_classes)


# Normalize Images #

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x,y : (normalization_layer(x), y)) # (map) Apply this function to every element of the dataset
valid_ds = valid_ds.map(lambda x,y : (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE # loading and training happen at the same time.

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
"""
With Prefetch

GPU trains Batch 1

 ↓

CPU loads Batch 2 simultaneously

↓

GPU immediately starts Batch 2 """


# MobileNetV2
base_model = MobileNetV2(
    weights = "imagenet",
    include_top = False,
    input_shape = (224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes,activation = "softmax")(x)

model = Model(inputs=base_model.input,outputs=outputs)

# Compile
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

model.summary()


# Callbacks
os.makedirs("models",exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/best_model.keras",
    monitor = "val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# Train
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs= EPOCHS,
    callbacks=[checkpoint, early_stop]
)


# Save Final Model
model.save("models/final_model.keras")

print("\nModel saved successfully!")


# Accuracy Graph
plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig("models/training_history.png")
plt.show()


# Loss Graph
plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.savefig("models/loss_history.png")
plt.show()
print(class_names)