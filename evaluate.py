import tensorflow as tf

model = tf.keras.models.load_model("models/best_model.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    r"D:\Hand gesture\dataset\test\test",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

normalization = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization(x), y))

loss, accuracy = model.evaluate(test_ds)

print(f"Test Accuracy: {accuracy:.4f}")