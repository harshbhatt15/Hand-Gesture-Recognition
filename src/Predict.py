import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing import image 

# Load Trained Model
model = tf.keras.models.load_model("models/best_model.keras")

# Class Names
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

# Image Path
img_path = r"D:\Hand gesture\dataset\test\test\0\902.jpg"

import matplotlib.pyplot as plt

img = image.load_img(img_path, target_size=(224, 224))

plt.imshow(img)
plt.title("Test Image")
plt.axis("off")
plt.show()

# Load Image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

print("Prediction probabilities:")
print(prediction)

predicted_index = np.argmax(prediction)

print("Predicted Index:", predicted_index)
print("Predicted Class:", class_names[predicted_index])
print("Confidence:", np.max(prediction) * 100)