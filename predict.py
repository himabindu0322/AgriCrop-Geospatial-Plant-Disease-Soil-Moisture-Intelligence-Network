import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==========================
# LOAD MODEL
# ==========================
model = load_model("models/disease_model.h5")

# ==========================
# LOAD CLASS NAMES
# ==========================
with open("classes.json", "r") as f:
    class_names = json.load(f)

# ==========================
# IMAGE PATH
# ==========================
img_path = "test.jpg"  # Replace with your image path

# Check if image exists
if not os.path.exists(img_path):
    print(f"Error: '{img_path}' not found.")
    exit()

# ==========================
# PREPROCESS IMAGE
# ==========================
img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# ==========================
# MAKE PREDICTION
# ==========================
prediction = model.predict(img_array)

predicted_index = np.argmax(prediction, axis=1)[0]
confidence = np.max(prediction) * 100

disease = class_names[str(predicted_index)]

# ==========================
# DISPLAY RESULT
# ==========================
print("\n===== Prediction Result =====")
print("Predicted Class Index:", predicted_index)
print("Disease:", disease)
print(f"Confidence: {confidence:.2f}%")
