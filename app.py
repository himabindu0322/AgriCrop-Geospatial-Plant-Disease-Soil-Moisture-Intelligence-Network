from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ==========================
# CONFIGURATION
# ==========================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model(
    "models/disease_model.h5"
)

# ==========================
# LOAD CLASS NAMES
# ==========================
with open("classes.json", "r") as f:
    class_names = json.load(f)

# ==========================
# HOME PAGE
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

# ==========================
# PREDICTION
# ==========================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image."

    # Secure filename
    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Save uploaded image
    file.save(filepath)

    # Load image
    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    # Convert to array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(
        img_array,
        axis=0
    )
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)

    predicted_index = np.argmax(
        prediction,
        axis=1
    )[0]

    disease = class_names[
        str(predicted_index)
    ]

    return render_template(
        "result.html",
        disease=disease,
        image=filepath
    )

# ==========================
# RUN APP
# ==========================
if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
