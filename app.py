from flask import Flask
from flask import render_template
from flask import request

import os
import numpy as np
import tensorflow as tf
import json

from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model(
    "models/disease_model.h5"
)

with open("classes.json", "r") as f:
    class_names = json.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    prediction = model.predict(
        img_array
    )

    predicted_index = np.argmax(
        prediction
    )

    disease = class_names[
        str(predicted_index)
    ]

    return render_template(
        "result.html",
        disease=disease,
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)
