import os
import pickle
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("model/model.keras")

with open("model/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    filename = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            pred = model.predict(img, verbose=0)

            prediction = class_names[np.argmax(pred)]
            confidence = round(np.max(pred) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename
    )


if __name__ == "__main__":
    app.run(debug=True)