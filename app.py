import os
import pickle
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

app = Flask(__name__)

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)

with open(CLASS_PATH, "rb") as f:
    class_names = pickle.load(f)


# =========================
# PREDICT FUNCTION
# =========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    pred = model.predict(img, verbose=0)

    idx = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    return class_names[idx], confidence


# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    filename = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template("index.html")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html")

        filename = os.path.basename(file.filename)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(filepath)

        prediction, confidence = predict_image(filepath)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename
    )


if __name__ == "__main__":
    app.run(debug=True)