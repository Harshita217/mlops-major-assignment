# app.py
from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = "models/savedmodel.pth"

def load_model():
    return joblib.load(MODEL_PATH)

# We assume images are uploaded as 64x64 grayscale (Olivetti format).
# For a general approach, we will convert to grayscale and resize to 64x64.
def preprocess_image(file_stream):
    img = Image.open(file_stream).convert("L")  # grayscale
    img = img.resize((64, 64))
    arr = np.array(img).reshape(1, -1) / 255.0
    return arr

model = None

@app.route("/", methods=["GET", "POST"])
def index():
    global model
    if model is None:
        model = load_model()

    if request.method == "POST":
        f = request.files.get("image")
        if not f:
            return "No file uploaded", 400
        arr = preprocess_image(f.stream)
        pred = model.predict(arr)[0]
        return render_template("index.html", prediction=int(pred))
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
