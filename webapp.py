from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

upload_folder = "static/uploads"
app.config["UPLOAD_FOLDER"] = upload_folder

os.makedirs(upload_folder, exist_ok=True)

model = load_model("ResNet50V2_Potato_Model.keras")

classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image selected"})

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "File name is empty"})

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    img = prepare_image(file_path)

    result = model.predict(img)
    index = np.argmax(result)
    prediction = classes[index]
    confidence = float(np.max(result) * 100)

    return jsonify({
        "prediction": prediction,
        "confidence": str(round(confidence, 2)) + "%",
        "image_path": "/" + file_path
    })

if __name__ == "__main__":
    app.run(debug=True)
