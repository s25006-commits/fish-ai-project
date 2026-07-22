
from flask import Flask, request, jsonify, render_template
import os
import uuid
from PIL import Image, ImageOps
from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/map')
def map_page():
    return render_template('map.html')



@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )
    file.save(filepath)

    label, conf, info = predict_image(filepath)

    return jsonify({
        "name": info["名前"],
     "confidence": f"{conf*100:.2f}",
    "feature": info["外見"],
    "habit": info["習性"],
    "種類識別": info["type"]
})
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))