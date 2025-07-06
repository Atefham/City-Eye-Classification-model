from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image  # 👈 Add this line


app = Flask(__name__)

# Load your trained model once at startup
model = load_model('city_eye_model.h5')  # Make sure model.h5 is in the same folder as this script

# Define class labels in the correct order used during training
class_names = [
    "Traffic Jam",
    "Garbage Accumulation",
    "Fires",
    "Accidents",
    "Broken Roads",
    "Floods",
    "Fallen Tree"
]

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Process image
        image = Image.open(file).resize((224, 224))
        image = np.array(image) / 255.0

        if image.shape != (224, 224, 3):
            return jsonify({"error": f"Invalid image shape: {image.shape}"}), 400

        image = np.expand_dims(image, axis=0)

        # Predict
        preds = model.predict(image)[0]
        top3_indices = preds.argsort()[-3:][::-1]
        top3 = {class_names[i]: float(preds[i]) for i in top3_indices}

        return jsonify(top3)
    except Exception as e:
        print(f"🔥 Error: {e}")  # This prints the error in terminal
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
