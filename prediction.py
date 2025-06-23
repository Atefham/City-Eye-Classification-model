import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Arabic class names
CLASS_NAMES = [
    "Traffic Jam",     # Traffic Jam
    "Garbage Accumulation",     # Garbage Accumulation
    "Fires",            # Fires
    "Accidents",       # Accidents
    "Broken Roads",       # Broken Roads
    "Floods",          # Floods
    "Fallen Tree"         # Tree Cutting
]

def preprocess(img_path, target_size=(224, 224)):
    """
    Preprocess image for prediction: load, resize, normalize.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)

def predict_and_show(model, img_array, top_k=3):
    """
    Predict using the model and show top-k class probabilities (Arabic).
    """
    preds = model.predict(img_array)[0]  # Shape: (7,)

    # Get top-k predictions
    top_indices = preds.argsort()[-top_k:][::-1]
    
    print("Top probablities")
    for i in top_indices:
        print(f"{CLASS_NAMES[i]}: {preds[i]:.4f}")



# ==== Load your trained model ====
model = tf.keras.models.load_model('city_eye_model.h5')

# ==== Predict on a new image ====
img_array = preprocess('images.jpeg')  # Path to your image
predict_and_show(model, img_array)
