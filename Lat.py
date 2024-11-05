from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import ViTFeatureExtractor, ViTModel
import torch
import pickle
import numpy as np
from PIL import Image
import io

app = Flask(name)
CORS(app)  # Enable CORS for all routes and origins

# Load the ViT feature extractor and model for feature extraction
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.eval()

# Function to extract features from an image using ViT
def extract_features(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = vit_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return features

# Function to load models and make predictions
def predict_creatinine_concentration(image):
    features = extract_features(image).reshape(1, -1)
    model_files = {
        "Histogram_Boosted_Trees": "Histogram_Boosted_Trees.pkl",
        "Random_Forest": "Random_Forest.pkl",
        "Decision_Tree": "Decision_Tree.pkl",
        "Linear_Regression": "Linear_Regression.pkl"
    }
    predictions = {}
    for model_name, model_file in model_files.items():
        with open(model_file, "rb") as file:
            model = pickle.load(file)
        concentration = model.predict(features)[0]
        predictions[model_name] = float(concentration)
    return predictions

# Flask endpoint to receive image and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    predictions = predict_creatinine_concentration(image)
    return jsonify(predictions)

# Run the Flask app
if name == 'main':
    app.run(debug=True, host="0.0.0.0")  # host set to 0.0.0.0 to accept all regions