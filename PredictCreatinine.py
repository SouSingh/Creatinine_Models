from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS  # Import CORS
import pickle
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Homepage route to check if the app is running
@app.route('/')
def home():
    return render_template_string("<h1>Welcome to the Creatinine Concentration Prediction API</h1><p>Status: Running</p>")

# Function to extract basic features from an image
def extract_features(image):
    # Convert image to RGB if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image to a fixed size (e.g., 32x32) to create a simple feature vector
    resized_image = image.resize((32, 32))
    
    # Convert the resized image to a numpy array and flatten it to create a feature vector
    feature_vector = np.array(resized_image).flatten()
    
    return feature_vector

# Function to load models and make predictions
def predict_creatinine_concentration(image):
    # Extract features from the image
    features = extract_features(image).reshape(1, -1)
    
    # Define model paths and names for loading
    model_files = {
        "Histogram_Boosted_Trees": "Histogram_Boosted_Trees.pkl",
        "Random_Forest": "Random_Forest.pkl",
        "Decision_Tree": "Decision_Tree.pkl",
        "Linear_Regression": "Linear_Regression.pkl"
    }
    
    # Dictionary to store predictions
    predictions = {}
    
    # Loop through each model, load it, and make a prediction
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

    # Get predictions
    predictions = predict_creatinine_concentration(image)
    return jsonify(predictions)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)  # host set to 0.0.0.0 to accept all regions
