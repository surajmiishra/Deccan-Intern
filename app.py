from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)

# Define expected feature order
FEATURE_ORDER = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt",
                 "Location_Rural", "Location_Suburban", "Location_Urban",
                 "Condition_Fair", "Condition_Good", "Condition_Poor", "Garage_Yes"]

# Load the trained model if the file exists
MODEL_PATH = "house_price_model.pkl"
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logging.info("✅ Model loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Error loading model: {str(e)}")
        model = None
else:
    logging.error("❌ Model file not found.")
    model = None  # Prevents using an uninitialized model

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "House Price Prediction API is running!"})

@app.route("/health", methods=["GET"])
def health():
    """Check if API and model are working correctly after deployment."""
    if model:
        return jsonify({"status": "healthy", "message": "Model is loaded."})
    else:
        return jsonify({"status": "error", "message": "Model failed to load."}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        logging.info("📩 Received request at /predict")

        # Parse incoming JSON request
        data = request.get_json()
        if not data:
            logging.error("❌ No JSON data received.")
            return jsonify({"error": "Invalid input: No data provided"}), 400

        # Ensure model is loaded
        if model is None:
            logging.error("❌ Model is not loaded.")
            return jsonify({"error": "Model is unavailable"}), 500

        # Extract features in expected order and handle missing values
        features = []
        for key in FEATURE_ORDER:
            if key in data:
                features.append(data[key])
            else:
                logging.warning(f"⚠️ Missing feature '{key}', setting to default (0).")
                features.append(0)  # Set missing values to 0

        features_array = np.array(features).reshape(1, -1)
        logging.info(f"🔍 Features received: {features_array}")

        # Make prediction
        prediction = model.predict(features_array)
        
        logging.info(f"✅ Prediction made: {prediction[0]}")
        return jsonify({"predicted_price": float(prediction[0])})

    except Exception as e:
        logging.error(f"⚠️ Error in prediction: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
