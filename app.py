from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("house_price_model.pkl")
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {str(e)}")
    model = None  # Prevents using an uninitialized model

# Define expected feature order
FEATURE_ORDER = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt",
                 "Location_Rural", "Location_Suburban", "Location_Urban",
                 "Condition_Fair", "Condition_Good", "Condition_Poor", "Garage_Yes"]

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
        
        # Parse incoming request
        data = request.get_json()
        if not data:
            logging.error("❌ No JSON data received.")
            return jsonify({"error": "Invalid input: No data provided"}), 400

        # Extract features in expected order
        features = np.array([data.get(key, 0) for key in FEATURE_ORDER]).reshape(1, -1)

        logging.info(f"🔍 Features received: {features}")

        # Ensure model is loaded
        if model is None:
            logging.error("❌ Model is not loaded.")
            return jsonify({"error": "Model is unavailable"}), 500

        # Make prediction
        prediction = model.predict(features)
        
        logging.info(f"✅ Prediction made: {prediction[0]}")
        return jsonify({"predicted_price": float(prediction[0])})
    
    except Exception as e:
        logging.error(f"⚠️ Error in prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
