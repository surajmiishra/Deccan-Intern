from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "House Price Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        logging.info("Received request at /predict")
        data = request.get_json()
        if not data:
            logging.error("No JSON data received")
            return jsonify({"error": "Invalid input: No data provided"}), 400
        
        # Define the expected feature order
        feature_order = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt",
                         "Location_Rural", "Location_Suburban", "Location_Urban",
                         "Condition_Fair", "Condition_Good", "Condition_Poor", "Garage_Yes"]
        
        features = np.array([data.get(key, 0) for key in feature_order]).reshape(1, -1)

        logging.info(f"Features received: {features}")

        # Make prediction
        prediction = model.predict(features)
        
        logging.info(f"Prediction made: {prediction[0]}")
        return jsonify({"predicted_price": float(prediction[0])})
    
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
