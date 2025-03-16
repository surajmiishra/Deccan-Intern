from flask import Flask, request, jsonify
import joblib
import numpy as np

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
        # Get JSON request data
        data = request.get_json()

        # Validate input
        required_fields = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt",
                           "Location_Rural", "Location_Suburban", "Location_Urban",
                           "Condition_Fair", "Condition_Good", "Condition_Poor", "Garage_Yes"]

        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Convert input values to float (ensures type safety)
        features = np.array([float(data[field]) for field in required_fields]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Return response
        return jsonify({"predicted_price": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
