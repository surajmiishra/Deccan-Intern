from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()
        
        # Extract features in correct order
        features = np.array([data["Area"], data["Bedrooms"], data["Bathrooms"],
                             data["Floors"], data["YearBuilt"], data["Location_Rural"],
                             data["Location_Suburban"], data["Location_Urban"],
                             data["Condition_Fair"], data["Condition_Good"],
                             data["Condition_Poor"], data["Garage_Yes"]]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return response
        return jsonify({"predicted_price": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

