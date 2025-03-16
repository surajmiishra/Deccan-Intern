# House Price Prediction API  

This repository contains a Flask-based REST API for predicting house prices based on various features. The model was trained using a **Gradient Boosting Regressor**, optimized for accuracy, and deployed as a web service.

---

## 1. Installation & Setup  

### **1.1 Clone the Repository**  
```bash
git clone https://github.com/surajmiishra/Deccan-Intern.git
cd house-price-api
```

### **1.2 Create a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **1.3 Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## 2. Model & Data  

- The trained model is stored as `house_price_model.pkl`.  
- Ensure that the dataset used for training aligns with the API's expected feature format.  

---

## 3. Running the API  

### **3.1 Start the Flask Server**  
```bash
python app.py
```
The API will start on **https://deccan-intern.onrender.com**.  

---

## 4. API Endpoints  

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`       | API status check |
| GET    | `/health` | Checks if the model is loaded properly |
| POST   | `/predict` | Predicts house prices based on input features |

---

## 5. Making Predictions  

### **Example Request (POST /predict)**  
```json
{
    "Area": 2500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Floors": 2,
    "YearBuilt": 2010,
    "Location_Rural": 0,
    "Location_Suburban": 1,
    "Location_Urban": 0,
    "Condition_Fair": 0,
    "Condition_Good": 1,
    "Condition_Poor": 0,
    "Garage_Yes": 1
}
```

### **Example Response**  
```json
{
    "predicted_price": 450000.0
}
```

---

## 6. Deployment (Optional)  

### **6.1 Docker Deployment**  
1. **Build the Docker Image**  
   ```bash
   docker build -t house-price-api .
   ```

2. **Run the Container**  
   ```bash
   docker run -p 10000:10000 house-price-api
   ```

### **6.2 Cloud Deployment**  
This API can be deployed on cloud platforms like:  
- **AWS EC2**  
- **Google Cloud Run**  
- **Heroku**  

---

## 7. License  

This project is licensed under the **MIT License**.

---

## 8. Contact  

For any issues or inquiries, feel free to reach out via **[surajm304@gmail.com]**.

