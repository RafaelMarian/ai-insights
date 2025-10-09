# app/predict_sales.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from encoders import PRODUCT_TYPE_MAP, TARGET_AGE_MAP, CLIMATE_MAP, SEASON_MAP, URBAN_MAP

app = Flask(__name__)

# Încarcă modelul + scaler
model = joblib.load("model/sales_model.pkl")
scaler = joblib.load("model/sales_scaler.pkl")

def encode_input(data):
    return [
        data["average_income"],
        data["average_age"],
        data["population_density"],
        data["elderly_ratio"],
        CLIMATE_MAP[data["climate"]],
        SEASON_MAP[data["season"]],
        PRODUCT_TYPE_MAP[data["product_type"]],
        TARGET_AGE_MAP[data["target_age_group"]],
        data["average_price"],
        URBAN_MAP[data["urbanization"]]
    ]

@app.route("/predict_sales", methods=["POST"])
def predict_sales():
    data = request.json
    X = np.array(encode_input(data)).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
