# app/main.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS




# 1. Inițializăm aplicația Flask
app = Flask(__name__)
CORS(app)
# 2. Încărcăm modelul antrenat
model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "✅ AI Insights API is running!"})

# 3. Endpoint pentru predicție
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Citim variabilele din JSON
        age = data.get("Age")
        income = data.get("Income")

        # Validare simplă
        if age is None or income is None:
            return jsonify({"error": "Missing Age or Income"}), 400

        # Pregătim datele pentru model
        input_data = np.array([[age, income]])

        # Facem predicția
        prediction = model.predict(input_data)[0]
        result = "Will Purchase" if prediction == 1 else "Will Not Purchase"

        return jsonify({
            "Age": age,
            "Income": income,
            "Prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
