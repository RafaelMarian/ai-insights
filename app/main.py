# app/main.py
import os, json
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from encoders import PRODUCT_TYPE_MAP, TARGET_AGE_MAP, CLIMATE_MAP, SEASON_MAP, URBAN_MAP, FEATURE_NAMES

APP_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(APP_DIR, "model")
DATA_DIR = os.path.join(APP_DIR, "data")

app = Flask(__name__)
CORS(app)

# load regions
with open(os.path.join(DATA_DIR, "regions.json"), "r", encoding="utf-8") as f:
    REGIONS = json.load(f)

# load model + scaler
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run train_model.py to create model.pkl")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/")
def home():
    return jsonify({"message": "✅ AI Insights API is running!"})

@app.route("/regions", methods=["GET"])
def regions():
    # return only region names to populate dropdown
    return jsonify([r["region"] for r in REGIONS])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # expected fields
        product_type = data.get("product_type")
        target_age_group = data.get("target_age_group")
        average_price = float(data.get("average_price", 20))
        region_name = data.get("region")
        season = data.get("season")

        # basic validation
        if not all([product_type, target_age_group, region_name, season]):
            return jsonify({"error": "Missing fields. Required: product_type, target_age_group, region, season"}), 400

        # find region
        region = next((r for r in REGIONS if r["region"] == region_name), None)
        if region is None:
            return jsonify({"error": f"Region '{region_name}' not found"}), 400

        # build feature vector in same order as FEATURE_NAMES
        feat = [
            region["average_income"],
            region["average_age"],
            region["population_density"],
            region["elderly_ratio"],
            CLIMATE_MAP.get(region["climate"], 1),
            SEASON_MAP.get(season, 0),
            PRODUCT_TYPE_MAP.get(product_type, 0),
            TARGET_AGE_MAP.get(target_age_group, 1),
            average_price,
            URBAN_MAP.get(region.get("urbanization", "Rural"), 0)
        ]
        X = np.array(feat).reshape(1, -1)
        Xs = scaler.transform(X)
        proba = model.predict_proba(Xs)[0][1]  # probability of class 1
        pred = int(proba > 0.5)

        # feature importance mapping (model.feature_importances_)
        importances = {}
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            for n, v in zip(FEATURE_NAMES, fi):
                importances[n] = float(v)

        result = {
            "Product": data.get("product_name", "Unnamed"),
            "Predicted_Success": f"{proba*100:.1f}%",
            "PredictionLabel": "Likely" if pred==1 else "Unlikely",
            "Interpretation": "Model-based estimate",
            "Feature_Importance": importances
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
