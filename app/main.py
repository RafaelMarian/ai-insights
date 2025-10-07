# app/main.py
import os, json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
from encoders import (
    PRODUCT_TYPE_MAP,
    TARGET_AGE_MAP,
    CLIMATE_MAP,
    SEASON_MAP,
    URBAN_MAP,
    FEATURE_NAMES,
)

# =========================
# App initialization
# =========================
APP_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(APP_DIR, "model")
DATA_DIR = os.path.join(APP_DIR, "data")
FRONTEND_BUILD_DIR = os.path.join(APP_DIR, "build")  # <-- React build folder

app = Flask(__name__, static_folder=FRONTEND_BUILD_DIR, static_url_path="/")
CORS(app)

# =========================
# Load Data & Model
# =========================
with open(os.path.join(DATA_DIR, "regions.json"), "r", encoding="utf-8") as f:
    REGIONS = json.load(f)

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ Model not found. Run train_model.py to create model.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# API ROUTES
# =========================
@app.route("/api/regions", methods=["GET"])
def regions():
    return jsonify([r["region"] for r in REGIONS])

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        product_type = data.get("product_type")
        target_age_group = data.get("target_age_group")
        average_price = float(data.get("average_price", 20))
        region_name = data.get("region")
        season = data.get("season")

        if not all([product_type, target_age_group, region_name, season]):
            return jsonify({"error": "Missing fields: product_type, target_age_group, region, season"}), 400

        region = next((r for r in REGIONS if r["region"] == region_name), None)
        if region is None:
            return jsonify({"error": f"Region '{region_name}' not found"}), 400

        # build feature vector
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
            URBAN_MAP.get(region.get("urbanization", "Rural"), 0),
        ]

        X = np.array(feat).reshape(1, -1)
        Xs = scaler.transform(X)
        proba = model.predict_proba(Xs)[0][1]
        pred = int(proba > 0.5)

        importances = {}
        if hasattr(model, "feature_importances_"):
            for n, v in zip(FEATURE_NAMES, model.feature_importances_):
                importances[n] = float(v)

        result = {
            "Product": data.get("product_name", "Unnamed"),
            "Predicted_Success": f"{proba*100:.1f}%",
            "PredictionLabel": "Likely" if pred == 1 else "Unlikely",
            "Interpretation": "Model-based estimate",
            "Feature_Importance": importances,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# FRONTEND SERVING
# =========================
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    """Serve React frontend build files"""
    if path != "" and os.path.exists(os.path.join(FRONTEND_BUILD_DIR, path)):
        return send_from_directory(FRONTEND_BUILD_DIR, path)
    else:
        return send_from_directory(FRONTEND_BUILD_DIR, "index.html")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask server on port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=True)
