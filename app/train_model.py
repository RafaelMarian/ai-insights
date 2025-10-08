# app/train_model.py
import os, json, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib
from encoders import PRODUCT_TYPE_MAP, TARGET_AGE_MAP, CLIMATE_MAP, SEASON_MAP, URBAN_MAP, FEATURE_NAMES

try:
    from xgboost import XGBClassifier
except ImportError:
    print("⚠️ xgboost not installed. Install with `pip install xgboost` to enable XGBClassifier.")
    XGBClassifier = None

# =====================
# Paths
# =====================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================
# Load regions & create synthetic dataset
# =====================
with open(os.path.join(DATA_DIR, "regions.json"), "r", encoding="utf-8") as f:
    regions = json.load(f)

rows = []
for region in regions:
    for product_type in PRODUCT_TYPE_MAP.keys():
        for season in SEASON_MAP.keys():
            for target in TARGET_AGE_MAP.keys():
                for _ in range(12):
                    avg_income = region["average_income"] * random.uniform(0.85, 1.15)
                    avg_age = region["average_age"] * random.uniform(0.9, 1.1)
                    pop_density = region["population_density"] * random.uniform(0.8, 1.2)
                    elderly_ratio = region["elderly_ratio"] * random.uniform(0.8, 1.2)
                    avg_price = random.uniform(5, 120)
                    climate = region["climate"]
                    urban = region.get("urbanization", "Rural")
                    rows.append({
                        "region": region["region"],
                        "average_income": avg_income,
                        "average_age": avg_age,
                        "population_density": pop_density,
                        "elderly_ratio": elderly_ratio,
                        "climate": climate,
                        "season": season,
                        "product_type": product_type,
                        "target_age_group": target,
                        "average_price": avg_price,
                        "urbanization": urban
                    })

df = pd.DataFrame(rows)

# =====================
# Synthetic target
# =====================
def compute_prob(row):
    p = 0.0
    if row["product_type"] == "Cold Medicine":
        p += 0.3 * row["elderly_ratio"]
    if row["climate"] == "Cold":
        p += 0.25
    if row["season"] == "Winter":
        p += 0.25
    income_score = (row["average_income"] - 1000) / 6000
    p += 0.2 * max(0, min(1, income_score))
    if row["average_price"] > 100:
        p -= 0.2
    elif row["average_price"] < 30:
        p += 0.05
    if row["urbanization"] == "Urban":
        p += 0.1
    elif row["urbanization"] == "Semi-urban":
        p += 0.05
    p += random.uniform(-0.05, 0.05)
    return max(0, min(1, p))

df["prob"] = df.apply(compute_prob, axis=1)
df["bought"] = (df["prob"] > 0.5).astype(int)

# =====================
# Encode features
# =====================
def encode_row(r):
    return [
        r["average_income"],
        r["average_age"],
        r["population_density"],
        r["elderly_ratio"],
        CLIMATE_MAP[r["climate"]],
        SEASON_MAP[r["season"]],
        PRODUCT_TYPE_MAP[r["product_type"]],
        TARGET_AGE_MAP[r["target_age_group"]],
        r["average_price"],
        URBAN_MAP[r["urbanization"]]
    ]

X = np.vstack(df.apply(encode_row, axis=1).values)
y = df["bought"].values

# =====================
# Scale features
# =====================
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# =====================
# Models to evaluate
# =====================
models = [
    RandomForestClassifier(n_estimators=150, random_state=42),
    GradientBoostingClassifier(n_estimators=150, random_state=42),
    LogisticRegression(max_iter=500)
]

if XGBClassifier:
    models.append(XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric='logloss'))

# =====================
# Evaluate & pick best model
# =====================
best_model = None
best_score = -1.0

print("🔹 Evaluating models...")
for m in models:
    scores = cross_val_score(m, Xs, y, cv=5)
    mean_score = np.mean(scores)
    print(f"{m.__class__.__name__}: Mean Accuracy = {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_model = m

# =====================
# Train best model on full dataset
# =====================
print(f"✅ Best model: {best_model.__class__.__name__} with accuracy {best_score:.4f}")
best_model.fit(Xs, y)

# =====================
# Save model & scaler
# =====================
joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"Trained model saved to {MODEL_DIR}/model.pkl")
