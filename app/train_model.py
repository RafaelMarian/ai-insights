# app/train_model.py
import json, os, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from encoders import PRODUCT_TYPE_MAP, TARGET_AGE_MAP, CLIMATE_MAP, SEASON_MAP, URBAN_MAP, FEATURE_NAMES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# load regions
with open(os.path.join(DATA_DIR, "regions.json"), "r", encoding="utf-8") as f:
    regions = json.load(f)

# build synthetic samples
rows = []
for region in regions:
    for product_type in PRODUCT_TYPE_MAP.keys():
        for season in SEASON_MAP.keys():
            for target in TARGET_AGE_MAP.keys():
                # create multiple samples with small noise
                for _ in range(12):
                    avg_income = region["average_income"] * random.uniform(0.85, 1.15)
                    avg_age = region["average_age"] * random.uniform(0.9, 1.1)
                    pop_density = region["population_density"] * random.uniform(0.8, 1.2)
                    elderly_ratio = region["elderly_ratio"] * random.uniform(0.8, 1.2)
                    avg_price = random.uniform(5, 120)  # RON
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

# create synthetic label using a hidden rule and noise
def compute_prob(row):
    p = 0.0
    # elderly favors cold medicine
    if row["product_type"] == "Cold Medicine":
        p += 0.25 * row["elderly_ratio"]
    # climate cold increases chances for cold medicine
    if row["climate"] == "Cold":
        p += 0.20
    # season winter increases
    if row["season"] == "Winter":
        p += 0.15
    # higher income slightly increases purchase
    income_score = (row["average_income"] - 1000) / 6000
    p += 0.15 * max(0, min(1, income_score))
    # price penalty for too high price
    if row["average_price"] > 80:
        p -= 0.20
    # urbanization effect
    if row["urbanization"] == "Urban":
        p += 0.05
    # normalize to [0,1]
    p = max(0, min(1, p + random.uniform(-0.05, 0.05)))
    return p

df["prob"] = df.apply(compute_prob, axis=1)
df["bought"] = (df["prob"] > 0.5).astype(int)

# prepare feature matrix
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

# scale and train
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(Xs, y)

# save model and scaler
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("Trained model saved to app/model/model.pkl")
