# scripts/train_sales_predictor.py
import os, json, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from encoders import PRODUCT_TYPE_MAP, TARGET_AGE_MAP, CLIMATE_MAP, SEASON_MAP, URBAN_MAP

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Încarcă date reale Cluj
df = pd.read_csv('data/residents_cluj.csv')
df.columns = df.columns.str.strip()
#RES_CSV = os.path.join(DATA_DIR, "data/residents_cluj.csv")
#df = pd.read_csv(RES_CSV, sep=",")  # ajustează separatorul dacă e cazul

# Encodează rândurile
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

# Scale și antrenare
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(Xs, y)

# Salvare model + scaler
joblib.dump(model, os.path.join(MODEL_DIR, "sales_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "sales_scaler.pkl"))

print("✅ Model de vânzări antrenat și salvat în model/sales_model.pkl")
