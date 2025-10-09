# app/train_model_real.py
import os, json, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib

# encoders maps from your project
from encoders import PRODUCT_TYPE_MAP, TARGET_AGE_MAP, CLIMATE_MAP, SEASON_MAP, URBAN_MAP, FEATURE_NAMES

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
    print("XGBoost not available (optional).")

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
MODEL_DIR = os.path.join(APP_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

RES_CSV = os.path.join(DATA_DIR, "residents_cluj.csv")
REGIONS_JSON = os.path.join(DATA_DIR, "regions.json")

# occupation rates (method 2)
OCC_RATE = {"Urban": 0.693, "Rural": 0.571}

if not os.path.exists(RES_CSV):
    raise FileNotFoundError("Put residents CSV at: " + RES_CSV)
if not os.path.exists(REGIONS_JSON):
    raise FileNotFoundError("Run prepare_cluj_regions.py first to create regions.json")

with open(REGIONS_JSON, "r", encoding="utf-8") as f:
    regions = json.load(f)
REGION = regions[0]  # Cluj

# read residents and expand ranges (same logic as prepare script)
def expand_age_row(age_str, urban_count, rural_count):
    import re
    s = str(age_str).strip()
    nums = list(map(int, re.findall(r"\d+", s)))
    if len(nums) == 0:
        return {}
    if len(nums) == 1:
        return {nums[0]: (int(urban_count), int(rural_count))}
    a, b = nums[0], nums[-1]
    ages = list(range(a, b+1))
    n = len(ages)
    u_each = urban_count / n
    r_each = rural_count / n
    return {age: (u_each, r_each) for age in ages}

df = pd.read_csv(RES_CSV)
df.columns = df.columns.str.strip()
age_map = {}
for _, row in df.iterrows():
    expanded = expand_age_row(row['age'], float(row['urban']), float(row['rural']))
    for age, (u, r) in expanded.items():
        if age not in age_map:
            age_map[age] = [0.0, 0.0]
        age_map[age][0] += u
        age_map[age][1] += r

# Build list of rows per (age, mediu)
rows = []
for age, (u, r) in age_map.items():
    # urban row
    if u > 0:
        rows.append({
            "age": age,
            "mediu": "Urban",
            "population": u,
            "employed": round(u * OCC_RATE["Urban"], 0),
            "employment_rate": (u * OCC_RATE["Urban"]) / u if u>0 else 0.0
        })
    # rural row
    if r > 0:
        rows.append({
            "age": age,
            "mediu": "Rural",
            "population": r,
            "employed": round(r * OCC_RATE["Rural"], 0),
            "employment_rate": (r * OCC_RATE["Rural"]) / r if r>0 else 0.0
        })

# product types / seasons / target age groups
product_types = list(PRODUCT_TYPE_MAP.keys())
target_age_groups = list(TARGET_AGE_MAP.keys())
seasons = list(SEASON_MAP.keys())

# create dataset by combinining demographic rows with product/season/price
data_rows = []
for d in rows:
    for ptype in product_types:
        for season in seasons:
            for target in target_age_groups:
                # sample average price (or you can use a small list)
                avg_price = random.uniform(10, 120)
                # build feature vector using same order as earlier main.py expects:
                # [average_income, average_age, population_density, elderly_ratio, climate_code, season_code,
                #  product_type_code, target_age_code, average_price, urban_flag]
                feat = {
                    "average_income": REGION.get("average_income", 4000),
                    "average_age": REGION.get("average_age", 40),
                    "population_density": REGION.get("population_density", 1),
                    "elderly_ratio": REGION.get("elderly_ratio", 0.12),
                    "climate": REGION.get("climate", "Temperate"),
                    "season": season,
                    "product_type": ptype,
                    "target_age_group": target,
                    "average_price": avg_price,
                    "urbanization": d["mediu"]
                }
                # Extend with per-age/per-median features (we'll incorporate employment via a rule)
                feat.update({
                    "age": d["age"],
                    "population_by_age": d["population"],
                    "employed_by_age": d["employed"],
                    "employment_rate_by_age": d["employment_rate"]
                })
                data_rows.append(feat)

dfX = pd.DataFrame(data_rows)

# create synthetic label using a realistic rule that uses employment_rate and elderly_ratio
def compute_prob_row(r):
    p = 0.0
    # product effect
    if r["product_type"] == "Cold Medicine":
        p += 0.35 * (r["employment_rate_by_age"])  # more employed -> more buying power
        # elderly effect
        if r["age"] >= 65:
            p += 0.25
    # climate/season
    if str(r["climate"]).lower().find("cold") >= 0 or r["season"] == "Winter":
        p += 0.25
    # price sensitivity
    if r["average_price"] > 100:
        p -= 0.25
    elif r["average_price"] < 30:
        p += 0.05
    # urban effect
    if r["urbanization"] == "Urban":
        p += 0.1
    # small noise
    p += random.uniform(-0.05, 0.05)
    return max(0, min(1, p))

dfX["prob"] = dfX.apply(compute_prob_row, axis=1)
dfX["bought"] = (dfX["prob"] > 0.5).astype(int)

# now encode features into matrix X using same encoding as main.py expects
def encode_row_for_model(r):
    # map climate -> code
    climate_code = CLIMATE_MAP.get(r["climate"], list(CLIMATE_MAP.values())[0])
    season_code = SEASON_MAP.get(r["season"], 0)
    product_code = PRODUCT_TYPE_MAP.get(r["product_type"], 0)
    target_code = TARGET_AGE_MAP.get(r["target_age_group"], 0)
    urban_flag = URBAN_MAP.get(r["urbanization"], 0)
    return [
        float(r["average_income"]),
        float(r["average_age"]),
        float(r["population_density"]),
        float(r["elderly_ratio"]),
        climate_code,
        season_code,
        product_code,
        target_code,
        float(r["average_price"]),
        urban_flag
    ]

X = np.vstack(dfX.apply(encode_row_for_model, axis=1).values)
y = dfX["bought"].values

# scale
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# models to try
models = [
    RandomForestClassifier(n_estimators=150, random_state=42),
    GradientBoostingClassifier(n_estimators=150, random_state=42),
    LogisticRegression(max_iter=500)
]
if XGBClassifier:
    models.append(XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric='logloss'))

best_model = None
best_score = -1.0

print("Evaluating models on real-ish dataset (Cluj)...")
for m in models:
    try:
        scores = cross_val_score(m, Xs, y, cv=5)
        mean_score = np.mean(scores)
        print(f"{m.__class__.__name__}: Mean Accuracy = {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_model = m
    except Exception as e:
        print("Error evaluating", m, e)

print(f"Best model: {best_model.__class__.__name__} (score {best_score:.4f})")
best_model.fit(Xs, y)

joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("Saved model and scaler to:", MODEL_DIR)
