# app/scripts/prepare_cluj_regions.py
import os, json, re
import pandas as pd
import numpy as np

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RES_CSV = os.path.join(DATA_DIR, "residents_cluj.csv")
OUT_JSON = os.path.join(DATA_DIR, "regions.json")

# Occupation rates (method 2)
OCC_RATE = {"Urban": 0.693, "Rural": 0.571}

if not os.path.exists(RES_CSV):
    raise FileNotFoundError(f"Place residents CSV at: {RES_CSV}")

df = pd.read_csv(RES_CSV)
df.columns = df.columns.str.strip()  # curăță spațiile și tab-urile
# Helper to parse age or range like "20-24"
def expand_age_row(age_str, urban_count, rural_count):
    s = str(age_str).strip()
    nums = list(map(int, re.findall(r"\d+", s)))
    if len(nums) == 0:
        return {}
    if len(nums) == 1:
        return {nums[0]: (int(urban_count), int(rural_count))}
    a, b = nums[0], nums[-1]
    ages = list(range(a, b+1))
    n = len(ages)
    # split counts equally
    u_each = urban_count / n
    r_each = rural_count / n
    return {age: (u_each, r_each) for age in ages}

# build dict age -> (urban_sum, rural_sum)
age_map = {}
for _, row in df.iterrows():
    age_key = row['age']
    u = float(row['urban'])
    r = float(row['rural'])
    expanded = expand_age_row(age_key, u, r)
    for age, (uu, rr) in expanded.items():
        if age not in age_map:
            age_map[age] = [0.0, 0.0]
        age_map[age][0] += uu
        age_map[age][1] += rr

# compute totals and stats
ages = sorted(age_map.keys())
pop_by_age = [(age, int(round(vals[0])), int(round(vals[1]))) for age, vals in age_map.items()]
total_urban = sum(v[1] for v in pop_by_age)
total_rural = sum(v[2] for v in pop_by_age)
total_pop = total_urban + total_rural

# average age (weighted)
num = sum(age * (u + r) for age, u, r in pop_by_age)
avg_age = num / total_pop

# elderly ratio (65+)
elderly = sum((u + r) for age, u, r in pop_by_age if age >= 65)
elderly_ratio = elderly / total_pop

# urbanization
urbanization = "Urban" if total_urban >= total_rural else "Rural"

# population_density placeholder (we don't have area) -> store total_pop (you can replace by /area later)
population_density = total_pop

# average_income placeholder -> replace with real value if you have
default_income = 4000  # RON per month (placeholder) — update if you have real data

region = {
    "region": "Cluj",
    "average_income": default_income,
    "average_age": round(avg_age, 1),
    "population_density": int(population_density),
    "elderly_ratio": round(elderly_ratio, 4),
    "climate": "Temperate",
    "urbanization": urbanization
}

# write regions.json (list)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump([region], f, ensure_ascii=False, indent=2)

print(f"Created/updated regions.json for Cluj at: {OUT_JSON}")
print("Summary:")
print(f" total_pop = {total_pop:,}")
print(f" avg_age = {avg_age:.2f}")
print(f" elderly_ratio = {elderly_ratio:.4f}")
print(f" urbanization = {urbanization}")
