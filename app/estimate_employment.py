import pandas as pd

# 1. Citim datele demografice
df = pd.read_csv('data/residents_cluj.csv')
df.columns = df.columns.str.strip()

# 2. Definim funcția de estimare în funcție de vârstă
def estimate_employment_rate(age):
    if 18 <= age <= 24:
        return 0.40
    elif 25 <= age <= 34:
        return 0.80
    elif 35 <= age <= 44:
        return 0.85
    elif 45 <= age <= 54:
        return 0.75
    elif 55 <= age <= 64:
        return 0.50
    elif 65 <= age <= 74:
        return 0.15
    else:
        return 0.05

# 3. Aplicăm funcția pe fiecare vârstă
df['employment_rate'] = df['age'].apply(estimate_employment_rate)

# 4. Calculăm numărul estimat de persoane ocupate
df['estimated_employed'] = (df['urban'] + df['rural']) * df['employment_rate']

# 5. Salvăm noul fișier
df.to_csv('data/cluj_employment_estimated.csv', index=False)

print("✅ Estimare completă salvată în data/cluj_employment_estimated.csv")
