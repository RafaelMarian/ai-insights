import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Citim datele generate anterior
df = pd.read_csv('data/cluj_employment_estimated.csv')
df.columns = df.columns.str.strip()

# 2. Grupăm datele pe grupe de vârstă
bins = [18, 24, 34, 44, 54, 64, 74, 85]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

grouped = df.groupby('age_group')['estimated_employed'].sum().reset_index()

# 3. Simulăm evoluția pentru anii următori
years = np.arange(2020, 2036)
data = []

for _, row in grouped.iterrows():
    base = row['estimated_employed']
    trend = np.random.uniform(-0.01, 0.02)  # variație anuală între -1% și +2%
    for year in years:
        value = base * ((1 + trend) ** (year - 2020))
        data.append({'year': year, 'age_group': row['age_group'], 'employed': value})

trend_df = pd.DataFrame(data)

# 4. Antrenăm un model simplu pentru fiecare grupă de vârstă
predictions = []

for group in labels:
    group_data = trend_df[trend_df['age_group'] == group]
    X = group_data[['year']]
    y = group_data['employed']

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(2025, 2036).reshape(-1, 1)
    y_pred = model.predict(future_years)

    for year, pred in zip(future_years.flatten(), y_pred):
        predictions.append({'year': year, 'age_group': group, 'predicted_employed': pred})

pred_df = pd.DataFrame(predictions)

# 5. Salvăm rezultatele
pred_df.to_csv('data/cluj_employment_predictions.csv', index=False)
print("✅ Predicțiile au fost salvate în data/cluj_employment_predictions.csv")

# 6. Vizualizare rapidă
for group in labels:
    subset = pred_df[pred_df['age_group'] == group]
    plt.plot(subset['year'], subset['predicted_employed'], label=group)

plt.title("Evoluția estimată a populației ocupate din Cluj (2025–2035)")
plt.xlabel("An")
plt.ylabel("Număr estimat persoane ocupate")
plt.legend()
plt.grid(True)
plt.show()
