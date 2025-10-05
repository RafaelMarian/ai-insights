# model/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Generăm date artificiale
def generate_data(num_samples=1000):
    np.random.seed(42)
    ages = np.random.randint(18, 70, num_samples)
    incomes = np.random.randint(20000, 120000, num_samples)

    # Regula simplă: clienții tineri cu venit mare cumpără mai probabil
    purchased = ((ages < 35) & (incomes > 50000)) | (incomes > 90000)
    purchased = purchased.astype(int)

    data = pd.DataFrame({
        'Age': ages,
        'Income': incomes,
        'Purchased': purchased
    })

    return data

# 2. Antrenăm modelul
def train_model():
    data = generate_data()
    X = data[['Age', 'Income']]
    y = data['Purchased']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Evaluare
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")

    # 4. Salvăm modelul într-un fișier .pkl
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    print("✅ Model saved as model/model.pkl")

if __name__ == "__main__":
    train_model()
