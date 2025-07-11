import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Искусственные данные
X = np.random.rand(200, 4)  # 4 признака
y = np.random.choice([0, 1], size=200)  # целевая переменная (риск: 0 или 1)

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Модель
model = LogisticRegression()
model.fit(X_scaled, y)

# Сохраняем
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Модель и scaler сохранены")