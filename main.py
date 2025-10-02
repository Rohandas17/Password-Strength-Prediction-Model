import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# --- 1. Data Loading and Initial Cleaning ---
data = pd.read_csv('passwords.csv', on_bad_lines='skip')
columns_to_drop = ["rank", "value", "time_unit", "offline_crack_sec", "rank_alt", "font_size"]
data = data.drop(columns=columns_to_drop, axis=1)
data.dropna(inplace=True)
data['password'] = data['password'].astype(str)

# --- 2. Feature Engineering ---
def extract_features(password):
    features = {}
    features['length'] = len(password)
    features['n_upper'] = len(re.findall(r'[A-Z]', password))
    features['n_lower'] = len(re.findall(r'[a-z]', password))
    features['n_digits'] = len(re.findall(r'[0-9]', password))
    features['n_special'] = len(re.findall(r'[^A-Za-z0-9]', password))
    return features

features_df = data['password'].apply(extract_features).apply(pd.Series)
data = pd.concat([data, features_df], axis=1)

# --- 3. Target Variable Scaling ---
min_max_scaler = MinMaxScaler()
data['strength_minmax'] = min_max_scaler.fit_transform(data[['strength']])
data = data.drop("strength", axis=1)

# --- 4. Prepare Data for Modeling ---
X = data.drop(['strength_minmax', 'password'], axis=1)
y = data['strength_minmax']
X_encoded = pd.get_dummies(X, columns=['category'], drop_first=True)
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# --- 5. Model Training ---
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.01,
    eval_metric='rmse',
    random_state=42
)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Model Evaluation ---
y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)

print("\n--- Regression Metrics (Validation Set) ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# --- 7. Visualization ---
print("\n--- Actual vs. Predicted Plot ---")
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.6)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--', lw=2)
plt.title('Actual vs. Predicted Strength Scores')
plt.xlabel('Actual Values (Normalized)')
plt.ylabel('Predicted Values (Normalized)')
plt.grid(True)

plt.show()
