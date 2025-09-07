import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------
# Data Preprocessing
# --------------------
def preprocess_data(df):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day'] = df['date'].dt.dayofweek
        df.drop(columns=['date'], inplace=True)
    df.fillna(df.mean(), inplace=True)
    X = df.drop(columns=['Appliances'])
    y = df['Appliances']
    return X, y

# --------------------
# Model Training
# --------------------
def train_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }

# --------------------
# Model Evaluation
# --------------------
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
        print(f"{name} ✅ -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

    results_df = pd.DataFrame(results).T
    print("\n📊 Model Performance Comparison:\n", results_df)

    best_model_name = results_df['RMSE'].idxmin()
    print(f"🏆 Best Model: {best_model_name}")
    return results, best_model_name, models[best_model_name]

# --------------------
# Save Model
# --------------------
def save_best_model(model, filename="best_energy_model.pkl"):
    joblib.dump(model, filename)
    print(f"✅ Best model saved as {filename}")

# --------------------
# Visualization
# --------------------
def visualize_results(best_model, best_model_name, X_test, y_test, df):
    y_pred = best_model.predict(X_test)

    # Prediction vs Actual
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Energy Consumption")
    plt.ylabel("Predicted Energy Consumption")
    plt.title(f"Prediction vs Actual ({best_model_name})")
    plt.show()

    # Feature Importance
    if best_model_name in ["Random Forest", "XGBoost", "LightGBM"]:
        importances = best_model.feature_importances_
        features = df.drop(columns=["Appliances"]).columns
        feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=False)

        plt.figure(figsize=(10,6))
        sns.barplot(x="Importance", y="Feature", data=feat_df.head(15))
        plt.title("Top 15 Important Features")
        plt.show()
