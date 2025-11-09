import os
from datetime import timedelta
import pandas as pd
import numpy as np
import mlflow
import joblib

# -------- Paths --------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "final", "karachi_aqi_final.parquet")
PRED_PATH = os.path.join(PROJECT_ROOT, "data", "predictions")
FEATURE_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
TARGET_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "target_scaler.pkl")

os.makedirs(PRED_PATH, exist_ok=True)

# ✅ Safe MLflow Setup (No /C:/ paths)
mlflow_uri = "file:./mlruns"
os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
os.environ["MLFLOW_ARTIFACT_URI"] = "file:./mlruns/artifacts"
mlflow.set_tracking_uri(mlflow_uri)
print(f"🧭 MLflow tracking set to: {mlflow_uri}")

# -------- Config --------
MODEL_NAME = "AQI_Predictor_XGBoost"   # or "AQI_Predictor_RandomForest"
MODEL_STAGE = "Staging"
TARGET_COL = "target_pm25_next"
FORECAST_HORIZON = 3

# -------- Load Model --------
print(f"🔗 Connecting to MLflow Model Registry: {MODEL_NAME} ({MODEL_STAGE})")
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri)
print("✅ Model loaded successfully!")

# -------- Load Scalers --------
if not os.path.exists(FEATURE_SCALER_PATH):
    raise FileNotFoundError(f"❌ Missing feature scaler at {FEATURE_SCALER_PATH}")
if not os.path.exists(TARGET_SCALER_PATH):
    raise FileNotFoundError(f"❌ Missing target scaler at {TARGET_SCALER_PATH}")

feature_scaler = joblib.load(FEATURE_SCALER_PATH)
target_scaler = joblib.load(TARGET_SCALER_PATH)
print("✅ Feature & Target scalers loaded successfully!")

# -------- Load Data --------
df = pd.read_parquet(DATA_PATH)
print(f"📥 Loaded data from: {DATA_PATH}")

if "event_timestamp" not in df.columns and "time" in df.columns:
    df.rename(columns={"time": "event_timestamp"}, inplace=True)
if "event_timestamp" not in df.columns:
    raise KeyError("❌ 'event_timestamp' missing in dataset")

df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
df = df.dropna(subset=["event_timestamp"]).sort_values("event_timestamp").reset_index(drop=True)

latest = df.iloc[-1:].copy()
print(f"📅 Latest timestamp: {latest['event_timestamp'].iloc[0]}")

# -------- Feature Prep --------
pm_candidates = [c for c in df.columns if "pm25" in c.lower() or "pm" in c.lower()]
pm_candidates = [c for c in pm_candidates if c != TARGET_COL]
feature_cols = [c for c in df.columns if c not in ["city", "event_timestamp", TARGET_COL]]

current_input = latest[feature_cols + ["event_timestamp", TARGET_COL]].copy().reset_index(drop=True)

# -------- Forecast Loop --------
forecasts = []

for step in range(1, FORECAST_HORIZON + 1):
    X = current_input[feature_cols]
    X_scaled = feature_scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    scaled_pred = model.predict(X_scaled_df)[0]
    pred_value = target_scaler.inverse_transform([[scaled_pred]])[0][0]

    next_date = current_input["event_timestamp"].iloc[0] + timedelta(days=1)
    forecasts.append({
        "forecast_day": f"Day+{step}",
        "predicted_date": next_date.date(),
        "predicted_AQI": round(pred_value, 2)
    })

    current_input["event_timestamp"] = next_date
    for pm_col in pm_candidates:
        if pm_col in current_input.columns:
            current_input[pm_col] = scaled_pred
    if TARGET_COL in current_input.columns:
        current_input[TARGET_COL] = scaled_pred

    lag_features = [c for c in current_input.columns if "lag" in c.lower()]
    for lag_col in lag_features:
        try:
            lag_num = int(''.join(filter(str.isdigit, lag_col)))
            if lag_num == 1:
                current_input[lag_col] = scaled_pred
            else:
                prev_lag = f"pm25_lag{lag_num - 1}"
                if prev_lag in current_input.columns:
                    current_input[lag_col] = current_input[prev_lag]
        except:
            pass

# -------- Save Forecast --------
forecast_df = pd.DataFrame(forecasts)
out_csv = os.path.join(PRED_PATH, "forecast_next_3_days.csv")
forecast_df.to_csv(out_csv, index=False)

print("\n📈 Forecast for next 3 days:")
print(forecast_df)
print(f"💾 Saved to: {out_csv}")
