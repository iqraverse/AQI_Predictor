import os
import pandas as pd
import mlflow
import mlflow.pyfunc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -------- 1️⃣ Setup Paths --------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "final", "karachi_aqi_final.parquet")
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlruns", "mlflow.db")

mlflow_uri = f"sqlite:///{MLFLOW_DB_PATH.replace(os.sep, '/')}"
mlflow.set_tracking_uri(mlflow_uri)

# -------- 2️⃣ Connect to MLflow Model Registry --------
print("🔗 Connecting to MLflow Model Registry...")

client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)

# Get latest staging model (prefer XGBoost if multiple; we pick first found)
model_names = ["AQI_Predictor_RandomForest", "AQI_Predictor_XGBoost"]
staging_models = []
for name in model_names:
    try:
        versions = client.get_latest_versions(name=name, stages=["Staging"])
        if versions:
            staging_models.append(versions[0])
    except Exception as e:
        print(f"ℹ️ Could not fetch versions for {name}: {e}")

if not staging_models:
    raise ValueError("❌ No model found in 'Staging' stage! Please promote one model first.")

latest_model = staging_models[0]
model_name = latest_model.name
model_version = latest_model.version
print(f"✅ Loaded Staging Model: {model_name} (Version {model_version})")

# -------- 3️⃣ Load Model from Registry --------
model_uri = f"models:/{model_name}/Staging"
model = mlflow.pyfunc.load_model(model_uri)
print(f"📦 Model loaded successfully from registry → {model_uri}")

# -------- 4️⃣ Load Sample Data (with robust rename) --------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Data file not found at {DATA_PATH}. Run feature_engineering first.")

df = pd.read_parquet(DATA_PATH)
print(f"📥 Loaded data from: {DATA_PATH}")
print(f"Initial columns: {list(df.columns)}")

# 🛠️ Fix: Rename 'time' -> 'event_timestamp' if needed
if "event_timestamp" not in df.columns and "time" in df.columns:
    df.rename(columns={"time": "event_timestamp"}, inplace=True)
    print("🔁 Renamed column 'time' -> 'event_timestamp'")

# If neither exists, raise informative error
if "event_timestamp" not in df.columns:
    raise KeyError("❌ 'event_timestamp' column not found. Ensure your parquet has 'event_timestamp' or 'time' column.")

# Ensure datetime dtype and drop bad rows
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
df = df.dropna(subset=["event_timestamp"]).sort_values("event_timestamp").reset_index(drop=True)

print(f"✅ Data ready. Rows: {len(df)}. Time range: {df['event_timestamp'].min()} -> {df['event_timestamp'].max()}")

# -------- 5️⃣ Prepare Evaluation Sample --------
TARGET_COL = "target_pm25_next"
if TARGET_COL not in df.columns:
    raise ValueError(f"❌ Target column '{TARGET_COL}' not found in data.")

# Use last 100 samples for evaluation (or fewer if not available)
n_eval = min(100, len(df))
sample_df = df.tail(n_eval).copy()

# Drop non-feature cols before predict
X_sample = sample_df.drop(columns=["city", "event_timestamp", TARGET_COL], errors="ignore")
y_actual = sample_df[TARGET_COL].values

print(f"🔎 Evaluation on last {n_eval} rows. Feature shape: {X_sample.shape}")

# -------- 6️⃣ Run Predictions --------
print("🔮 Generating predictions...")
try:
    y_pred = model.predict(X_sample)
except Exception as e:
    # Helpful debug message if model expects different input schema
    raise RuntimeError(f"❌ Model prediction failed. Check feature names & scaling. Underlying error: {e}")

# If model returns 2D array, flatten
y_pred = np.asarray(y_pred).ravel()

# Align lengths (safety)
if len(y_pred) != len(y_actual):
    raise ValueError(f"❌ Prediction length mismatch: preds={len(y_pred)} vs actuals={len(y_actual)}")

# -------- 7️⃣ Evaluate --------
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print("\n📊 Evaluation Results (on last {} rows):".format(n_eval))
print(f"R² Score  : {r2:.3f}")
print(f"RMSE      : {rmse:.3f}")
print(f"MAE       : {mae:.3f}")

# -------- 8️⃣ Plot Prediction vs Actual --------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_actual, y=y_pred, color="royalblue", s=60, alpha=0.6, label="Predictions")
sns.lineplot(x=[y_actual.min(), y_actual.max()], y=[y_actual.min(), y_actual.max()],
             color="red", lw=2, label="Perfect Fit")
plt.title(f"Prediction vs Actual — {model_name} (R²={r2:.3f})", fontsize=13, weight="bold")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.legend()
plt.grid(True, alpha=0.3)

plot_path = os.path.join(PROJECT_ROOT, "models", f"pred_vs_actual_{model_name}.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n📈 Prediction comparison plot saved at: {plot_path}")
print("\n✅ Prediction script executed successfully using MLflow Model Registry (Staging model). 🚀")
