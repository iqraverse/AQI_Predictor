import os
import platform
os.environ["FEAST_USAGE"] = "pandas"

import pandas as pd
import numpy as np
from feast import FeatureStore
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient

# -------- 1️⃣ Setup Paths --------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FEATURE_REPO_PATH = os.path.join(PROJECT_ROOT, "feature_repo")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "final", "karachi_aqi_final.parquet")

os.makedirs(MODELS_DIR, exist_ok=True)

# -------- 2️⃣ Connect to Feast --------
print("🔗 Connecting to Feast Feature Store...")
store = FeatureStore(repo_path=FEATURE_REPO_PATH)

# -------- 3️⃣ Load Local Data (time-series aware) --------
print(f"📥 Loading data from: {DATA_PATH}")
df_local = pd.read_parquet(DATA_PATH)

# 🛠️ Fix: Rename 'time' → 'event_timestamp' if necessary
if "time" in df_local.columns and "event_timestamp" not in df_local.columns:
    df_local.rename(columns={"time": "event_timestamp"}, inplace=True)

# ✅ Convert timestamp to datetime
df_local["event_timestamp"] = pd.to_datetime(df_local["event_timestamp"], errors="coerce")
df_local = df_local.dropna(subset=["event_timestamp"]).sort_values("event_timestamp").reset_index(drop=True)

print(f"📋 Columns: {list(df_local.columns)}")
print(f"🕒 Date Range: {df_local['event_timestamp'].min()} → {df_local['event_timestamp'].max()}")
print(f"📊 Total rows: {len(df_local)}")

# ✅ Time-series split: 80% train, 20% test
train_size = int(0.8 * len(df_local))
train_df = df_local.iloc[:train_size].copy()
test_df = df_local.iloc[train_size:].copy()

print(f"📊 Train: {len(train_df)}, Test: {len(test_df)}")

# -------- 4️⃣ Feature setup --------
feature_view_name = "karachi_aqi_features"
feature_view = store.get_feature_view(feature_view_name)
TARGET_COL = "target_pm25_next"

features_list = [
    f"{feature_view_name}:{f.name}"
    for f in feature_view.schema
    if f.name not in ["city", TARGET_COL]
]

print("📦 Feature schema loaded from Feast")

# -------- 5️⃣ Prepare Data --------
train_df = train_df.drop(columns=["city", "event_timestamp"], errors="ignore")
test_df = test_df.drop(columns=["city", "event_timestamp"], errors="ignore")

if TARGET_COL not in train_df.columns:
    raise ValueError(f"❌ Target column '{TARGET_COL}' not found in dataset!")

X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

print(f"🧩 Training: {X_train.shape}, Testing: {X_test.shape}")

# -------- 6️⃣ Scale Target Variable --------
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1)).ravel()

scaler_path = os.path.join(MODELS_DIR, "target_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"💾 Target scaler saved → {scaler_path}")

# -------- 7️⃣ Initialize MLflow (Cross-platform Safe Setup) --------
mlflow_uri = "file:./mlruns"
os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
os.environ["MLFLOW_ARTIFACT_URI"] = "file:./mlruns/artifacts"

mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("AQI_Feast_Training_TimeSeries")

print(f"🗃️ MLflow tracking initialized → {mlflow_uri}")

# -------- 8️⃣ Train RandomForest --------
with mlflow.start_run(run_name="RandomForest_Tuned") as rf_run:
    print("\n🌲 Fine-tuning RandomForest (time-series)...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid=param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train_scaled)

    best_rf = grid_search.best_estimator_
    rf_preds_scaled = best_rf.predict(X_test)
    rf_preds = scaler.inverse_transform(rf_preds_scaled.reshape(-1, 1)).ravel()

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    print(f"✅ RF Best Params: {grid_search.best_params_}")
    print(f"✅ RF → R²: {rf_r2:.3f}, RMSE: {rf_rmse:.3f}, MAE: {rf_mae:.3f}")

    rf_signature = infer_signature(X_train, best_rf.predict(X_train.head(5)))

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({"R2": rf_r2, "RMSE": rf_rmse, "MAE": rf_mae})
    mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="RandomForest_Model",
        input_example=X_train.head(5),
        signature=rf_signature
    )

# -------- 9️⃣ Train XGBoost --------
with mlflow.start_run(run_name="XGBoost_Model") as xgb_run:
    print("\n⚡ Training XGBoost (time-series)...")
    xgb = XGBRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
    )
    xgb.fit(X_train, y_train_scaled)
    xgb_preds_scaled = xgb.predict(X_test)
    xgb_preds = scaler.inverse_transform(xgb_preds_scaled.reshape(-1, 1)).ravel()

    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)

    print(f"✅ XGB → R²: {xgb_r2:.3f}, RMSE: {xgb_rmse:.3f}, MAE: {xgb_mae:.3f}")

    xgb_signature = infer_signature(X_train, xgb.predict(X_train.head(5)))

    mlflow.log_metrics({"R2": xgb_r2, "RMSE": xgb_rmse, "MAE": xgb_mae})
    mlflow.sklearn.log_model(
        sk_model=xgb,
        artifact_path="XGBoost_Model",
        input_example=X_train.head(5),
        signature=xgb_signature
    )

# -------- 🔟 Prediction vs Actual Plot --------
rf_df = pd.DataFrame({"Actual": y_test, "Predicted": rf_preds})
xgb_df = pd.DataFrame({"Actual": y_test, "Predicted": xgb_preds})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(ax=axes[0], x="Actual", y="Predicted", data=rf_df, color="forestgreen", alpha=0.6, s=40)
sns.lineplot(ax=axes[0], x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color="red", lw=2)
axes[0].set_title(f"Random Forest (R²={rf_r2:.3f})")

sns.scatterplot(ax=axes[1], x="Actual", y="Predicted", data=xgb_df, color="royalblue", alpha=0.6, s=40)
sns.lineplot(ax=axes[1], x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color="red", lw=2)
axes[1].set_title(f"XGBoost (R²={xgb_r2:.3f})")

plt.suptitle("📈 Actual vs Predicted AQI Comparison", fontsize=15)
plot_path = os.path.join(MODELS_DIR, "rf_xgb_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

# -------- 1️⃣1️⃣ Save & Log --------
metrics = {
    "RandomForest": {"RMSE": rf_rmse, "MAE": rf_mae, "R2": rf_r2},
    "XGBoost": {"RMSE": xgb_rmse, "MAE": xgb_mae, "R2": xgb_r2},
}
metrics_df = pd.DataFrame(metrics).T
metrics_csv = os.path.join(MODELS_DIR, "model_metrics_feast.csv")
metrics_df.to_csv(metrics_csv)

joblib.dump(best_rf, os.path.join(MODELS_DIR, "random_forest_tuned.pkl"))
joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost_model.pkl"))

mlflow.log_artifact(metrics_csv)
mlflow.log_artifact(plot_path)
mlflow.log_artifact(scaler_path)

# -------- 🧱 MODEL REGISTRY INTEGRATION --------
print("\n🔗 Registering models into MLflow Model Registry...")

client = MlflowClient(tracking_uri=mlflow_uri)

rf_registered = mlflow.register_model(
    model_uri=f"runs:/{rf_run.info.run_id}/RandomForest_Model",
    name="AQI_Predictor_RandomForest"
)
xgb_registered = mlflow.register_model(
    model_uri=f"runs:/{xgb_run.info.run_id}/XGBoost_Model",
    name="AQI_Predictor_XGBoost"
)

print(f"✅ Registered RandomForest model → Version: {rf_registered.version}")
print(f"✅ Registered XGBoost model → Version: {xgb_registered.version}")

if rf_r2 > xgb_r2:
    client.transition_model_version_stage(
        name="AQI_Predictor_RandomForest",
        version=rf_registered.version,
        stage="Staging",
        archive_existing_versions=True
    )
    print("🏆 RandomForest moved to 'Staging' stage.")
else:
    client.transition_model_version_stage(
        name="AQI_Predictor_XGBoost",
        version=xgb_registered.version,
        stage="Staging",
        archive_existing_versions=True
    )
    print("🏆 XGBoost moved to 'Staging' stage.")

# -------- 🏁 FINAL OUTPUT --------
best_model = "RandomForest" if rf_r2 > xgb_r2 else "XGBoost"

print("\n💾 Models, metrics, plots, and registry updates saved successfully!")
print("📁 Outputs:")
print("   → models/random_forest_tuned.pkl")
print("   → models/xgboost_model.pkl")
print("   → models/model_metrics_feast.csv")
print("   → models/target_scaler.pkl")
print("   → models/rf_xgb_comparison.png")
print(f"\n🏆 Best Model: {best_model} (R² = {max(rf_r2, xgb_r2):.3f})")
print("\n🏁 Model training + registration completed successfully using Feast + MLflow 🚀")
