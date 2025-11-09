import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import joblib

CLEAN_FILE = "data/clean/karachi_aqi_clean.csv"
FINAL_FILE = "data/final/karachi_aqi_final.parquet"
SCALER_PATH = "models/scaler.pkl"
TARGET_SCALER_PATH = "models/target_scaler.pkl"

os.makedirs("data/final", exist_ok=True)
os.makedirs("models", exist_ok=True)

def feature_engineering():
    # ---------- Load ----------
    df = pd.read_csv(CLEAN_FILE)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df.dropna(subset=["time"], inplace=True)
    df.sort_values("time", inplace=True)

    # ---------- Log Transform ----------
    if "co" in df.columns:
        df["co"] = np.log1p(df["co"])

    # ---------- Temporal Features ----------
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # ---------- PM2.5 Lag & Rolling ----------
    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm25_lag2"] = df["pm25"].shift(2)
    df["pm25_roll3h"] = df["pm25"].rolling(window=3).mean()
    df["pm25_roll6h"] = df["pm25"].rolling(window=6).mean()
    df["pm25_change_rate"] = df["pm25"].diff(periods=3)

    # ---------- Weather Feature Lags ----------
    if "humidity" in df.columns:
        df["humidity_lag12h"] = df["humidity"].shift(12)
    if "wind_speed" in df.columns:
        df["wind_speed_lag24h"] = df["wind_speed"].shift(24)

    # ---------- Environmental Metrics ----------
    if "evapotranspiration" in df.columns:
        df["evapotranspiration_lag3"] = df["evapotranspiration"].shift(3)
        df["evapotranspiration_roll6h"] = df["evapotranspiration"].rolling(window=6).mean()
    if "shortwave_radiation" in df.columns:
        df["shortwave_radiation_lag3"] = df["shortwave_radiation"].shift(3)
        df["shortwave_radiation_roll6h"] = df["shortwave_radiation"].rolling(window=6).mean()

    # ---------- Target Feature ----------
    df["target_pm25_next"] = df["pm25"].shift(-1)
    df.dropna(inplace=True)

    # ---------- Multicollinearity ----------
    num_df = df.select_dtypes("number")
    vif = pd.DataFrame({
        "feature": num_df.columns,
        "VIF": [variance_inflation_factor(num_df.values, i)
                for i in range(len(num_df.columns))]
    })
    high_vif = vif[vif["VIF"] > 10]["feature"].tolist()
    for f in high_vif:
        if f == "pm10":
            df.drop(columns=["pm10"], inplace=True, errors="ignore")

    # ---------- Separate Scaling ----------
    feature_cols = [c for c in df.select_dtypes("number").columns if c != "target_pm25_next"]
    target_col = "target_pm25_next"

    # Scale Features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, SCALER_PATH)

    # Scale Target Separately
    target_scaler = StandardScaler()
    df[[target_col]] = target_scaler.fit_transform(df[[target_col]])
    joblib.dump(target_scaler, TARGET_SCALER_PATH)

    print(f"✅ Feature scaler saved → {SCALER_PATH}")
    print(f"✅ Target scaler saved → {TARGET_SCALER_PATH}")

    # ---------- Visualization ----------
    plt.figure(figsize=(9, 6))
    sns.heatmap(df[feature_cols + [target_col]].corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (After Feature Engineering)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["pm25"], label="PM2.5 (Original)")
    plt.plot(df["time"], df["target_pm25_next"], label="Target (Scaled Next PM2.5)", alpha=0.7)
    plt.legend()
    plt.title("PM2.5 Trend and Target Shift")
    plt.tight_layout()
    plt.show()

    # ---------- Save ----------
    df.to_parquet(FINAL_FILE, index=False)
    print(f"\n✅ Final feature-engineered data saved → {FINAL_FILE}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    feature_engineering()






# streamlit-based two modes

# CI/CD automation 
# feature script (daily): automation via github Actions ("") fetches raw weather data from api 
#Training script (every 3 days): automated via github actions (" ") retrain models every 3 days 