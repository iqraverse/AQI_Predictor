import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_FILE = "data/raw/karachi_aqi_data.csv"
CLEAN_FILE = "data/clean/karachi_aqi_clean.csv"
os.makedirs("data/clean", exist_ok=True)

def eda_and_clean():
    # ---------- Load ----------
    df = pd.read_csv(RAW_FILE)
    print(f"Raw shape: {df.shape}")

    # ---------- Rename columns ----------
    rename_dict = {
        "pm2_5": "pm25",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "carbon_monoxide": "co",
        "ozone": "o3",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed"
    }
    df.rename(columns=rename_dict, inplace=True, errors="ignore")

    # ---------- Remove duplicates ----------
    df.drop_duplicates(inplace=True)

    # ---------- Timestamp ----------
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df.dropna(subset=["time"], inplace=True)
    df.sort_values("time", inplace=True)

    # ---------- Fix CO units ----------
    if "co" in df.columns and df["co"].max() > 100000:
        df["co"] = df["co"] / 1000

    # ---------- Handle missing ----------
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # ---------- Outlier removal ----------
    for col in ["pm25","pm10","no2","so2","o3","co"]:
        if col in df.columns:
            q1, q99 = df[col].quantile([0.01, 0.99])
            df = df[(df[col] >= q1) & (df[col] <= q99)]

    # ---------- Time continuity check ----------
    df = df.set_index("time").sort_index()

    # infer time frequency
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        print("⚠️ Frequency not detected, assuming daily.")
        inferred_freq = "D"

    full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=inferred_freq)
    missing_timestamps = full_time_index.difference(df.index)

    if len(missing_timestamps) > 0:
        print(f"🕒 Found {len(missing_timestamps)} missing timestamps. Filling with NaNs and forward filling...")
        df = df.reindex(full_time_index)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
    else:
        print("✅ Time continuity verified — no missing timestamps found.")

    df.reset_index(inplace=True)
    df.rename(columns={"index": "time"}, inplace=True)

    # ---------- EDA ----------
    print("\nSummary stats:\n", df.describe())

    plt.figure(figsize=(8, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Value Heatmap")
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (Cleaned Data)")
    plt.show()

    for col in ["pm25","pm10","no2","so2","o3","co"]:
        if col in df.columns:
            plt.figure(figsize=(6, 3))
            sns.histplot(df[col], bins=40, kde=True)
            plt.title(f"{col} Distribution After Cleaning")
            plt.tight_layout()
            plt.show()

    df.to_csv(CLEAN_FILE, index=False)
    print(f"\n✅ Cleaned data saved → {CLEAN_FILE}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    eda_and_clean()
