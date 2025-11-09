import requests, pandas as pd
from datetime import date, timedelta
import os

LAT, LON = 24.8607, 67.0011
START_DATE = "2024-01-01"
END_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

AIR_URL = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LAT}&longitude={LON}"
    f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    f"&start_date={START_DATE}&end_date={END_DATE}&timezone=auto"
)

WEATHER_URL = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
    f"&start_date={START_DATE}&end_date={END_DATE}&timezone=auto"
)

OUT = "data/raw/karachi_aqi_data.csv"
os.makedirs("data/raw", exist_ok=True)

def fetch_data():
    aq = requests.get(AIR_URL, timeout=60).json().get("hourly", {})
    wx = requests.get(WEATHER_URL, timeout=60).json().get("hourly", {})
    df_aq, df_wx = pd.DataFrame(aq), pd.DataFrame(wx)
    df_aq["time"] = pd.to_datetime(df_aq["time"])
    df_wx["time"] = pd.to_datetime(df_wx["time"])
    df = pd.merge(df_aq, df_wx, on="time", how="inner").sort_values("time")

    # ✅ Fix CO units (mg/m³ → µg/m³) if needed
    if "carbon_monoxide" in df.columns:
        max_co = df["carbon_monoxide"].max()
        if max_co < 1000:  # values too small → still in mg/m³
            print("🔁 Converting CO mg/m³ → µg/m³")
            df["carbon_monoxide"] = df["carbon_monoxide"] * 1000
        elif max_co > 100000:  # too high → accidentally converted twice
            print("⚠️ CO seems over-converted (dividing by 1000 to fix)")
            df["carbon_monoxide"] = df["carbon_monoxide"] / 1000
        else:
            print("✅ CO values already in correct µg/m³ range")

    df.to_csv(OUT, index=False)
    print("✅ Karachi AQI + Weather data saved:", OUT)

if __name__ == "__main__":
    fetch_data()

