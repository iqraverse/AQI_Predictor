import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ---------- Paths ----------
PROJECT_ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "final", "karachi_aqi_final.parquet")
FORECAST_PATH = os.path.join(PROJECT_ROOT, "data", "predictions", "forecast_next_3_days.csv")

# ---------- Page Config ----------
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    page_icon="🌆",
    layout="wide"
)

st.title("🌫️ Karachi Air Quality Index (AQI) Dashboard")
st.markdown("### 📈 Real-time Monitoring, Trends & Forecasts")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    if "time" in df.columns and "event_timestamp" not in df.columns:
        df.rename(columns={"time": "event_timestamp"}, inplace=True)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    df = df.dropna(subset=["event_timestamp"])
    df = df.sort_values("event_timestamp")
    return df

df = load_data()
st.sidebar.success("✅ Data Loaded Successfully")

# ---------- Sidebar Filters ----------
st.sidebar.header("🔍 Filter Options")
start_date = st.sidebar.date_input("Start Date", df["event_timestamp"].min().date())
end_date = st.sidebar.date_input("End Date", df["event_timestamp"].max().date())

mask = (df["event_timestamp"].dt.date >= start_date) & (df["event_timestamp"].dt.date <= end_date)
filtered_df = df.loc[mask]

# ---------- KPI Cards ----------
st.subheader("🌡️ Key Air Quality Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Average PM2.5", f"{filtered_df['pm25'].mean():.2f}")
col2.metric("Average AQI (Target)", f"{filtered_df['target_pm25_next'].mean():.2f}")
col3.metric("Data Points", f"{len(filtered_df)}")

# ---------- Time Series Plot ----------
st.markdown("### 🕒 PM2.5 Trend Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filtered_df["event_timestamp"], filtered_df["pm25"], label="PM2.5", color="orange")
ax.plot(filtered_df["event_timestamp"], filtered_df["target_pm25_next"], label="Next PM2.5", alpha=0.6)
ax.set_xlabel("Date")
ax.set_ylabel("PM2.5 Concentration")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# ---------- Correlation Heatmap ----------
st.markdown("### 🔬 Feature Correlation Heatmap")
corr = filtered_df.select_dtypes("number").corr()
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)

# ---------- Forecast Section ----------
st.markdown("### 📅 3-Day AQI Forecast (From Trained Model)")
if os.path.exists(FORECAST_PATH):
    forecast_df = pd.read_csv(FORECAST_PATH)
    st.dataframe(forecast_df, use_container_width=True)

    # Plot forecast trend
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(forecast_df["forecast_day"], forecast_df["predicted_AQI"], marker="o", color="crimson")
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Predicted AQI")
    ax.set_title("Predicted AQI for Next 3 Days")
    st.pyplot(fig)
else:
    st.warning("⚠️ Forecast file not found. Please run `forecast_next_days.py` first.")

# ---------- Raw Data View ----------
with st.expander("📋 View Cleaned Dataset"):
    st.dataframe(filtered_df.head(50), use_container_width=True)

# ---------- Footer ----------
st.markdown("""
---
**Developed by Mahjabeen’s AQI Predictor Project**  
🌍 Powered by Python · Streamlit · Feast · MLflow · XGBoost
""")
