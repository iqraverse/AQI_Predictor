import os
import pandas as pd
from feast import FeatureStore, Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

# ---------- Paths ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "final", "karachi_aqi_final.parquet")

# ✅ Safety Check: ensure file exists
if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(f"❌ Parquet file not found at {PARQUET_PATH}. Please run feature_engineering.py first.")

# ---------- Load and Prepare Data ----------
print(f"📥 Loading latest feature-engineered data from: {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH)

# ✅ Validation check: ensure new cleaned dataset is being used
if len(df) > 1000:
    raise ValueError(f"❌ Unexpected data size ({len(df)} rows) — likely loading old file instead of new 653-row Parquet!")

# ✅ Rename timestamp if needed
if "time" in df.columns and "event_timestamp" not in df.columns:
    df.rename(columns={"time": "event_timestamp"}, inplace=True)

# ✅ Ensure timestamps are correctly formatted
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")

# ✅ Add entity column (City)
if "city" not in df.columns:
    df["city"] = "Karachi"

# ✅ Drop duplicates and invalid timestamps
df = df.drop_duplicates(subset=["event_timestamp"]).dropna(subset=["event_timestamp"])

# ---------- Save Clean Copy ----------
clean_parquet_path = os.path.abspath(PARQUET_PATH)
print(f"✅ Cleaned Parquet ready for Feast: {clean_parquet_path}")
print(f"✅ Rows: {len(df)} | Columns: {len(df.columns)}")
print(f"🕒 Data Time Range: {df['event_timestamp'].min()} → {df['event_timestamp'].max()}")

# ---------- Define Feast Data Source ----------
source = FileSource(
    path=clean_parquet_path,
    event_timestamp_column="event_timestamp",
)

# ---------- Define Entity ----------
entity = Entity(
    name="city",
    join_keys=["city"],
    value_type=ValueType.STRING,
    description="City name for air quality features",
)

# ---------- Define Feature View ----------
numeric_cols = [
    col for col in df.select_dtypes(include=["number"]).columns
    if col not in ["event_timestamp"]
]

feature_view = FeatureView(
    name="karachi_aqi_features",
    entities=[entity],
    ttl=None,
    schema=[Field(name=col, dtype=Float32) for col in numeric_cols],
    online=True,
    source=source,
    tags={"source": "feature_engineered_data", "city": "Karachi"},
)

# ---------- Register Features to Feast ----------
store = FeatureStore(repo_path=os.path.dirname(__file__))
store.apply([entity, feature_view])

print("\n✅ Karachi AQI features successfully registered to Feast Feature Store!")
print(f"📦 Uploaded data rows: {len(df)} | Columns: {len(df.columns)}")
