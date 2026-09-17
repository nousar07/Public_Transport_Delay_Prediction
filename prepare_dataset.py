import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PREPARE FINAL ASSAM DATASET FOR MACHINE LEARNING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = (
    DATA_DIR /
    "assam_transport_delay_dataset_final_v2.csv"
)

OUTPUT_FILE = (
    DATA_DIR /
    "assam_prepared_dataset_final_v2.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\n" + "=" * 70)
print("PREPARING FINAL ASSAM DATASET")
print("=" * 70)

print(f"Original shape: {df.shape}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "starting_point",
    "destination_point",
    "route_id",
    "transport_type",
    "route_name",
    "route_path",
    "road_distance_km",
    "road_type",
    "highway_number",
    "date",
    "time",
    "weather_condition",
    "temperature_C",
    "traffic_congestion_index",
    "holiday",
    "weekday",
    "season",
    "actual_arrival_delay_min"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# DATE FEATURES
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

if df["date"].isna().any():
    raise ValueError("Invalid date values found.")


df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day"] = df["date"].dt.day

df["day_of_week_num"] = (
    df["date"].dt.dayofweek
)


# ============================================================
# TIME FEATURES
# ============================================================

time_parsed = pd.to_datetime(
    df["time"],
    format="%H:%M:%S",
    errors="coerce"
)

if time_parsed.isna().any():
    raise ValueError("Invalid time values found.")


df["hour"] = time_parsed.dt.hour

df["minute"] = time_parsed.dt.minute


# ============================================================
# REMOVE ORIGINAL DATE/TIME
# ============================================================

df.drop(
    columns=["date", "time"],
    inplace=True
)


# ============================================================
# REMOVE RAW TEXT ROUTE DESCRIPTION
# ============================================================
# These fields are retained in the original dataset for the app,
# but are not directly passed to the regression model.

df.drop(
    columns=[
        "route_name",
        "route_path"
    ],
    inplace=True
)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

categorical_columns = [
    "starting_point",
    "destination_point",
    "route_id",
    "transport_type",
    "road_type",
    "highway_number",
    "weather_condition",
    "weekday",
    "season"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = [
    str(col).strip()
    for col in df.columns
]


# ============================================================
# HANDLE NUMERIC VALUES
# ============================================================

numeric_columns = [
    "road_distance_km",
    "temperature_C",
    "traffic_congestion_index",
    "holiday",
    "year",
    "month",
    "day",
    "day_of_week_num",
    "hour",
    "minute",
    "actual_arrival_delay_min"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# MISSING VALUE CHECK
# ============================================================

missing_values = df.isna().sum()

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) > 0:

    print("\nMissing values found:")

    print(
        missing_values
        .sort_values(ascending=False)
        .to_string()
    )

    raise ValueError(
        "Missing values detected after preprocessing."
    )


# ============================================================
# TARGET CHECK
# ============================================================

TARGET = "actual_arrival_delay_min"

if TARGET not in df.columns:
    raise ValueError(
        "Target column not found."
    )


# ============================================================
# TARGET LAST
# ============================================================

feature_columns = [
    col
    for col in df.columns
    if col != TARGET
]

df = df[
    feature_columns + [TARGET]
]


# ============================================================
# VALIDATION
# ============================================================

if df.shape[0] != 5000:

    raise ValueError(
        f"Expected 5000 rows, got {df.shape[0]}"
    )


if df[TARGET].isna().any():

    raise ValueError(
        "Target contains missing values."
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETED")
print("=" * 70)

print(
    f"Prepared shape : {df.shape}"
)

print(
    f"Feature count  : {df.shape[1] - 1}"
)

print(
    f"Target         : {TARGET}"
)

print(
    f"Target mean    : "
    f"{df[TARGET].mean():.4f}"
)

print(
    f"Target std     : "
    f"{df[TARGET].std():.4f}"
)

print(
    f"Target min     : "
    f"{df[TARGET].min():.4f}"
)

print(
    f"Target max     : "
    f"{df[TARGET].max():.4f}"
)

print("\nFirst 10 columns:")

print(
    df.columns[:10].tolist()
)

print("\nLast 10 columns:")

print(
    df.columns[-10:].tolist()
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("\nValidation PASSED.")