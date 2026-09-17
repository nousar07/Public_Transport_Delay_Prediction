import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# FINAL ASSAM TRANSPORT DELAY DATASET
# Generated from verified ASTC bus routes
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ROUTE_FILE = DATA_DIR / "assam_routes_final_verified.csv"
OUTPUT_FILE = DATA_DIR / "assam_transport_delay_dataset_final_v2.csv"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
N_ROWS = 5000

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# LOAD VERIFIED ROUTES
# ============================================================

routes = pd.read_csv(ROUTE_FILE)

print("\n" + "=" * 70)
print("GENERATING FINAL ASSAM TRANSPORT DELAY DATASET")
print("=" * 70)

print(f"Verified route records loaded: {len(routes)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_route_columns = [
    "route_id",
    "transport_type",
    "starting_point",
    "destination_point",
    "route_name",
    "route_path",
    "road_distance_km"
]

missing_columns = [
    col for col in required_route_columns
    if col not in routes.columns
]

if missing_columns:
    raise ValueError(
        f"Missing route columns: {missing_columns}"
    )


if routes["road_distance_km"].isna().any():
    raise ValueError(
        "Route master contains missing distances."
    )


if not (routes["transport_type"] == "Bus").all():
    raise ValueError(
        "Final route master must contain Bus only."
    )


# ============================================================
# GENERATE ROUTE SELECTION
# ============================================================

# Every verified route is given representation in the dataset.
route_indices = rng.choice(
    len(routes),
    size=N_ROWS,
    replace=True
)

sampled_routes = routes.iloc[
    route_indices
].reset_index(drop=True)


# ============================================================
# DATE AND TIME
# ============================================================

start_date = pd.Timestamp("2025-01-01")

random_days = rng.integers(
    0,
    365,
    size=N_ROWS
)

dates = start_date + pd.to_timedelta(
    random_days,
    unit="D"
)

hours = rng.integers(
    5,
    23,
    size=N_ROWS
)

minutes = rng.choice(
    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    size=N_ROWS
)

times = [
    f"{h:02d}:{m:02d}:00"
    for h, m in zip(hours, minutes)
]


# ============================================================
# WEATHER
# ============================================================

weather_conditions = [
    "Clear",
    "Cloudy",
    "Rain",
    "Heavy Rain",
    "Storm"
]

weather_probabilities = [
    0.35,
    0.30,
    0.20,
    0.10,
    0.05
]

weather = rng.choice(
    weather_conditions,
    size=N_ROWS,
    p=weather_probabilities
)


# ============================================================
# TEMPERATURE
# ============================================================

month_values = dates.month.to_numpy()

base_temperature = np.select(
    [
        np.isin(month_values, [12, 1, 2]),
        np.isin(month_values, [3, 4, 5]),
        np.isin(month_values, [6, 7, 8]),
        np.isin(month_values, [9, 10, 11])
    ],
    [
        18,
        27,
        29,
        26
    ],
    default=25
)

temperature = (
    base_temperature
    + rng.normal(0, 3, N_ROWS)
)

temperature = np.round(
    temperature,
    1
)


# ============================================================
# TRAFFIC CONGESTION
# ============================================================

traffic = rng.normal(
    50,
    20,
    N_ROWS
)

traffic = np.clip(
    traffic,
    0,
    100
)

traffic = np.round(
    traffic,
    1
)


# ============================================================
# HOLIDAY
# ============================================================

holiday = rng.choice(
    [0, 1],
    size=N_ROWS,
    p=[0.90, 0.10]
)


# ============================================================
# DERIVED CALENDAR FEATURES
# ============================================================

weekday = dates.day_name()

season = np.select(
    [
        np.isin(month_values, [12, 1, 2]),
        np.isin(month_values, [3, 4, 5]),
        np.isin(month_values, [6, 7, 8]),
        np.isin(month_values, [9, 10, 11])
    ],
    [
        "Winter",
        "Spring",
        "Monsoon",
        "Autumn"
    ],
    default="Autumn"
)


# ============================================================
# DELAY GENERATION
# ============================================================
# Synthetic target created for ML experimentation.
#
# The target is influenced by:
# - route distance
# - traffic congestion
# - weather
# - holiday
# - time of day
# - random variation
#
# This is a simulated dataset, not real historical
# ASTC delay measurements.
# ============================================================

distance = sampled_routes[
    "road_distance_km"
].to_numpy()


# Base effect from distance
distance_effect = (
    distance / 100.0
) * 1.8


# Traffic effect
traffic_effect = (
    traffic / 100.0
) * 5.0


# Weather effect
weather_effect_map = {
    "Clear": 0.0,
    "Cloudy": 0.8,
    "Rain": 2.0,
    "Heavy Rain": 4.0,
    "Storm": 6.0
}

weather_effect = np.array([
    weather_effect_map[w]
    for w in weather
])


# Time-of-day effect
time_effect = np.select(
    [
        np.isin(hours, [7, 8, 9]),
        np.isin(hours, [17, 18, 19, 20])
    ],
    [
        3.0,
        4.0
    ],
    default=0.5
)


# Holiday effect
holiday_effect = holiday * 1.5


# Random noise
noise = rng.normal(
    0,
    2.2,
    N_ROWS
)


# Final delay
delay = (
    distance_effect
    + traffic_effect
    + weather_effect
    + time_effect
    + holiday_effect
    + noise
)


# Keep delays realistic for this simulated project dataset
delay = np.clip(
    delay,
    -1,
    60
)

delay = np.round(
    delay,
    2
)


# ============================================================
# FINAL DATAFRAME
# ============================================================

df = pd.DataFrame({

    "starting_point":
        sampled_routes["starting_point"],

    "destination_point":
        sampled_routes["destination_point"],

    "route_id":
        sampled_routes["route_id"],

    "transport_type":
        sampled_routes["transport_type"],

    "route_name":
        sampled_routes["route_name"],

    "route_path":
        sampled_routes["route_path"],

    "road_distance_km":
        sampled_routes["road_distance_km"],

    "road_type":
        sampled_routes["road_type"],

    "highway_number":
        sampled_routes["highway_number"],

    "date":
        dates.strftime("%Y-%m-%d"),

    "time":
        times,

    "weather_condition":
        weather,

    "temperature_C":
        temperature,

    "traffic_congestion_index":
        traffic,

    "holiday":
        holiday,

    "weekday":
        weekday,

    "season":
        season,

    "actual_arrival_delay_min":
        delay
})


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(df) != N_ROWS:
    raise ValueError(
        "Incorrect number of rows generated."
    )


if df["route_id"].isna().any():
    raise ValueError(
        "Missing route IDs found."
    )


if df["road_distance_km"].isna().any():
    raise ValueError(
        "Missing route distances found."
    )


if not (
    df["transport_type"] == "Bus"
).all():
    raise ValueError(
        "Dataset contains non-Bus records."
    )


# Make sure every dataset route exists
# in the verified route master.

valid_route_ids = set(
    routes["route_id"]
)

dataset_route_ids = set(
    df["route_id"]
)

unknown_routes = (
    dataset_route_ids
    - valid_route_ids
)

if unknown_routes:
    raise ValueError(
        f"Unknown route IDs found: {unknown_routes}"
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
print("DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Rows                 : {len(df)}")
print(f"Columns              : {len(df.columns)}")

print(
    f"Unique routes        : "
    f"{df['route_id'].nunique()}"
)

print(
    f"Unique starting pts  : "
    f"{df['starting_point'].nunique()}"
)

print(
    f"Unique destinations   : "
    f"{df['destination_point'].nunique()}"
)

print(
    f"Transport types      : "
    f"{df['transport_type'].unique().tolist()}"
)

print(
    f"Delay mean           : "
    f"{df['actual_arrival_delay_min'].mean():.2f}"
)

print(
    f"Delay median         : "
    f"{df['actual_arrival_delay_min'].median():.2f}"
)

print(
    f"Delay minimum        : "
    f"{df['actual_arrival_delay_min'].min():.2f}"
)

print(
    f"Delay maximum        : "
    f"{df['actual_arrival_delay_min'].max():.2f}"
)

print("\nRoute distribution:")
print(
    df["route_id"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nDataset columns:")
print(
    df.columns.tolist()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nValidation PASSED.")