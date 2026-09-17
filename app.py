import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Assam Public Transport Delay Prediction",
    page_icon="🚌",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 17px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 22px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 10px;
}

.result-box {
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 15px;
}

.delay-number {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
}

.delay-label {
    text-align: center;
    font-size: 18px;
    margin-bottom: 15px;
}

.small-note {
    font-size: 13px;
    color: #666;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

ROUTE_FILE = "data/assam_routes_final_verified.csv"

MODEL_FILE = "models/final_assam_delay_model_v2.pkl"

MODEL_COLUMNS_FILE = "models/final_assam_model_columns_v2.pkl"

NLP_VECTORIZER_FILE = "models/final_assam_tfidf_vectorizer_v2.pkl"


# ============================================================
# LOAD ROUTE DATA
# ============================================================

@st.cache_data
def load_routes():

    df = pd.read_csv(ROUTE_FILE)

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    return model


# ============================================================
# LOAD MODEL COLUMNS
# ============================================================

@st.cache_resource
def load_model_columns():

    with open(MODEL_COLUMNS_FILE, "rb") as f:
        columns = pickle.load(f)

    return columns


# ============================================================
# LOAD NLP VECTORIZER
# ============================================================

@st.cache_resource
def load_nlp_vectorizer():

    try:

        with open(NLP_VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)

        return vectorizer

    except Exception:

        return None


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    routes_df = load_routes()
    model = load_model()
    model_columns = load_model_columns()
    nlp_vectorizer = load_nlp_vectorizer()

except Exception as e:

    st.error("Unable to load project files.")

    st.code(str(e))

    st.stop()


# ============================================================
# WEATHER COORDINATES
# ============================================================

CITY_COORDINATES = {

    "Guwahati": (26.1445, 91.7362),

    "Nagaon": (26.3500, 92.6830),

    "Jorhat": (26.7509, 94.2037),

    "Dibrugarh": (27.4728, 94.9120),

    "Haflong": (25.1648, 93.0174),

    "Margherita": (27.2844, 95.6678),

    "Tinsukia": (27.4891, 95.3599),

    "Biswanath Chariali": (26.7271, 93.1478),

}


# ============================================================
# GET LIVE TEMPERATURE
# ============================================================

@st.cache_data(ttl=1800)
def get_temperature(city):

    if city not in CITY_COORDINATES:

        return None

    latitude, longitude = CITY_COORDINATES[city]

    try:

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            "&current=temperature_2m"
        )

        response = requests.get(
            url,
            timeout=8
        )

        if response.status_code == 200:

            data = response.json()

            temperature = data["current"]["temperature_2m"]

            return round(float(temperature), 1)

    except Exception:

        pass

    return None


# ============================================================
# WEATHER CONDITION
# ============================================================

WEATHER_OPTIONS = [
    "Clear",
    "Cloudy",
    "Rain",
    "Heavy Rain",
    "Storm"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:

        return "Winter"

    elif month in [3, 4, 5]:

        return "Spring"

    elif month in [6, 7, 8, 9]:

        return "Monsoon"

    else:

        return "Autumn"


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def prepare_prediction_input(
    selected_route,
    journey_date,
    journey_time,
    weather_condition,
    temperature,
    traffic,
    holiday
):

    row = {}

    # --------------------------------------------------------
    # ROUTE INFORMATION
    # --------------------------------------------------------

    row["road_distance_km"] = float(
        selected_route["road_distance_km"]
    )

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    row["temperature_C"] = float(
        temperature
    )

    # --------------------------------------------------------
    # TRAFFIC
    # --------------------------------------------------------

    row["traffic_congestion_index"] = float(
        traffic
    )

    # --------------------------------------------------------
    # HOLIDAY
    # --------------------------------------------------------

    row["holiday"] = int(
        holiday
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    row["year"] = journey_date.year

    row["month"] = journey_date.month

    row["day"] = journey_date.day

    row["day_of_week_num"] = journey_date.weekday()

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    row["hour"] = journey_time.hour

    row["minute"] = journey_time.minute

    # --------------------------------------------------------
    # TRANSPORT
    # --------------------------------------------------------

    transport = selected_route["transport_type"]

    row[
        "transport_type_" + str(transport)
    ] = 1

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    row[
        "weather_condition_" +
        str(weather_condition)
    ] = 1

    # --------------------------------------------------------
    # WEEKDAY
    # --------------------------------------------------------

    weekday_name = journey_date.strftime("%A")

    row[
        "weekday_" + weekday_name
    ] = 1

    # --------------------------------------------------------
    # SEASON
    # --------------------------------------------------------

    season = get_season(
        journey_date.month
    )

    row[
        "season_" + season
    ] = 1

    # --------------------------------------------------------
    # ROUTE ID
    # --------------------------------------------------------

    route_id = str(
        selected_route["route_id"]
    )

    row[
        "route_id_" + route_id
    ] = 1

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame([row])

    # --------------------------------------------------------
    # ADD MISSING MODEL COLUMNS
    # --------------------------------------------------------

    for column in model_columns:

        if column not in input_df.columns:

            input_df[column] = 0

    # --------------------------------------------------------
    # REMOVE EXTRA COLUMNS
    # --------------------------------------------------------

    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    return input_df


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚌 Assam Public Transport Delay Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Data Science + NLP Industrial Internship Project'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SECTION 1 — TRANSPORT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '1. Transport Type'
    '</div>',
    unsafe_allow_html=True
)

transport_types = sorted(
    routes_df["transport_type"]
    .dropna()
    .unique()
    .tolist()
)

selected_transport = st.selectbox(
    "Select transport type",
    transport_types
)


# ============================================================
# FILTER BY TRANSPORT
# ============================================================

transport_routes = routes_df[
    routes_df["transport_type"]
    == selected_transport
]


# ============================================================
# SECTION 2 — STARTING POINT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '2. Starting Point'
    '</div>',
    unsafe_allow_html=True
)

starting_points = sorted(
    transport_routes["starting_point"]
    .dropna()
    .unique()
    .tolist()
)

selected_start = st.selectbox(
    "Select starting point",
    starting_points
)


# ============================================================
# SECTION 3 — DESTINATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '3. Destination Point'
    '</div>',
    unsafe_allow_html=True
)

destination_routes = transport_routes[
    transport_routes["starting_point"]
    == selected_start
]

destinations = sorted(
    destination_routes["destination_point"]
    .dropna()
    .unique()
    .tolist()
)

selected_destination = st.selectbox(
    "Select destination point",
    destinations
)


# ============================================================
# SELECTED ROUTE
# ============================================================

matching_routes = destination_routes[
    destination_routes["destination_point"]
    == selected_destination
]

if matching_routes.empty:

    st.warning(
        "No verified route found for this selection."
    )

    st.stop()


selected_route = matching_routes.iloc[0]


# ============================================================
# SECTION 4 — ROUTE DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '4. Selected Route Details'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Route ID",
        str(selected_route["route_id"])
    )

with col2:

    st.metric(
        "Distance",
        f'{selected_route["road_distance_km"]:.2f} km'
    )

with col3:

    st.metric(
        "Transport",
        str(selected_route["transport_type"])
    )


st.write(
    "**Route Name:**",
    selected_route["route_name"]
)

st.write(
    "**Route Path:**",
    selected_route["route_path"]
)

st.write(
    "**Road Type:**",
    selected_route["road_type"]
)

st.write(
    "**Highway Number:**",
    selected_route["highway_number"]
)


# ============================================================
# SECTION 5 — JOURNEY DATE & TIME
# ============================================================

st.markdown(
    '<div class="section-title">'
    '5. Journey Date & Time'
    '</div>',
    unsafe_allow_html=True
)

date_col, time_col = st.columns(2)

with date_col:

    journey_date = st.date_input(
        "Journey date",
        value=datetime.now().date()
    )

with time_col:

    journey_time = st.time_input(
        "Journey time",
        value=datetime.now().replace(
            second=0,
            microsecond=0
        ).time()
    )


# ============================================================
# SECTION 6 — WEATHER
# ============================================================

st.markdown(
    '<div class="section-title">'
    '6. Weather Condition'
    '</div>',
    unsafe_allow_html=True
)

weather_condition = st.selectbox(
    "Select weather condition",
    WEATHER_OPTIONS
)


# ============================================================
# SECTION 7 — TEMPERATURE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '7. Temperature'
    '</div>',
    unsafe_allow_html=True
)


# IMPORTANT:
# Temperature is now based on selected STARTING POINT.

temperature = get_temperature(
    selected_start
)


if temperature is not None:

    st.metric(
        "Current temperature at starting point",
        f"{temperature:.1f} °C"
    )

else:

    st.warning(
        "Live temperature could not be retrieved."
    )

    temperature = st.number_input(
        "Enter temperature manually",
        min_value=-10.0,
        max_value=50.0,
        value=25.0,
        step=0.1
    )


st.caption(
    f"Temperature location: {selected_start}"
)


# ============================================================
# SECTION 8 — TRAFFIC
# ============================================================

st.markdown(
    '<div class="section-title">'
    '8. Traffic Congestion'
    '</div>',
    unsafe_allow_html=True
)

traffic = st.slider(
    "Traffic congestion index",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    help=(
        "0 = very low traffic, "
        "100 = extremely heavy traffic"
    )
)


# ============================================================
# TRAFFIC DESCRIPTION
# ============================================================

if traffic < 25:

    traffic_label = "Low"

elif traffic < 50:

    traffic_label = "Moderate"

elif traffic < 75:

    traffic_label = "High"

else:

    traffic_label = "Very High"


st.info(
    f"Traffic level: **{traffic_label}**"
)


# ============================================================
# SECTION 9 — HOLIDAY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '9. Holiday'
    '</div>',
    unsafe_allow_html=True
)

holiday_option = st.radio(
    "Is the journey on a holiday?",
    ["No", "Yes"],
    horizontal=True
)

holiday = (
    1
    if holiday_option == "Yes"
    else 0
)


# ============================================================
# SECTION 10 — INPUT SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '10. Journey Summary'
    '</div>',
    unsafe_allow_html=True
)

summary_col1, summary_col2 = st.columns(2)

with summary_col1:

    st.write(
        f"**Transport:** {selected_transport}"
    )

    st.write(
        f"**Starting Point:** {selected_start}"
    )

    st.write(
        f"**Destination:** {selected_destination}"
    )

    st.write(
        f"**Route ID:** {selected_route['route_id']}"
    )

    st.write(
        f"**Distance:** "
        f"{selected_route['road_distance_km']:.2f} km"
    )

with summary_col2:

    st.write(
        f"**Date:** "
        f"{journey_date.strftime('%d-%m-%Y')}"
    )

    st.write(
        f"**Time:** "
        f"{journey_time.strftime('%H:%M')}"
    )

    st.write(
        f"**Weather:** {weather_condition}"
    )

    st.write(
        f"**Temperature:** {temperature:.1f} °C"
    )

    st.write(
        f"**Traffic:** {traffic_label}"
    )

    st.write(
        f"**Holiday:** {holiday_option}"
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Transport Delay",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------------------------

        input_data = prepare_prediction_input(
            selected_route=selected_route,
            journey_date=journey_date,
            journey_time=journey_time,
            weather_condition=weather_condition,
            temperature=temperature,
            traffic=traffic,
            holiday=holiday
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        prediction = float(
            prediction
        )

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🎯 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # MAIN DELAY RESULT
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="delay-number">'
            f'{prediction:.2f} minutes'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="delay-label">'
            'Predicted Arrival Delay'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if prediction < 2:

            status = "🟢 Very Low Delay"

        elif prediction < 5:

            status = "🟢 Low Delay"

        elif prediction < 10:

            status = "🟡 Moderate Delay"

        elif prediction < 15:

            status = "🟠 High Delay"

        else:

            status = "🔴 Very High Delay"

        st.subheader(status)

        # ----------------------------------------------------
        # RESULT DETAILS
        # ----------------------------------------------------

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Transport",
                selected_transport
            )

            st.metric(
                "Route",
                f"{selected_start} → "
                f"{selected_destination}"
            )

        with result_col2:

            st.metric(
                "Distance",
                f'{selected_route["road_distance_km"]:.2f} km'
            )

            st.metric(
                "Temperature",
                f"{temperature:.1f} °C"
            )

        with result_col3:

            st.metric(
                "Weather",
                weather_condition
            )

            st.metric(
                "Traffic",
                traffic_label
            )

        # ----------------------------------------------------
        # JOURNEY DETAILS
        # ----------------------------------------------------

        st.write("### 📋 Journey Details")

        result_details = pd.DataFrame({
            "Parameter": [
                "Transport Type",
                "Starting Point",
                "Destination",
                "Route ID",
                "Route Name",
                "Road Distance",
                "Journey Date",
                "Journey Time",
                "Weather Condition",
                "Temperature",
                "Traffic Level",
                "Holiday"
            ],

            "Value": [
                selected_transport,
                selected_start,
                selected_destination,
                selected_route["route_id"],
                selected_route["route_name"],
                f'{selected_route["road_distance_km"]:.2f} km',
                journey_date.strftime("%d-%m-%Y"),
                journey_time.strftime("%H:%M"),
                weather_condition,
                f"{temperature:.1f} °C",
                f"{traffic_label} ({traffic}/100)",
                holiday_option
            ]
        })

        st.table(
            result_details
        )

        # ----------------------------------------------------
        # FINAL MESSAGE
        # ----------------------------------------------------

        st.success(
            f"The model predicts an arrival delay of "
            f"approximately **{prediction:.2f} minutes** "
            f"for this journey."
        )

    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Assam Public Transport Delay Prediction | "
    "Data Science + NLP Industrial Internship Project"
)

st.caption(
    "Training data contains simulated delay values "
    "generated using verified route information. "
    "Predictions are model estimates and are not "
    "official ASTC real-time predictions."
)