"""Streamlit interface for Assam public transport delay predictions."""

from datetime import datetime
import pickle

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Assam Transit | Delay Predictor",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root { --ink:#17313a; --muted:#61777d; --line:#dbe9e7; --surface:#fff; --wash:#f4f9f8; --teal:#247a78; --teal-dark:#165d5b; --sky:#e8f2f8; }
.stApp { background:linear-gradient(135deg,#f7fbfa 0%,#eef6f8 55%,#f8faf7 100%); }
.block-container { max-width:1160px; padding-top:2.4rem; padding-bottom:3rem; }
h1,h2,h3,p,label { color:var(--ink) !important; }
[data-testid="stHeader"] { background:rgba(247,251,250,.88); }
.hero { padding:2rem 2.2rem; border:1px solid var(--line); border-radius:22px; background:linear-gradient(120deg,#fff 20%,#eaf6f4 100%); box-shadow:0 12px 32px rgba(31,91,91,.08); }
.eyebrow,.section-kicker { color:var(--teal) !important; font-size:.78rem; font-weight:750; letter-spacing:.11em; text-transform:uppercase; }
.eyebrow { margin-bottom:.5rem; }.section-kicker { margin:1.8rem 0 .2rem; }
.hero h1 { font-size:clamp(2rem,4vw,3rem); margin:0; letter-spacing:-.04em; }
.hero p { color:var(--muted) !important; font-size:1.03rem; margin:.65rem 0 0; max-width:650px; }
.section-heading { color:var(--ink) !important; font-size:1.35rem; font-weight:700; margin:0 0 1rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--line) !important; border-radius:16px !important; background:rgba(255,255,255,.82); box-shadow:0 4px 16px rgba(25,83,82,.035); }
[data-testid="stMetric"] { background:var(--wash); border-radius:12px; padding:.75rem .85rem; }
[data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:.78rem; }
[data-testid="stMetricValue"] { color:var(--ink) !important; font-size:1.35rem; }
div[data-baseweb="select"] > div,div[data-baseweb="input"] > div { border-color:#c9ddda !important; background:#fbfdfd !important; border-radius:10px !important; }
div[data-baseweb="select"] *,div[data-baseweb="input"] *,input { color:var(--ink) !important; }
.stButton > button,.stButton > button * { border:0; border-radius:11px; background:var(--teal); color:#fff !important; font-weight:700; min-height:3rem; box-shadow:0 6px 14px rgba(36,122,120,.18); }
.stButton > button:hover { background:var(--teal-dark); color:#fff; }
.route-strip { background:var(--sky); border:1px solid #d2e3eb; border-radius:13px; padding:1rem 1.1rem; color:var(--ink); }.route-strip strong { color:var(--teal-dark); }
.result-card { padding:1.7rem; border:1px solid #b9ded7; border-radius:18px; background:linear-gradient(135deg,#fafffe,#e5f4f0); text-align:center; }
.result-label { color:var(--teal-dark) !important; font-size:.78rem; font-weight:750; letter-spacing:.1em; text-transform:uppercase; }
.result-number { color:var(--ink) !important; font-size:clamp(2.4rem,6vw,4rem); font-weight:800; letter-spacing:-.06em; margin:.35rem 0; }
.result-note,.footer-note { color:var(--muted) !important; }.result-note { margin:0; }.footer-note { font-size:.82rem; text-align:center; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

ROUTE_FILE = "data/assam_routes_final_verified.csv"
MODEL_FILE = "models/final_assam_delay_model_v2.pkl"
MODEL_COLUMNS_FILE = "models/final_assam_model_columns_v2.pkl"
CITY_COORDINATES = {
    "Guwahati": (26.1445, 91.7362), "Nagaon": (26.3500, 92.6830), "Jorhat": (26.7509, 94.2037),
    "Dibrugarh": (27.4728, 94.9120), "Haflong": (25.1648, 93.0174), "Margherita": (27.2844, 95.6678),
    "Tinsukia": (27.4891, 95.3599), "Biswanath Chariali": (26.7271, 93.1478),
}
WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Heavy Rain", "Storm"]


@st.cache_data
def load_routes():
    return pd.read_csv(ROUTE_FILE)


@st.cache_resource
def load_model_assets():
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
    with open(MODEL_COLUMNS_FILE, "rb") as file:
        columns = pickle.load(file)
    return model, columns


@st.cache_data(ttl=1800, show_spinner=False)
def get_temperature(city):
    if city not in CITY_COORDINATES:
        return None
    latitude, longitude = CITY_COORDINATES[city]
    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": latitude, "longitude": longitude, "current": "temperature_2m"}, timeout=8)
        response.raise_for_status()
        return round(float(response.json()["current"]["temperature_2m"]), 1)
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None


def get_season(month):
    if month in (12, 1, 2): return "Winter"
    if month in (3, 4, 5): return "Spring"
    if month in (6, 7, 8, 9): return "Monsoon"
    return "Autumn"


def prepare_prediction_input(route, journey_date, journey_time, weather, temperature, traffic, holiday, model_columns):
    row = {
        "road_distance_km": float(route["road_distance_km"]), "temperature_C": float(temperature),
        "traffic_congestion_index": float(traffic), "holiday": int(holiday), "year": journey_date.year,
        "month": journey_date.month, "day": journey_date.day, "day_of_week_num": journey_date.weekday(),
        "hour": journey_time.hour, "minute": journey_time.minute,
        f"transport_type_{route['transport_type']}": 1, f"weather_condition_{weather}": 1,
        f"weekday_{journey_date.strftime('%A')}": 1, f"season_{get_season(journey_date.month)}": 1,
        f"route_id_{route['route_id']}": 1,
    }
    return pd.DataFrame([row]).reindex(columns=model_columns, fill_value=0)


def traffic_level(value):
    if value < 25: return "Low", "Easy-flowing roads"
    if value < 50: return "Moderate", "Some slow sections expected"
    if value < 75: return "High", "Busy roads may affect timing"
    return "Very high", "Significant congestion expected"


def delay_status(prediction):
    if prediction < 2: return "Very low delay", "Your journey is expected to run close to schedule."
    if prediction < 5: return "Low delay", "A small delay is possible."
    if prediction < 10: return "Moderate delay", "Allow a little extra time for your journey."
    if prediction < 15: return "High delay", "Consider leaving with a comfortable time buffer."
    return "Very high delay", "A larger time buffer is recommended."


try:
    routes_df = load_routes()
    model, model_columns = load_model_assets()
except ModuleNotFoundError as error:
    st.error("The saved prediction model needs a newer compatible version of scikit-learn.")
    st.code("python -m pip install -r requirements.txt\npython -m streamlit run app.py", language="powershell")
    st.caption("Run these commands in the project folder, then refresh this page.")
    st.exception(error)
    st.stop()
except Exception as error:
    st.error("The prediction files could not be loaded. Check that the data and model folders are present.")
    st.exception(error)
    st.stop()

st.markdown('''<section class="hero"><div class="eyebrow">Assam transit companion</div><h1>Plan your journey with a little more certainty.</h1><p>Select a verified route and journey conditions to receive a calm, easy-to-read delay estimate.</p></section>''', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Journey</div><div class="section-heading">Where are you travelling?</div>', unsafe_allow_html=True)
with st.container(border=True):
    first_col, second_col, third_col = st.columns(3)
    with first_col:
        transport_types = sorted(routes_df["transport_type"].dropna().unique().tolist())
        selected_transport = st.selectbox("Transport type", transport_types)
    transport_routes = routes_df[routes_df["transport_type"] == selected_transport]
    with second_col:
        starting_points = sorted(transport_routes["starting_point"].dropna().unique().tolist())
        selected_start = st.selectbox("Starting point", starting_points)
    destination_routes = transport_routes[transport_routes["starting_point"] == selected_start]
    with third_col:
        destinations = sorted(destination_routes["destination_point"].dropna().unique().tolist())
        selected_destination = st.selectbox("Destination", destinations)

matching_routes = destination_routes[destination_routes["destination_point"] == selected_destination]
if matching_routes.empty:
    st.warning("No verified route is available for this selection. Please choose another journey.")
    st.stop()
selected_route = matching_routes.iloc[0]

st.markdown('<div class="section-kicker">Verified route</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown(f'<div class="route-strip"><strong>{selected_start} → {selected_destination}</strong> &nbsp;·&nbsp; {selected_route["route_name"]}</div>', unsafe_allow_html=True)
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Service", selected_route["route_id"])
    metric_b.metric("Road distance", f'{selected_route["road_distance_km"]:.0f} km')
    metric_c.metric("Road type", selected_route["road_type"])
    with st.expander("View full route information"):
        st.write(f"**Route path:** {selected_route['route_path']}")
        st.write(f"**Highway:** {selected_route['highway_number']}")

st.markdown('<div class="section-kicker">Journey conditions</div><div class="section-heading">When and what should we factor in?</div>', unsafe_allow_html=True)
with st.container(border=True):
    date_col, time_col, weather_col = st.columns(3)
    now = datetime.now()
    with date_col: journey_date = st.date_input("Journey date", value=now.date())
    with time_col: journey_time = st.time_input("Journey time", value=now.replace(second=0, microsecond=0).time())
    with weather_col: weather_condition = st.selectbox("Weather condition", WEATHER_OPTIONS)
    live_temperature = get_temperature(selected_start)
    settings_col, traffic_col, holiday_col = st.columns((1, 1.5, 1))
    with settings_col:
        if live_temperature is None:
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
            st.caption("Enter a temperature because live weather is unavailable.")
        else:
            temperature = live_temperature
            st.metric("Live temperature", f"{temperature:.1f} °C")
            st.caption(f"Updated for {selected_start}")
    with traffic_col:
        traffic = st.slider("Traffic congestion", 0, 100, 50, help="0 is clear roads; 100 is extremely heavy traffic.")
        traffic_name, traffic_help = traffic_level(traffic)
        st.caption(f"**{traffic_name}:** {traffic_help}")
    with holiday_col:
        holiday_option = st.radio("Is it a holiday?", ["No", "Yes"], horizontal=True)
        holiday = int(holiday_option == "Yes")

st.markdown('<div class="section-kicker">Ready when you are</div>', unsafe_allow_html=True)
predict_button = st.button("Predict expected delay", type="primary", use_container_width=True)

if predict_button:
    try:
        input_data = prepare_prediction_input(selected_route, journey_date, journey_time, weather_condition, temperature, traffic, holiday, model_columns)
        prediction = float(model.predict(input_data)[0])
        status, guidance = delay_status(prediction)
        st.markdown('<div class="section-kicker">Your estimate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-card"><div class="result-label">Predicted arrival delay</div><div class="result-number">{prediction:.1f} <span style="font-size:.48em; letter-spacing:0">minutes</span></div><p class="result-note">{guidance}</p></div>', unsafe_allow_html=True)
        st.info(f"**{status.capitalize()}** · This is a model estimate based on the selected journey conditions.")
        result_one, result_two, result_three, result_four = st.columns(4)
        result_one.metric("Route", f"{selected_start} → {selected_destination}")
        result_two.metric("Distance", f'{selected_route["road_distance_km"]:.0f} km')
        result_three.metric("Weather", weather_condition)
        result_four.metric("Traffic", traffic_level(traffic)[0])
        details = pd.DataFrame({"Journey detail": ["Service", "Date & time", "Temperature", "Holiday", "Route name"], "Selected value": [selected_route["route_id"], f"{journey_date.strftime('%d %b %Y')} · {journey_time.strftime('%H:%M')}", f"{temperature:.1f} °C", holiday_option, selected_route["route_name"]]})
        with st.expander("View estimate details"):
            st.dataframe(details, hide_index=True, use_container_width=True)
    except Exception as error:
        st.error("The estimate could not be calculated. Please review the selected journey and try again.")
        st.exception(error)

st.markdown('<p class="footer-note">This project uses simulated training delay data and verified route information. Estimates are not official ASTC real-time predictions.</p>', unsafe_allow_html=True)
