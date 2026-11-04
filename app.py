
import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Normalization used in the original dataset (hour.csv)
TEMP_MAX, ATEMP_MAX, HUM_MAX, WIND_MAX = 41.0, 50.0, 100.0, 67.0
WEATHER = {
    "Clear / few clouds": 1,
    "Mist / cloudy": 2,
    "Light rain or snow": 3,
    "Heavy rain / storm": 3,   # merged into 3: only 3 such hours existed in training
}
PEAK_HOURS = [7, 8, 17, 18]


# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    model = joblib.load("bike_demand_model.joblib")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    return model, meta


def build_features(meta, date, hours, holiday, weathersit, temp_c, feels_c, hum_pct, wind_kmh):
    """Apply exactly the same preprocessing as in training, for one or many hours."""
    pre = meta["preprocessing"]
    weekend = date.weekday() >= 5
    hum = hum_pct / HUM_MAX
    if hum == 0:
        hum = pre["hum_zero_fill"]
    rows = [{
        "yr": 0 if date.year <= 2011 else 1,
        "mnth": date.month,
        "hr": h,
        "holiday": int(holiday),
        "workingday": int(not weekend and not holiday),
        "weathersit": weathersit,
        "hum": hum,
        "windspeed": wind_kmh / WIND_MAX,
        "temp_avg": (temp_c / TEMP_MAX + feels_c / ATEMP_MAX) / 2,
        "low_demand_hour": int(h in pre["low_demand_hours"]),
    } for h in hours]
    return pd.DataFrame(rows)[meta["features_in_order"]]


def predict(model, X):
    return np.clip(model.predict(X), 0, None).round().astype(int)


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Bike Demand Planner", page_icon="🚲", layout="wide")
st.title("🚲 Bike Demand Planner")
st.caption("Estimate how many bikes riders will need, hour by hour, from the date and the weather forecast.")

try:
    model, meta = load_model()
except FileNotFoundError:
    st.error("Model files not found. Put `bike_demand_model.joblib` and `model_metadata.json` next to app.py.")
    st.stop()

with st.sidebar:
    st.header("When")
    date = st.date_input("Date", dt.date.today())
    hour = st.slider("Hour of day", 0, 23, 8, format="%d:00")
    holiday = st.checkbox("Public holiday")

    st.header("Weather forecast")
    weather_label = st.selectbox("Conditions", list(WEATHER))
    temp_c = st.slider("Temperature (°C)", -8, 41, 22)
    feels_c = st.slider("Feels-like temperature (°C)", -16, 50, temp_c)
    hum_pct = st.slider("Humidity (%)", 0, 100, 60)
    wind_kmh = st.slider("Wind speed (km/h)", 0, 57, 12)

    st.header("Planning")
    buffer_pct = st.slider("Safety buffer (%)", 0, 50, 15,
                           help="The model tends to under-predict at peaks. "
                                "A buffer reduces the risk of riders finding no bike.")

weathersit = WEATHER[weather_label]
is_working = date.weekday() < 5 and not holiday
day_type = "Working day" if is_working else ("Holiday" if holiday else "Weekend")

if date.year > 2012:
    st.info("The model was trained on 2011–2012 data and treats later years as 2012. "
            "If ridership has grown since, real demand may be higher than shown.")

# ---- selected hour
X_hour = build_features(meta, date, [hour], holiday, weathersit, temp_c, feels_c, hum_pct, wind_kmh)
pred_hour = int(predict(model, X_hour)[0])
plan_hour = int(np.ceil(pred_hour * (1 + buffer_pct / 100)))

# ---- whole day, same weather
hours = list(range(24))
X_day = build_features(meta, date, hours, holiday, weathersit, temp_c, feels_c, hum_pct, wind_kmh)
day = pd.DataFrame({"Hour": hours, "Expected rentals": predict(model, X_day)})
day["With buffer"] = np.ceil(day["Expected rentals"] * (1 + buffer_pct / 100)).astype(int)
peak = day.loc[day["Expected rentals"].idxmax()]

st.subheader(f"{date:%A, %d %b %Y} · {hour:02d}:00 · {day_type}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected rentals this hour", f"{pred_hour:,}")
c2.metric(f"Bikes to have ready (+{buffer_pct}%)", f"{plan_hour:,}")
c3.metric("Busiest hour of the day", f"{int(peak.Hour):02d}:00", f"{int(peak['Expected rentals']):,} rentals",
          delta_color="off")
c4.metric("Total rentals for the day", f"{int(day['Expected rentals'].sum()):,}")

if is_working and hour in PEAK_HOURS:
    st.warning("Commute peak hour: demand is typically 2.6× higher than midday. "
               "Rebalance bikes to commuter stations before this hour.")
elif hour in meta["preprocessing"]["low_demand_hours"]:
    st.success("Low-demand night hour: a good window to move bikes and do maintenance.")

st.markdown("#### Expected demand across the day")
st.caption("Uses the same weather for every hour, so treat it as a planning shape, not an exact forecast.")
st.bar_chart(day.set_index("Hour")[["Expected rentals"]], height=320)

with st.expander("Hourly table"):
    st.dataframe(day, hide_index=True, use_container_width=True)
    st.download_button("Download as CSV", day.to_csv(index=False), file_name=f"demand_{date}.csv")

with st.expander("About the model"):
    m = meta["test_metrics_before_retrain"]
    st.markdown(
        f"- **Model:** {meta['model']}\n"
        f"- **Accuracy on unseen months:** average error ≈ {m['MAE']:.0f} rentals/hour, R² = {m['R2']:.2f}\n"
        "- **Biggest drivers:** hour of day, working day vs not, temperature\n"
        "- Predictions are *rentals*, which can be lower than true demand when no bike was available."
    )
