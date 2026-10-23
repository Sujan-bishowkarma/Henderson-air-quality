# Henderson Air Quality Dashboard (Streamlit)
# Reads data from the Supabase cloud database through its REST API.
# Run with:  streamlit run app.py

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

st.set_page_config(page_title="Henderson Air Quality", layout="wide")


# ---------- connection ----------

# keys come from .env on your computer, or from Streamlit secrets when deployed online
load_dotenv()

def get_secret(name):
    value = os.getenv(name)
    if value is None:
        value = st.secrets[name]
    return value

@st.cache_resource
def get_client():
    return create_client(get_secret("SUPABASE_URL"), get_secret("SUPABASE_KEY"))


# ---------- load data through the API ----------

# the API returns max 1000 rows per request, so read in pages of 1000
@st.cache_data(ttl=3600)
def load_table(table, start, end, pollutant=None):
    sb = get_client()
    rows, page, first = [], 1000, 0
    while True:
        query = (sb.table(table).select("*")
                   .gte("datetime", start.isoformat())
                   .lt("datetime", end.isoformat()))
        if pollutant:
            query = query.eq("pollutant", pollutant)
        batch = query.order("datetime").range(first, first + page - 1).execute().data
        rows += batch
        if len(batch) < page:
            break
        first += page
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ---------- sidebar: user choices ----------

st.sidebar.title("Settings")
pollutant = st.sidebar.radio("Pollutant", ["PM10", "NO2"])
col = pollutant.lower()

dates = st.sidebar.date_input(
    "Date range",
    value=(date(2019, 1, 1), date(2019, 12, 31)),
    min_value=date(2006, 1, 1),
    max_value=date(2023, 2, 28),
)
if len(dates) != 2:
    st.info("Select a start and end date.")
    st.stop()
start, end = dates
end_excl = end + timedelta(days=1)

if (end - start).days > 3 * 365:
    st.sidebar.warning("Long ranges take longer to load from the API.")


# ---------- header ----------

st.title("Henderson Air Quality Dashboard")
st.caption(f"Hourly {pollutant} and weather at Henderson monitoring site · {start} to {end}")

with st.spinner("Loading data from the cloud database..."):
    df = load_table("air_quality", start, end_excl)

if df.empty:
    st.warning("No data for this period.")
    st.stop()

df = df.set_index("datetime").sort_index()


# ---------- key numbers ----------

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Average {pollutant}", f"{df[col].mean():.1f} µg/m³")
k2.metric(f"Highest hour", f"{df[col].max():.1f} µg/m³")
k3.metric("Average wind speed", f"{df['ws'].mean():.1f} m/s")
k4.metric("Hours with data", f"{df[col].notna().mean() * 100:.0f}%")


# ---------- trend over time ----------

st.subheader(f"{pollutant} over time")

# daily averages for long periods, hourly for short ones
if (end - start).days > 60:
    trend = df[col].resample("D").mean()
    label = "Daily average"
else:
    trend = df[col]
    label = "Hourly"

fig = px.line(trend, labels={"value": f"{pollutant} (µg/m³)", "datetime": "Date"})
fig.update_layout(showlegend=False, title=label, height=350)
st.plotly_chart(fig, width="stretch")


# ---------- patterns ----------

st.subheader("Patterns")
p1, p2 = st.columns(2)

by_hour = df.groupby(df.index.hour)[col].mean()
fig = px.line(x=by_hour.index, y=by_hour.values, markers=True,
              labels={"x": "Hour of day", "y": f"{pollutant} (µg/m³)"},
              title="Average by hour of day")
p1.plotly_chart(fig, width="stretch")

bins = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 11]
names = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", "4-5", "5-6", "6+"]
by_wind = df.groupby(pd.cut(df["ws"], bins=bins, labels=names, include_lowest=True),
                     observed=True)[col].mean()
fig = px.bar(x=by_wind.index.astype(str), y=by_wind.values,
             labels={"x": "Wind speed (m/s)", "y": f"{pollutant} (µg/m³)"},
             title="Average by wind speed")
p2.plotly_chart(fig, width="stretch")



# ---------- seasons ----------

st.subheader("Seasons")

# NZ seasons: summer = Dec-Feb, winter = Jun-Aug
season_map = {12: "Summer", 1: "Summer", 2: "Summer",
              3: "Autumn", 4: "Autumn", 5: "Autumn",
              6: "Winter", 7: "Winter", 8: "Winter",
              9: "Spring", 10: "Spring", 11: "Spring"}
order = ["Summer", "Autumn", "Winter", "Spring"]
df["season"] = df.index.month.map(season_map)

s1, s2 = st.columns(2)

# average per season
by_season = df.groupby("season")[col].mean().reindex(order).dropna()
fig = px.bar(x=by_season.index, y=by_season.values,
             color=by_season.index,
             color_discrete_map={"Summer": "#f4a259", "Autumn": "#bc4b51",
                                 "Winter": "#5b8e7d", "Spring": "#8cb369"},
             labels={"x": "Season", "y": f"{pollutant} (µg/m³)"},
             title="Average by season")
fig.update_layout(showlegend=False)
s1.plotly_chart(fig, width="stretch")

# summer vs winter: daily pattern
sw = df[df["season"].isin(["Summer", "Winter"])]
if sw["season"].nunique() < 2:
    s2.info("Choose a date range that includes both summer and winter (e.g. a full year) to compare them.")
else:
    profile = sw.groupby(["season", sw.index.hour])[col].mean().reset_index()
    profile.columns = ["Season", "Hour", col]
    fig = px.line(profile, x="Hour", y=col, color="Season", markers=True,
                  color_discrete_map={"Summer": "#f4a259", "Winter": "#5b8e7d"},
                  labels={col: f"{pollutant} (µg/m³)", "Hour": "Hour of day"},
                  title="Summer vs winter: average by hour of day")
    s2.plotly_chart(fig, width="stretch")

    summer = sw.loc[sw["season"] == "Summer", col].mean()
    winter = sw.loc[sw["season"] == "Winter", col].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Summer average", f"{summer:.1f} µg/m³")
    c2.metric("Winter average", f"{winter:.1f} µg/m³")
    c3.metric("Winter vs summer", f"{(winter / summer - 1) * 100:+.0f}%")

# ---------- model results (test set: last 20% of hours) ----------

st.subheader("Next-hour forecast: actual vs predicted")

p_start, p_end = max(start, date(2017, 7, 15)), min(end_excl, date(2023, 2, 1))

if p_start >= p_end:
    st.info("Model predictions cover the test set (mid-2017/2018 to January 2023). Choose dates in that period to see them.")
else:
    pred = load_table("predictions", p_start, p_end, pollutant=pollutant)
    if pred.empty:
        st.info("No predictions for this period.")
    else:
        model_name = pred["model"].iloc[0]
        err = pred["predicted"] - pred["actual"]
        mae = err.abs().mean()
        rmse = np.sqrt((err ** 2).mean())
        r2 = 1 - (err ** 2).sum() / ((pred["actual"] - pred["actual"].mean()) ** 2).sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", model_name)
        m2.metric("MAE", f"{mae:.2f}")
        m3.metric("RMSE", f"{rmse:.2f}")
        m4.metric("R²", f"{r2:.3f}")

        long = pred.melt(id_vars="datetime", value_vars=["actual", "predicted"],
                         var_name="Series", value_name=pollutant)
        fig = px.line(long, x="datetime", y=pollutant, color="Series",
                      color_discrete_map={"actual": "blue", "predicted": "orange"},
                      labels={pollutant: f"{pollutant} (µg/m³)", "datetime": "Date"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Metrics are for the selected period only. Tip: zoom in on the chart to see hourly detail.")


# ---------- live next-hour prediction ----------

st.subheader("Next-hour prediction")
st.caption("Pick a date and hour. The app reads the previous 24 hours from the cloud database, "
           "builds the same features used in training, and predicts the next hour with the saved best model.")

MODEL_DIR = Path(__file__).parent / "models"

@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / f"best_model_{name}.pkl")

def feature_names(model):
    # scikit-learn and LightGBM store the training column names differently
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(model.feature_name_)

def build_features(window, t, now, temp, rh, ws):
    # same features as make_features() in the modelling notebook
    s = window[t.lower()]
    row = {
        "hour": now.hour,
        "month": now.month,
        "dayofweek": now.weekday(),
        "is_weekend": int(now.weekday() >= 5),
        "Temp": temp,
        "RH": rh,
        "WS": ws,
        f"{t}_lag0": s.get(now),
        f"{t}_lag1": s.get(now - timedelta(hours=1)),
        f"{t}_lag2": s.get(now - timedelta(hours=2)),
        f"{t}_lag3": s.get(now - timedelta(hours=3)),
        f"{t}_lag24": s.get(now - timedelta(hours=24)),
        f"{t}_roll3": s.loc[now - timedelta(hours=2): now].mean(),
        f"{t}_roll24": s.loc[now - timedelta(hours=23): now].mean(),
    }
    return pd.DataFrame([row])

l1, l2 = st.columns(2)
pick_day = l1.date_input("Date", value=date(2019, 7, 10),
                         min_value=date(2006, 1, 2), max_value=date(2023, 2, 28), key="live_day")
pick_hour = l2.selectbox("Current hour", list(range(24)), index=17,
                         format_func=lambda h: f"{h:02d}:00", key="live_hour")
now = datetime.combine(pick_day, time(pick_hour))
nxt = now + timedelta(hours=1)

window = load_table("air_quality", now - timedelta(hours=24), nxt + timedelta(hours=1))
if window.empty:
    st.warning("No data around this time.")
    st.stop()
window = window.set_index("datetime").sort_index()

if now not in window.index or window.loc[now, ["temp", "rh", "ws"]].isna().any():
    st.warning("Weather data is missing at this hour. Try another date (2019 has complete data).")
    st.stop()

# what-if: start from the real weather, let the user change it
with st.expander("What if the weather were different? (optional)"):
    w1, w2, w3 = st.columns(3)
    temp = w1.slider("Temperature (°C)", -2.0, 32.0, float(window.loc[now, "temp"]), 0.5)
    rh = w2.slider("Humidity (%)", 10.0, 100.0, float(window.loc[now, "rh"]), 1.0)
    ws = w3.slider("Wind speed (m/s)", 0.0, 11.0, float(window.loc[now, "ws"]), 0.1)

results = []
for t in ["PM10", "NO2"]:
    X = build_features(window, t, now, temp, rh, ws)
    if X.isna().any(axis=None):
        results.append((t, None, None, X))
        continue
    model = load_model(t)
    pred = float(model.predict(X[feature_names(model)])[0])
    actual = window[t.lower()].get(nxt)
    results.append((t, max(pred, 0), actual, X))

cols = st.columns(2)
for c, (t, pred, actual, X) in zip(cols, results):
    with c:
        st.markdown(f"**{t} at {nxt:%d %b %Y %H:00}**")
        if pred is None:
            st.info("Not enough past data (gaps in the previous 24 hours) to predict.")
            continue
        now_val = X[f"{t}_lag0"].iloc[0]
        st.metric("Predicted", f"{pred:.1f} µg/m³", f"{pred - now_val:+.1f} vs now")
        if actual is not None and pd.notna(actual):
            st.caption(f"Now: {now_val:.1f} · Actual next hour: {actual:.1f} µg/m³")
        recent = window[t.lower()].loc[:now]
        fig = px.line(x=recent.index, y=recent.values, markers=True,
                      labels={"x": "", "y": f"{t} (µg/m³)"})
        fig.add_scatter(x=[nxt], y=[pred], mode="markers", name="Predicted",
                        marker=dict(size=12, color="orange", symbol="star"))
        if actual is not None and pd.notna(actual):
            fig.add_scatter(x=[nxt], y=[actual], mode="markers", name="Actual",
                            marker=dict(size=10, color="blue"))
        fig.update_layout(height=280, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

st.caption("Data: Henderson air quality monitoring site. Stored in Supabase (Sydney), read-only API access.")
