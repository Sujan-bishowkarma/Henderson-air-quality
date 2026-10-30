# Henderson Air Quality Dashboard (Streamlit) — revised prediction tables
# Reads data from the Supabase cloud database through its REST API.
# Run with:  streamlit run app.py

import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

st.set_page_config(page_title="Henderson Air Quality", layout="wide")

# monitoring site location (APPROXIMATE: Henderson, Auckland - replace with the exact site coordinates)
SITE_LAT, SITE_LON = -36.8786, 174.6302


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
def load_table(table, start, end, pollutant=None, horizon=None):
    sb = get_client()
    rows, page, first = [], 1000, 0
    while True:
        query = (sb.table(table).select("*")
                   .gte("datetime", start.isoformat())
                   .lt("datetime", end.isoformat()))
        if pollutant:
            query = query.eq("pollutant", pollutant)
        if horizon:
            query = query.eq("horizon_hours", horizon)
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
    value=(date(2023, 1, 23), date(2023, 1, 31)),
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


# ---------- wind direction polar plots ----------

st.subheader("Wind direction and pollution")
st.caption("Wind direction was only recorded from September 2022 to February 2023, so these plots always use that period. "
           "Angle = direction the wind blows from (north at the top), distance from centre = wind speed, colour = average concentration.")

WD_START, WD_END = date(2022, 9, 1), date(2023, 3, 1)
SPEED_EDGES = [0, 1, 2, 3, 4, 6, 11]
DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

wd = load_table("wind_direction", WD_START, WD_END)
if wd.empty:
    st.info("No wind direction data found in the database.")
else:
    aq = load_table("air_quality", WD_START, WD_END)
    wind = aq.merge(wd, on="datetime").dropna(subset=["wd", "ws"])

    # 16 direction sectors of 22.5 degrees, and wind speed bands
    wind["sector"] = (((wind["wd"] + 11.25) // 22.5) % 16).astype(int)
    wind["band"] = pd.cut(wind["ws"], SPEED_EDGES, labels=False, include_lowest=True, right=False)

    def polar_plot(col, name):
        g = (wind.dropna(subset=[col, "band"])
                 .groupby(["sector", "band"])[col].agg(["mean", "count"]).reset_index())
        g = g[g["count"] >= 3]                       # skip cells with too few hours
        lo = [SPEED_EDGES[int(b)] for b in g["band"]]
        hi = [SPEED_EDGES[int(b) + 1] for b in g["band"]]
        fig = go.Figure(go.Barpolar(
            theta=g["sector"] * 22.5,
            r=[h - l for h, l in zip(hi, lo)],
            base=lo,
            width=22.5,
            marker=dict(color=g["mean"], colorscale="YlOrRd", cmin=0,
                        colorbar=dict(title="µg/m³"), line=dict(color="white", width=0.5)),
            customdata=list(zip([DIRS[int(x)] for x in g["sector"]], lo, hi, g["mean"], g["count"])),
            hovertemplate="%{customdata[0]}, %{customdata[1]}–%{customdata[2]} m/s<br>"
                          "Average: %{customdata[3]:.1f} µg/m³<br>Hours: %{customdata[4]}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{name} by wind direction and speed",
            height=460, margin=dict(t=60, b=30, l=30, r=30),
            polar=dict(angularaxis=dict(direction="clockwise", rotation=90,
                                        tickmode="array", tickvals=[0, 90, 180, 270],
                                        ticktext=["N", "E", "S", "W"]),
                       radialaxis=dict(range=[0, SPEED_EDGES[-1]], ticksuffix=" m/s",
                                       tickvals=[2, 4, 6, 8, 10], angle=45, tickangle=45,
                                       tickfont=dict(size=10))),
        )
        return fig

    # site map + polar plot of the selected pollutant only
    c_map, c_polar = st.columns([1, 1.6])
    with c_map:
        st.markdown("**Monitoring site**")
        # newer Plotly versions use Scattermap; older ones only have Scattermapbox
        marker_args = dict(lat=[SITE_LAT], lon=[SITE_LON], mode="markers+text",
                           marker=dict(size=16, color="#D9692B"),
                           text=["Henderson monitoring site"], textposition="top center",
                           hovertext=["Henderson monitoring site (HEN)"], hoverinfo="text")
        map_settings = dict(style="open-street-map", center=dict(lat=SITE_LAT, lon=SITE_LON), zoom=11)
        if hasattr(go, "Scattermap"):
            site_map = go.Figure(go.Scattermap(**marker_args))
            site_map.update_layout(map=map_settings)
        else:
            site_map = go.Figure(go.Scattermapbox(**marker_args))
            site_map.update_layout(mapbox=map_settings)
        site_map.update_layout(height=380, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(site_map, width="stretch")
        st.caption(f"Henderson, Auckland · {len(wind):,} hours with wind direction")
    c_polar.plotly_chart(polar_plot(col, "NO₂" if pollutant == "NO2" else "PM10"), width="stretch")

# ---------- revised validation-selected model results (held-out test period) ----------

st.subheader("Forecast: actual vs predicted")

# PM10 also has averaged forecasts (tutor feedback): averaging removes hourly instrument noise
horizon = 1
if pollutant == "PM10":
    horizon = st.radio("PM10 forecast", [3, 1, 24], horizontal=True,
                       format_func=lambda h: {1: "Next hour", 3: "Next 3-hour average (main)",
                                              24: "Next 24-hour average"}[h])

p_start, p_end = max(start, date(2019, 9, 24)), min(end_excl, date(2023, 3, 1))

if p_start >= p_end:
    st.info("Revised model predictions cover 24 September 2019 to February 2023. Choose dates in that period to see them.")
else:
    if horizon == 1:
        pred = load_table("predictions", p_start, p_end, pollutant=pollutant)
    else:
        pred = load_table("predictions_pm10_avg", p_start, p_end, horizon=horizon)
        st.caption(f"Date = when the forecast is made. Actual = the real average PM10 over the following {horizon} hours.")
    if pred.empty:
        st.info("No predictions for this period.")
    else:
        model_name = pred["model"].iloc[0]
        # Metrics for the selected date range, not necessarily the full test set.
        pred = pred.copy()
        pred["actual"] = pd.to_numeric(pred["actual"], errors="coerce")
        pred["predicted"] = pd.to_numeric(pred["predicted"], errors="coerce")
        pred = pred.dropna(subset=["actual", "predicted"])
        if pred.empty:
            st.info("No valid actual/predicted pairs for the chosen dates.")
            st.stop()
        err = pred["predicted"] - pred["actual"]
        mae = err.abs().mean()
        rmse = np.sqrt((err ** 2).mean())
        denom = ((pred["actual"] - pred["actual"].mean()) ** 2).sum()
        r2 = 1 - (err ** 2).sum() / denom if denom > 0 else float("nan")

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
        st.caption("These metrics are recalculated for the selected dates, not the full held-out test set. Zoom in to inspect hourly forecasts.")


# ---------- historical what-if forecasting using validated model packages ----------

st.subheader(f"Historical what-if {pollutant} forecast")
st.caption(
    "Historical simulation only — not a real-time monitoring feed. "
    "Forecasts below use locally saved, validation-selected models. "
    "The preceding charts display published Supabase test-set predictions."
)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILES = {
    "NO2_next_hour": "best_model_NO2.pkl",
    "PM10_next_hour": "best_model_PM10.pkl",
    "PM10_next_3h_average": "pm10_3h_model.pkl",
    "PM10_next_24h_average": "pm10_24h_model.pkl",
}

@st.cache_resource
def load_validated_model(task):
    package = joblib.load(MODEL_DIR / MODEL_FILES[task])
    if not isinstance(package, dict) or not all(k in package for k in ("model", "features", "name")):
        raise ValueError("Model file must contain model, features, and name")
    if package["name"] != "LightGBM":
        raise ValueError("Expected the final validated LightGBM model")
    return package


def make_model_row(past, now, pollutant, weather):
    """Use measurements at/before now only, with the training feature definitions."""
    s = past[pollutant.lower()]
    row = {
        "hour": now.hour,
        "month": now.month,
        "dayofweek": now.weekday(),
        "is_weekend": int(now.weekday() >= 5),
        "Temp": weather["Temp"],
        "RH": weather["RH"],
        "WS": weather["WS"],
    }
    if pollutant == "PM10":
        row["fireworks_days"] = int(now.month == 11 and now.day in (4, 5, 6))
        row["era_2017"] = int(now.year >= 2017)
    for lag in (0, 1, 2, 3, 24):
        row[f"{pollutant}_lag{lag}"] = s.get(now - timedelta(hours=lag), np.nan)
    last3 = s.loc[now - timedelta(hours=2): now]
    last24 = s.loc[now - timedelta(hours=23): now]
    if pollutant == "PM10":
        row["PM10_roll3"] = last3.mean() if last3.notna().sum() >= 2 else np.nan
        row["PM10_roll24"] = last24.mean() if last24.notna().sum() >= 18 else np.nan
        row["PM10_std24"] = last24.std() if last24.notna().sum() >= 18 else np.nan
        other = past["no2"]
        other24 = other.loc[now - timedelta(hours=23): now]
        row["NO2_lag0"] = other.get(now, np.nan)
        row["NO2_roll24"] = other24.mean() if other24.notna().sum() >= 18 else np.nan
    else:
        row["NO2_roll3"] = last3.mean() if last3.notna().sum() == 3 else np.nan
        row["NO2_roll24"] = last24.mean() if last24.notna().sum() == 24 else np.nan
    return pd.DataFrame([row])


pick1, pick2, pick3 = st.columns(3)
chosen_day = pick1.date_input(
    "Forecast date", value=date(2023, 1, 25),
    min_value=date(2006, 1, 2), max_value=date(2023, 2, 27), key="forecast_date"
)
chosen_hour = pick2.selectbox("Forecast origin hour", list(range(24)), index=12,
                              format_func=lambda h: f"{h:02d}:00")
if pollutant == "NO2":
    task = "NO2_next_hour"
    pick3.caption("Forecast: next hour")
else:
    chosen_horizon = pick3.selectbox("Forecast horizon", [1, 3, 24], index=1,
                                     format_func=lambda h: {1: "Next hour", 3: "Next 3-hour average", 24: "Next 24-hour average"}[h])
    task = {1: "PM10_next_hour", 3: "PM10_next_3h_average", 24: "PM10_next_24h_average"}[chosen_horizon]

now = datetime.combine(chosen_day, time(chosen_hour))
forecast_hours = 1 if pollutant == "NO2" else chosen_horizon

# Include preceding measurements for lag features, and subsequent measurements
# solely to show the actual outcome AFTER the prediction has been calculated.
window = load_table("air_quality", now - timedelta(hours=30),
                    now + timedelta(hours=forecast_hours + 1))
if window.empty:
    st.info("No observations available near this date.")
else:
    window = window.set_index("datetime").sort_index()
    past = window.loc[:now].copy()  # never use measurements later than now as features
    required = ["temp", "rh", "ws", pollutant.lower()]
    if now not in past.index or past.loc[now, required].isna().any():
        st.info("Current-hour weather or pollutant data is missing. Select another hour.")
    else:
        with st.expander("What if the weather were different? (optional)"):
            a, b, c = st.columns(3)
            temp = a.slider("Temperature (°C)", -10.0, 40.0, float(past.loc[now, "temp"]), 0.5)
            rh = b.slider("Humidity (%)", 0.0, 100.0, float(past.loc[now, "rh"]), 1.0)
            ws = c.slider("Wind speed (m/s)", 0.0, 25.0, float(past.loc[now, "ws"]), 0.1)
        try:
            package = load_validated_model(task)
            X = make_model_row(past, now, pollutant, {"Temp": temp, "RH": rh, "WS": ws})
            features = package["features"]
            missing = [f for f in features if f not in X.columns]
            if missing:
                st.error(f"Model and dashboard feature definitions differ: {missing}")
            elif X[features].isna().any(axis=None):
                st.info("Some required past-hour features are missing; select another hour.")
            else:
                estimate = max(0.0, float(package["model"].predict(X[features])[0]))
                st.metric(f"Predicted {pollutant} ({forecast_hours}h)", f"{estimate:.2f} µg/m³")
                future = window.loc[now + timedelta(hours=1): now + timedelta(hours=forecast_hours), pollutant.lower()]
                if len(future) == forecast_hours and future.notna().all():
                    observed = float(future.mean())
                    st.caption(f"Observed following {forecast_hours}h: {observed:.2f} µg/m³")
                else:
                    st.caption("Actual outcome unavailable or incomplete for this period.")
                st.caption(f"Saved model: {package['name']} · {len(features)} input features")
        except (FileNotFoundError, ValueError, KeyError, AttributeError, ImportError) as exc:
            st.error(f"Cannot load or evaluate the updated model: {exc}")
            st.info("Check that all four revised .pkl files are in the models folder and LightGBM is installed.")

st.caption(
    "Henderson historical measurements · Supabase REST API (read-only). "
    "Interactive forecasts use historical stored inputs, which should be checked against "
    "the final forward-only preprocessing rules before claiming exact reproduction "
    "of the notebook's held-out scores. Monitoring site map location is approximate."
)
