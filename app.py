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

# ---------- model results (test set: last 20% of hours) ----------

st.subheader("Forecast: actual vs predicted")

# PM10 also has averaged forecasts (tutor feedback): averaging removes hourly instrument noise
horizon = 1
if pollutant == "PM10":
    horizon = st.radio("PM10 forecast", [3, 1, 24], horizontal=True,
                       format_func=lambda h: {1: "Next hour", 3: "Next 3-hour average (main)",
                                              24: "Next 24-hour average"}[h])

p_start, p_end = max(start, date(2017, 7, 15)), min(end_excl, date(2023, 3, 1))

if p_start >= p_end:
    st.info("Model predictions cover the test set (mid-2017/2019 to February 2023). Choose dates in that period to see them.")
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


# ---------- live prediction (only the selected pollutant) ----------

st.subheader(f"Live {pollutant} prediction")
st.caption("Pick a date and hour. The app reads the previous 24 hours from the cloud database, "
           "builds the same features used in training, and predicts with the saved best model.")

MODEL_DIR = Path(__file__).parent / "models"

@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / f"best_model_{name}.pkl")

@st.cache_resource
def load_avg_model(hours):
    path = MODEL_DIR / f"pm10_{hours}h_model.pkl"
    return joblib.load(path) if path.exists() else None

def feature_names(model):
    # scikit-learn and LightGBM store the training column names differently
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(model.feature_name_)

def build_features(window, t, now, temp, rh, ws):
    # same features as make_features() in the modelling notebook (next hour)
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

def build_features_avg(window, now, temp, rh, ws):
    # same features as make_pm10_avg() in the modelling notebook (3-hour average)
    s, n = window["pm10"], window["no2"]
    last24 = s.loc[now - timedelta(hours=23): now]
    no2_24 = n.loc[now - timedelta(hours=23): now]
    last3 = s.loc[now - timedelta(hours=2): now]
    row = {
        "hour": now.hour,
        "month": now.month,
        "dayofweek": now.weekday(),
        "is_weekend": int(now.weekday() >= 5),
        "fireworks_days": int(now.month == 11 and now.day in (4, 5, 6)),
        "era_2017": int(now.year >= 2017),
        "Temp": temp, "RH": rh, "WS": ws,
        "PM10_lag0": s.get(now),
        "PM10_lag1": s.get(now - timedelta(hours=1)),
        "PM10_lag2": s.get(now - timedelta(hours=2)),
        "PM10_lag3": s.get(now - timedelta(hours=3)),
        "PM10_lag24": s.get(now - timedelta(hours=24)),
        "PM10_roll3": last3.mean() if last3.notna().sum() >= 2 else np.nan,
        "PM10_roll24": last24.mean() if last24.notna().sum() >= 18 else np.nan,
        "PM10_std24": last24.std() if last24.notna().sum() >= 18 else np.nan,
        "NO2_lag0": n.get(now),
        "NO2_roll24": no2_24.mean() if no2_24.notna().sum() >= 18 else np.nan,
    }
    return pd.DataFrame([row]).astype(float)

def forecast_chart(recent, x_pred, pred, x_actual=None, actual=None, label="Predicted"):
    fig = px.line(x=recent.index, y=recent.values, markers=True, labels={"x": "", "y": f"{pollutant} (µg/m³)"})
    fig.add_scatter(x=x_pred, y=[pred] * len(x_pred), mode="lines+markers", name=label,
                    line=dict(color="orange", width=3), marker=dict(size=10, color="orange", symbol="star"))
    if actual is not None and pd.notna(actual):
        fig.add_scatter(x=x_actual, y=[actual] * len(x_actual), mode="lines+markers", name="Actual",
                        line=dict(color="blue", width=3, dash="dot"), marker=dict(size=8, color="blue"))
    fig.update_layout(height=300, showlegend=True, margin=dict(t=10),
                      legend=dict(orientation="h", y=1.1))
    return fig

l1, l2 = st.columns(2)
pick_day = l1.date_input("Date", value=date(2019, 7, 10),
                         min_value=date(2006, 1, 2), max_value=date(2023, 2, 28), key="live_day")
pick_hour = l2.selectbox("Current hour", list(range(24)), index=17,
                         format_func=lambda h: f"{h:02d}:00", key="live_hour")
now = datetime.combine(pick_day, time(pick_hour))
nxt = now + timedelta(hours=1)

window = load_table("air_quality", now - timedelta(hours=24), now + timedelta(hours=4))
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

series = window[col]
recent = series.loc[:now]
now_val = series.get(now)

# next hour (both pollutants)
def next_hour_block(box):
    with box:
        st.markdown(f"**Next hour: {nxt:%d %b %Y %H:00}**")
        X = build_features(window, pollutant, now, temp, rh, ws)
        if X.isna().any(axis=None):
            st.info("Not enough past data (gaps in the previous 24 hours) to predict.")
            return
        model = load_model(pollutant)
        pred = max(float(model.predict(X[feature_names(model)])[0]), 0)
        actual = series.get(nxt)
        st.metric("Predicted", f"{pred:.1f} µg/m³", f"{pred - now_val:+.1f} vs now")
        if actual is not None and pd.notna(actual):
            st.caption(f"Now: {now_val:.1f} · Actual next hour: {actual:.1f} µg/m³")
        st.plotly_chart(forecast_chart(recent, [nxt], pred, [nxt], actual), width="stretch")

# next 3-hour average (PM10 only: the main PM10 forecast)
def three_hour_block(box):
    with box:
        h1, h3 = now + timedelta(hours=1), now + timedelta(hours=3)
        st.markdown(f"**Next 3-hour average: {h1:%H:00}–{h3:%H:00} (main)**")
        saved = load_avg_model(3)
        if saved is None:
            st.info("Add models/pm10_3h_model.pkl (saved by the modelling notebook) to show this forecast.")
            return
        X = build_features_avg(window, now, temp, rh, ws)
        if pd.isna(X["PM10_lag0"].iloc[0]):
            st.info("PM10 is missing at this hour, so the forecast can't be made.")
            return
        pred = max(float(saved["model"].predict(X[saved["features"]])[0]), 0)
        next3 = series.loc[h1:h3]
        actual = next3.mean() if next3.notna().sum() >= 2 else None
        st.metric("Predicted average", f"{pred:.1f} µg/m³", f"{pred - now_val:+.1f} vs now")
        if actual is not None:
            st.caption(f"Now: {now_val:.1f} · Actual 3-hour average: {actual:.1f} µg/m³")
        hours = [h1, now + timedelta(hours=2), h3]
        st.plotly_chart(forecast_chart(recent, hours, pred, hours, actual, label="Predicted average"),
                        width="stretch")
        st.caption(f"Model: {saved['name']}")

if now_val is None or pd.isna(now_val):
    st.info(f"{pollutant} is missing at this hour. Try another hour.")
elif pollutant == "PM10":
    c1, c2 = st.columns(2)
    three_hour_block(c1)
    next_hour_block(c2)
    st.caption("Hourly PM10 is noisy, so the 3-hour average is the main PM10 forecast. "
               "The next-hour forecast is shown for comparison.")
else:
    next_hour_block(st.container())

st.caption("Data: Henderson air quality monitoring site. Stored in Supabase (Sydney), read-only API access.")
