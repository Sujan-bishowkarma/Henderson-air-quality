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

st.subheader(f"Historical {pollutant} forecast demonstration")
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
    """Read persisted feature order across sklearn/LightGBM versions."""
    for attr in ("feature_names_in_", "feature_name_"):
        try:
            names = getattr(model, attr, None)
            if names is not None and len(names):
                return list(names)
        except (AttributeError, TypeError, ValueError):
            pass
    booster = getattr(model, "booster_", None)
    if booster is not None and hasattr(booster, "feature_name"):
        return list(booster.feature_name())
    raise ValueError("Saved model does not expose the training feature names. Re-export it from the notebook.")


def predict_saved_model(model, X):
    """Predict using saved model; fall back to LightGBM Booster on wrapper compatibility errors."""
    try:
        return float(np.asarray(model.predict(X)).reshape(-1)[0])
    except AttributeError as exc:
        booster = getattr(model, "booster_", None)
        if booster is None or not hasattr(booster, "predict"):
            raise ValueError(
                "The saved model cannot be used with the deployed library versions. "
                "Re-save the fitted model using matching versions. "
                f"Original error: {exc}"
            ) from exc
        try:
            return float(np.asarray(booster.predict(X)).reshape(-1)[0])
        except Exception as inner:
            raise ValueError(f"LightGBM Booster prediction failed: {inner}") from inner

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
        try:
            model = load_model(pollutant)
            required = feature_names(model)
            missing = [c for c in required if c not in X.columns]
            if missing:
                st.error("Missing model features: " + ", ".join(missing))
                return
            pred = max(predict_saved_model(model, X[required]), 0.0)
        except (AttributeError, ValueError, KeyError, FileNotFoundError, ImportError) as exc:
            st.error(f"Unable to calculate {pollutant} next-hour forecast: {exc}")
            st.info("Check the deployed model file and pin the same LightGBM/scikit-learn versions used when saving it.")
            return
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
        try:
            saved = load_avg_model(3)
        except (AttributeError, ValueError, FileNotFoundError, ImportError) as exc:
            st.error(f"Unable to load PM10 three-hour model: {exc}")
            return
        if saved is None:
            st.info("Add models/pm10_3h_model.pkl (saved by the modelling notebook) to show this forecast.")
            return
        X = build_features_avg(window, now, temp, rh, ws)
        if pd.isna(X["PM10_lag0"].iloc[0]):
            st.info("PM10 is missing at this hour, so the forecast can't be made.")
            return
        try:
            pred = max(predict_saved_model(saved["model"], X[saved["features"]]), 0.0)
        except (AttributeError, ValueError, KeyError) as exc:
            st.error(f"Unable to calculate PM10 three-hour forecast: {exc}")
            return
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

# ---------- end-of-dataset future forecasts: next hour / next 24h ----------
# A forward-only demonstration: all model inputs are sourced at or before the
# latest available air-quality timestamp. This is NOT a real-time 2026 forecast.

st.divider()
st.header("Future Air Quality Forecast")
st.caption(
    "Forecasts beyond the latest stored observation, using saved LightGBM models. "
    "If the database ends in February 2023, these are forecasts from that historical "
    "endpoint—not today's Auckland air quality."
)

@st.cache_data(ttl=3600)
def get_latest_observation_time():
    result = (get_client().table("air_quality")
              .select("datetime")
              .order("datetime", desc=True)
              .limit(1).execute())
    records = result.data or []
    return pd.Timestamp(records[0]["datetime"]) if records else None


def safely_predict_forward(model, required_features, values):
    """Ensure inference uses complete historical features in training order."""
    missing = [f for f in required_features if f not in values.columns]
    if missing:
        raise ValueError("Model input columns missing: " + ", ".join(missing))
    X = values.loc[:, required_features]
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Historical inputs contain missing/non-finite features.")
    return max(0.0, predict_saved_model(model, X))


def future_history_chart(history, origin, target, prediction, hours):
    """Screenshot-style graph: blue history and orange forecast star.

    For 24h, the star represents ONE mean across the shaded 24h period,
    not a sequence of hourly values. No unavailable future actual is plotted.
    """
    past = history[target.lower()].loc[:origin].tail(25).dropna()
    first = origin + timedelta(hours=1)
    end = origin + timedelta(hours=hours)
    # Represent the 24h average at its period-end timestamp, not every hour.
    point_time = first if hours == 1 else end
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[point_time], y=[prediction], mode="markers",
        name="Predicted", marker=dict(color="#FF9900", size=15, symbol="star"),
        hovertemplate=("Predicted: %{y:.2f} µg/m³<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=past.index, y=past.values, mode="lines+markers", name="Actual (historical)",
        line=dict(color="#0866D8", width=2),
        marker=dict(color="#0866D8", size=5),
        hovertemplate="%{x|%d %b %Y %H:%M}<br>Observed: %{y:.2f} µg/m³<extra></extra>",
    ))
    if hours == 24:
        fig.add_vrect(
            x0=first, x1=end, fillcolor="#FFB347", opacity=0.14,
            line_width=0,
            annotation_text="Next 24h: mean over shaded period",
            annotation_position="top left",
        )
    fig.add_vline(x=origin.timestamp() * 1000, line_color="#777777", line_dash="dot")
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title=f"{target} (µg/m³)",
        height=350,
        hovermode="closest",
        legend=dict(orientation="h", y=1.15, x=0),
        margin=dict(t=35, b=45, l=50, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E8EEF4", zeroline=False)
    return fig


def render_forecast_result(history, origin, target, prediction, hours):
    """Screenshot-style summary (date, predicted value, delta, and graph)."""
    start_future = origin + timedelta(hours=1)
    end_future = origin + timedelta(hours=hours)
    if hours == 1:
        st.markdown(f"**Next hour: {start_future:%d %b %Y %H:%M}**")
    else:
        st.markdown(
            f"**Next 24-hour average: {start_future:%d %b %Y %H:%M} "
            f"to {end_future:%d %b %Y %H:%M}**"
        )
    current_value = history[target.lower()].loc[:origin].dropna()
    latest = float(current_value.iloc[-1]) if not current_value.empty else None
    delta = (f"{prediction - latest:+.1f} vs last observed hour"
             if latest is not None else None)
    st.metric("Predicted" if hours == 1 else "Predicted 24h average",
              f"{prediction:.2f} µg/m³", delta=delta,
              delta_color="off" if hours == 24 else "normal")
    if latest is not None:
        st.caption(f"Last observed {target}: {latest:.2f} µg/m³ · "
                   "No future actual measurement is available for comparison.")
    else:
        st.caption("No future actual measurement is available for comparison.")
    st.plotly_chart(
        future_history_chart(history, origin, target, prediction, hours),
        width="stretch",
    )
    if hours == 24:
        st.caption(
            "The orange star is the predicted **average** across the shaded "
            "24-hour interval; it is not an individual hourly prediction."
        )
    st.caption("Blue line = previously observed readings · Orange star = future forecast · "
               "Dotted line = forecast origin")

try:
    last_stored = get_latest_observation_time()
    if last_stored is None:
        st.warning("No observations were found in the database.")
    else:
        forecast_origin = last_stored.to_pydatetime()
        # Only query records through the origin (exclusive upper bound).
        lookback = load_table(
            "air_quality",
            forecast_origin - timedelta(hours=48),
            forecast_origin + timedelta(microseconds=1),
        )
        if lookback.empty:
            st.warning("Not enough historical data to generate a future forecast.")
        else:
            lookback = lookback.set_index("datetime").sort_index()
            lookback = lookback.loc[lookback.index <= last_stored]
            origin = lookback.index.max().to_pydatetime()
            st.write(f"**Forecast starts after:** {origin:%d %B %Y at %H:%M}")
            if origin != forecast_origin:
                st.warning("Latest observation could not be recovered from the historical window.")
            else:
                weather_at_origin = lookback.loc[last_stored]
                if isinstance(weather_at_origin, pd.DataFrame):
                    weather_at_origin = weather_at_origin.iloc[-1]
                required_weather = ("temp", "rh", "ws")
                if any(v not in weather_at_origin.index or pd.isna(weather_at_origin[v])
                       for v in required_weather):
                    st.warning("Latest weather inputs are incomplete; forecasts cannot run safely.")
                else:
                    origin_weather = tuple(float(weather_at_origin[v]) for v in required_weather)
                    tab_hour, tab_day = st.tabs(["Next 1 Hour", "Next 24 Hours"])

                    with tab_hour:
                        st.markdown("**One-hour-ahead pollutant concentration**")
                        c_pm10, c_no2 = st.columns(2)
                        for holder, target in [(c_pm10, "PM10"), (c_no2, "NO2")]:
                            with holder:
                                st.markdown(f"#### {target}")
                                try:
                                    path = MODEL_DIR / f"best_model_{target}.pkl"
                                    if not path.is_file():
                                        raise FileNotFoundError(f"Missing {path.name}")
                                    fitted = load_model(target)
                                    values = build_features(lookback, target, origin, *origin_weather)
                                    prediction = safely_predict_forward(
                                        fitted, feature_names(fitted), values
                                    )
                                    render_forecast_result(lookback, origin, target, prediction, 1)
                                except (ValueError, KeyError, AttributeError, FileNotFoundError) as exc:
                                    st.warning(f"Cannot calculate {target} forecast: {exc}")

                    with tab_day:
                        st.markdown("**PM10 average concentration over the next 24 hours**")
                        st.caption(
                            "A 24-hour mean is different from a 24-hour sequence of "
                            "individual hourly forecasts. No saved 24-hour NO₂ model "
                            "has been confirmed in this project."
                        )
                        try:
                            path = MODEL_DIR / "pm10_24h_model.pkl"
                            if not path.is_file():
                                raise FileNotFoundError(f"Missing {path.name}")
                            saved = load_avg_model(24)
                            if not (isinstance(saved, dict)
                                    and "model" in saved and "features" in saved):
                                raise ValueError(
                                    "24-hour model must contain 'model' and 'features'."
                                )
                            values = build_features_avg(lookback, origin, *origin_weather)
                            prediction = safely_predict_forward(
                                saved["model"], list(saved["features"]), values
                            )
                            render_forecast_result(lookback, origin, "PM10", prediction, 24)
                        except (ValueError, KeyError, AttributeError, FileNotFoundError) as exc:
                            st.warning(f"Cannot calculate 24-hour PM10 forecast: {exc}")

                    st.caption(
                        "Both tabs use weather measured at the forecast origin, not a "
                        "future weather forecast. Forecast values are not verified here "
                        "against the final training notebook. The results are an academic "
                        "end-of-data demonstration, not official public-health warnings."
                    )
except Exception as exc:
    st.error("Future forecasting cannot access the cloud database. Check Supabase settings.")
    st.caption(f"Technical details: {type(exc).__name__}: {exc}")

st.caption("Data: Henderson air quality monitoring site. Supabase REST API (read-only).")
