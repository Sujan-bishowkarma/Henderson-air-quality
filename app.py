# Henderson Air Quality Dashboard (Streamlit)
# Reads data from the Supabase cloud database through its REST API.
# Run with:  streamlit run app.py
# Deploy this file at the repository ROOT as app.py.
# IMPORTANT: This is an application, NOT a model-training notebook.
# Historical data is loaded from Supabase; do not add laptop-local CSV paths.
# To display the 24 hourly forecast line, deploy 46 files named
# direct_{PM10|NO2}_{2..24}h.pkl alongside the existing forward_* models.


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

# ---------- model results: six algorithms' historical held-out predictions ----------

st.subheader("Forecast: actual vs predicted")
st.caption(
    "Historical held-out test results (not future forecasts). "
    "Select an algorithm to compare its own predictions with observed values."
)

PRESENTATION_MODELS = (
    "Linear Regression", "Decision Tree", "Random Forest",
    "XGBoost", "LightGBM", "Prophet"
)
MODEL_CODES = {
    "Linear Regression": "linear_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Prophet": "prophet",
}

@st.cache_data(ttl=3600)
def load_six_model_predictions(task, method, start_time, end_time):
    """Fetch only the selected task/model/date range via the public read-only API.

    Supabase PostgREST paginates at 1,000 rows per page. Filtering on the server
    prevents fetching all 351,684 stored predictions when switching a model.
    """
    sb = get_client()
    rows = []
    offset = 0
    page_size = 1000
    while True:
        batch = (
            sb.table("model_comparison_predictions")
            .select("datetime,actual,predicted")
            .eq("task", task)
            .eq("model", method)
            .gte("datetime", pd.Timestamp(start_time).isoformat())
            .lt("datetime", pd.Timestamp(end_time).isoformat())
            .order("datetime")
            .range(offset, offset + page_size - 1)
            .execute()
            .data or []
        )
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    result = pd.DataFrame(rows)
    if not result.empty:
        result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
        result["actual"] = pd.to_numeric(result["actual"], errors="coerce")
        result["predicted"] = pd.to_numeric(result["predicted"], errors="coerce")
        result = result.dropna(subset=["datetime", "actual", "predicted"])
        result = result.sort_values("datetime")
    return result

historical_model = st.selectbox(
    "Select evaluation model",
    PRESENTATION_MODELS,
    index=PRESENTATION_MODELS.index("LightGBM"),
    key="historical_model_choice",
    help="The graph and test-period metrics change with this model selection."
)

horizon = 1
if pollutant == "PM10":
    horizon = st.radio(
        "PM10 evaluation horizon",
        [3, 1, 24],
        horizontal=True,
        format_func=lambda h: {
            1: "Next hour",
            3: "Next 3-hour average",
            24: "Next 24-hour average",
        }[h],
    )

task = ("NO2_next_hour" if pollutant == "NO2"
        else {1: "PM10_next_hour",
              3: "PM10_next_3h_average",
              24: "PM10_next_24h_average"}[horizon])

st.caption(f"Evaluation target: {task.replace('_', ' ')} · "
           "Metrics are recalculated for the selected date range.")

try:
    comparison = load_six_model_predictions(task, historical_model, start, end_excl)
    if comparison.empty:
        st.info(
            "No stored historical test predictions for this model, "
            "forecast target and selected date range. "
            "Try September 2019 – February 2023."
        )
    else:
        error = comparison["predicted"] - comparison["actual"]
        mae = float(error.abs().mean())
        rmse = float(np.sqrt(error.pow(2).mean()))
        total = float((comparison["actual"] - comparison["actual"].mean()).pow(2).sum())
        r2 = 1 - float(error.pow(2).sum()) / total if total > 0 else np.nan

        a, b, c, d = st.columns(4)
        a.metric("Model", historical_model)
        b.metric("MAE", f"{mae:.3f} µg/m³")
        c.metric("RMSE", f"{rmse:.3f} µg/m³")
        d.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comparison["datetime"], y=comparison["actual"],
            mode="lines", name="Actual (historical)",
            line=dict(color="#1268D3", width=1.8)
        ))
        fig.add_trace(go.Scatter(
            x=comparison["datetime"], y=comparison["predicted"],
            mode="lines", name=f"Predicted ({historical_model})",
            line=dict(color="#F39A1F", width=1.8)
        ))
        fig.update_layout(
            height=385, xaxis_title="Forecast origin date and time",
            yaxis_title=f"{pollutant} (µg/m³)",
            legend=dict(orientation="h", y=1.12, x=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Displayed {len(comparison):,} historical test predictions from "
            f"{comparison['datetime'].min():%d %b %Y} to "
            f"{comparison['datetime'].max():%d %b %Y}. "
            "These are NOT live forecasts. Actual values were measured after "
            "the prediction origin and are shown only for model evaluation."
        )
except Exception as exc:
    st.error("Could not load the model comparison predictions from Supabase.")
    st.caption(
        "Check that the table contains the exported CSV, RLS permits SELECT "
        "for the dashboard's API role, and the selected date range includes test data."
    )

# The former historical single-date forecast demonstration has been removed
# from the main dashboard. Historical model evaluation above is retained.
# Saved models are still loaded by the future forecasting sections below.
MODEL_DIR = Path(__file__).parent / "models"

# ---------- validated, forward-only demonstration ----------
# These four bundles are trained separately using the accompanying trainer.
# They use past pollutant measurements + calendar features and DO NOT require
# missing wind speed or future weather inputs.

st.divider()
st.header("Future Air Quality Forecast")
future_model = st.selectbox(
    "Select future forecasting model",
    PRESENTATION_MODELS,
    index=PRESENTATION_MODELS.index("LightGBM"),
    key="future_model_choice",
    help="Only models with compatible saved hourly forecast bundles can produce future predictions."
)
st.caption(f"Forecasting method selected: **{future_model}**. "
           "LightGBM is the deployed default; other models need their own trained and tested hourly forecast files.")
st.caption(
    "Next-hour predictions and 24 individual hourly forecasts for PM10 or NO₂. "
    "An optional tab preserves the original next-24-hour average model. "
    "Forecasts start AFTER the latest Supabase observation, so if the dataset ends "
    "in February 2023 these are historical-origin forecasts, not current 2026 readings."
)

@st.cache_data(ttl=3600)
def latest_forward_origin():
    response = (get_client().table("air_quality").select("datetime")
                .order("datetime", desc=True).limit(1).execute())
    rows = response.data or []
    return pd.Timestamp(rows[0]["datetime"]) if rows else None


def next_forward_features(history, target, origin):
    """Rebuild the exact feature names/order used in the in-notebook 4-model training cell.

    The notebook defines rolling summaries at the forecast origin with full
    3/24-hour windows. Only observations up to origin are accessed.
    """
    other = "NO2" if target == "PM10" else "PM10"
    now = pd.Timestamp(origin)
    s = pd.to_numeric(history[target.lower()], errors="coerce")
    other_s = pd.to_numeric(history[other.lower()], errors="coerce")
    h3 = s.loc[now-pd.Timedelta(hours=2):now]
    h24 = s.loc[now-pd.Timedelta(hours=23):now]
    row = {
        "hour": now.hour,
        "dayofweek": now.dayofweek,
        "month": now.month,
        "dayofyear": now.dayofyear,
    }
    for lag in (0, 1, 2, 3, 24):
        row[f"{target}_lag{lag}"] = s.get(now-pd.Timedelta(hours=lag), np.nan)
    row[f"{target}_roll3"] = h3.mean() if h3.notna().sum() == 3 else np.nan
    row[f"{target}_roll24"] = h24.mean() if h24.notna().sum() == 24 else np.nan
    row[f"{target}_std24"] = h24.std() if h24.notna().sum() == 24 else np.nan
    row[f"{other}_lag0"] = other_s.get(now, np.nan)
    row[f"{other}_lag1"] = other_s.get(now-pd.Timedelta(hours=1), np.nan)
    return pd.DataFrame([row], index=[now]).astype(float)


def forward_graph(history, origin, target, value, hours):
    """Connect last observed value to the next-hour forecast without inventing future actuals."""
    past = history[target.lower()].loc[:origin].tail(25).dropna()
    if past.empty:
        raise ValueError("No historical observations available for this chart")
    origin_time = pd.Timestamp(origin)
    forecast_time = origin_time + pd.Timedelta(hours=hours)
    last_observed = float(past.iloc[-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=past.index, y=past.values, name="Actual (historical)",
        mode="lines+markers", line=dict(color="#1268D3", width=2.3),
        marker=dict(size=5, color="#1268D3"),
        hovertemplate="%{x|%d %b %Y %H:%M}<br>Observed: %{y:.2f} µg/m³<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        # The first point is only a visual anchor: it is the LAST REAL READING.
        x=[origin_time, forecast_time], y=[last_observed, float(value)],
        name="Forecast (connected)", mode="lines+markers",
        line=dict(color="#F39A1F", width=2.8, dash="dash"),
        marker=dict(color="#F39A1F", size=[1, 15], symbol=["circle", "star"]),
        customdata=["Last observed (not a forecast)", "Forecast"],
        hovertemplate="%{x|%d %b %Y %H:%M}<br>%{customdata}: %{y:.2f} µg/m³<extra></extra>"
    ))
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=origin_time, x1=origin_time, y0=0, y1=1,
                  line=dict(color="#777777", dash="dash", width=1.5))
    fig.add_annotation(x=origin_time, y=1, xref="x", yref="paper",
                       text="Last observed hour", showarrow=False,
                       xanchor="right", yanchor="bottom",
                       font=dict(size=11, color="#666666"))
    fig.update_layout(
        height=370, margin=dict(l=32, r=24, t=55, b=35),
        xaxis_title="Date and time", yaxis_title=f"{target} (µg/m³)",
        legend=dict(orientation="h", y=1.16, x=0), hovermode="closest"
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def forecast_file(target, horizon, method, kind="next"):
    """Safe, explicit file naming convention; never label LightGBM as another model."""
    if method == "LightGBM":
        basename = (f"forward_{target}_1h.pkl" if kind == "next"
                    else f"direct_{target}_{horizon}h.pkl")
    else:
        code = MODEL_CODES[method]
        basename = (f"forward_{code}_{target}_1h.pkl" if kind == "next"
                    else f"direct_{code}_{target}_{horizon}h.pkl")
    return MODEL_DIR / basename


def forward_card(history, origin, target, hours, method):
    label = "Next-hour concentration" if hours == 1 else "Next-24-hour average"
    st.markdown(f"#### {target} — {label} ({method})")
    path = forecast_file(target, hours, method, kind="next")
    if not path.is_file():
        st.info(f"{method} next-hour forecast is not deployed: {path.name}")
        st.caption("Train and test this algorithm on the same features/forecast target, then upload its model bundle. No forecast is shown until that is done.")
        return
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or not all(k in bundle for k in
                ("model", "features", "target", "horizon_hours")):
            raise ValueError("Saved file must contain model, features, target and horizon_hours.")
        if bundle["target"] != target or int(bundle["horizon_hours"]) != hours:
            raise ValueError("Bundle target or horizon does not match.")
        if method != "LightGBM" and bundle.get("model_name") != method:
            raise ValueError("Bundle's model_name must match selected forecasting model.")
        row = next_forward_features(history, target, origin)
        cols = list(bundle["features"])
        missing = [c for c in cols if c not in row.columns]
        if missing:
            raise ValueError("Missing features: " + ", ".join(missing))
        if row[cols].isna().any(axis=None):
            raise ValueError("Required recent pollutant readings are missing. Choose a valid forecast origin or provide verified observations.")
        if not hasattr(bundle["model"], "predict"):
            raise ValueError("Saved model does not have a predict method.")
        # Use the same features and column order as the notebook's saved model.
        prediction = float(np.asarray(bundle["model"].predict(row[cols])).ravel()[0])
        if not np.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction.")
        prediction = max(0., prediction)
        first = pd.Timestamp(origin) + pd.Timedelta(hours=1)
        last = pd.Timestamp(origin) + pd.Timedelta(hours=hours)
        period = (f"{first:%d %b %Y %H:%M}" if hours == 1 else
                  f"{first:%d %b %Y %H:%M} – {last:%d %b %Y %H:%M}")
        st.caption("Predicted period: " + period)
        st.metric("Forecast" if hours == 1 else "Forecast 24h mean",
                  f"{prediction:.2f} µg/m³")
        st.plotly_chart(forward_graph(history, origin, target, prediction, hours),
                        width="stretch")
        if hours == 24:
            st.caption("Orange star = ONE predicted average over 24 hours, "
                       "not 24 individual hourly concentrations.")
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        st.error(f"{target} {hours}h forecast unavailable: {exc}")


# ---------- 24 separate, direct hourly forecasts ----------
# Existing forward_{pollutant}_1h.pkl is reused. New direct-horizon
# models direct_{pollutant}_2h.pkl ... direct_{pollutant}_24h.pkl
# must be trained before deployment. These are NOT 24h-mean models.

def hourly_24_forecasts(history, origin, target, method):
    features = next_forward_features(history, target, origin)
    future_times, predictions = [], []
    for horizon in range(1, 25):
        filename = forecast_file(target, horizon, method,
                                 kind="next" if horizon == 1 else "direct")
        if not filename.is_file():
            raise FileNotFoundError(
                f"{filename.name} is missing. Train/upload the 2–24h hourly models first."
            )
        bundle = joblib.load(filename)
        if not isinstance(bundle, dict) or not all(
            k in bundle for k in ("model", "features", "target", "horizon_hours")
        ):
            raise ValueError(f"Invalid saved model bundle: {filename.name}")
        if bundle["target"] != target or int(bundle["horizon_hours"]) != horizon:
            raise ValueError(f"Incorrect forecast horizon in {filename.name}")
        if horizon >= 2 and bundle.get("forecast_kind") != "direct_hourly":
            raise ValueError(f"{filename.name} must be a direct hourly model, not a 24h mean model")
        if method != "LightGBM" and bundle.get("model_name") != method:
            raise ValueError(f"{filename.name} has a different or unlabelled model type")
        columns = list(bundle["features"])
        if any(c not in features.columns for c in columns):
            raise ValueError(f"Feature names do not match {filename.name}")
        row = features[columns]
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            raise ValueError("A required past pollutant reading is missing")
        model = bundle["model"]
        if not hasattr(model, "predict"):
            raise ValueError(f"Model in {filename.name} has no predict method")
        estimate = float(np.asarray(model.predict(row)).ravel()[0])
        if not np.isfinite(estimate):
            raise ValueError(f"Non-finite forecast from {filename.name}")
        future_times.append(pd.Timestamp(origin) + pd.Timedelta(hours=horizon))
        predictions.append(max(0.0, estimate))
    return pd.Series(predictions, index=pd.DatetimeIndex(future_times), name=target)


def hourly_24_plot(history, origin, target, predictions):
    """Blue observed history joined visually to 24 orange hourly predictions."""
    past = history[target.lower()].loc[:origin].tail(25).dropna()
    if past.empty:
        raise ValueError("No historical measurements available for plotting")
    boundary = pd.Timestamp(origin)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=past.index, y=past.values,
        name="Actual (historical)", mode="lines+markers",
        line=dict(color="#1268D3", width=2.4),
        marker=dict(size=5, color="#1268D3"),
        hovertemplate="%{x|%d %b %Y, %H:%M}<br>Observed: %{y:.2f} µg/m³<extra></extra>"
    ))
    # The orange line is anchored at the final BLUE actual reading for visual
    # continuity. That boundary value is NEVER labelled as a future forecast.
    future_x = [boundary] + list(predictions.index)
    future_y = [float(past.iloc[-1])] + [float(v) for v in predictions.values]
    fig.add_trace(go.Scatter(
        x=future_x, y=future_y,
        name="Forecast (24 hourly values)", mode="lines+markers",
        line=dict(color="#F39A1F", width=2.8, dash="dash"),
        marker=dict(size=[1] + [7]*len(predictions),
                    color="#F39A1F", symbol=["circle"] + ["diamond"]*len(predictions)),
        customdata=["Last observed (not forecast)"] + ["Forecast"]*len(predictions),
        hovertemplate="%{x|%d %b %Y, %H:%M}<br>%{customdata}: %{y:.2f} µg/m³<extra></extra>"
    ))
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=boundary, x1=boundary, y0=0, y1=1,
                  line=dict(color="#777777", dash="dash", width=1.5))
    fig.add_annotation(x=boundary, y=1, xref="x", yref="paper",
                       text="Last observed hour", showarrow=False,
                       xanchor="right", yanchor="bottom",
                       font=dict(size=11, color="#666666"))
    fig.update_layout(
        height=400, margin=dict(l=36, r=22, t=48, b=30),
        xaxis_title="Date and time", yaxis_title=f"{target} (µg/m³)",
        legend=dict(orientation="h", y=1.17, x=0),
        hovermode="x unified"
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def hourly_24_card(history, origin, target, method):
    st.subheader(f"{target} — 24 individual hourly forecasts ({method})")
    st.caption("Each orange point represents one forecast hour, not a 24-hour average. "
               "Blue points are measured values only, never future estimates.")
    try:
        values = hourly_24_forecasts(history, origin, target, method)
        a, b, c = st.columns(3)
        a.metric("Hour 1", f"{values.iloc[0]:.2f} µg/m³")
        b.metric("Hour 12", f"{values.iloc[11]:.2f} µg/m³")
        c.metric("Hour 24", f"{values.iloc[23]:.2f} µg/m³")
        st.plotly_chart(hourly_24_plot(history, origin, target, values), width="stretch")
        st.caption(f"Forecast period: {values.index[0]:%d %b %Y %H:%M} "
                   f"to {values.index[-1]:%d %b %Y %H:%M}. "
                   "These are direct forecasts from a single historical origin; "
                   "no unknown future pollutant readings are fed into the model.")
        with st.expander("Show all 24 hourly predictions"):
            st.dataframe(pd.DataFrame({"Forecast time": values.index,
                                       f"{target} predicted (µg/m³)": values.values}),
                         hide_index=True, use_container_width=True)
    except (FileNotFoundError, ValueError, KeyError, AttributeError,
            TypeError, OSError) as exc:
        st.warning(f"{method} 24-hour forecast line unavailable: {exc}")
        st.caption("Keep the earlier dashboard sections unchanged; new hourly models "
                   "need to be trained and uploaded before this chart can appear.")

try:
    end_time = latest_forward_origin()
    if end_time is None:
        st.warning("No observations available in Supabase.")
    else:
        # 48 prior hours and NO future records. Fill only absent timestamps, NOT values.
        start_time = end_time - pd.Timedelta(hours=48)
        window = load_table("air_quality", start_time.to_pydatetime(),
                            (end_time+pd.Timedelta(microseconds=1)).to_pydatetime())
        if window.empty:
            st.warning("No historical data near the forecast origin.")
        else:
            window = (window.set_index("datetime").sort_index()
                      .loc[lambda x: x.index <= end_time])
            window = window[~window.index.duplicated(keep="last")]
            window = window.reindex(pd.date_range(end_time-pd.Timedelta(hours=48),
                                                  end_time, freq="h"))
            if len(window) < 25 or not {'pm10','no2'}.issubset(window.columns):
                st.warning("Not enough pollutant history to make a valid forecast.")
            else:
                st.write(f"**Forecast origin:** {end_time:%d %B %Y, %H:%M}")
                st.caption("Models use only pollutant history and calendar information. "
                           "Missing wind speed is not replaced with invented measurements.")
                # Respect the existing pollutant filter in the sidebar.
                # No extra selection is needed in the forecast section.
                st.caption(f"Showing forecasts for: {pollutant}")
                one_hour, hourly_day = st.tabs(
                    ["Next 1 Hour", "Next 24 Hours (hourly)"]
                )
                with one_hour:
                    forward_card(window, end_time, pollutant, 1, future_model)
                with hourly_day:
                    hourly_24_card(window, end_time, pollutant, future_model)
                st.info("Demonstration only. These are forecasts beyond the stored dataset "
                        "endpoint, not live forecasts or official health warnings. "
                        "The 24-hour tab requires "
                        "24 separately predicted future hourly values.")
except Exception as exc:
    st.warning("Cannot load forward forecast inputs from Supabase.")
    st.caption(f"Details: {type(exc).__name__}: {exc}")

st.caption("Data: Henderson air-quality monitoring site · Supabase REST API (read-only).")
