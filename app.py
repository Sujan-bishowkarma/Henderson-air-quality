# CELL 1 — Load your existing prepared data and measured-value flags
from pathlib import Path
import pandas as pd

data_path = Path(r"C:\Users\suraj\Downloads\henderson_clean_final_data.csv")
flags_path = Path(r"C:\Users\suraj\Downloads\henderson_measured_flags.csv")

for path in (data_path, flags_path):
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}. Check the folder/file name.")

df = pd.read_csv(data_path, index_col="datetime", parse_dates=True).sort_index()
measured = pd.read_csv(flags_path, index_col="datetime", parse_dates=True).sort_index()
for col in ("PM10", "NO2"):
    if col not in df.columns or col not in measured.columns:
        raise KeyError(f"Missing required column {col}")
    parsed = measured[col].astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        raise ValueError(f"Unknown measured flag values for {col}")
    measured[col] = parsed.astype(bool)
if df.index.has_duplicates or measured.index.has_duplicates:
    raise ValueError("Duplicate timestamps; correct preprocessing first")
if not df.index.equals(measured.index):
    raise ValueError("Data and measured flags indexes do not match")
print("Loaded", len(df), "hourly records:", df.index.min(), "to", df.index.max())


# CELL 2 — Train and evaluate NEW 2–24 hour direct forecasts for each pollutant
# Existing forward_{pollutant}_1h.pkl and forward_{pollutant}_24h.pkl remain untouched.
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

source = df[["PM10", "NO2"]].copy().sort_index()
source.index = pd.to_datetime(source.index)
observed = measured[["PM10", "NO2"]].copy().sort_index()
observed.index = pd.to_datetime(observed.index)

hourly_index = pd.date_range(source.index.min(), source.index.max(), freq="h")
source = source.reindex(hourly_index)
observed = observed.reindex(hourly_index).fillna(False).astype(bool)
for pollutant in ("PM10", "NO2"):
    source[pollutant] = pd.to_numeric(source[pollutant], errors="coerce")
    source.loc[~observed[pollutant], pollutant] = np.nan

def features_at_origin(target):
    other = "NO2" if target == "PM10" else "PM10"
    X = pd.DataFrame(index=source.index)
    X["hour"] = source.index.hour
    X["dayofweek"] = source.index.dayofweek
    X["month"] = source.index.month
    X["dayofyear"] = source.index.dayofyear
    for lag in (0, 1, 2, 3, 24):
        X[f"{target}_lag{lag}"] = source[target].shift(lag)
    X[f"{target}_roll3"] = source[target].rolling(3, min_periods=3).mean()
    X[f"{target}_roll24"] = source[target].rolling(24, min_periods=24).mean()
    X[f"{target}_std24"] = source[target].rolling(24, min_periods=24).std()
    X[f"{other}_lag0"] = source[other]
    X[f"{other}_lag1"] = source[other].shift(1)
    return X

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
results = []
train_boundary = hourly_index[int(len(hourly_index)*0.60)]
val_boundary = hourly_index[int(len(hourly_index)*0.80)]

for pollutant in ("PM10", "NO2"):
    full_features = features_at_origin(pollutant)
    for horizon in range(2, 25):
        # ONE future hourly value at t+h (not the average of a window).
        y_future = source[pollutant].shift(-horizon)
        actually_measured = observed[pollutant].shift(-horizon).fillna(False).astype(bool)
        usable = full_features.notna().all(axis=1) & y_future.notna() & actually_measured
        X, y = full_features.loc[usable], y_future.loc[usable]
        target_time = X.index + pd.Timedelta(hours=horizon)

        # Exclude examples whose future target crosses the next split boundary.
        train_mask = target_time < train_boundary
        valid_mask = (X.index >= train_boundary) & (target_time < val_boundary)
        test_mask = X.index >= val_boundary
        if min(train_mask.sum(), valid_mask.sum(), test_mask.sum()) < 50:
            raise ValueError(f"Not enough observations for {pollutant}, hour {horizon}")

        config = dict(n_estimators=300, learning_rate=0.05,
                      num_leaves=31, random_state=42, verbosity=-1)
        model = LGBMRegressor(**config)
        model.fit(X.loc[train_mask], y.loc[train_mask])
        val_pred = np.clip(model.predict(X.loc[valid_mask]), 0, None)
        validation_rmse = float(np.sqrt(mean_squared_error(y.loc[valid_mask], val_pred)))

        # Refit using train+validation; evaluate held-out test once.
        final = LGBMRegressor(**config)
        train_and_val = train_mask | valid_mask
        final.fit(X.loc[train_and_val], y.loc[train_and_val])
        test_pred = np.clip(final.predict(X.loc[test_mask]), 0, None)
        test_actual = y.loc[test_mask]
        mae = float(mean_absolute_error(test_actual, test_pred))
        rmse = float(np.sqrt(mean_squared_error(test_actual, test_pred)))
        r2 = float(r2_score(test_actual, test_pred))

        path = model_dir / f"direct_{pollutant}_{horizon}h.pkl"
        joblib.dump({
            "model": final,
            "features": list(X.columns),
            "target": pollutant,
            "horizon_hours": horizon,
            "forecast_kind": "direct_hourly",
            "training_through": str(X.index[train_and_val].max()),
            "validation_RMSE": validation_rmse,
            "held_out_test": {"MAE": mae, "RMSE": rmse, "R2": r2},
        }, path)
        results.append({"Pollutant": pollutant, "Hour Ahead": horizon,
                        "Validation RMSE": validation_rmse,
                        "Test MAE": mae, "Test RMSE": rmse, "Test R2": r2})
        print(f"Saved {path.name} | test R² = {r2:.3f}")

result_table = pd.DataFrame(results)
display(result_table.round(3))
result_table.to_csv(model_dir / "direct_hourly_2_to_24_metrics.csv", index=False)
print("Saved 46 models and metrics to", model_dir.resolve())
