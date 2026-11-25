"""Turn raw BDG2 files into one tidy table: one row = (building, hour), with model features.

Forecasting set-up (day-ahead):
    At midnight we forecast all 24 hourly cooling loads of the coming day.
    So every load-based feature must be at least 24 hours old ("lag >= 24").
    Weather at the target hour is used as a stand-in for a weather forecast
    (the usual assumption in building-load studies; real forecasts add some error).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

DATA = Path(__file__).resolve().parent.parent / "data"

MAX_MISSING = 0.05   # keep buildings with < 5% missing hours
MAX_ZERO = 0.05      # ... and < 5% zero readings (a dead meter reads 0)
BALANCE_POINT_C = 18.0  # outdoor temperature above which cooling is usually needed

FEATURES = [
    # weather (what drives cooling physically)
    "airTemperature", "dewTemperature", "windSpeed", "cooling_degree",
    "temp_mean_24h", "temp_max_24h",
    # calendar (what drives occupancy)
    "hour", "dayofweek", "is_weekend", "is_holiday", "month", "dayofyear",
    # history of the building's own load (all >= 24 h old)
    "lag_24", "lag_48", "lag_168", "mean_prev_day", "max_prev_day", "same_hour_mean_7d",
    # static building information
    "building_code", "site_code", "use_code", "log_sqm",
]
CATEGORICAL = ["building_code", "site_code", "use_code"]


def select_buildings(cw: pd.DataFrame) -> list[str]:
    values = cw.drop(columns="timestamp")
    missing = values.isna().mean()
    zero = (values == 0).mean()
    keep = missing[(missing < MAX_MISSING) & (zero < MAX_ZERO)].index
    return sorted(keep)


def build_table() -> pd.DataFrame:
    cw = pd.read_csv(DATA / "chilledwater_cleaned.csv", parse_dates=["timestamp"])
    weather = pd.read_csv(DATA / "weather.csv", parse_dates=["timestamp"])
    meta = pd.read_csv(DATA / "metadata.csv")

    buildings = select_buildings(cw)
    long = cw[["timestamp", *buildings]].melt(id_vars="timestamp", var_name="building_id", value_name="load")
    long["site_id"] = long["building_id"].str.split("_").str[0]

    # --- weather: fill short gaps per site, add physically meaningful features
    wcols = ["airTemperature", "dewTemperature", "windSpeed"]
    weather = weather[["timestamp", "site_id", *wcols]].sort_values(["site_id", "timestamp"])
    weather[wcols] = weather.groupby("site_id")[wcols].transform(lambda s: s.interpolate(limit=6))
    weather["cooling_degree"] = (weather["airTemperature"] - BALANCE_POINT_C).clip(lower=0)
    g = weather.groupby("site_id")["airTemperature"]
    # buildings have thermal mass: yesterday's heat is still in the walls
    weather["temp_mean_24h"] = g.transform(lambda s: s.rolling(24, min_periods=12).mean())
    weather["temp_max_24h"] = g.transform(lambda s: s.rolling(24, min_periods=12).max())
    df = long.merge(weather, on=["timestamp", "site_id"], how="left")

    # --- calendar
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = ts.dt.month
    df["dayofyear"] = ts.dt.dayofyear
    hol = USFederalHolidayCalendar().holidays(start="2016-01-01", end="2017-12-31")
    df["is_holiday"] = ts.dt.normalize().isin(hol).astype(int)

    # --- normalise each building by its 2016 (training-year) mean, so one global model
    #     can learn from big and small buildings together. Uses training data only.
    scale = df[ts.dt.year == 2016].groupby("building_id")["load"].mean()
    df["scale"] = df["building_id"].map(scale)
    df["y"] = df["load"] / df["scale"]

    # --- lag features (computed on the normalised load, per building, hourly grid is complete)
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    gy = df.groupby("building_id")["y"]
    df["lag_24"] = gy.shift(24)
    df["lag_48"] = gy.shift(48)
    df["lag_168"] = gy.shift(168)
    df["mean_prev_day"] = gy.transform(lambda s: s.shift(24).rolling(24, min_periods=18).mean())
    df["max_prev_day"] = gy.transform(lambda s: s.shift(24).rolling(24, min_periods=18).max())
    same_hour = [gy.shift(24 * k) for k in range(1, 8)]
    df["same_hour_mean_7d"] = pd.concat(same_hour, axis=1).mean(axis=1)

    # --- static info
    meta = meta.set_index("building_id")
    df["use"] = df["building_id"].map(meta["primaryspaceusage"]).fillna("unknown")
    df["log_sqm"] = np.log1p(df["building_id"].map(meta["sqm"]))
    for col, src in [("building_code", "building_id"), ("site_code", "site_id"), ("use_code", "use")]:
        df[col] = df[src].astype("category").cat.codes

    float_cols = [c for c in FEATURES if c not in CATEGORICAL] + ["y", "load", "scale"]
    df[float_cols] = df[float_cols].astype("float32")
    return df
