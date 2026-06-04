"""Feature engineering pipeline for weekly influenza forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_fourier_week_features(
    frame: pd.DataFrame,
    week_col: str = "iso_week",
    period: int = 52,
    k_terms: int = 2,
) -> pd.DataFrame:
    out = frame.copy()
    week = out[week_col].astype(float).to_numpy()
    for k in range(1, k_terms + 1):
        out[f"sin_w{k}"] = np.sin(2 * np.pi * k * week / period)
        out[f"cos_w{k}"] = np.cos(2 * np.pi * k * week / period)
    return out


def make_lag_features(
    frame: pd.DataFrame,
    col: str,
    lags: list[int] | tuple[int, ...],
    prefix: str | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    feature_prefix = prefix or col
    for lag in lags:
        out[f"{feature_prefix}_lag{lag}"] = out[col].shift(lag)
    return out


def make_rolling_features(
    frame: pd.DataFrame,
    col: str,
    windows: list[int] | tuple[int, ...],
    prefix: str | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    feature_prefix = prefix or col
    for window in windows:
        out[f"{feature_prefix}_rollmean{window}"] = out[col].rolling(window).mean()
        out[f"{feature_prefix}_rollstd{window}"] = out[col].rolling(window).std()
    return out


def add_epidemic_dynamics_features(
    frame: pd.DataFrame,
    target_col: str = "inc_per_10k",
    growth_lags: list[int] | tuple[int, ...] = (1, 2, 4),
) -> pd.DataFrame:
    out = frame.copy()
    target = out[target_col].astype(float)

    for lag in growth_lags:
        target_lag = target.shift(lag)
        out[f"y_diff{lag}"] = target - target_lag

    out["y_accel"] = (target - target.shift(1)) - (target.shift(1) - target.shift(2))
    return out


def build_supervised(
    df: pd.DataFrame,
    *,
    target_col: str = "inc_per_10k",
    temp_cols: list[str] | tuple[str, ...] = ("temp_mean",),
    horizon_weeks: int = 4,
    y_lags: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    temp_lags: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4),
    y_roll_windows: list[int] | tuple[int, ...] = (4, 5, 6, 7, 8, 9),
    temp_roll_windows: list[int] | tuple[int, ...] = (4, 5, 6, 7),
    fourier_k: int = 2,
    growth_lags: list[int] | tuple[int, ...] = (1, 2, 4),
) -> tuple[pd.DataFrame, list[str]]:
    """Build a direct multi-step supervised table for weekly influenza forecasting."""
    data = df.copy()

    data = add_fourier_week_features(data, week_col="iso_week", period=52, k_terms=fourier_k)
    data = make_lag_features(data, target_col, y_lags, prefix="y")
    data = make_rolling_features(data, target_col, y_roll_windows, prefix="y")
    data = add_epidemic_dynamics_features(
        data,
        target_col=target_col,
        growth_lags=growth_lags,
    )

    for temp_col in temp_cols:
        data = make_lag_features(data, temp_col, temp_lags, prefix=temp_col)
        data = make_rolling_features(data, temp_col, temp_roll_windows, prefix=temp_col)

    for horizon in range(1, horizon_weeks + 1):
        data[f"y_h{horizon}"] = data[target_col].shift(-horizon)

    drop_cols = {
        "datetime",
        target_col,
        "total_population",
        "total_cases_formula",
        "rh_min",
        "rh_max",
    }
    drop_cols |= {f"y_h{horizon}" for horizon in range(1, horizon_weeks + 1)}
    feature_cols = [col for col in data.columns if col not in drop_cols]

    return data, feature_cols


def feature_group(feature_name: str) -> str:
    """Return a stable semantic feature group for registry and schema output."""
    if feature_name in {"iso_year", "iso_week"}:
        return "calendar_features"
    if feature_name.startswith(("sin_w", "cos_w")):
        return "fourier_seasonality"
    if feature_name.startswith("y_lag"):
        return "target_lags"
    if feature_name.startswith("y_roll"):
        return "target_rolling_stats"
    if feature_name.startswith(("y_diff", "y_accel")):
        return "epidemic_dynamics"
    if "_lag" in feature_name and feature_name.startswith("temp_"):
        return "temperature_lags"
    if "_roll" in feature_name and feature_name.startswith("temp_"):
        return "temperature_rolling_stats"
    if feature_name.startswith("temp_"):
        return "weather_current"
    if feature_name.startswith("rh_"):
        return "humidity_current"
    return "other"
