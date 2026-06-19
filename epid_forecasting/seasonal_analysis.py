"""Deterministic seasonal comparison utilities for influenza surveillance data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SeasonalAnalysisConfig:
    """Configuration for epidemic-season based analysis."""

    datetime_col: str = "datetime"
    target_col: str = "inc_per_10k"
    season_start_week: int = 40
    smooth_window: int = 3
    n_last_seasons: int = 3
    round_digits: int = 3


def _round(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, digits)


def _require_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")


def _week_order(season_start_week: int) -> list[int]:
    if not 1 <= int(season_start_week) <= 53:
        raise ValueError("season_start_week must be in [1, 53].")
    return list(range(int(season_start_week), 54)) + list(range(1, int(season_start_week)))


def _season_label(timestamp: pd.Timestamp, season_start_week: int) -> tuple[int, str]:
    iso = timestamp.isocalendar()
    iso_year = int(iso.year)
    iso_week = int(iso.week)
    start_year = iso_year if iso_week >= int(season_start_week) else iso_year - 1
    return start_year, f"{start_year}-{start_year + 1}"


def prepare_epidemic_season_frame(frame: pd.DataFrame, config: SeasonalAnalysisConfig | None = None) -> pd.DataFrame:
    """Normalize weekly data and add epidemic-season coordinates."""
    cfg = config or SeasonalAnalysisConfig()
    _require_columns(frame, [cfg.datetime_col, cfg.target_col])
    if cfg.smooth_window < 1:
        raise ValueError("smooth_window must be positive.")

    data = frame[[cfg.datetime_col, cfg.target_col]].copy()
    data[cfg.datetime_col] = pd.to_datetime(data[cfg.datetime_col], errors="coerce")
    data[cfg.target_col] = pd.to_numeric(data[cfg.target_col], errors="coerce")
    data = data.dropna(subset=[cfg.datetime_col, cfg.target_col]).sort_values(cfg.datetime_col).reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid rows remain after datetime/target normalization.")
    if data[cfg.datetime_col].duplicated().any():
        duplicates = data.loc[data[cfg.datetime_col].duplicated(keep=False), cfg.datetime_col].head(10).tolist()
        raise ValueError(f"Duplicate weekly dates are not allowed: {duplicates!r}")

    labels = data[cfg.datetime_col].apply(lambda value: _season_label(pd.Timestamp(value), cfg.season_start_week))
    iso = data[cfg.datetime_col].dt.isocalendar()
    week_to_position = {week: idx for idx, week in enumerate(_week_order(cfg.season_start_week))}

    data = data.assign(
        season_start_year=[int(item[0]) for item in labels],
        season_label=[str(item[1]) for item in labels],
        iso_year=iso.year.astype(int),
        iso_week=iso.week.astype(int),
    )
    data["season_week_pos"] = data["iso_week"].map(week_to_position).astype(float)

    weekly = (
        data.groupby(["season_start_year", "season_label", "iso_week", "season_week_pos"], as_index=False)
        .agg(value=(cfg.target_col, "mean"), date=(cfg.datetime_col, "min"))
        .sort_values(["season_start_year", "season_week_pos"])
        .reset_index(drop=True)
    )
    weekly["smoothed_value"] = weekly.groupby("season_label")["value"].transform(
        lambda series: series.rolling(window=cfg.smooth_window, center=True, min_periods=1).mean()
    )
    return weekly


def _interpolate_x_at_y(x0: float, y0: float, x1: float, y1: float, target_y: float) -> float:
    if abs(float(y1) - float(y0)) < 1e-12:
        return float(x0)
    return float(x0 + (target_y - y0) * (x1 - x0) / (y1 - y0))


def _nearest_crossing(xs: np.ndarray, ys: np.ndarray, peak_idx: int, half_height: float, *, side: str) -> float | None:
    if side == "left":
        pairs = [(j, j + 1) for j in range(int(peak_idx) - 1, -1, -1)]
    elif side == "right":
        pairs = [(j, j + 1) for j in range(int(peak_idx), len(xs) - 1)]
    else:
        raise ValueError("side must be 'left' or 'right'.")

    for i0, i1 in pairs:
        y0 = float(ys[i0])
        y1 = float(ys[i1])
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == half_height:
            return float(xs[i0])
        if y1 == half_height:
            return float(xs[i1])
        if (y0 - half_height) * (y1 - half_height) < 0:
            return _interpolate_x_at_y(float(xs[i0]), y0, float(xs[i1]), y1, half_height)
    return None


def _wave_status(left_x: float | None, right_x: float | None) -> str:
    if left_x is None and right_x is None:
        return "both_censored"
    if left_x is None:
        return "left_censored"
    if right_x is None:
        return "right_censored"
    return "complete"


def _secondary_peak_ratio(values: np.ndarray, peak_idx: int, digits: int) -> float | None:
    if len(values) < 3:
        return None
    peaks: list[float] = []
    for idx in range(1, len(values) - 1):
        if idx == int(peak_idx):
            continue
        if values[idx] >= values[idx - 1] and values[idx] >= values[idx + 1]:
            peaks.append(float(values[idx]))
    if not peaks:
        peaks = [float(value) for idx, value in enumerate(values) if idx != int(peak_idx)]
    if not peaks:
        return None
    main = float(values[int(peak_idx)])
    if main <= 0 or not np.isfinite(main):
        return None
    return _round(max(peaks) / main, digits)


def _extract_wave(season_frame: pd.DataFrame, cfg: SeasonalAnalysisConfig) -> JsonObject:
    season_frame = season_frame.sort_values("season_week_pos").reset_index(drop=True)
    xs = season_frame["season_week_pos"].to_numpy(dtype=float)
    ys = season_frame["smoothed_value"].to_numpy(dtype=float)
    raw = season_frame["value"].to_numpy(dtype=float)
    if len(xs) == 0 or not np.isfinite(ys).any():
        raise ValueError("Cannot extract a wave from an empty or non-finite season.")

    peak_idx = int(np.nanargmax(ys))
    peak_value = float(ys[peak_idx])
    half_height = peak_value * 0.5
    left_x = _nearest_crossing(xs, ys, peak_idx, half_height, side="left")
    right_x = _nearest_crossing(xs, ys, peak_idx, half_height, side="right")

    if left_x is not None and right_x is not None:
        fwhm_weeks = _round(right_x - left_x, cfg.round_digits)
        left_span = _round(xs[peak_idx] - left_x, cfg.round_digits)
        right_span = _round(right_x - xs[peak_idx], cfg.round_digits)
        asymmetry_ratio = _round((right_span / left_span) if left_span else None, cfg.round_digits)
        fwhm_lower_bound = None
    else:
        fwhm_weeks = None
        left_span = _round(xs[peak_idx] - left_x, cfg.round_digits) if left_x is not None else None
        right_span = _round(right_x - xs[peak_idx], cfg.round_digits) if right_x is not None else None
        asymmetry_ratio = None
        fwhm_lower_bound = _round(xs[-1] - left_x, cfg.round_digits) if left_x is not None and right_x is None else None

    return {
        "season_start_year": int(season_frame["season_start_year"].iloc[0]),
        "season_label": str(season_frame["season_label"].iloc[0]),
        "wave_status": _wave_status(left_x, right_x),
        "observed_until": pd.Timestamp(season_frame["date"].max()).date().isoformat(),
        "n_observed_weeks": int(len(season_frame)),
        "peak_week": int(season_frame.loc[peak_idx, "iso_week"]),
        "peak_date": pd.Timestamp(season_frame.loc[peak_idx, "date"]).date().isoformat(),
        "peak_value": _round(peak_value, cfg.round_digits),
        "half_height_value": _round(half_height, cfg.round_digits),
        "left_half_cross_pos": _round(left_x, cfg.round_digits),
        "right_half_cross_pos": _round(right_x, cfg.round_digits),
        "fwhm_weeks": fwhm_weeks,
        "fwhm_lower_bound_weeks": fwhm_lower_bound,
        "left_span_weeks": left_span,
        "right_span_weeks": right_span,
        "asymmetry_ratio": asymmetry_ratio,
        "season_burden": _round(float(np.nansum(raw)), cfg.round_digits),
        "secondary_peak_ratio": _secondary_peak_ratio(ys, peak_idx, cfg.round_digits),
        "plot_points": [
            {
                "week": int(row["iso_week"]),
                "pos": _round(float(row["season_week_pos"]), cfg.round_digits),
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "value_raw": _round(float(row["value"]), cfg.round_digits),
                "value_smooth": _round(float(row["smoothed_value"]), cfg.round_digits),
            }
            for _, row in season_frame.iterrows()
        ],
    }


def _latest_vs_previous(latest: JsonObject, previous: list[JsonObject], digits: int) -> list[JsonObject]:
    rows: list[JsonObject] = []
    latest_peak = float(latest["peak_value"])
    latest_burden = float(latest["season_burden"])
    for item in previous:
        prev_peak = float(item["peak_value"])
        prev_burden = float(item["season_burden"])
        latest_width = item_width = None
        if latest.get("fwhm_weeks") is not None:
            latest_width = float(latest["fwhm_weeks"])
        if item.get("fwhm_weeks") is not None:
            item_width = float(item["fwhm_weeks"])
        rows.append(
            {
                "latest_season": latest["season_label"],
                "previous_season": item["season_label"],
                "peak_diff_abs": _round(latest_peak - prev_peak, digits),
                "peak_diff_pct": _round((latest_peak - prev_peak) / prev_peak * 100.0, 1) if prev_peak else None,
                "peak_week_diff": int(latest["peak_week"] - item["peak_week"]),
                "season_burden_diff_pct": _round((latest_burden - prev_burden) / prev_burden * 100.0, 1)
                if prev_burden
                else None,
                "width_diff_weeks": _round(latest_width - item_width, digits)
                if latest_width is not None and item_width is not None
                else None,
                "latest_width_censored": latest.get("fwhm_weeks") is None,
            }
        )
    return rows


def compare_recent_epidemic_waves(
    frame: pd.DataFrame,
    *,
    season_start_week: int = 40,
    smooth_window: int = 3,
    n_last_seasons: int = 3,
    target_col: str = "inc_per_10k",
) -> JsonObject:
    """Compare recent epidemic waves by peak, timing, FWHM-like width and burden."""
    cfg = SeasonalAnalysisConfig(
        season_start_week=season_start_week,
        smooth_window=smooth_window,
        n_last_seasons=n_last_seasons,
        target_col=target_col,
    )
    if cfg.n_last_seasons < 2:
        raise ValueError("n_last_seasons must be at least 2.")
    weekly = prepare_epidemic_season_frame(frame, cfg)
    seasons = sorted(int(value) for value in weekly["season_start_year"].unique())
    if len(seasons) < cfg.n_last_seasons:
        raise ValueError(f"Need at least {cfg.n_last_seasons} seasons; found {len(seasons)}.")
    selected = weekly.loc[weekly["season_start_year"].isin(seasons[-cfg.n_last_seasons :])].copy()
    waves = [_extract_wave(group, cfg) for _, group in selected.groupby("season_start_year", sort=True)]
    latest = waves[-1]
    previous = waves[:-1]
    peak_ranking = sorted(
        [{"season_label": wave["season_label"], "peak_value": wave["peak_value"]} for wave in waves],
        key=lambda item: float(item["peak_value"]),
        reverse=True,
    )
    width_ranking = sorted(
        [
            {"season_label": wave["season_label"], "fwhm_weeks": wave["fwhm_weeks"]}
            for wave in waves
            if wave.get("fwhm_weeks") is not None
        ],
        key=lambda item: float(item["fwhm_weeks"]),
        reverse=True,
    )
    return {
        "series_name": cfg.target_col,
        "comparison_mode": "recent_epidemic_waves",
        "season_definition": f"ISO weeks {cfg.season_start_week}..{cfg.season_start_week - 1}",
        "smoothing": {"method": "centered_rolling_mean", "window_weeks": cfg.smooth_window},
        "width_definition": "FWHM-like width on the smoothed weekly curve",
        "season_labels": [wave["season_label"] for wave in waves],
        "latest_wave_status": latest["wave_status"],
        "waves": waves,
        "peak_ranking": peak_ranking,
        "width_ranking_complete": width_ranking,
        "latest_vs_previous": _latest_vs_previous(latest, previous, cfg.round_digits),
        "allowed_claims": [
            "peak height comparison",
            "peak timing comparison",
            "FWHM-like width comparison when complete",
            "right-censored width caveat for incomplete latest season",
            "seasonal burden comparison",
        ],
        "forbidden_claims": [
            "causal explanations absent from the dataset",
            "formal epidemic threshold declarations",
            "age-group conclusions without age-group data",
        ],
    }
