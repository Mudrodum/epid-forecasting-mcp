import pandas as pd

from epid_forecasting.weather_source import aggregate_hourly_weather_to_weekly, merge_influenza_weather_weekly


def test_aggregate_hourly_weather_to_weekly_uses_monday_anchor():
    hourly = pd.DataFrame(
        {
            "time": pd.date_range("2025-09-29", periods=48, freq="h"),
            "temp": [10.0] * 24 + [12.0] * 24,
            "rh": [80.0] * 48,
        }
    )
    weekly = aggregate_hourly_weather_to_weekly(hourly)
    assert weekly.loc[0, "week_start"].date().isoformat() == "2025-09-29"
    assert weekly.loc[0, "n_hours"] == 48
    assert weekly.loc[0, "temp_mean"] == 11.0


def test_merge_influenza_weather_weekly_adds_weather_columns():
    influenza = pd.DataFrame(
        {
            "datetime": ["2025-09-29"],
            "iso_year": [2025],
            "iso_week": [40],
            "total_population": [1000],
            "total_cases_formula": [10],
            "inc_per_10k": [100.0],
        }
    )
    weather = pd.DataFrame(
        {
            "week_start": ["2025-09-29"],
            "temp_mean": [3.0],
            "temp_max": [5.0],
            "temp_min": [1.0],
            "rh_mean": [70.0],
            "rh_max": [90.0],
            "rh_min": [50.0],
            "n_hours": [168],
        }
    )
    merged = merge_influenza_weather_weekly(influenza, weather)
    assert merged.loc[0, "temp_mean"] == 3.0
    assert merged.loc[0, "rh_mean"] == 70.0
