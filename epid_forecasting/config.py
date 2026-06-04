"""Configuration constants for the EpidForecasting MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PACKAGE_ROOT / "data" / "influenza_weather_spb_dataset.csv"
DEFAULT_ARTIFACT_DIR = PACKAGE_ROOT / "artifacts"


@dataclass(frozen=True)
class ForecastConfig:
    """Default model configuration for weekly influenza forecasting."""

    horizon_weeks: int = 4
    test_weeks: int = 52
    calib_weeks: int = 52
    alpha: float = 0.20
    target_transform: str = "none"
    target_col: str = "inc_per_10k"
    datetime_col: str = "datetime"
    temp_cols: tuple[str, ...] = ("temp_mean",)
    y_lags: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    temp_lags: tuple[int, ...] = (0, 1, 2, 3, 4)
    y_roll_windows: tuple[int, ...] = (4, 5, 6, 7, 8, 9)
    temp_roll_windows: tuple[int, ...] = (4, 5, 6, 7)
    fourier_k: int = 2
    growth_lags: tuple[int, ...] = (1, 2, 4)
    point_random_state: int = 42
    prod_random_state: int = 142
