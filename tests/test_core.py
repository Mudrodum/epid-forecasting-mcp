from pathlib import Path

import numpy as np

from epid_forecasting.config import DEFAULT_DATA_PATH, ForecastConfig
from epid_forecasting.features import build_supervised
from epid_forecasting.service import EpidForecastingService


class DummyPoissonModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full(x.shape[0], self.value, dtype=float)


def test_dataset_description_uses_static_csv():
    service = EpidForecastingService(data_path=DEFAULT_DATA_PATH)
    desc = service.describe_dataset()
    assert desc["rows"] == 800
    assert desc["date_min"] == "2011-01-03"
    assert desc["date_max"] == "2026-04-27"
    assert desc["target_variable"] == "inc_per_10k"


def test_feature_pipeline_uses_expected_feature_count():
    service = EpidForecastingService(data_path=DEFAULT_DATA_PATH)
    df = service.load_data()
    cfg = ForecastConfig()
    data, feature_cols = build_supervised(
        df,
        target_col=cfg.target_col,
        temp_cols=cfg.temp_cols,
        horizon_weeks=cfg.horizon_weeks,
        y_lags=cfg.y_lags,
        temp_lags=cfg.temp_lags,
        y_roll_windows=cfg.y_roll_windows,
        temp_roll_windows=cfg.temp_roll_windows,
        fourier_k=cfg.fourier_k,
        growth_lags=cfg.growth_lags,
    )
    assert len(feature_cols) == 45
    assert "y_lag0" in feature_cols
    assert "temp_mean_rollstd7" in feature_cols
    assert all(f"y_h{h}" in data.columns for h in range(1, 5))


def test_compact_workflow_returns_analytical_result_and_reuses_fitted_state(monkeypatch):
    calls = {"count": 0}

    def dummy_fit(_x_train, y_train_list, *, random_state=42):
        calls["count"] += 1
        return [DummyPoissonModel(float(np.mean(y_train))) for y_train in y_train_list]

    monkeypatch.setattr("epid_forecasting.service.fit_models_hist_gbdt", dummy_fit)

    service = EpidForecastingService(data_path=DEFAULT_DATA_PATH)
    result = service.run_influenza_forecasting(origin_date="2026-04-27")
    repeated = service.run_influenza_forecasting(origin_date="2025-04-07")

    assert calls["count"] == 2
    assert result["task"] == "four-week influenza incidence forecasting"
    assert result["forecast_origin_date"] == "2026-04-27"
    assert repeated["forecast_origin_date"] == "2025-04-07"
    assert len(result["forecast"]) == 4
    assert set(result["forecast"][0]) >= {"inc_per_10k_prediction", "pi80_lower", "pi80_upper"}
    assert "result_delivery" not in result
