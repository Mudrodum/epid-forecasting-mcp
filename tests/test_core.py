from pathlib import Path

import numpy as np
import pytest

from epid_forecasting.config import DEFAULT_DATA_PATH, ForecastConfig
from epid_forecasting.features import build_supervised
from epid_forecasting.service import EpidForecastingService
from epid_forecasting.storage import S3ConfigurationError, S3StorageConfig


class DummyPoissonModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full(x.shape[0], self.value, dtype=float)

    def get_params(self):
        return {
            "loss": "poisson",
            "learning_rate": 0.01,
            "max_iter": 8000,
            "min_samples_leaf": 40,
            "l2_regularization": 5.0,
            "early_stopping": True,
            "max_depth": None,
            "max_leaf_nodes": None,
        }


class FakeArtifactStorage:
    def __init__(self):
        self.uploads: dict[str, bytes] = {}

    def upload_file(self, file_path: Path, object_key: str) -> str:
        self.uploads[object_key] = file_path.read_bytes()
        return f"s3://test-bucket/{object_key}"


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


def test_train_forecast_arbitrary_origin_and_s3_export_smoke(tmp_path: Path, monkeypatch):
    def dummy_fit(_x_train, y_train_list, *, random_state=42):
        return [DummyPoissonModel(float(np.mean(y_train))) for y_train in y_train_list]

    monkeypatch.setattr("epid_forecasting.service.fit_models_hist_gbdt", dummy_fit)

    service = EpidForecastingService(data_path=DEFAULT_DATA_PATH, artifact_dir=tmp_path)
    result = service.train_forecast_models(persist_artifacts=False)
    assert result["training_status"] == "trained"
    assert result["split"]["n_test"] == 52
    assert len(result["conformal_interval"]["radii_by_horizon"]) == 4
    assert np.isfinite(result["conformal_interval"]["radii_by_horizon"]).all()

    latest = service.forecast_next_4_weeks()
    assert latest["origin_date"] == "2026-04-27"
    assert len(latest["forecast"]) == 4
    assert set(latest["forecast"][0]) >= {"inc_per_10k_pred", "pi80_lower", "pi80_upper"}

    historical = service.forecast_next_4_weeks(origin_date="2025-04-07")
    assert historical["origin_date"] == "2025-04-07"
    assert len(historical["forecast"]) == 4

    storage = FakeArtifactStorage()
    exported = service.export_forecast_results(user_id="user-1", session_id="session-1", storage=storage)
    assert exported["storage"] == "s3"
    assert exported["forecast_origin_date"] == "2026-04-27"
    assert set(exported["artifacts"]) == {
        "metrics",
        "test_predictions",
        "forecast",
        "feature_list",
        "history_plus_forecast",
        "model_registry",
    }
    expected_prefix = f"s3://test-bucket/user-1/session-1/epid_forecasting/{exported['run_id']}/"
    assert all(uri.startswith(expected_prefix) for uri in exported["artifacts"].values())
    assert len(storage.uploads) == 6
    assert any(key.endswith("/forecast_next_4w.csv") for key in storage.uploads)


def test_s3_export_requires_explicit_configuration(monkeypatch):
    for name in ("S3_ENDPOINT_URL", "S3_BUCKET_NAME", "S3_ACCESS_KEY", "S3_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(S3ConfigurationError, match="S3 export requires configured environment variables"):
        S3StorageConfig.from_environment()
