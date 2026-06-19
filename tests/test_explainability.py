import types

import numpy as np
import pandas as pd

from epid_forecasting.config import ForecastConfig
from epid_forecasting.explainability import compute_forecast_shap_explainability
from epid_forecasting.service import TrainingState


class DummyModel:
    def predict(self, x):
        return x[:, 0] + 1.0


class DummyTreeExplainer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, x):
        return np.ones_like(x, dtype=float)


def test_compute_forecast_shap_explainability_builds_tables(monkeypatch):
    fake_shap = types.SimpleNamespace(TreeExplainer=DummyTreeExplainer)
    monkeypatch.setitem(__import__("sys").modules, "shap", fake_shap)
    config = ForecastConfig(horizon_weeks=2, test_weeks=2, calib_weeks=1)
    data_valid = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-06", periods=6, freq="7D"),
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [2, 3, 4, 5, 6, 7],
            "y_h1": [1, 2, 3, 4, 5, 6],
            "y_h2": [2, 3, 4, 5, 6, 7],
        }
    )
    train_mask = np.array([True, True, True, False, False, False])
    calib_mask = np.array([False, False, False, True, False, False])
    test_mask = np.array([False, False, False, False, True, True])
    state = TrainingState(
        config=config,
        data=data_valid.copy(),
        supervised_data=data_valid.copy(),
        data_valid=data_valid,
        feature_cols=["f1", "f2"],
        train_mask=train_mask,
        calib_mask=calib_mask,
        test_mask=test_mask,
        evaluation_models=[DummyModel(), DummyModel()],
        production_models=[DummyModel(), DummyModel()],
        conformal_radii=np.array([1.0, 1.0]),
        per_h_metrics=pd.DataFrame(),
        overall_metrics={},
        per_h_interval_metrics=pd.DataFrame(),
        overall_interval_metrics={},
    )
    result = compute_forecast_shap_explainability(state, max_test_samples=2, background_size=2)
    assert set(result.global_importance["horizon_weeks"]) == {1, 2}
    assert len(result.local_values) == 8
    assert result.summary["method"] == "SHAP"
    assert result.summary["by_horizon"]["1"][0]["feature"] == "f1"
