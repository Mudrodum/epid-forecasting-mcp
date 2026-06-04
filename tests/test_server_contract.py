import asyncio

import numpy as np

import epid_forecasting_server as server
from epid_forecasting.config import DEFAULT_DATA_PATH
from epid_forecasting.service import EpidForecastingService


class DummyPoissonModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full(x.shape[0], self.value, dtype=float)


class FakeArtifactStore:
    def save_forecasting_run(self, *, result, user_id, session_id):
        assert user_id == "mark"
        assert session_id == "session-01"
        assert len(result["forecast"]) == 4
        return {
            "run_id": "run-01",
            "storage_prefix": "mark/session-01/epid_forecasting/run-01",
            "download_access": "presigned_urls",
            "presigned_url_expiration_seconds": 3600,
            "artifacts": {"forecast": {"download_url": "https://example.test/forecast.csv"}},
        }


def test_mcp_surface_contains_only_two_agent_facing_tools():
    tools = asyncio.run(server.mcp.list_tools())
    assert [tool.name for tool in tools] == [
        "describe_influenza_dataset",
        "run_influenza_forecasting",
    ]
    run_tool = next(tool for tool in tools if tool.name == "run_influenza_forecasting")
    assert set(run_tool.parameters["properties"]) == {"session_id", "user_id", "origin_date"}
    assert set(run_tool.parameters["required"]) == {"session_id", "user_id"}


def test_main_tool_returns_inline_result_and_presigned_artifact_contract(monkeypatch):
    def dummy_fit(_x_train, y_train_list, *, random_state=42):
        return [DummyPoissonModel(float(np.mean(y_train))) for y_train in y_train_list]

    monkeypatch.setattr("epid_forecasting.service.fit_models_hist_gbdt", dummy_fit)
    monkeypatch.setattr(server, "service", EpidForecastingService(data_path=DEFAULT_DATA_PATH))
    monkeypatch.setattr(server, "_artifact_store", lambda: FakeArtifactStore())

    response = server.run_influenza_forecasting(
        session_id="session-01", user_id="mark", origin_date="2026-04-27"
    )
    metadata = response["metadata"]
    assert metadata["forecast_origin_date"] == "2026-04-27"
    assert len(metadata["forecast"]) == 4
    assert metadata["result_delivery"]["client_download_access"] == "temporary_presigned_urls"
    assert metadata["storage_prefix"] == "mark/session-01/epid_forecasting/run-01"
