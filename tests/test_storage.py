import json
from io import BytesIO

import pytest

from epid_forecasting.storage import S3ForecastArtifactStore, S3StorageSettings


class FakeS3Client:
    def __init__(self):
        self.uploads = []
        self.signed = []

    def upload_fileobj(self, buffer, bucket, key, ExtraArgs=None):
        assert isinstance(buffer, BytesIO)
        self.uploads.append((bucket, key, buffer.read(), ExtraArgs))

    def generate_presigned_url(self, method, Params, ExpiresIn):
        self.signed.append((method, Params, ExpiresIn))
        return f"https://example.test/{Params['Key']}?expires={ExpiresIn}"


def _sample_result():
    return {
        "task": "four-week influenza incidence forecasting",
        "target_variable": "inc_per_10k",
        "forecast_horizon_weeks": 4,
        "forecast_origin_date": "2026-04-27",
        "fixed_configuration": {"model_family": "HistGradientBoostingRegressor"},
        "holdout_evaluation": {"point_metrics": {"overall": {"mae_overall": 1.2}}},
        "forecast": [
            {
                "origin_date": "2026-04-27",
                "target_date": "2026-05-04",
                "horizon_weeks": 1,
                "inc_per_10k_prediction": 2.1,
                "pi80_lower": 1.1,
                "pi80_upper": 3.1,
            }
        ],
        "forecast_uncertainty_bounds": {"method": "calibration_residual_bounds_transferred_to_production_refit"},
    }


def test_s3_artifact_contract_uses_session_scoped_prefix_and_presigned_links():
    client = FakeS3Client()
    settings = S3StorageSettings(
        endpoint_url="http://s3.test",
        access_key="key",
        secret_key="secret",
        bucket_name="forecast-results",
        presigned_expiration_seconds=3600,
    )
    store = S3ForecastArtifactStore(settings, client=client)
    metadata = store.save_forecasting_run(
        result=_sample_result(), user_id="mark", session_id="session-01", run_id="run-01"
    )

    prefix = "mark/session-01/epid_forecasting/run-01"
    assert metadata["storage_prefix"] == prefix
    assert metadata["download_access"] == "presigned_urls"
    assert len(client.uploads) == 3
    assert [item[1] for item in client.uploads] == [
        f"{prefix}/forecast.csv",
        f"{prefix}/metrics.json",
        f"{prefix}/run_summary.json",
    ]
    assert len(client.signed) == 3
    assert all(call[2] == 3600 for call in client.signed)
    assert metadata["artifacts"]["forecast"]["s3_uri"] == f"s3://forecast-results/{prefix}/forecast.csv"
    assert "download_url" in metadata["artifacts"]["metrics"]

    summary_bytes = client.uploads[2][2]
    summary = json.loads(summary_bytes.decode("utf-8"))
    assert summary["artifacts"]["forecast_s3_path"].endswith("/forecast.csv")


def test_artifact_prefix_rejects_path_injection_in_identifiers():
    store = S3ForecastArtifactStore(
        S3StorageSettings("http://s3.test", "key", "secret", "bucket"),
        client=FakeS3Client(),
    )
    with pytest.raises(ValueError):
        store.save_forecasting_run(
            result=_sample_result(), user_id="../mark", session_id="session-01", run_id="run-01"
        )
