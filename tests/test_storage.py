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


def test_s3_store_persists_influenza_db_dataset_artifacts():
    import pandas as pd

    client = FakeS3Client()
    store = S3ForecastArtifactStore(
        S3StorageSettings("http://s3.test", "key", "secret", "forecast-results"),
        client=client,
    )
    weekly = pd.DataFrame({"datetime": ["2025-09-29"], "inc_per_10k": [1.2]})
    cases = pd.DataFrame({"datetime": ["2025-09-29"], "sars_total_cases": [100]})
    age_groups = pd.DataFrame({"datetime": ["2025-09-29"], "age_group": ["7-14"], "inc_per_10k": [4.5]})

    metadata = store.save_influenza_db_dataset(
        weekly=weekly,
        cases=cases,
        age_groups=age_groups,
        summary={"city": {"slug": "spb"}},
        user_id="mark",
        session_id="session-01",
        run_id="run-01",
    )

    prefix = "mark/session-01/epid_forecasting/influenza_db/run-01"
    assert metadata["storage_prefix"] == prefix
    assert [item[1] for item in client.uploads] == [
        f"{prefix}/weekly.csv",
        f"{prefix}/cases.csv",
        f"{prefix}/age_groups.csv",
        f"{prefix}/summary.json",
    ]
    assert metadata["artifacts"]["age_groups"]["s3_uri"] == f"s3://forecast-results/{prefix}/age_groups.csv"


def test_s3_store_persists_bulletin_context_artifacts():
    import pandas as pd

    client = FakeS3Client()
    store = S3ForecastArtifactStore(
        S3StorageSettings("http://s3.test", "key", "secret", "forecast-results"),
        client=client,
    )
    weekly = pd.DataFrame({"datetime": ["2025-09-29"], "inc_per_10k": [1.2]})
    age_groups = pd.DataFrame({"datetime": ["2025-09-29"], "age_group": ["7-14"], "inc_per_10k": [4.5]})
    context = {"purpose": "structured_evidence_for_external_bulletin_writing"}

    metadata = store.save_bulletin_context(
        context=context,
        markdown="# Context\n",
        weekly=weekly,
        age_groups=age_groups,
        user_id="mark",
        session_id="session-01",
        run_id="run-01",
    )

    prefix = "mark/session-01/epid_forecasting/bulletin_context/run-01"
    assert metadata["storage_prefix"] == prefix
    assert [item[1] for item in client.uploads] == [
        f"{prefix}/bulletin_context.json",
        f"{prefix}/bulletin_context.md",
        f"{prefix}/weekly.csv",
        f"{prefix}/age_groups.csv",
    ]
    assert metadata["artifacts"]["bulletin_context_markdown"]["s3_uri"] == f"s3://forecast-results/{prefix}/bulletin_context.md"


def test_s3_store_persists_br_calibration_artifacts():
    import pandas as pd

    client = FakeS3Client()
    store = S3ForecastArtifactStore(
        S3StorageSettings("http://s3.test", "key", "secret", "forecast-results"),
        client=client,
    )
    metadata = store.save_br_calibration_run(
        kind="forecast",
        trajectory=pd.DataFrame({"datetime": ["2025-10-06"], "fitted_cases": [12.0]}),
        parameter_samples=pd.DataFrame({"sample_id": [1], "alpha_total": [0.4]}),
        parameter_summary={"best_fit": {"alpha_total": 0.4}},
        diagnostics={"r2_observed_vs_fitted": 0.2},
        configuration={"method": "abc"},
        limitations=["Auxiliary model."],
        figures={"forecast_ru": {"png": b"png", "pdf": b"pdf"}},
        user_id="mark",
        session_id="session-01",
        run_id="run-01",
    )

    prefix = "mark/session-01/epid_forecasting/br_calibration/forecast/run-01"
    assert metadata["storage_prefix"] == prefix
    assert [item[1] for item in client.uploads] == [
        f"{prefix}/trajectory.csv",
        f"{prefix}/parameter_samples.csv",
        f"{prefix}/run_summary.json",
        f"{prefix}/forecast_ru.png",
        f"{prefix}/forecast_ru.pdf",
    ]
    assert metadata["artifacts"]["forecast_ru_png"]["s3_uri"] == f"s3://forecast-results/{prefix}/forecast_ru.png"


def test_s3_store_persists_br_artifacts_within_bulletin_context_prefix():
    import pandas as pd

    client = FakeS3Client()
    store = S3ForecastArtifactStore(
        S3StorageSettings("http://s3.test", "key", "secret", "forecast-results"),
        client=client,
    )
    metadata = store.save_bulletin_context(
        context={"purpose": "structured_evidence_for_external_bulletin_writing", "forecast_model": {"engine": "br"}},
        markdown="# Context\n",
        weekly=pd.DataFrame({"datetime": ["2025-09-29"], "inc_per_10k": [1.2]}),
        age_groups=pd.DataFrame({"datetime": ["2025-09-29"], "age_group": ["7-14"], "inc_per_10k": [4.5]}),
        br_trajectory=pd.DataFrame({"datetime": ["2025-10-06"], "fitted_cases": [12.0]}),
        br_parameter_samples=pd.DataFrame({"alpha_total": [0.4], "beta_total": [0.2]}),
        br_summary={"parameter_summary": {"best_fit": {"alpha_total": 0.4}}},
        br_figures={"br_forecast_ru": {"png": b"png", "pdf": b"pdf"}},
        user_id="mark",
        session_id="session-01",
        run_id="run-01",
    )

    prefix = "mark/session-01/epid_forecasting/bulletin_context/run-01"
    uploaded_keys = [item[1] for item in client.uploads]
    assert f"{prefix}/br_trajectory.csv" in uploaded_keys
    assert f"{prefix}/br_parameter_samples.csv" in uploaded_keys
    assert f"{prefix}/br_summary.json" in uploaded_keys
    assert f"{prefix}/br_forecast_ru.png" in uploaded_keys
    assert metadata["artifacts"]["br_forecast_ru_pdf"]["s3_uri"] == f"s3://forecast-results/{prefix}/br_forecast_ru.pdf"
