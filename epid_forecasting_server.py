"""FastMCP entry point for compact influenza forecasting tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from dotenv import find_dotenv, load_dotenv

from epid_forecasting.config import DEFAULT_DATA_PATH
from epid_forecasting.service import EpidForecastingService
from epid_forecasting.storage import S3ForecastArtifactStore, S3StorageSettings

PROJECT_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(PROJECT_ENV_PATH, override=False)
load_dotenv(find_dotenv(usecwd=True), override=False)

DATA_PATH = Path(os.getenv("EPID_DATA_PATH", str(DEFAULT_DATA_PATH)))

service = EpidForecastingService(data_path=DATA_PATH)
mcp = FastMCP("EpidForecasting")


def _ok(answer: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"answer": answer, "metadata": metadata}


def _artifact_store() -> S3ForecastArtifactStore:
    return S3ForecastArtifactStore(S3StorageSettings.from_env())


@mcp.tool()
def describe_influenza_dataset() -> dict[str, Any]:
    """Return a compact description of the bundled weekly influenza dataset."""
    metadata = service.describe_dataset()
    compact = {
        "dataset": "weekly influenza incidence and weather data for Saint Petersburg",
        "target_variable": metadata["target_variable"],
        "target_description": metadata["target_description"],
        "rows": metadata["rows"],
        "date_range": {
            "start": metadata["date_min"],
            "end": metadata["date_max"],
        },
        "forecast_horizon_weeks": 4,
        "missing_values": metadata["missing_values"],
    }
    return _ok(
        answer=(
            f"Loaded {compact['rows']} weekly observations from "
            f"{compact['date_range']['start']} to {compact['date_range']['end']}; "
            f"target variable is {compact['target_variable']}."
        ),
        metadata=compact,
    )


@mcp.tool()
def run_influenza_forecasting(
    session_id: str,
    user_id: str,
    origin_date: str | None = None,
) -> dict[str, Any]:
    """Run the fixed four-week influenza forecasting workflow and persist result artifacts.

    Args:
        session_id: Session identifier used in the S3 artifact prefix.
        user_id: User identifier used in the S3 artifact prefix.
        origin_date: Optional observed dataset week in YYYY-MM-DD format. When omitted,
            forecasting starts from the latest feature-complete observation.

    Returns:
        Inline numerical results plus S3 object references and temporary presigned
        URLs for downloading forecast, metrics, and run-summary artifacts.
    """
    analytics = service.run_influenza_forecasting(origin_date=origin_date)
    artifact_metadata = _artifact_store().save_forecasting_run(
        result=analytics,
        user_id=user_id,
        session_id=session_id,
    )
    metadata = {
        **analytics,
        "result_delivery": {
            "mode": "inline_summary_plus_s3_artifacts",
            "storage": "s3_compatible",
            "authentication": "server_side_s3_credentials",
            "client_download_access": "temporary_presigned_urls",
        },
        **artifact_metadata,
    }
    return _ok(
        answer=(
            "Completed the fixed four-week forecasting workflow for "
            f"origin date {analytics['forecast_origin_date']}; "
            "the numerical result is included inline and full artifacts are available "
            "through temporary download URLs."
        ),
        metadata=metadata,
    )


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7331, path="/mcp")
