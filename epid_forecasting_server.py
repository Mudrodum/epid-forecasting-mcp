"""FastMCP entry point for the EpidForecasting MCP server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from fastmcp import FastMCP

from epid_forecasting.config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_PATH
from epid_forecasting.service import EpidForecastingService

load_dotenv(find_dotenv(usecwd=True), override=False)

DATA_PATH = Path(os.getenv("EPID_DATA_PATH", str(DEFAULT_DATA_PATH)))
ARTIFACT_DIR = Path(os.getenv("EPID_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))

service = EpidForecastingService(data_path=DATA_PATH, artifact_dir=ARTIFACT_DIR)
mcp = FastMCP("EpidForecasting")


def _ok(answer: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"answer": answer, "metadata": metadata}


@mcp.tool()
def describe_dataset() -> dict[str, Any]:
    """Describe the static weekly influenza/weather dataset used by the server."""
    metadata = service.describe_dataset()
    return _ok(
        answer=(
            f"Dataset loaded from {metadata['static_dataset_path']} with {metadata['rows']} rows, "
            f"date range {metadata['date_min']} to {metadata['date_max']}, target inc_per_10k."
        ),
        metadata=metadata,
    )


@mcp.tool()
def get_feature_schema() -> dict[str, Any]:
    """Return the engineered feature schema used by the forecasting models."""
    metadata = service.get_feature_schema()
    return _ok(
        answer=f"Feature schema contains {metadata['n_features']} engineered features.",
        metadata=metadata,
    )


@mcp.tool()
def train_forecast_models(
    test_weeks: int = 52,
    calib_weeks: int = 52,
    alpha: float = 0.20,
    target_transform: str = "none",
    persist_artifacts: bool = True,
) -> dict[str, Any]:
    """Train direct 1-4 week HistGradientBoostingRegressor models and conformal intervals.

    Args:
        test_weeks: Number of latest valid origin weeks reserved for the test block.
        calib_weeks: Number of valid origin weeks reserved for split conformal calibration.
        alpha: Miscoverage level for split conformal prediction intervals. Default 0.20 gives nominal 80% intervals.
        target_transform: Either "none" or "log1p". The default is the validated production configuration.
        persist_artifacts: If true, save the trained model bundle under the artifact directory.
    """
    metadata = service.train_forecast_models(
        test_weeks=test_weeks,
        calib_weeks=calib_weeks,
        alpha=alpha,
        target_transform=target_transform,
        persist_artifacts=persist_artifacts,
    )
    split = metadata["split"]
    return _ok(
        answer=(
            "Training completed: "
            f"train {split['train_start']}..{split['train_end']} (n={split['n_train']}), "
            f"calib {split['calib_start']}..{split['calib_end']} (n={split['n_calib']}), "
            f"test {split['test_start']}..{split['test_end']} (n={split['n_test']})."
        ),
        metadata=metadata,
    )


@mcp.tool()
def forecast_next_4_weeks(origin_date: str | None = None) -> dict[str, Any]:
    """Forecast inc_per_10k for the next 4 weekly horizons.

    Args:
        origin_date: Optional dataset week in YYYY-MM-DD format. If omitted, the latest week with complete features is used.
    """
    metadata = service.forecast_next_4_weeks(origin_date=origin_date)
    return _ok(
        answer=f"Generated 4-week forecast from origin date {metadata['origin_date']}.",
        metadata=metadata,
    )


@mcp.tool()
def backtest_forecast_models() -> dict[str, Any]:
    """Return point and split-conformal interval backtest metrics for the trained models."""
    metadata = service.backtest_forecast_models()
    return _ok(
        answer="Backtest metrics returned for train-only and train-plus-calibration refit stages.",
        metadata=metadata,
    )


@mcp.tool()
def get_model_registry() -> dict[str, Any]:
    """Return a machine-readable registry of trained direct multi-step models."""
    metadata = service.get_model_registry()
    return _ok(
        answer=f"Model registry contains {len(metadata['models'])} direct horizon models.",
        metadata=metadata,
    )


@mcp.tool()
def export_forecast_results(
    user_id: str,
    session_id: str,
    origin_date: str | None = None,
    history_tail_n: int = 40,
) -> dict[str, Any]:
    """Export metrics, predictions, features, registry, and forecast files to S3-compatible storage.

    Args:
        user_id: Identifier used as the top-level S3 artifact namespace.
        session_id: Identifier used as the session-level S3 artifact namespace.
        origin_date: Optional dataset week in YYYY-MM-DD format for the forecast export.
        history_tail_n: Number of historical observations included in history_plus_forecast_40.csv.
    """
    metadata = service.export_forecast_results(
        user_id=user_id,
        session_id=session_id,
        origin_date=origin_date,
        history_tail_n=history_tail_n,
    )
    return _ok(
        answer="Forecast artifacts exported to S3-compatible object storage.",
        metadata=metadata,
    )


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7331, path="/mcp")
