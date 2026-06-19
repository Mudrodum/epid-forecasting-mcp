"""FastMCP entry point for compact influenza forecasting tools."""

from __future__ import annotations

import os

import pandas as pd
from pathlib import Path
from typing import Any, Literal

from dotenv import find_dotenv, load_dotenv
from fastmcp import FastMCP

from epid_forecasting.br_calibration import (
    BRCalibrationConfig,
    render_br_forecast_figure,
    render_br_parameter_figures,
    run_br_calibration,
)
from epid_forecasting.bulletin_context import build_bulletin_context, render_bulletin_context_markdown
from epid_forecasting.config import DEFAULT_DATA_PATH
from epid_forecasting.explainability import compute_forecast_shap_explainability as compute_shap_for_state
from epid_forecasting.influenza_db import (
    InfluenzaDbRequest,
    InfluenzaDbSettings,
    compare_age_groups,
    fetch_influenza_db_bundle,
    list_supported_cities,
    summarize_influenza_db_bundle,
)
from epid_forecasting.seasonal_analysis import compare_recent_epidemic_waves
from epid_forecasting.service import EpidForecastingService
from epid_forecasting.storage import S3ForecastArtifactStore, S3StorageSettings
from epid_forecasting.weather_source import (
    load_weather_aligned_to_influenza,
    load_weather_until_date,
    merge_influenza_weather_weekly,
)

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


def _influenza_db_settings() -> InfluenzaDbSettings:
    return InfluenzaDbSettings.from_env()


def _db_request(
    *,
    city: str,
    begin_year: int,
    begin_week: int,
    end_year: int | None,
    end_week: int | None,
) -> InfluenzaDbRequest:
    return InfluenzaDbRequest.until_latest(
        city=city,
        begin_year=begin_year,
        begin_week=begin_week,
        end_year=end_year,
        end_week=end_week,
    )


def _parse_horizons(horizons: list[int] | None) -> list[int] | None:
    if horizons is None:
        return None
    parsed = sorted({int(item) for item in horizons})
    return parsed or None


@mcp.tool()
def describe_influenza_dataset() -> dict[str, Any]:
    """Return a compact description of the bundled weekly influenza dataset."""
    metadata = service.describe_dataset()
    compact = {
        "dataset": "weekly influenza incidence and weather data for Saint Petersburg",
        "target_variable": metadata["target_variable"],
        "target_description": metadata["target_description"],
        "rows": metadata["rows"],
        "date_range": {"start": metadata["date_min"], "end": metadata["date_max"]},
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
def list_influenza_db_cities() -> dict[str, Any]:
    """Return city slugs supported by the NII influenza DB endpoint."""
    cities = list_supported_cities()
    return _ok(
        answer=f"Influenza DB city registry contains {len(cities)} supported city entries.",
        metadata={"cities": cities},
    )


@mcp.tool()
def export_influenza_db_dataset(
    session_id: str,
    user_id: str,
    city: str = "spb",
    begin_year: int = 2011,
    begin_week: int = 1,
    end_year: int | None = None,
    end_week: int | None = None,
) -> dict[str, Any]:
    """Export NII influenza DB surveillance tables to S3-compatible storage."""
    request = _db_request(city=city, begin_year=begin_year, begin_week=begin_week, end_year=end_year, end_week=end_week)
    bundle = fetch_influenza_db_bundle(request, _influenza_db_settings())
    summary = summarize_influenza_db_bundle(bundle)
    artifact_metadata = _artifact_store().save_influenza_db_dataset(
        weekly=bundle.weekly,
        cases=bundle.cases,
        age_groups=bundle.age_groups,
        summary=summary,
        user_id=user_id,
        session_id=session_id,
    )
    metadata = {
        **summary,
        "result_delivery": {
            "mode": "inline_summary_plus_s3_artifacts",
            "storage": "s3_compatible",
            "authentication": "server_side_s3_and_influenza_db_credentials",
            "client_download_access": "temporary_presigned_urls",
        },
        **artifact_metadata,
    }
    return _ok(
        answer=(
            "Exported influenza DB data for "
            f"{summary['city']['name_ru']} from {summary['date_range']['start']} "
            f"to {summary['date_range']['end']} as normalized S3 artifacts."
        ),
        metadata=metadata,
    )


@mcp.tool()
def export_weather_source_dataset(
    session_id: str,
    user_id: str,
    city: str = "spb",
    start_date: str = "2023-01-01",
    end_date: str = "2026-05-31",
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    """Fetch Open-Meteo hourly weather, aggregate it weekly, and export artifacts to S3."""
    weather = load_weather_until_date(
        city,
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )
    summary = weather.summary()
    artifact_metadata = _artifact_store().save_weather_dataset(
        hourly=weather.hourly,
        weekly=weather.weekly,
        location=weather.location.to_dict(),
        summary=summary,
        user_id=user_id,
        session_id=session_id,
    )
    metadata = {
        **summary,
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
            f"Exported Open-Meteo weather artifacts for {weather.location.query} "
            f"from {summary['date_range']['start']} to {summary['date_range']['end']}."
        ),
        metadata=metadata,
    )


@mcp.tool()
def compare_influenza_age_groups_from_db(
    city: str = "spb",
    season: str | None = None,
    begin_year: int = 2011,
    begin_week: int = 1,
    end_year: int | None = None,
    end_week: int | None = None,
    peak_width_fraction: float = 0.5,
) -> dict[str, Any]:
    """Compare age groups using live NII influenza DB data."""
    request = _db_request(city=city, begin_year=begin_year, begin_week=begin_week, end_year=end_year, end_week=end_week)
    bundle = fetch_influenza_db_bundle(request, _influenza_db_settings())
    db_summary = summarize_influenza_db_bundle(bundle)
    comparison = compare_age_groups(bundle.age_groups, season=season, peak_width_fraction=peak_width_fraction)
    metadata = {
        "city": db_summary["city"],
        "request": db_summary["request"],
        "source_url": db_summary["source_url"],
        "age_groups": db_summary["age_groups"],
        **comparison,
    }
    return _ok(
        answer=(
            "Compared influenza incidence age groups for "
            f"{metadata['city']['name_ru']} in season {metadata['season']}."
        ),
        metadata=metadata,
    )


@mcp.tool()
def generate_br_model_forecast(
    session_id: str,
    user_id: str,
    city: str = "spb",
    begin_year: int = 2024,
    begin_week: int = 40,
    end_year: int | None = None,
    end_week: int | None = None,
    forecast_type: str = "total",
    method: str = "mcmc",
    forecast_duration_weeks: int = 4,
    calibration_window_weeks: int | None = 26,
    posterior_samples: int = 200,
    abc_candidates: int = 3000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Calibrate a compact Baroyan-Rvachev-style model from NII DB data and export forecast figures.

    This is an auxiliary mechanistic analysis, separate from the GBDT plus
    conformal four-week forecast. ``forecast_type`` accepts ``total`` or
    ``age``; ``method`` accepts ``mcmc``, ``abc``, ``annealing``, or
    ``optuna`` (a compatibility alias for deterministic optimization).
    """

    request = _db_request(
        city=city,
        begin_year=begin_year,
        begin_week=begin_week,
        end_year=end_year,
        end_week=end_week,
    )
    bundle = fetch_influenza_db_bundle(request, _influenza_db_settings())
    config = BRCalibrationConfig(
        forecast_type=forecast_type,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        forecast_duration_weeks=forecast_duration_weeks,
        calibration_window_weeks=calibration_window_weeks,
        posterior_samples=posterior_samples,
        abc_candidates=abc_candidates,
        random_state=random_state,
    )
    result = run_br_calibration(bundle.cases, config=config)
    figures = {
        "forecast_ru": render_br_forecast_figure(result, language="ru"),
        "forecast_en": render_br_forecast_figure(result, language="en"),
    }
    artifact_metadata = _artifact_store().save_br_calibration_run(
        kind="forecast",
        trajectory=result.trajectory,
        parameter_samples=result.parameter_samples,
        parameter_summary=result.parameter_summary,
        diagnostics=result.diagnostics,
        configuration=result.configuration,
        limitations=result.limitations,
        figures=figures,
        user_id=user_id,
        session_id=session_id,
    )
    metadata = {
        "city": summarize_influenza_db_bundle(bundle)["city"],
        "request": summarize_influenza_db_bundle(bundle)["request"],
        "source_url": bundle.redacted_source_url,
        **result.to_public_dict(),
        "result_delivery": {
            "mode": "inline_summary_plus_s3_artifacts",
            "storage": "s3_compatible",
            "authentication": "server_side_s3_and_influenza_db_credentials",
            "client_download_access": "temporary_presigned_urls",
        },
        **artifact_metadata,
    }
    return _ok(
        answer=(
            "Calibrated a compact Baroyan-Rvachev-style mechanistic influenza model and "
            f"exported {forecast_duration_weeks}-week forecast figures for {metadata['city']['name_ru']}."
        ),
        metadata=metadata,
    )


@mcp.tool()
def estimate_br_model_parameters(
    session_id: str,
    user_id: str,
    city: str = "spb",
    begin_year: int = 2024,
    begin_week: int = 40,
    end_year: int | None = None,
    end_week: int | None = None,
    forecast_type: str = "total",
    method: str = "mcmc",
    calibration_window_weeks: int | None = 26,
    posterior_samples: int = 200,
    abc_candidates: int = 3000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Estimate BR-model alpha/beta parameters from NII DB data and export their distributions.

    The returned alpha/beta distributions are calibration parameters for the
    auxiliary mechanistic model. They are not direct biological measurements.
    """

    request = _db_request(
        city=city,
        begin_year=begin_year,
        begin_week=begin_week,
        end_year=end_year,
        end_week=end_week,
    )
    bundle = fetch_influenza_db_bundle(request, _influenza_db_settings())
    config = BRCalibrationConfig(
        forecast_type=forecast_type,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        forecast_duration_weeks=1,
        calibration_window_weeks=calibration_window_weeks,
        posterior_samples=posterior_samples,
        abc_candidates=abc_candidates,
        random_state=random_state,
    )
    result = run_br_calibration(bundle.cases, config=config)
    figures = render_br_parameter_figures(result)
    artifact_metadata = _artifact_store().save_br_calibration_run(
        kind="parameter_estimation",
        trajectory=result.trajectory,
        parameter_samples=result.parameter_samples,
        parameter_summary=result.parameter_summary,
        diagnostics=result.diagnostics,
        configuration=result.configuration,
        limitations=result.limitations,
        figures={
            "alpha_distribution": figures["alpha"],
            "beta_distribution": figures["beta"],
        },
        user_id=user_id,
        session_id=session_id,
    )
    metadata = {
        "city": summarize_influenza_db_bundle(bundle)["city"],
        "request": summarize_influenza_db_bundle(bundle)["request"],
        "source_url": bundle.redacted_source_url,
        **result.to_public_dict(),
        "result_delivery": {
            "mode": "inline_summary_plus_s3_artifacts",
            "storage": "s3_compatible",
            "authentication": "server_side_s3_and_influenza_db_credentials",
            "client_download_access": "temporary_presigned_urls",
        },
        **artifact_metadata,
    }
    return _ok(
        answer=(
            "Estimated alpha/beta distributions for the compact Baroyan-Rvachev-style "
            f"mechanistic model for {metadata['city']['name_ru']} and uploaded the figures to S3."
        ),
        metadata=metadata,
    )


@mcp.tool()
def compute_forecast_shap_explainability(
    session_id: str,
    user_id: str,
    origin_date: str | None = None,
    max_test_samples: int | None = 64,
    background_size: int = 128,
    top_features_per_horizon: int = 8,
    worst_cases_per_horizon: int = 5,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    """Compute SHAP explainability for the fixed bundled SPB forecasting workflow and store artifacts."""
    # Ensure the fixed forecast state exists; origin_date is accepted to match the forecast tool signature,
    # but explainability is computed on the holdout evaluation rows rather than on one future origin row.
    analytics = service.run_influenza_forecasting(origin_date=origin_date)
    state = service._ensure_state()
    shap_result = compute_shap_for_state(
        state,
        horizons=_parse_horizons(horizons),
        max_test_samples=max_test_samples,
        background_size=background_size,
        top_features_per_horizon=top_features_per_horizon,
        worst_cases_per_horizon=worst_cases_per_horizon,
    )
    artifact_metadata = _artifact_store().save_shap_explainability(
        global_importance=shap_result.global_importance,
        local_values=shap_result.local_values,
        worst_cases=shap_result.worst_cases,
        summary=shap_result.summary,
        user_id=user_id,
        session_id=session_id,
    )
    metadata = {
        "forecast_origin_date": analytics["forecast_origin_date"],
        **shap_result.to_public_dict(),
        "result_delivery": {
            "mode": "inline_summary_plus_s3_artifacts",
            "storage": "s3_compatible",
            "authentication": "server_side_s3_credentials",
            "client_download_access": "temporary_presigned_urls",
        },
        **artifact_metadata,
    }
    return _ok(
        answer="Computed SHAP forecast-driver explainability and uploaded SHAP artifacts to S3.",
        metadata=metadata,
    )


def _br_parameter_meanings(forecast_type: str) -> dict[str, dict[str, str]]:
    """Describe the fitted BR parameters without treating them as direct measurements."""
    meanings: dict[str, dict[str, str]] = {
        "alpha_total": {
            "label": "Initial susceptible fraction",
            "interpretation": (
                "Estimated fraction of the aggregate population treated as susceptible at the start of the "
                "calibration window. It is a model initialization parameter, not a directly observed immune-status measure."
            ),
        },
        "beta_total": {
            "label": "Aggregate transmission/contact coefficient",
            "interpretation": (
                "Fitted coefficient governing how infectious pressure is converted into new infections in the "
                "aggregate renewal recurrence. It is not a directly measured contact rate or reproduction number."
            ),
        },
    }
    if forecast_type == "age":
        meanings = {
            "alpha_0_14": {
                "label": "Initial susceptible fraction, 0-14 years",
                "interpretation": "Fitted initial susceptible fraction for the 0-14 model group.",
            },
            "alpha_15_plus": {
                "label": "Initial susceptible fraction, 15+ years",
                "interpretation": "Fitted initial susceptible fraction for the 15+ model group.",
            },
            "beta_0_14_to_0_14": {
                "label": "Within/cross-group coupling coefficient",
                "interpretation": "Fitted age-structured transmission/contact coupling. The implementation label specifies its matrix position; it is not a directly observed contact rate.",
            },
            "beta_0_14_to_15_plus": {
                "label": "Cross-group coupling coefficient",
                "interpretation": "Fitted age-structured transmission/contact coupling. The implementation label specifies its matrix position; it is not a directly observed contact rate.",
            },
            "beta_15_plus_to_0_14": {
                "label": "Cross-group coupling coefficient",
                "interpretation": "Fitted age-structured transmission/contact coupling. The implementation label specifies its matrix position; it is not a directly observed contact rate.",
            },
            "beta_15_plus_to_15_plus": {
                "label": "Within/cross-group coupling coefficient",
                "interpretation": "Fitted age-structured transmission/contact coupling. The implementation label specifies its matrix position; it is not a directly observed contact rate.",
            },
        }
    return meanings


def _br_bulletin_payload(result: Any) -> dict[str, Any]:
    """Convert a BR calibration result into compact inline bulletin evidence."""
    trajectory = result.trajectory.copy()
    trajectory["datetime"] = pd.to_datetime(trajectory["datetime"], errors="coerce")
    forecast = trajectory.loc[trajectory["is_forecast"].fillna(False)].copy()
    origin = trajectory.loc[~trajectory["is_forecast"].fillna(False), "datetime"].max()
    columns = [
        "datetime",
        "group",
        "fitted_cases",
        "pi80_lower_cases",
        "pi80_upper_cases",
        "fitted_inc_per_10k",
        "pi80_lower_inc_per_10k",
        "pi80_upper_inc_per_10k",
    ]
    available_columns = [column for column in columns if column in forecast.columns]
    forecast_rows = forecast.loc[:, available_columns].sort_values(["datetime", "group"]).to_dict(orient="records")
    configuration = dict(result.configuration)
    return {
        "configuration": configuration,
        "forecast_origin_date": None if pd.isna(origin) else origin.date().isoformat(),
        "forecast_horizon_weeks": int(configuration.get("forecast_duration_weeks", 0)),
        "forecast": forecast_rows,
        "parameter_summary": dict(result.parameter_summary),
        "parameter_meanings": _br_parameter_meanings(str(configuration.get("forecast_type", "total"))),
        "diagnostics": dict(result.diagnostics),
        "limitations": list(result.limitations),
    }


@mcp.tool()
def prepare_influenza_bulletin_context(
    session_id: str,
    user_id: str,
    city: str = "spb",
    begin_year: int = 2011,
    begin_week: int = 1,
    end_year: int | None = None,
    end_week: int | None = None,
    season: str | None = None,
    forecast_engine: Literal["gbdt", "br"] = "gbdt",
    include_weather: bool = True,
    include_forecast: bool = True,
    include_shap: bool = True,
    origin_date: str | None = None,
    trend_window_weeks: int = 4,
    peak_width_fraction: float = 0.5,
    season_start_week: int = 40,
    smooth_window: int = 3,
    n_last_seasons: int = 3,
    weather_latitude: float | None = None,
    weather_longitude: float | None = None,
    weather_timezone: str | None = None,
    shap_max_test_samples: int | None = 64,
    shap_background_size: int = 128,
    br_forecast_type: Literal["total", "age"] = "total",
    br_method: Literal["mcmc", "abc", "annealing", "optuna"] = "mcmc",
    br_forecast_duration_weeks: int = 4,
    br_calibration_window_weeks: int | None = 26,
    br_posterior_samples: int = 200,
    br_abc_candidates: int = 3000,
    br_random_state: int = 42,
) -> dict[str, Any]:
    """Prepare inline bulletin evidence using default GBDT or explicitly requested BR forecasting.

    ``forecast_engine='gbdt'`` is the default and produces split-conformal
    forecasts plus SHAP evidence. ``forecast_engine='br'`` runs the compact
    mechanistic model instead, excludes SHAP, and returns alpha/beta calibration
    evidence. The BR implementation has no separately estimated gamma parameter.
    """
    if forecast_engine not in {"gbdt", "br"}:
        raise ValueError("forecast_engine must be 'gbdt' or 'br'.")

    request = _db_request(city=city, begin_year=begin_year, begin_week=begin_week, end_year=end_year, end_week=end_week)
    bundle = fetch_influenza_db_bundle(request, _influenza_db_settings())
    db_summary = summarize_influenza_db_bundle(bundle)
    age_group_comparison = compare_age_groups(bundle.age_groups, season=season, peak_width_fraction=peak_width_fraction)
    wave_comparison = compare_recent_epidemic_waves(
        bundle.weekly,
        season_start_week=season_start_week,
        smooth_window=smooth_window,
        n_last_seasons=n_last_seasons,
        target_col="inc_per_10k",
    )

    weather_bundle = None
    merged_weekly = None
    weather_summary = None
    forecast_result: dict[str, Any] | None = None
    forecast_state = None
    shap_result = None
    br_result = None
    br_payload: dict[str, Any] | None = None
    br_figures: dict[str, dict[str, bytes]] | None = None

    if include_weather:
        weather_bundle = load_weather_aligned_to_influenza(
            request.city,
            bundle.weekly,
            latitude=weather_latitude,
            longitude=weather_longitude,
            timezone=weather_timezone,
        )
        weather_summary = {"status": "included", **weather_bundle.summary()}
        if forecast_engine == "gbdt":
            merged_weekly = merge_influenza_weather_weekly(bundle.weekly, weather_bundle.weekly)

    if include_forecast:
        if forecast_engine == "gbdt":
            if merged_weekly is not None:
                forecast_result, forecast_state = service.run_influenza_forecasting_for_frame(
                    merged_weekly, origin_date=origin_date
                )
            elif request.city == "spb":
                forecast_result = service.run_influenza_forecasting(origin_date=origin_date)
                forecast_state = service._ensure_state()
        else:
            br_config = BRCalibrationConfig(
                forecast_type=br_forecast_type,
                method=br_method,
                forecast_duration_weeks=br_forecast_duration_weeks,
                calibration_window_weeks=br_calibration_window_weeks,
                posterior_samples=br_posterior_samples,
                abc_candidates=br_abc_candidates,
                random_state=br_random_state,
            )
            br_result = run_br_calibration(bundle.cases, config=br_config)
            br_payload = _br_bulletin_payload(br_result)
            parameter_figures = render_br_parameter_figures(br_result)
            br_figures = {
                "br_forecast_ru": render_br_forecast_figure(br_result, language="ru"),
                "br_alpha_distribution": parameter_figures["alpha"],
                "br_beta_distribution": parameter_figures["beta"],
            }

    if forecast_engine == "gbdt" and include_shap and forecast_state is not None:
        shap_result = compute_shap_for_state(
            forecast_state,
            max_test_samples=shap_max_test_samples,
            background_size=shap_background_size,
        )
    shap_summary = shap_result.summary if shap_result is not None else None

    context = build_bulletin_context(
        city=db_summary["city"],
        request=db_summary["request"],
        source_url=db_summary["source_url"],
        weekly=bundle.weekly,
        age_group_comparison=age_group_comparison,
        wave_comparison=wave_comparison,
        forecast_result=forecast_result,
        date_range=db_summary["date_range"],
        parameters={
            "season": age_group_comparison["season"],
            "forecast_engine": forecast_engine,
            "include_weather": include_weather,
            "include_forecast": include_forecast,
            "include_shap_requested": include_shap,
            "include_shap_effective": bool(forecast_engine == "gbdt" and include_shap),
            "origin_date": origin_date,
            "trend_window_weeks": trend_window_weeks,
            "peak_width_fraction": peak_width_fraction,
            "season_start_week": season_start_week,
            "smooth_window": smooth_window,
            "n_last_seasons": n_last_seasons,
            "shap_max_test_samples": shap_max_test_samples,
            "shap_background_size": shap_background_size,
            "br_forecast_type": br_forecast_type if forecast_engine == "br" else None,
            "br_method": br_method if forecast_engine == "br" else None,
            "br_forecast_duration_weeks": br_forecast_duration_weeks if forecast_engine == "br" else None,
            "br_calibration_window_weeks": br_calibration_window_weeks if forecast_engine == "br" else None,
        },
        weather_summary=weather_summary,
        shap_summary=shap_summary,
        forecast_engine=forecast_engine,
        mechanistic_result=br_payload,
    )
    if include_forecast and forecast_engine == "gbdt" and forecast_result is None:
        context["short_term_forecast"] = {
            "status": "not_included",
            "reason": "GBDT forecasting requires either include_weather=true for source DB data or the fixed bundled SPB workflow.",
        }
        context["limitations"].append("GBDT forecast was skipped because no suitable merged weekly weather table was available.")
    if include_forecast and forecast_engine == "br" and br_payload is None:
        context["short_term_forecast"] = {
            "status": "not_included",
            "reason": "BR calibration was disabled or unavailable.",
        }
        context["limitations"].append("BR forecast was skipped because mechanistic calibration did not complete.")
    if forecast_engine == "gbdt" and include_shap and shap_result is None:
        context["forecast_explainability"] = {
            "status": "not_included",
            "reason": "SHAP requires a fitted GBDT forecast state; forecasting was disabled or unavailable.",
        }

    markdown = render_bulletin_context_markdown(context)
    artifact_metadata = _artifact_store().save_bulletin_context(
        context=context,
        markdown=markdown,
        weekly=bundle.weekly,
        age_groups=bundle.age_groups,
        weather_hourly=weather_bundle.hourly if weather_bundle is not None else None,
        weather_weekly=weather_bundle.weekly if weather_bundle is not None else None,
        merged_weekly=merged_weekly,
        shap_global_importance=shap_result.global_importance if shap_result is not None else None,
        shap_local_values=shap_result.local_values if shap_result is not None else None,
        shap_worst_cases=shap_result.worst_cases if shap_result is not None else None,
        br_trajectory=br_result.trajectory if br_result is not None else None,
        br_parameter_samples=br_result.parameter_samples if br_result is not None else None,
        br_summary={
            "configuration": br_result.configuration,
            "parameter_summary": br_result.parameter_summary,
            "diagnostics": br_result.diagnostics,
            "limitations": br_result.limitations,
        }
        if br_result is not None
        else None,
        br_figures=br_figures,
        user_id=user_id,
        session_id=session_id,
    )
    context_summary = {
        "schema_version": context["schema_version"],
        "purpose": context["purpose"],
        "forecast_engine": forecast_engine,
        "latest_week": context["current_situation"]["latest_week"],
        "weather_status": context["weather_source"].get("status"),
        "age_group_season": context["age_group_patterns"]["season"],
        "wave_seasons": context["epidemic_wave_comparison"]["season_labels"],
        "forecast_status": context["short_term_forecast"].get("status"),
    }
    if forecast_engine == "gbdt":
        context_summary["shap_status"] = context["forecast_explainability"].get(
            "status", "included" if shap_result is not None else "not_included"
        )
    else:
        context_summary["mechanistic_parameters_status"] = context["mechanistic_model_interpretation"].get("status")
        context_summary["gamma_status"] = context["mechanistic_model_interpretation"].get("gamma", {}).get("status")

    metadata = {
        "city": db_summary["city"],
        "request": db_summary["request"],
        "date_range": db_summary["date_range"],
        "source_url": db_summary["source_url"],
        "bulletin_context": context,
        "context_summary": context_summary,
        "result_delivery": {
            "mode": "inline_bulletin_context_plus_s3_artifacts",
            "storage": "s3_compatible",
            "authentication": "server_side_s3_influenza_db_and_weather_credentials",
            "client_download_access": "temporary_presigned_urls",
        },
        **artifact_metadata,
    }
    if forecast_engine == "br":
        answer = (
            "Prepared and returned an inline influenza bulletin evidence packet using the compact BR mechanistic "
            f"forecast engine for {db_summary['city']['name_ru']}; alpha/beta evidence and full artifacts were uploaded to S3."
        )
    else:
        answer = (
            "Prepared and returned an inline influenza bulletin evidence packet using the GBDT forecast engine with "
            f"SHAP support for {db_summary['city']['name_ru']}; full JSON/Markdown and tabular artifacts were uploaded to S3."
        )
    return _ok(answer=answer, metadata=metadata)


@mcp.tool()
def run_influenza_forecasting(session_id: str, user_id: str, origin_date: str | None = None) -> dict[str, Any]:
    """Run the fixed four-week influenza forecasting workflow and persist result artifacts."""
    analytics = service.run_influenza_forecasting(origin_date=origin_date)
    artifact_metadata = _artifact_store().save_forecasting_run(result=analytics, user_id=user_id, session_id=session_id)
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
            f"origin date {analytics['forecast_origin_date']}; full artifacts are available through temporary download URLs."
        ),
        metadata=metadata,
    )


@mcp.tool()
def compare_epidemic_waves(season_start_week: int = 40, smooth_window: int = 3, n_last_seasons: int = 3) -> dict[str, Any]:
    """Compare recent epidemic waves by peak height, peak timing, width and burden."""
    metadata = service.compare_epidemic_waves(
        season_start_week=season_start_week,
        smooth_window=smooth_window,
        n_last_seasons=n_last_seasons,
    )
    return _ok(
        answer=f"Compared {len(metadata['waves'])} recent epidemic waves: " + ", ".join(metadata["season_labels"]) + ".",
        metadata=metadata,
    )



if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7331, path="/mcp")
