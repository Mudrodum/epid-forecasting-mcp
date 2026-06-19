"""Structured bulletin-context assembly for external narrative generation.

This module does not generate prose with an LLM. It prepares a compact,
machine-readable evidence packet that an external agent such as Codex can use
when drafting surveillance comments or a bulletin.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

ForecastEngine = Literal["gbdt", "br"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _round(value: Any, digits: int = 3) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, digits)


def _recent_situation(weekly: pd.DataFrame, *, trend_window_weeks: int = 4) -> dict[str, Any]:
    """Summarize the latest aggregate incidence level and short recent trend."""
    if trend_window_weeks < 1:
        raise ValueError("trend_window_weeks must be positive.")
    required = {"datetime", "inc_per_10k", "iso_year", "iso_week"}
    missing = sorted(required - set(weekly.columns))
    if missing:
        raise ValueError(f"weekly frame is missing required columns: {missing}.")

    df = weekly.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["inc_per_10k"] = pd.to_numeric(df["inc_per_10k"], errors="coerce")
    df = df.dropna(subset=["datetime", "inc_per_10k"]).sort_values("datetime").reset_index(drop=True)
    if df.empty:
        raise ValueError("weekly frame has no valid rows for bulletin context.")

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else None
    recent = df.tail(trend_window_weeks)
    previous_window = df.iloc[max(0, len(df) - 2 * trend_window_weeks) : max(0, len(df) - trend_window_weeks)]
    latest_incidence = _round(latest["inc_per_10k"])
    previous_incidence = _round(previous["inc_per_10k"]) if previous is not None else None
    week_over_week_change_abs = (
        _round(float(latest["inc_per_10k"]) - float(previous["inc_per_10k"])) if previous is not None else None
    )
    week_over_week_change_pct = (
        _round((float(latest["inc_per_10k"]) - float(previous["inc_per_10k"])) / float(previous["inc_per_10k"]) * 100.0, 1)
        if previous is not None and float(previous["inc_per_10k"]) != 0.0
        else None
    )
    recent_mean = _round(recent["inc_per_10k"].mean())
    previous_window_mean = _round(previous_window["inc_per_10k"].mean()) if not previous_window.empty else None
    recent_vs_previous_window_pct = (
        _round((float(recent_mean) - float(previous_window_mean)) / float(previous_window_mean) * 100.0, 1)
        if recent_mean is not None and previous_window_mean not in (None, 0.0)
        else None
    )
    return _jsonable(
        {
            "latest_week": {
                "date": pd.Timestamp(latest["datetime"]).date().isoformat(),
                "iso_year": int(latest["iso_year"]),
                "iso_week": int(latest["iso_week"]),
                "inc_per_10k": latest_incidence,
            },
            "previous_week": None
            if previous is None
            else {
                "date": pd.Timestamp(previous["datetime"]).date().isoformat(),
                "iso_year": int(previous["iso_year"]),
                "iso_week": int(previous["iso_week"]),
                "inc_per_10k": previous_incidence,
            },
            "week_over_week_change_abs": week_over_week_change_abs,
            "week_over_week_change_pct": week_over_week_change_pct,
            "recent_window_weeks": int(trend_window_weeks),
            "recent_mean_inc_per_10k": recent_mean,
            "previous_window_mean_inc_per_10k": previous_window_mean,
            "recent_vs_previous_window_pct": recent_vs_previous_window_pct,
        }
    )


def _forecast_brief(forecast_result: dict[str, Any] | None) -> dict[str, Any]:
    if forecast_result is None:
        return {
            "status": "not_included",
            "reason": "Forecasting was disabled or unavailable for this bulletin context.",
        }
    return _jsonable(
        {
            "status": "included",
            "forecast_engine": "gbdt",
            "forecast_origin_date": forecast_result.get("forecast_origin_date"),
            "target_variable": forecast_result.get("target_variable"),
            "forecast_horizon_weeks": forecast_result.get("forecast_horizon_weeks"),
            "forecast": forecast_result.get("forecast", []),
            "holdout_evaluation": forecast_result.get("holdout_evaluation"),
            "uncertainty_bounds": forecast_result.get("forecast_uncertainty_bounds"),
        }
    )


def _mechanistic_forecast_brief(mechanistic_result: dict[str, Any] | None) -> dict[str, Any]:
    if mechanistic_result is None:
        return {
            "status": "not_included",
            "reason": "Mechanistic BR forecasting was disabled or unavailable for this bulletin context.",
        }
    diagnostics = mechanistic_result.get("diagnostics", {})
    sample_kind = diagnostics.get("sample_kind", "calibration_parameter_samples")
    if sample_kind == "conditional_gaussian_log1p_pseudo_posterior":
        interpretation = (
            "The 80% interval reflects quantiles across conditional Gaussian log1p-residual pseudo-posterior "
            "parameter draws; it is not a split-conformal prediction interval."
        )
    else:
        interpretation = (
            "The 80% interval reflects quantiles across objective-based calibration parameter samples; "
            "it is not a split-conformal prediction interval or a Bayesian posterior interval."
        )
    return _jsonable(
        {
            "status": "included",
            "forecast_engine": "br",
            "forecast_origin_date": mechanistic_result.get("forecast_origin_date"),
            "target_variable": "estimated influenza cases and incidence per 10,000",
            "forecast_horizon_weeks": mechanistic_result.get("forecast_horizon_weeks"),
            "forecast": mechanistic_result.get("forecast", []),
            "uncertainty_bounds": {
                "method": diagnostics.get("forecast_interval_method", "parameter_sample_quantiles"),
                "sample_kind": sample_kind,
                "lower_column": "pi80_lower_inc_per_10k",
                "upper_column": "pi80_upper_inc_per_10k",
                "interpretation": interpretation,
            },
        }
    )


def _forecast_model_summary(
    *,
    forecast_engine: ForecastEngine,
    forecast_result: dict[str, Any] | None,
    mechanistic_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if forecast_engine == "gbdt":
        return {
            "engine": "gbdt",
            "label": "Direct multi-horizon gradient boosting with split-conformal uncertainty intervals",
            "model_family": (forecast_result or {}).get("fixed_configuration", {}).get(
                "model_family", "HistGradientBoostingRegressor"
            ),
            "forecast_strategy": (forecast_result or {}).get("fixed_configuration", {}).get(
                "forecast_strategy", "direct_multi_step"
            ),
            "explanation_section": "forecast_explainability",
        }
    configuration = (mechanistic_result or {}).get("configuration", {})
    return {
        "engine": "br",
        "label": "Compact Baroyan-Rvachev-style mechanistic renewal model",
        "model_family": configuration.get("model_family", "compact_baroyan_rvachev_style_renewal_model"),
        "forecast_type": configuration.get("forecast_type"),
        "calibration_method": configuration.get("method"),
        "explanation_section": "mechanistic_model_interpretation",
    }


def _mechanistic_interpretation(mechanistic_result: dict[str, Any] | None) -> dict[str, Any]:
    if mechanistic_result is None:
        return {
            "status": "not_included",
            "reason": "Mechanistic BR calibration was disabled or unavailable for this context.",
        }
    return _jsonable(
        {
            "status": "included",
            "method": "calibrated_compact_baroyan_rvachev_style_renewal_model",
            "configuration": mechanistic_result.get("configuration", {}),
            "parameter_summary": mechanistic_result.get("parameter_summary", {}),
            "parameter_meanings": mechanistic_result.get("parameter_meanings", {}),
            "gamma": {
                "status": "not_estimated",
                "reason": (
                    "This compact BR implementation uses a fixed infectivity kernel rather than a separately "
                    "estimated recovery/removal parameter gamma."
                ),
            },
            "calibration_diagnostics": mechanistic_result.get("diagnostics", {}),
            "limitations": mechanistic_result.get("limitations", []),
        }
    )


def build_bulletin_context(
    *,
    city: dict[str, Any],
    request: dict[str, Any],
    source_url: str,
    weekly: pd.DataFrame,
    age_group_comparison: dict[str, Any],
    wave_comparison: dict[str, Any],
    forecast_result: dict[str, Any] | None,
    date_range: dict[str, Any],
    parameters: dict[str, Any],
    weather_summary: dict[str, Any] | None = None,
    shap_summary: dict[str, Any] | None = None,
    forecast_engine: ForecastEngine = "gbdt",
    mechanistic_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured evidence packet consumed by external writers.

    ``forecast_engine`` selects either the validated GBDT workflow with SHAP
    attribution or the compact BR mechanistic workflow with alpha/beta
    calibration evidence. The two explanation sections are intentionally
    mutually exclusive.
    """
    if forecast_engine not in {"gbdt", "br"}:
        raise ValueError("forecast_engine must be 'gbdt' or 'br'.")

    if forecast_engine == "gbdt":
        forecast_section = _forecast_brief(forecast_result)
        explanation_section_name = "forecast_explainability"
        explanation_section = shap_summary or {
            "status": "not_included",
            "reason": "SHAP explainability was disabled or unavailable for this context.",
        }
        suggested_sections = [
            "current_situation",
            "short_term_forecast",
            "forecast_explainability",
            "epidemic_wave_comparison",
            "age_group_patterns",
            "limitations",
        ]
    else:
        forecast_section = _mechanistic_forecast_brief(mechanistic_result)
        explanation_section_name = "mechanistic_model_interpretation"
        explanation_section = _mechanistic_interpretation(mechanistic_result)
        suggested_sections = [
            "current_situation",
            "short_term_forecast",
            "mechanistic_model_interpretation",
            "epidemic_wave_comparison",
            "age_group_patterns",
            "limitations",
        ]

    limitations = [
        "This is a structured evidence packet, not a finished bulletin.",
        "Narrative comments must be generated outside the MCP server from the supplied evidence only.",
        "Influenza DB authentication is server-side and not included in metadata.",
        "Wave comparison describes geometry of the observed incidence curve; it does not establish causal drivers.",
        "Age-group conclusions are limited to age-specific incidence/cases available in the DB response.",
    ]
    if weather_summary is not None:
        if forecast_engine == "gbdt":
            limitations.append(
                "Weather covariates are obtained from Open-Meteo and used as weekly environmental predictors; "
                "they do not establish causal attribution."
            )
        else:
            limitations.append(
                "Weather observations are included as contextual evidence only; the compact BR calibration does not use "
                "weather covariates as model inputs."
            )
    if forecast_engine == "gbdt" and shap_summary is not None:
        limitations.append(
            "SHAP values explain the fitted forecast model on held-out feature-complete rows; they are model-attribution "
            "evidence, not causal effects."
        )
    if forecast_engine == "gbdt" and forecast_section["status"] == "included":
        limitations.append(
            "The GBDT forecast uses the fixed direct multi-horizon workflow; its uncertainty bounds are calibration-derived "
            "split-conformal bounds and are not causal estimates."
        )
    if forecast_engine == "br" and forecast_section["status"] == "included":
        limitations.extend(
            [
                "The BR forecast interval is derived from calibration parameter samples and is not a split-conformal interval.",
                "Alpha and beta are fitted model parameters, not directly observed biological quantities.",
                "No separately estimated gamma parameter is available in this compact BR implementation; infectiousness duration "
                "is represented by a fixed kernel.",
            ]
        )

    context = {
        "schema_version": "epid_forecasting.bulletin_context.v3",
        "prepared_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "purpose": "structured_evidence_for_external_bulletin_writing",
        "city": city,
        "request": request,
        "date_range": date_range,
        "source": {
            "influenza_db_url": source_url,
            "token_policy": "redacted_server_side_only",
        },
        "parameters": parameters,
        "forecast_model": _forecast_model_summary(
            forecast_engine=forecast_engine,
            forecast_result=forecast_result,
            mechanistic_result=mechanistic_result,
        ),
        "writing_constraints": [
            "Use only the quantitative evidence in this context and referenced artifacts.",
            "Do not infer causal explanations from correlations, SHAP attributions, or curve geometry.",
            "Do not declare formal epidemic thresholds unless a threshold is explicitly present in evidence.",
            "Keep uncertainty and data-coverage caveats visible.",
            "For the BR engine, do not describe alpha or beta as directly measured biological constants.",
        ],
        "suggested_report_sections": suggested_sections,
        "current_situation": _recent_situation(weekly, trend_window_weeks=int(parameters.get("trend_window_weeks", 4))),
        "weather_source": weather_summary or {"status": "not_included"},
        "short_term_forecast": forecast_section,
        "epidemic_wave_comparison": wave_comparison,
        "age_group_patterns": age_group_comparison,
        "limitations": limitations,
    }
    context[explanation_section_name] = explanation_section
    return _jsonable(context)


def render_bulletin_context_markdown(context: dict[str, Any]) -> str:
    """Render the evidence packet as Markdown for human inspection."""
    city = context.get("city", {})
    current = context.get("current_situation", {})
    latest = current.get("latest_week", {}) or {}
    forecast = context.get("short_term_forecast", {})
    weather = context.get("weather_source", {})
    model = context.get("forecast_model", {})
    waves = context.get("epidemic_wave_comparison", {})
    age = context.get("age_group_patterns", {})
    lines = [
        "# Influenza surveillance bulletin context",
        "",
        "This file is a structured evidence packet, not a final bulletin.",
        "",
        "## City and period",
        "",
        f"- City: {city.get('name_ru', city.get('slug', 'unknown'))} (`{city.get('slug', 'unknown')}`)",
        f"- Date range: {context.get('date_range', {}).get('start')} to {context.get('date_range', {}).get('end')}",
        f"- Prepared at UTC: {context.get('prepared_at_utc')}",
        "",
        "## Forecast model",
        "",
        f"- Engine: {model.get('engine')}",
        f"- Model: {model.get('label')}",
        f"- Forecast type: {model.get('forecast_type', 'aggregate incidence')}",
        "",
        "## Current situation evidence",
        "",
        f"- Latest week: {latest.get('date')} / ISO week {latest.get('iso_year')}-W{latest.get('iso_week')}",
        f"- Latest incidence per 10,000: {latest.get('inc_per_10k')}",
        f"- Week-over-week absolute change: {current.get('week_over_week_change_abs')}",
        f"- Week-over-week percent change: {current.get('week_over_week_change_pct')}",
        f"- Recent {current.get('recent_window_weeks')} week mean incidence: {current.get('recent_mean_inc_per_10k')}",
        "",
        "## Weather-source evidence",
        "",
        f"- Status: {weather.get('status', 'included' if weather else 'not_included')}",
        f"- Source: {weather.get('source')}",
        f"- Location: {weather.get('location')}",
        "",
        "## Forecast evidence",
        "",
        f"- Status: {forecast.get('status')}",
    ]
    if forecast.get("status") == "included":
        lines.extend(
            [
                f"- Forecast origin date: {forecast.get('forecast_origin_date')}",
                f"- Forecast horizon weeks: {forecast.get('forecast_horizon_weeks')}",
                "- Forecast rows are available in `short_term_forecast.forecast` in the JSON context.",
            ]
        )
    else:
        lines.append(f"- Reason: {forecast.get('reason')}")

    if model.get("engine") == "br":
        mechanical = context.get("mechanistic_model_interpretation", {})
        parameters = mechanical.get("parameter_summary", {})
        lines.extend(
            [
                "",
                "## Mechanistic model interpretation evidence",
                "",
                f"- Status: {mechanical.get('status')}",
                f"- Method: {mechanical.get('method')}",
                "- Alpha/beta estimates are available in `mechanistic_model_interpretation.parameter_summary`.",
                f"- Gamma: {mechanical.get('gamma', {}).get('status')} — {mechanical.get('gamma', {}).get('reason')}",
            ]
        )
        for name, value in (parameters.get("best_fit") or {}).items():
            lines.append(f"- Best-fit {name}: {value}")
    else:
        shap = context.get("forecast_explainability", {})
        lines.extend(
            [
                "",
                "## SHAP forecast-driver evidence",
                "",
                f"- Status: {shap.get('status', 'included' if shap.get('method') else 'not_included')}",
                f"- Method: {shap.get('method')}",
                f"- Key insight: {shap.get('key_insight') or shap.get('summary', {}).get('key_insight')}",
                "- Detailed SHAP rows are available in `forecast_explainability.by_horizon` and referenced artifacts.",
            ]
        )

    lines.extend(
        [
            "",
            "## Epidemic-wave comparison evidence",
            "",
            f"- Compared seasons: {', '.join(waves.get('season_labels', []))}",
            f"- Latest wave status: {waves.get('latest_wave_status')}",
            "- Detailed wave rows are available in `epidemic_wave_comparison.waves`.",
            "",
            "## Age-group evidence",
            "",
            f"- Season: {age.get('season')}",
            "- Ranked age-group rows are available in `age_group_patterns.age_group_summary`.",
            "",
            "## Evidence-use constraints",
            "",
            *[f"- {item}" for item in context.get("writing_constraints", [])],
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in context.get("limitations", [])])
    lines.append("")
    return "\n".join(lines)
