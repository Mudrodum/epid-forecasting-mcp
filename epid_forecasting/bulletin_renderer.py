"""Deterministic Markdown/HTML/PDF renderer for influenza bulletin evidence packets.

The module deliberately does not call an LLM and does not perform epidemiological
calculations. A caller supplies the authored bulletin text; this renderer combines
that text with the previously persisted evidence packet and produces auditable
Markdown, HTML and PDF artifacts.
"""

from __future__ import annotations

import base64
import html
import os
import re
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape

RENDERER_SCHEMA_VERSION = "epid_forecasting.rendered_bulletin.v1"

PLACEHOLDERS: dict[str, str] = {
    "{{FORECAST_FIGURE}}": "forecast_figure",
    "{{WAVES_FIGURE}}": "waves_figure",
    "{{AGE_GROUPS_FIGURE}}": "age_groups_figure",
    "{{FORECAST_TABLE}}": "forecast_table",
    "{{AGE_GROUPS_TABLE}}": "age_groups_table",
    "{{SHAP_FIGURE}}": "shap_figure",
    "{{SHAP_TABLE}}": "shap_table",
    "{{MECHANISTIC_PARAMETERS}}": "mechanistic_parameters",
}


# A production bulletin is operational only when the forecast origin is close to
# the last observed surveillance week. The renderer cannot recompute a forecast,
# but it can make an old forecast visibly non-operational rather than silently
# presenting it as a current 1-4 week outlook.
OPERATIONAL_FORECAST_MAX_AGE_DAYS = 14

_FIGURE_DESCRIPTIONS: dict[str, str] = {
    "forecast_figure": "Наблюдаемая заболеваемость и краткосрочный прогноз.",
    "waves_figure": "Сравнение последних эпидемических волн.",
    "age_groups_figure": "Возрастная динамика заболеваемости в выбранном сезоне.",
    "shap_figure": "Глобальная важность признаков SHAP.",
    "br_forecast_ru": "Механистический прогноз BR.",
    "br_alpha_distribution": "Распределение калибровочного параметра alpha.",
    "br_beta_distribution": "Распределение калибровочного параметра beta.",
}

_DEFAULT_EVIDENCE_ORDER = [
    "forecast_figure",
    "forecast_table",
    "waves_figure",
    "age_groups_figure",
    "age_groups_table",
    "shap_figure",
    "shap_table",
    "mechanistic_parameters",
    "br_forecast_ru",
    "br_alpha_distribution",
    "br_beta_distribution",
]

_SHAP_FEATURE_GROUP_LABELS = {
    "calendar_features": "Календарные признаки",
    "fourier_seasonality": "Сезонные гармоники",
    "target_lags": "Лаги заболеваемости",
    "target_rolling_stats": "Скользящие статистики заболеваемости",
    "epidemic_dynamics": "Динамика эпидемического процесса",
    "temperature_lags": "Температурные лаги",
    "temperature_rolling_stats": "Скользящие статистики температуры",
    "weather_current": "Текущие погодные показатели",
    "humidity_current": "Текущая влажность",
    "other": "Прочие признаки",
    # Compatibility aliases from early explainability artifacts.
    "incidence_lag": "Лаги заболеваемости",
    "incidence_rolling": "Скользящие статистики заболеваемости",
    "incidence_rolling_stats": "Скользящие статистики заболеваемости",
    "seasonality": "Сезонность",
}

_SHAP_DIRECTION_LABELS = {
    "increases_prediction": "Повышает прогноз",
    "decreases_prediction": "Снижает прогноз",
    "neutral": "Нейтрально",
}

_FONT_NAME = "EpidBulletinUnicode"
_FONT_BOLD_NAME = "EpidBulletinUnicodeBold"
_FONT_REGISTERED = False


@dataclass(frozen=True)
class BulletinRenderOutput:
    """Binary bulletin deliverables and deterministic rendering metadata."""

    markdown: str
    html: str
    pdf: bytes
    figures: dict[str, bytes]
    manifest: dict[str, Any]


def _font_candidates(*, bold: bool) -> list[Path]:
    windows_dir = Path(os.environ.get("WINDIR", r"C:\\Windows")) / "Fonts"
    candidates = [
        os.getenv("BULLETIN_PDF_FONT_BOLD_PATH" if bold else "BULLETIN_PDF_FONT_PATH", ""),
        str(windows_dir / ("arialbd.ttf" if bold else "arial.ttf")),
        str(windows_dir / ("calibrib.ttf" if bold else "calibri.ttf")),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    return [Path(item) for item in candidates if item]


def _register_unicode_fonts() -> tuple[str, str]:
    """Register a local Unicode TTF for Cyrillic report output without shipping a font file."""

    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return _FONT_NAME, _FONT_BOLD_NAME

    regular = next((path for path in _font_candidates(bold=False) if path.is_file()), None)
    bold = next((path for path in _font_candidates(bold=True) if path.is_file()), None)
    if regular is None or bold is None:
        raise RuntimeError(
            "A Unicode TrueType font is required for Cyrillic PDF rendering. Set BULLETIN_PDF_FONT_PATH and "
            "BULLETIN_PDF_FONT_BOLD_PATH, or install a standard system font such as Arial/DejaVu Sans."
        )
    pdfmetrics.registerFont(TTFont(_FONT_NAME, str(regular)))
    pdfmetrics.registerFont(TTFont(_FONT_BOLD_NAME, str(bold)))
    _FONT_REGISTERED = True
    return _FONT_NAME, _FONT_BOLD_NAME


def _to_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)


def _figure_to_png(figure: Any) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _as_date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype="datetime64[ns]")


def _figure_forecast(
    context: dict[str, Any],
    weekly: pd.DataFrame,
    br_trajectory: pd.DataFrame | None,
    forecast_timeliness: dict[str, Any],
) -> bytes | None:
    short_forecast = context.get("short_term_forecast", {})
    if short_forecast.get("status") != "included":
        return None

    history = weekly.copy()
    if "datetime" not in history or "inc_per_10k" not in history:
        return None
    history["datetime"] = _as_date_series(history, "datetime")
    history["inc_per_10k"] = _to_numeric(history, "inc_per_10k")
    history = history.dropna(subset=["datetime", "inc_per_10k"]).sort_values("datetime").tail(32)
    if history.empty:
        return None

    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    axis.plot(history["datetime"], history["inc_per_10k"], marker="o", label="Наблюдения")
    engine = short_forecast.get("forecast_engine", context.get("forecast_model", {}).get("engine"))

    if engine == "br":
        trajectory = br_trajectory.copy() if br_trajectory is not None else pd.DataFrame(short_forecast.get("forecast", []))
        if trajectory.empty:
            return _figure_to_png(figure)
        trajectory["datetime"] = _as_date_series(trajectory, "datetime")
        forecast = trajectory.loc[trajectory.get("is_forecast", False).fillna(False)].copy() if "is_forecast" in trajectory else trajectory
        if forecast.empty:
            forecast = trajectory
        forecast = forecast.dropna(subset=["datetime"]).sort_values("datetime")
        point_col = "fitted_inc_per_10k"
        lower_col = "pi80_lower_inc_per_10k"
        upper_col = "pi80_upper_inc_per_10k"
        if point_col in forecast:
            axis.plot(forecast["datetime"], _to_numeric(forecast, point_col), marker="o", label="BR-прогноз")
        if lower_col in forecast and upper_col in forecast:
            axis.fill_between(
                forecast["datetime"],
                _to_numeric(forecast, lower_col),
                _to_numeric(forecast, upper_col),
                alpha=0.25,
                label="80% интервал параметрической неопределенности",
            )
        title = "Краткосрочный механистический прогноз BR"
    else:
        forecast = pd.DataFrame(short_forecast.get("forecast", []))
        if not forecast.empty and "target_date" in forecast:
            forecast["target_date"] = _as_date_series(forecast, "target_date")
            forecast = forecast.dropna(subset=["target_date"]).sort_values("target_date")
            axis.plot(
                forecast["target_date"],
                _to_numeric(forecast, "inc_per_10k_prediction"),
                marker="o",
                label="GBDT-прогноз",
            )
            if {"pi80_lower", "pi80_upper"}.issubset(forecast.columns):
                axis.fill_between(
                    forecast["target_date"],
                    _to_numeric(forecast, "pi80_lower"),
                    _to_numeric(forecast, "pi80_upper"),
                    alpha=0.25,
                    label="80% split-conformal интервал",
                )
        title = (
            f"Архивный прогноз GBDT (origin: {forecast_timeliness.get('forecast_origin_date')})"
            if forecast_timeliness.get("is_stale")
            else "Краткосрочный прогноз GBDT"
        )

    axis.set_title(title)
    axis.set_ylabel("На 10 тыс. населения")
    axis.set_xlabel("Дата")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    figure.autofmt_xdate()
    return _figure_to_png(figure)


def _figure_waves(context: dict[str, Any]) -> bytes | None:
    waves = context.get("epidemic_wave_comparison", {}).get("waves", [])
    if not waves:
        return None
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    plotted = False
    for wave in waves:
        points = pd.DataFrame(wave.get("plot_points", []))
        if points.empty or not {"pos", "value_smooth"}.issubset(points.columns):
            continue
        axis.plot(
            _to_numeric(points, "pos"),
            _to_numeric(points, "value_smooth"),
            label=str(wave.get("season_label", "Сезон")),
        )
        peak_value = wave.get("peak_value")
        peak_pos = None
        peak_week = wave.get("peak_week")
        if peak_week is not None and "week" in points.columns:
            matches = points.loc[pd.to_numeric(points["week"], errors="coerce") == float(peak_week)]
            if not matches.empty:
                peak_pos = float(pd.to_numeric(matches["pos"], errors="coerce").iloc[0])
        if peak_pos is None:
            smooth = _to_numeric(points, "value_smooth")
            if not smooth.empty and smooth.notna().any():
                peak_pos = float(_to_numeric(points, "pos").iloc[int(smooth.idxmax())])
        if peak_pos is not None and peak_value is not None:
            axis.scatter([peak_pos], [float(peak_value)], s=24)
        plotted = True
    if not plotted:
        plt.close(figure)
        return None
    axis.set_title("Сравнение последних эпидемических волн")
    axis.set_xlabel("Эпидемическая неделя сезона")
    axis.set_ylabel("На 10 тыс. населения")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    return _figure_to_png(figure)


def _figure_age_groups(context: dict[str, Any], age_groups: pd.DataFrame) -> bytes | None:
    if age_groups.empty or not {"datetime", "age_group", "inc_per_10k"}.issubset(age_groups.columns):
        return None
    requested_season = context.get("age_group_patterns", {}).get("season")
    frame = age_groups.copy()
    frame["datetime"] = _as_date_series(frame, "datetime")
    frame["inc_per_10k"] = _to_numeric(frame, "inc_per_10k")
    if requested_season and "season" in frame:
        frame = frame.loc[frame["season"].astype(str) == str(requested_season)].copy()
    frame = frame.dropna(subset=["datetime", "inc_per_10k"]).sort_values("datetime")
    if frame.empty:
        return None

    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    for age_group, group in frame.groupby("age_group", sort=True):
        axis.plot(group["datetime"], group["inc_per_10k"], label=str(age_group))
    axis.set_title("Возрастная динамика заболеваемости")
    axis.set_xlabel("Дата")
    axis.set_ylabel("На 10 тыс. населения")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncol=3, loc="best")
    figure.autofmt_xdate()
    return _figure_to_png(figure)


def _figure_shap(global_importance: pd.DataFrame | None) -> bytes | None:
    if global_importance is None or global_importance.empty:
        return None
    required = {"horizon_weeks", "feature", "rank", "mean_abs_shap"}
    if not required.issubset(global_importance.columns):
        return None
    frame = global_importance.copy()
    frame["horizon_weeks"] = _to_numeric(frame, "horizon_weeks")
    preferred_horizon = 1 if (frame["horizon_weeks"] == 1).any() else int(frame["horizon_weeks"].dropna().min())
    top = frame.loc[frame["horizon_weeks"] == preferred_horizon].sort_values("rank").head(10).copy()
    if top.empty:
        return None
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    labels = top["feature"].astype(str).tolist()[::-1]
    values = _to_numeric(top, "mean_abs_shap").tolist()[::-1]
    axis.barh(labels, values)
    axis.set_title(f"SHAP: наиболее важные признаки, горизонт h={preferred_horizon}")
    axis.set_xlabel("Средний абсолютный SHAP-вклад")
    axis.grid(True, axis="x", alpha=0.25)
    return _figure_to_png(figure)


def _format_number(value: Any, digits: int = 3) -> str:
    """Format numeric table values using the Russian decimal separator."""

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    try:
        return f"{float(value):.{digits}f}".replace(".", ",")
    except (TypeError, ValueError):
        return str(value)


def _translate_shap_feature_group(value: Any) -> str:
    raw = str(value or "-")
    return _SHAP_FEATURE_GROUP_LABELS.get(raw, raw.replace("_", " "))


def _translate_shap_direction(value: Any) -> str:
    raw = str(value or "-")
    return _SHAP_DIRECTION_LABELS.get(raw, raw.replace("_", " "))


def _forecast_timeliness(context: dict[str, Any]) -> dict[str, Any]:
    """Describe whether a stored forecast is operationally current for a bulletin."""

    short = context.get("short_term_forecast", {})
    latest = context.get("current_situation", {}).get("latest_week", {}).get("date")
    origin = short.get("forecast_origin_date")
    result: dict[str, Any] = {
        "status": "unknown",
        "latest_observation_date": latest,
        "forecast_origin_date": origin,
        "age_days": None,
        "max_operational_age_days": OPERATIONAL_FORECAST_MAX_AGE_DAYS,
        "is_stale": False,
        "warning": None,
    }
    if short.get("status") != "included":
        result["status"] = "not_included"
        return result
    if not latest or not origin:
        return result
    latest_ts = pd.to_datetime(latest, errors="coerce")
    origin_ts = pd.to_datetime(origin, errors="coerce")
    if pd.isna(latest_ts) or pd.isna(origin_ts):
        return result
    age_days = int((latest_ts.normalize() - origin_ts.normalize()).days)
    result["age_days"] = age_days
    result["status"] = "current" if age_days <= OPERATIONAL_FORECAST_MAX_AGE_DAYS else "stale"
    result["is_stale"] = age_days > OPERATIONAL_FORECAST_MAX_AGE_DAYS
    if result["is_stale"]:
        result["warning"] = (
            "Внимание: прогноз построен от "
            f"{origin_ts.date().isoformat()} и отстает от последней наблюдаемой недели "
            f"{latest_ts.date().isoformat()} на {age_days} дней. "
            "Он приведен как архивный модельный результат и не является текущим оперативным прогнозом."
        )
    return result


def _forecast_table_rows(context: dict[str, Any], forecast_timeliness: dict[str, Any] | None = None) -> list[list[str]]:
    short = context.get("short_term_forecast", {})
    if short.get("status") != "included":
        return [["Прогноз не включен в evidence packet."]]
    engine = short.get("forecast_engine", context.get("forecast_model", {}).get("engine"))
    rows = short.get("forecast", [])
    if engine == "br":
        table = [["Горизонт", "Дата", "Группа", "Прогноз /10 тыс.", "80% интервал"]]
        for index, row in enumerate(rows, start=1):
            table.append(
                [
                    f"h={index}",
                    str(row.get("datetime", "-")),
                    str(row.get("group", "total")),
                    _format_number(row.get("fitted_inc_per_10k")),
                    f"[{_format_number(row.get('pi80_lower_inc_per_10k'))}; {_format_number(row.get('pi80_upper_inc_per_10k'))}]",
                ]
            )
        return table
    forecast_label = "Архивный прогноз /10 тыс." if (forecast_timeliness or {}).get("is_stale") else "Прогноз /10 тыс."
    table = [["Горизонт", "Дата", forecast_label, "80% интервал"]]
    for row in rows:
        table.append(
            [
                f"h={row.get('horizon_weeks', '-')}",
                str(row.get("target_date", "-")),
                _format_number(row.get("inc_per_10k_prediction")),
                f"[{_format_number(row.get('pi80_lower'))}; {_format_number(row.get('pi80_upper'))}]",
            ]
        )
    return table


def _age_group_table_rows(context: dict[str, Any]) -> list[list[str]]:
    summary = context.get("age_group_patterns", {}).get("age_group_summary", [])
    if not summary:
        return [["Возрастная сводка не включена в evidence packet."]]
    table = [["Группа", "Случаи", "Пик /10 тыс.", "Пик, нед.", "Сезонная нагрузка", "Доля"]]
    total_cases = sum(float(row.get("total_cases", 0) or 0) for row in summary)
    for row in summary:
        cases = float(row.get("total_cases", 0) or 0)
        share = cases / total_cases * 100.0 if total_cases else None
        table.append(
            [
                str(row.get("age_group", "-")),
                _format_number(cases, 0),
                _format_number(row.get("peak_inc_per_10k"), 2),
                str(row.get("peak_iso_week", "-")),
                _format_number(row.get("seasonal_burden_inc_week"), 2),
                _format_number(share, 1),
            ]
        )
    return table


def _mechanistic_parameter_rows(context: dict[str, Any]) -> list[list[str]]:
    payload = context.get("mechanistic_model_interpretation", {})
    if payload.get("status") != "included":
        return [["Механистическая интерпретация не включена в evidence packet."]]
    summary = payload.get("parameter_summary", {})
    estimate = summary.get("optimizer_best_fit") or summary.get("best_fit") or {}
    bounds = summary.get("optimizer_parameter_bounds") or payload.get("calibration_diagnostics", {}).get(
        "optimizer_parameter_bounds", {}
    )
    table = [["Параметр", "Оценка", "Статус границы", "Интерпретация"]]
    meanings = payload.get("parameter_meanings", {})
    for name, value in estimate.items():
        bound = bounds.get(name, {}) if isinstance(bounds, dict) else {}
        if bound.get("at_upper_bound"):
            boundary = "верхняя граница"
        elif bound.get("at_lower_bound"):
            boundary = "нижняя граница"
        else:
            boundary = "внутри диапазона"
        table.append(
            [
                str(name),
                _format_number(value),
                boundary,
                str(meanings.get(name, {}).get("label", "Калибровочный параметр")),
            ]
        )
    gamma = payload.get("gamma", {})
    table.append(["gamma", "не оценивается", str(gamma.get("status", "-")), str(gamma.get("reason", "-"))])
    return table


def _shap_table_rows(global_importance: pd.DataFrame | None) -> list[list[str]]:
    if global_importance is None or global_importance.empty:
        return [["SHAP-таблица не включена в evidence packet."]]
    table = [["Горизонт", "Признак", "Группа признаков", "Средний |SHAP|", "Направление"]]
    frame = global_importance.copy().sort_values(["horizon_weeks", "rank"])
    for horizon in sorted(pd.to_numeric(frame["horizon_weeks"], errors="coerce").dropna().astype(int).unique()):
        top = frame.loc[pd.to_numeric(frame["horizon_weeks"], errors="coerce") == horizon].head(3)
        for _, row in top.iterrows():
            table.append(
                [
                    f"h={horizon}",
                    str(row.get("feature", "-")),
                    _translate_shap_feature_group(row.get("feature_group", "-")),
                    _format_number(row.get("mean_abs_shap"), 4),
                    _translate_shap_direction(row.get("direction", "-")),
                ]
            )
    return table


def _title_and_period(context: dict[str, Any], title: str | None) -> tuple[str, str]:
    latest = context.get("current_situation", {}).get("latest_week", {})
    default_title = "ЕЖЕНЕДЕЛЬНЫЙ БЮЛЛЕТЕНЬ ПО ГРИППУ"
    report_title = title.strip() if isinstance(title, str) and title.strip() else default_title
    year = latest.get("iso_year")
    week = latest.get("iso_week")
    day = latest.get("date")
    city = context.get("city", {}).get("name_ru")
    period_parts = []
    if week is not None and year is not None:
        period_parts.append(f"за {week}-ю неделю {year} года")
    if day:
        period_parts.append(f"по данным на {day}")
    if city:
        period_parts.append(str(city))
    return report_title, "; ".join(period_parts)


def _make_figures(
    context: dict[str, Any],
    weekly: pd.DataFrame,
    age_groups: pd.DataFrame,
    shap_global_importance: pd.DataFrame | None,
    br_trajectory: pd.DataFrame | None,
    existing_br_figures: dict[str, bytes] | None,
    forecast_timeliness: dict[str, Any],
) -> dict[str, bytes]:
    figures: dict[str, bytes] = {}
    forecast = _figure_forecast(context, weekly, br_trajectory, forecast_timeliness)
    if forecast is not None:
        figures["forecast_figure"] = forecast
    waves = _figure_waves(context)
    if waves is not None:
        figures["waves_figure"] = waves
    ages = _figure_age_groups(context, age_groups)
    if ages is not None:
        figures["age_groups_figure"] = ages

    if context.get("forecast_model", {}).get("engine") == "gbdt":
        shap = _figure_shap(shap_global_importance)
        if shap is not None:
            figures["shap_figure"] = shap
    for key, payload in (existing_br_figures or {}).items():
        if payload:
            figures.setdefault(key, payload)
    return figures


def _asset_is_available(name: str, figures: dict[str, bytes], asset_tables: dict[str, list[list[str]]]) -> bool:
    return name in figures or name in asset_tables


def _planned_asset_order(
    *,
    context: dict[str, Any],
    writer_markdown: str,
    figures: dict[str, bytes],
    asset_tables: dict[str, list[list[str]]],
    append_missing_evidence: bool,
) -> list[str]:
    """Return each renderable evidence asset once, in its actual display order."""

    order: list[str] = []
    for kind, payload in _parse_markdown_blocks(writer_markdown):
        if kind != "asset":
            continue
        name = str(payload)
        if name not in order and _asset_is_available(name, figures, asset_tables):
            order.append(name)
    if append_missing_evidence:
        engine = context.get("forecast_model", {}).get("engine")
        fallback = [
            "forecast_figure",
            "forecast_table",
            "waves_figure",
            "age_groups_figure",
            "age_groups_table",
        ]
        fallback += (
            ["mechanistic_parameters", "br_forecast_ru", "br_alpha_distribution", "br_beta_distribution"]
            if engine == "br"
            else ["shap_figure", "shap_table"]
        )
        for name in fallback:
            if name not in order and _asset_is_available(name, figures, asset_tables):
                order.append(name)
    return order


def _figure_captions(
    *,
    asset_order: list[str],
    figures: dict[str, bytes],
    forecast_timeliness: dict[str, Any],
) -> dict[str, str]:
    """Number figures in the sequence in which they are actually displayed."""

    captions: dict[str, str] = {}
    number = 0
    for asset in asset_order:
        if asset not in figures or asset in captions:
            continue
        description = _FIGURE_DESCRIPTIONS.get(asset, asset)
        if asset == "forecast_figure" and forecast_timeliness.get("is_stale"):
            description = (
                "Архивный GBDT-прогноз: наблюдаемая заболеваемость и прогноз, "
                f"построенный от {forecast_timeliness.get('forecast_origin_date')}."
            )
        number += 1
        captions[asset] = f"Рис. {number}. {description}"
    return captions


def _strip_redundant_document_preamble(markdown: str, *, title: str, period: str) -> str:
    """Remove a copied title/date preamble because the renderer owns the title block."""

    lines = _sanitize_markdown_text(markdown).splitlines()
    title_norm = re.sub(r"\s+", " ", title.strip().lower())
    period_norm = re.sub(r"\s+", " ", period.strip().lower())
    output: list[str] = []
    preamble = True
    title_removed = False
    for raw_line in lines:
        line = raw_line.strip()
        stripped_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
        normalized = re.sub(r"\s+", " ", stripped_heading.lower())
        if preamble and re.fullmatch(r"(?:страница|page)\s+\d+", line, flags=re.IGNORECASE):
            continue
        if preamble and not title_removed and normalized and (
            normalized == title_norm or normalized.startswith(title_norm + ":")
        ):
            title_removed = True
            continue
        if preamble and title_removed and normalized and (
            normalized == period_norm or re.fullmatch(r"за .+? неделю .+", normalized)
        ):
            continue
        if line:
            preamble = False
        output.append(raw_line)
    return "\n".join(output).strip()


def _html_for_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    head = rows[0]
    body = rows[1:]
    cells = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in head)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in body
    )
    return f"<table><thead><tr>{cells}</tr></thead><tbody>{body_rows}</tbody></table>"


def _html_figure(key: str, figures: dict[str, bytes], captions: dict[str, str]) -> str:
    payload = figures.get(key)
    if payload is None:
        return ""
    data = base64.b64encode(payload).decode("ascii")
    caption = html.escape(captions.get(key, key))
    return f'<figure><img src="data:image/png;base64,{data}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


_INVALID_RENDER_CHARACTERS = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFFFE\uFFFF]")


def _sanitize_markdown_text(value: Any) -> str:
    """Remove non-rendering control characters before HTML/PDF conversion."""

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return _INVALID_RENDER_CHARACTERS.sub("", text)


def _inline_markdown(value: Any, *, target: str) -> str:
    """Render the small Markdown subset accepted from an external bulletin author.

    The renderer intentionally supports only presentation markup (bold, italics
    and inline code). HTML submitted by a caller is always escaped first.
    """

    if target not in {"html", "reportlab"}:
        raise ValueError(f"Unsupported inline Markdown target: {target}")

    code_tokens: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code = html.escape(match.group(1), quote=False)
        if target == "html":
            markup = f"<code>{code}</code>"
        else:
            markup = f'<font name="Courier">{code}</font>'
        token = f"@@EPIDCODE{len(code_tokens)}TOKEN@@"
        code_tokens.append(markup)
        return token

    escaped = re.sub(r"`([^`\n]+)`", stash_code, _sanitize_markdown_text(value))
    escaped = html.escape(escaped, quote=False)

    if target == "html":
        bold_open, bold_close = "<strong>", "</strong>"
        italic_open, italic_close = "<em>", "</em>"
    else:
        bold_open, bold_close = "<b>", "</b>"
        italic_open, italic_close = "<i>", "</i>"

    escaped = re.sub(
        r"(?<!\\)(\*\*|__)(.+?)\1",
        lambda match: f"{bold_open}{match.group(2)}{bold_close}",
        escaped,
    )
    escaped = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        lambda match: f"{italic_open}{match.group(1)}{italic_close}",
        escaped,
    )
    escaped = re.sub(
        r"(?<!_)_([^_\n]+)_(?!_)",
        lambda match: f"{italic_open}{match.group(1)}{italic_close}",
        escaped,
    )
    for index, markup in enumerate(code_tokens):
        escaped = escaped.replace(f"@@EPIDCODE{index}TOKEN@@", markup)
    return escaped


def _inline_markdown_to_html(value: Any) -> str:
    return _inline_markdown(value, target="html")


def _inline_markdown_to_reportlab(value: Any) -> str:
    return _inline_markdown(value, target="reportlab")


def _is_markdown_table_separator(line: str) -> bool:
    cells = _split_markdown_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells)


def _split_markdown_table_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return [cell.strip() for cell in value.split("|")]


def _parse_markdown_blocks(markdown: str) -> list[tuple[str, Any]]:
    """Parse a deliberately small, deterministic Markdown block subset."""

    lines = _sanitize_markdown_text(markdown).splitlines()
    blocks: list[tuple[str, Any]] = []
    paragraph: list[str] = []
    list_kind: str | None = None
    list_items: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(part.strip() for part in paragraph if part.strip())
            if text:
                blocks.append(("paragraph", text))
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_kind
        if list_kind and list_items:
            blocks.append((list_kind, list(list_items)))
        list_kind = None
        list_items.clear()

    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        if line in PLACEHOLDERS:
            flush_paragraph()
            flush_list()
            blocks.append(("asset", PLACEHOLDERS[line]))
            index += 1
            continue
        if not line:
            flush_paragraph()
            flush_list()
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            blocks.append(("heading", (min(3, len(heading.group(1))), heading.group(2).strip())))
            index += 1
            continue

        if index + 1 < len(lines) and "|" in line and _is_markdown_table_separator(lines[index + 1].strip()):
            flush_paragraph()
            flush_list()
            rows = [_split_markdown_table_row(line)]
            index += 2
            while index < len(lines):
                table_line = lines[index].strip()
                if not table_line or "|" not in table_line:
                    break
                rows.append(_split_markdown_table_row(table_line))
                index += 1
            width = len(rows[0])
            normalized = [(row + [""] * width)[:width] for row in rows]
            blocks.append(("markdown_table", normalized))
            continue

        unordered = re.match(r"^[-*+]\s+(.+)$", line)
        ordered = re.match(r"^\d+[.)]\s+(.+)$", line)
        if unordered or ordered:
            flush_paragraph()
            kind = "unordered_list" if unordered else "ordered_list"
            item = (unordered or ordered).group(1).strip()
            if list_kind is not None and list_kind != kind:
                flush_list()
            list_kind = kind
            list_items.append(item)
            index += 1
            continue

        flush_list()
        paragraph.append(line)
        index += 1

    flush_paragraph()
    flush_list()
    return blocks


def _html_for_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "".join(f"<th>{_inline_markdown_to_html(cell)}</th>" for cell in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{_inline_markdown_to_html(cell)}</td>" for cell in row) + "</tr>"
        for row in rows[1:]
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _markdown_to_html_blocks(
    markdown: str,
    assets: dict[str, str],
) -> str:
    blocks: list[str] = []
    for kind, payload in _parse_markdown_blocks(markdown):
        if kind == "asset":
            rendered = assets.get(str(payload), "")
            if rendered:
                blocks.append(rendered)
        elif kind == "paragraph":
            blocks.append(f"<p>{_inline_markdown_to_html(payload)}</p>")
        elif kind == "heading":
            level, text = payload
            blocks.append(f"<h{level + 1}>{_inline_markdown_to_html(text)}</h{level + 1}>")
        elif kind == "unordered_list":
            items = "".join(f"<li>{_inline_markdown_to_html(item)}</li>" for item in payload)
            blocks.append(f"<ul>{items}</ul>")
        elif kind == "ordered_list":
            items = "".join(f"<li>{_inline_markdown_to_html(item)}</li>" for item in payload)
            blocks.append(f"<ol>{items}</ol>")
        elif kind == "markdown_table":
            blocks.append(_html_for_markdown_table(payload))
    return "\n".join(blocks)

def _render_html(
    *,
    context: dict[str, Any],
    writer_markdown: str,
    title: str,
    period: str,
    figures: dict[str, bytes],
    forecast_rows: list[list[str]],
    age_rows: list[list[str]],
    shap_rows: list[list[str]],
    mechanical_rows: list[list[str]],
    append_missing_evidence: bool,
    asset_order: list[str],
    figure_captions: dict[str, str],
    forecast_timeliness: dict[str, Any],
) -> str:
    assets = {
        "forecast_table": _html_for_table(forecast_rows),
        "age_groups_table": _html_for_table(age_rows),
        "shap_table": _html_for_table(shap_rows),
        "mechanistic_parameters": _html_for_table(mechanical_rows),
    }
    for name in figure_captions:
        assets[name] = _html_figure(name, figures, figure_captions)
    body = _markdown_to_html_blocks(writer_markdown, assets)
    used = {PLACEHOLDERS[line.strip()] for line in writer_markdown.splitlines() if line.strip() in PLACEHOLDERS}
    if append_missing_evidence:
        pending = [name for name in asset_order if name not in used and assets.get(name)]
        if pending:
            appendix = ["<h2>Приложение. Расчетные материалы</h2>"] + [assets[name] for name in pending]
            body += "\n" + "\n".join(appendix)
    warning = forecast_timeliness.get("warning")
    warning_html = f'<p class="warning">{html.escape(str(warning))}</p>' if warning else ""
    provenance = html.escape(
        "Текст предоставлен внешним автором. Расчеты и изображения собраны из сохраненного evidence packet; renderer не выполняет повторный прогноз."
    )
    return f"""<!doctype html>
<html lang=\"ru\">
<head>
<meta charset=\"utf-8\">
<title>{html.escape(title)}</title>
<style>
@page {{ size: A4; margin: 18mm; }}
body {{ font-family: Arial, 'DejaVu Sans', sans-serif; line-height: 1.45; color: #111; max-width: 850px; margin: auto; }}
h1 {{ text-align: center; font-size: 24px; margin-bottom: 0.2em; }}
h2 {{ margin-top: 1.6em; font-size: 18px; }}
h3 {{ margin-top: 1.2em; font-size: 15px; }}
.subtitle {{ text-align: center; color: #444; margin-bottom: 2em; }}
figure {{ margin: 1.2em 0; page-break-inside: avoid; }}
img {{ max-width: 100%; height: auto; }}
figcaption {{ font-size: 0.9em; color: #333; margin-top: 0.3em; }}
table {{ width: 100%; border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }}
th, td {{ border: 1px solid #777; padding: 5px; vertical-align: top; }}
th {{ background: #efefef; }}
.provenance {{ margin-top: 2em; font-size: 0.8em; color: #555; }}
.warning {{ border-left: 4px solid #9b5b00; background: #fff4df; padding: 9px 11px; margin: 0 0 1.2em; font-weight: 600; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class=\"subtitle\">{html.escape(period)}</div>
{warning_html}
{body}
<p class=\"provenance\">{provenance}</p>
</body>
</html>"""


def _table_flowable(
    rows: list[list[str]],
    available_width: float,
    *,
    interpret_markdown: bool = False,
) -> Table:
    cell_style = ParagraphStyle("cell", fontName=_FONT_NAME, fontSize=7.2, leading=8.5)

    def render_cell(cell: Any) -> str:
        if interpret_markdown:
            return _inline_markdown_to_reportlab(cell)
        return escape(_sanitize_markdown_text(cell))

    normalized = [[Paragraph(render_cell(cell), cell_style) for cell in row] for row in rows]
    columns = max(1, len(rows[0]))
    table = Table(normalized, colWidths=[available_width / columns] * columns, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD_NAME),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _image_flowable(png: bytes, available_width: float) -> Image:
    image = Image(BytesIO(png))
    original_width, original_height = image.imageWidth, image.imageHeight
    scale = min(1.0, available_width / original_width)
    image.drawWidth = original_width * scale
    image.drawHeight = original_height * scale
    return image


def _pdf_page_number(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont(_FONT_NAME, 8)
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Страница {doc.page}")
    canvas.restoreState()


def _render_pdf(
    *,
    context: dict[str, Any],
    writer_markdown: str,
    title: str,
    period: str,
    figures: dict[str, bytes],
    forecast_rows: list[list[str]],
    age_rows: list[list[str]],
    shap_rows: list[list[str]],
    mechanical_rows: list[list[str]],
    append_missing_evidence: bool,
    asset_order: list[str],
    figure_captions: dict[str, str],
    forecast_timeliness: dict[str, Any],
) -> bytes:
    _register_unicode_fonts()
    buffer = BytesIO()
    margin = 1.6 * cm
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=1.5 * cm,
        bottomMargin=1.6 * cm,
        title=title,
        author="EpidForecasting MCP renderer",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BulletinTitle",
        parent=styles["Title"],
        fontName=_FONT_BOLD_NAME,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "BulletinSubtitle",
        parent=styles["Normal"],
        fontName=_FONT_NAME,
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=18,
    )
    heading_styles = {
        1: ParagraphStyle("Heading1RU", parent=styles["Heading1"], fontName=_FONT_BOLD_NAME, fontSize=15, leading=19, spaceBefore=10, spaceAfter=6),
        2: ParagraphStyle("Heading2RU", parent=styles["Heading2"], fontName=_FONT_BOLD_NAME, fontSize=13, leading=16, spaceBefore=9, spaceAfter=5),
        3: ParagraphStyle("Heading3RU", parent=styles["Heading3"], fontName=_FONT_BOLD_NAME, fontSize=11, leading=14, spaceBefore=7, spaceAfter=4),
    }
    body_style = ParagraphStyle(
        "BodyRU", parent=styles["BodyText"], fontName=_FONT_NAME, fontSize=9.5, leading=13.3, spaceAfter=7
    )
    bullet_style = ParagraphStyle(
        "BulletRU", parent=body_style, leftIndent=12, firstLineIndent=-8, bulletIndent=4
    )
    caption_style = ParagraphStyle(
        "CaptionRU", parent=styles["Normal"], fontName=_FONT_NAME, fontSize=8, leading=10, spaceBefore=3, spaceAfter=9
    )
    note_style = ParagraphStyle(
        "NoteRU", parent=styles["Normal"], fontName=_FONT_NAME, fontSize=7.8, leading=10, textColor=colors.darkgrey, spaceBefore=10
    )
    warning_style = ParagraphStyle(
        "ForecastWarningRU",
        parent=body_style,
        fontName=_FONT_BOLD_NAME,
        borderColor=colors.HexColor("#9B5B00"),
        borderWidth=0.8,
        borderPadding=7,
        backColor=colors.HexColor("#FFF4DF"),
        spaceBefore=0,
        spaceAfter=10,
    )
    story: list[Any] = [
        Paragraph(escape(title), title_style),
        Paragraph(escape(period), subtitle_style),
    ]
    if forecast_timeliness.get("warning"):
        story.append(Paragraph(escape(str(forecast_timeliness["warning"])), warning_style))
    available_width = A4[0] - 2 * margin
    asset_tables = {
        "forecast_table": forecast_rows,
        "age_groups_table": age_rows,
        "shap_table": shap_rows,
        "mechanistic_parameters": mechanical_rows,
    }
    used: set[str] = set()
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(item.strip() for item in paragraph if item.strip())
            if text:
                story.append(Paragraph(escape(text), body_style))
            paragraph.clear()

    def add_asset(asset: str) -> None:
        used.add(asset)
        if asset in asset_tables:
            story.append(_table_flowable(asset_tables[asset], available_width))
            story.append(Spacer(1, 0.18 * cm))
            return
        image_bytes = figures.get(asset)
        if image_bytes:
            story.append(_image_flowable(image_bytes, available_width))
            story.append(Paragraph(escape(figure_captions.get(asset, asset)), caption_style))

    for kind, payload in _parse_markdown_blocks(writer_markdown):
        if kind == "asset":
            flush_paragraph()
            add_asset(str(payload))
        elif kind == "paragraph":
            flush_paragraph()
            story.append(Paragraph(_inline_markdown_to_reportlab(payload), body_style))
        elif kind == "heading":
            flush_paragraph()
            level, text = payload
            story.append(Paragraph(_inline_markdown_to_reportlab(text), heading_styles[level]))
        elif kind in {"unordered_list", "ordered_list"}:
            flush_paragraph()
            for item in payload:
                story.append(Paragraph(_inline_markdown_to_reportlab(item), bullet_style, bulletText="•"))
        elif kind == "markdown_table":
            flush_paragraph()
            story.append(_table_flowable(payload, available_width, interpret_markdown=True))
            story.append(Spacer(1, 0.18 * cm))
    flush_paragraph()

    if append_missing_evidence:
        pending = [name for name in asset_order if name not in used and (name in asset_tables or name in figures)]
        if pending:
            story.append(PageBreak())
            story.append(Paragraph("Приложение. Расчетные материалы", heading_styles[1]))
            for asset in pending:
                add_asset(asset)

    story.append(
        Paragraph(
            escape(
                "Текст предоставлен внешним автором. Расчеты и изображения собраны из сохраненного evidence packet; "
                "renderer не выполняет повторный прогноз и не обращается к внешним источникам данных."
            ),
            note_style,
        )
    )
    document.build(story, onFirstPage=_pdf_page_number, onLaterPages=_pdf_page_number)
    return buffer.getvalue()


def render_influenza_bulletin(
    *,
    context: dict[str, Any],
    writer_markdown: str,
    weekly: pd.DataFrame,
    age_groups: pd.DataFrame,
    merged_weekly: pd.DataFrame | None = None,
    shap_global_importance: pd.DataFrame | None = None,
    br_trajectory: pd.DataFrame | None = None,
    existing_br_figures: dict[str, bytes] | None = None,
    title: str | None = None,
    append_missing_evidence: bool = True,
) -> BulletinRenderOutput:
    """Render an authored bulletin with figures/tables from one saved evidence packet.

    ``writer_markdown`` is narrative content authored outside the MCP server. The
    optional placeholders listed in :data:`PLACEHOLDERS` control where evidence is
    placed. Without placeholders, all unavailable placements are appended in an
    auditable evidence appendix.
    """

    if not isinstance(writer_markdown, str) or not writer_markdown.strip():
        raise ValueError("writer_markdown must be a non-empty Markdown or plain-text bulletin draft.")
    if not isinstance(context, dict) or not context:
        raise ValueError("context must be a non-empty bulletin-context object.")
    _ = merged_weekly  # Reserved for future figures; retained so renderer input matches persisted context artifacts.

    report_title, period = _title_and_period(context, title)
    cleaned_markdown = _strip_redundant_document_preamble(writer_markdown, title=report_title, period=period)
    forecast_timeliness = _forecast_timeliness(context)
    figures = _make_figures(
        context,
        weekly,
        age_groups,
        shap_global_importance,
        br_trajectory,
        existing_br_figures,
        forecast_timeliness,
    )
    forecast_rows = _forecast_table_rows(context, forecast_timeliness)
    age_rows = _age_group_table_rows(context)
    shap_rows = _shap_table_rows(shap_global_importance)
    mechanical_rows = _mechanistic_parameter_rows(context)
    asset_tables = {
        "forecast_table": forecast_rows,
        "age_groups_table": age_rows,
        "shap_table": shap_rows,
        "mechanistic_parameters": mechanical_rows,
    }
    asset_order = _planned_asset_order(
        context=context,
        writer_markdown=cleaned_markdown,
        figures=figures,
        asset_tables=asset_tables,
        append_missing_evidence=append_missing_evidence,
    )
    figure_captions = _figure_captions(
        asset_order=asset_order,
        figures=figures,
        forecast_timeliness=forecast_timeliness,
    )
    rendered_html = _render_html(
        context=context,
        writer_markdown=cleaned_markdown,
        title=report_title,
        period=period,
        figures=figures,
        forecast_rows=forecast_rows,
        age_rows=age_rows,
        shap_rows=shap_rows,
        mechanical_rows=mechanical_rows,
        append_missing_evidence=append_missing_evidence,
        asset_order=asset_order,
        figure_captions=figure_captions,
        forecast_timeliness=forecast_timeliness,
    )
    rendered_pdf = _render_pdf(
        context=context,
        writer_markdown=cleaned_markdown,
        title=report_title,
        period=period,
        figures=figures,
        forecast_rows=forecast_rows,
        age_rows=age_rows,
        shap_rows=shap_rows,
        mechanical_rows=mechanical_rows,
        append_missing_evidence=append_missing_evidence,
        asset_order=asset_order,
        figure_captions=figure_captions,
        forecast_timeliness=forecast_timeliness,
    )
    manifest = {
        "schema_version": RENDERER_SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "title": report_title,
        "period": period,
        "forecast_engine": context.get("forecast_model", {}).get("engine"),
        "source_bulletin_context_schema_version": context.get("schema_version"),
        "append_missing_evidence": bool(append_missing_evidence),
        "forecast_timeliness": forecast_timeliness,
        "asset_display_order": asset_order,
        "figure_captions": figure_captions,
        "available_figures": sorted(figures),
        "available_tables": {
            "forecast_table_rows": max(0, len(forecast_rows) - 1),
            "age_groups_table_rows": max(0, len(age_rows) - 1),
            "shap_table_rows": max(0, len(shap_rows) - 1),
            "mechanistic_parameter_rows": max(0, len(mechanical_rows) - 1),
        },
        "placeholders": dict(PLACEHOLDERS),
        "renderer_constraints": [
            "The renderer does not call an LLM.",
            "The renderer does not repeat forecasting, BR calibration, SHAP, database, or weather retrieval.",
            "Narrative text is supplied by the caller and stored unchanged as bulletin.md.",
        ],
    }
    return BulletinRenderOutput(
        markdown=cleaned_markdown,
        html=rendered_html,
        pdf=rendered_pdf,
        figures=figures,
        manifest=manifest,
    )
