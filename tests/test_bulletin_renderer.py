from __future__ import annotations

from pathlib import Path

import pandas as pd

from epid_forecasting.bulletin_renderer import PLACEHOLDERS, render_influenza_bulletin


def _context() -> dict:
    return {
        "schema_version": "epid_forecasting.bulletin_context.v3",
        "city": {"name_ru": "Санкт-Петербург", "slug": "spb"},
        "forecast_model": {"engine": "gbdt"},
        "current_situation": {
            "latest_week": {"date": "2026-04-19", "iso_year": 2026, "iso_week": 16, "inc_per_10k": 2.639}
        },
        "short_term_forecast": {
            "status": "included",
            "forecast_engine": "gbdt",
            "forecast": [
                {
                    "horizon_weeks": 1,
                    "target_date": "2026-04-20",
                    "inc_per_10k_prediction": 1.369,
                    "pi80_lower": 0.0,
                    "pi80_upper": 3.476,
                },
                {
                    "horizon_weeks": 2,
                    "target_date": "2026-04-27",
                    "inc_per_10k_prediction": 1.322,
                    "pi80_lower": 0.0,
                    "pi80_upper": 1.920,
                },
            ],
        },
        "epidemic_wave_comparison": {
            "waves": [
                {
                    "season_label": "2024-2025",
                    "peak_week": 51,
                    "peak_value": 4.0,
                    "plot_points": [
                        {"pos": 0, "week": 40, "value_smooth": 1.0},
                        {"pos": 1, "week": 41, "value_smooth": 4.0},
                        {"pos": 2, "week": 42, "value_smooth": 2.0},
                    ],
                },
                {
                    "season_label": "2025-2026",
                    "peak_week": 51,
                    "peak_value": 3.0,
                    "plot_points": [
                        {"pos": 0, "week": 40, "value_smooth": 0.5},
                        {"pos": 1, "week": 41, "value_smooth": 3.0},
                        {"pos": 2, "week": 42, "value_smooth": 1.0},
                    ],
                },
            ]
        },
        "age_group_patterns": {
            "season": "2025-2026",
            "age_group_summary": [
                {
                    "age_group": "3-6",
                    "total_cases": 1230,
                    "peak_inc_per_10k": 360.16,
                    "peak_iso_week": 51,
                    "seasonal_burden_inc_week": 205.9,
                },
                {
                    "age_group": "7-14",
                    "total_cases": 1717,
                    "peak_inc_per_10k": 315.07,
                    "peak_iso_week": 51,
                    "seasonal_burden_inc_week": 130.04,
                },
            ],
        },
    }


def test_renderer_creates_utf8_html_pdf_and_deterministic_figures(tmp_path: Path):
    weekly = pd.DataFrame(
        {
            "datetime": ["2026-03-30", "2026-04-06", "2026-04-13", "2026-04-20"],
            "inc_per_10k": [2.2, 1.9, 2.4, 2.639],
        }
    )
    age_groups = pd.DataFrame(
        {
            "datetime": ["2026-03-30", "2026-04-06", "2026-03-30", "2026-04-06"],
            "age_group": ["3-6", "3-6", "7-14", "7-14"],
            "inc_per_10k": [200.0, 360.0, 140.0, 315.0],
            "season": ["2025-2026"] * 4,
        }
    )
    shap = pd.DataFrame(
        {
            "horizon_weeks": [1, 1, 2],
            "feature": ["y_lag0", "y_lag1", "season_cos_1"],
            "feature_group": ["incidence_lag", "incidence_lag", "seasonality"],
            "rank": [1, 2, 1],
            "mean_abs_shap": [0.40, 0.20, 0.30],
            "direction": ["increases_prediction", "decreases_prediction", "neutral"],
        }
    )
    draft = """## 1. Текущая ситуация
Заболеваемость изменилась по сравнению с предыдущей неделей.

## 2. Прогноз
{{FORECAST_FIGURE}}
{{FORECAST_TABLE}}

## 3. Волны
{{WAVES_FIGURE}}

## 4. Возрастные группы
{{AGE_GROUPS_FIGURE}}
{{AGE_GROUPS_TABLE}}

## 5. Интерпретация модели
{{SHAP_FIGURE}}
"""

    output = render_influenza_bulletin(
        context=_context(),
        writer_markdown=draft,
        weekly=weekly,
        age_groups=age_groups,
        shap_global_importance=shap,
    )

    assert output.pdf.startswith(b"%PDF")
    assert "Санкт-Петербург" in output.html
    assert "data:image/png;base64" in output.html
    assert {"forecast_figure", "waves_figure", "age_groups_figure", "shap_figure"}.issubset(output.figures)
    assert output.manifest["schema_version"] == "epid_forecasting.rendered_bulletin.v1"
    assert output.manifest["placeholders"] == PLACEHOLDERS

    target = tmp_path / "bulletin.pdf"
    target.write_bytes(output.pdf)
    assert target.stat().st_size > 1_000


def test_renderer_converts_author_markdown_before_html_and_pdf():
    weekly = pd.DataFrame(
        {
            "datetime": ["2026-04-06", "2026-04-13", "2026-04-20"],
            "inc_per_10k": [1.9, 2.4, 2.639],
        }
    )
    age_groups = pd.DataFrame(
        {
            "datetime": ["2026-04-06", "2026-04-13"],
            "age_group": ["3-6", "3-6"],
            "inc_per_10k": [200.0, 360.0],
            "season": ["2025-2026", "2025-2026"],
        }
    )
    draft = """## **Краткое резюме**
**Низкий уровень** показателя `inc_per_10k` требует *обычного* мониторинга.

- Первый **маркер**.
- Второй: `y_lag0`.

| Показатель | Значение |
| --- | ---: |
| **Пик** | `0,186` |
| Среднее | 0,192 |

{{FORECAST_TABLE}}
"""

    output = render_influenza_bulletin(
        context=_context(),
        writer_markdown=draft,
        weekly=weekly,
        age_groups=age_groups,
        append_missing_evidence=False,
    )

    assert "<strong>Краткое резюме</strong>" in output.html
    assert "<strong>Низкий уровень</strong>" in output.html
    assert "<code>inc_per_10k</code>" in output.html
    assert "<em>обычного</em>" in output.html
    assert "<ul><li>Первый <strong>маркер</strong>.</li>" in output.html
    assert "<table>" in output.html
    assert "**Низкий уровень**" not in output.html
    assert "`inc_per_10k`" not in output.html
    assert output.pdf.startswith(b"%PDF")


def test_renderer_marks_stale_forecast_and_numbers_figures_by_placeholder_order():
    import copy

    context = copy.deepcopy(_context())
    context["short_term_forecast"]["forecast_origin_date"] = "2026-03-02"
    weekly = pd.DataFrame(
        {
            "datetime": ["2026-03-30", "2026-04-06", "2026-04-13", "2026-04-20"],
            "inc_per_10k": [2.2, 1.9, 2.4, 2.639],
        }
    )
    age_groups = pd.DataFrame(
        {
            "datetime": ["2026-03-30", "2026-04-06", "2026-03-30", "2026-04-06"],
            "age_group": ["3-6", "3-6", "7-14", "7-14"],
            "inc_per_10k": [200.0, 360.0, 140.0, 315.0],
            "season": ["2025-2026"] * 4,
        }
    )
    shap = pd.DataFrame(
        {
            "horizon_weeks": [1, 1, 2],
            "feature": ["y_lag0", "y_lag1", "season_cos_1"],
            "feature_group": ["incidence_lag", "incidence_lag", "seasonality"],
            "rank": [1, 2, 1],
            "mean_abs_shap": [0.40, 0.20, 0.30],
            "direction": ["increases_prediction", "decreases_prediction", "neutral"],
        }
    )
    draft = """# ЕЖЕНЕДЕЛЬНЫЙ БЮЛЛЕТЕНЬ ПО ГРИППУ
за 16-ю неделю 2026 года; по данным на 2026-04-19; Санкт-Петербург

## Интерпретация
{{SHAP_FIGURE}}

## Волны
{{WAVES_FIGURE}}

## Архивный прогноз
{{FORECAST_FIGURE}}
{{FORECAST_TABLE}}
{{SHAP_TABLE}}
"""

    output = render_influenza_bulletin(
        context=context,
        writer_markdown=draft,
        weekly=weekly,
        age_groups=age_groups,
        shap_global_importance=shap,
        append_missing_evidence=False,
    )

    assert output.manifest["forecast_timeliness"]["status"] == "stale"
    assert output.manifest["forecast_timeliness"]["age_days"] == 48
    assert output.manifest["asset_display_order"] == [
        "shap_figure",
        "waves_figure",
        "forecast_figure",
        "forecast_table",
        "shap_table",
    ]
    assert output.manifest["figure_captions"] == {
        "shap_figure": "Рис. 1. Глобальная важность признаков SHAP.",
        "waves_figure": "Рис. 2. Сравнение последних эпидемических волн.",
        "forecast_figure": "Рис. 3. Архивный GBDT-прогноз: наблюдаемая заболеваемость и прогноз, построенный от 2026-03-02.",
    }
    assert output.html.count("<h1>") == 1
    assert "Внимание: прогноз построен от 2026-03-02" in output.html
    assert "Архивный прогноз /10 тыс." in output.html
    assert "Лаги заболеваемости" in output.html
    assert "Повышает прогноз" in output.html
    assert "0,4000" in output.html
    assert "0.4000" not in output.html
    assert output.pdf.startswith(b"%PDF")
