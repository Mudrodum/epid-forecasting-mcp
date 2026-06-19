import pandas as pd

from epid_forecasting.bulletin_context import build_bulletin_context, render_bulletin_context_markdown


def _weekly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": ["2025-09-29", "2025-10-06", "2025-10-13", "2025-10-20"],
            "iso_year": [2025, 2025, 2025, 2025],
            "iso_week": [40, 41, 42, 43],
            "inc_per_10k": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _base_kwargs() -> dict:
    return {
        "city": {"slug": "spb", "name_ru": "Санкт-Петербург"},
        "request": {"begin_year": 2025, "begin_week": 40, "end_year": 2025, "end_week": 43},
        "source_url": "https://example.test?auth=%3Credacted%3E",
        "weekly": _weekly(),
        "age_group_comparison": {"season": "2025-2026", "age_group_summary": []},
        "wave_comparison": {"season_labels": ["2024-2025", "2025-2026"], "waves": []},
        "forecast_result": None,
        "date_range": {"start": "2025-09-29", "end": "2025-10-20"},
        "parameters": {"trend_window_weeks": 2},
    }


def test_build_gbdt_bulletin_context_creates_evidence_packet_without_prose_generation():
    context = build_bulletin_context(**_base_kwargs())

    assert context["purpose"] == "structured_evidence_for_external_bulletin_writing"
    assert context["current_situation"]["latest_week"]["inc_per_10k"] == 4.0
    assert context["short_term_forecast"]["status"] == "not_included"
    assert context["forecast_model"]["engine"] == "gbdt"
    assert context["schema_version"] == "epid_forecasting.bulletin_context.v3"
    assert "forecast_explainability" in context
    assert "mechanistic_model_interpretation" not in context
    assert "recommended_prompt" not in context
    markdown = render_bulletin_context_markdown(context)
    assert markdown.startswith("# Influenza surveillance bulletin context")
    assert "SHAP forecast-driver evidence" in markdown
    assert "Recommended prompt for Codex" not in markdown


def test_build_br_bulletin_context_replaces_shap_with_mechanistic_parameters():
    mechanistic_result = {
        "configuration": {
            "model_family": "compact_baroyan_rvachev_style_renewal_model",
            "forecast_type": "total",
            "method": "mcmc",
            "forecast_duration_weeks": 4,
        },
        "forecast_origin_date": "2025-10-20",
        "forecast_horizon_weeks": 4,
        "forecast": [
            {
                "datetime": "2025-10-27",
                "group": "total",
                "fitted_inc_per_10k": 4.2,
                "pi80_lower_inc_per_10k": 3.0,
                "pi80_upper_inc_per_10k": 5.5,
            }
        ],
        "parameter_summary": {
            "best_fit": {"alpha_total": 0.4, "beta_total": 0.2},
            "posterior_quantiles": {},
        },
        "parameter_meanings": {"alpha_total": {"label": "Initial susceptible fraction"}},
        "diagnostics": {"r2_observed_vs_fitted": 0.8},
        "limitations": ["Auxiliary model."],
    }
    context = build_bulletin_context(
        **_base_kwargs(),
        forecast_engine="br",
        mechanistic_result=mechanistic_result,
    )

    assert context["forecast_model"]["engine"] == "br"
    assert context["short_term_forecast"]["forecast_engine"] == "br"
    assert "forecast_explainability" not in context
    assert context["mechanistic_model_interpretation"]["parameter_summary"]["best_fit"]["alpha_total"] == 0.4
    assert context["mechanistic_model_interpretation"]["gamma"]["status"] == "not_estimated"
    assert "mechanistic_model_interpretation" in context["suggested_report_sections"]
    markdown = render_bulletin_context_markdown(context)
    assert "Mechanistic model interpretation evidence" in markdown
    assert "SHAP forecast-driver evidence" not in markdown
