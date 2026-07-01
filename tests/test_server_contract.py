import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd

import epid_forecasting_server as server
from epid_forecasting.config import DEFAULT_DATA_PATH
from epid_forecasting.service import EpidForecastingService


class DummyPoissonModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full(x.shape[0], self.value, dtype=float)


class FakeArtifactStore:
    def save_bulletin_context(self, *, context, markdown, weekly, age_groups, user_id, session_id, **kwargs):
        assert user_id == "mark"
        assert session_id == "session-01"
        assert "current_situation" in context
        assert markdown.startswith("# Influenza surveillance bulletin context")
        return {
            "run_id": "context-run-01",
            "storage_prefix": "mark/session-01/epid_forecasting/bulletin_context/context-run-01",
            "download_access": "presigned_urls",
            "presigned_url_expiration_seconds": 3600,
            "artifacts": {
                "bulletin_context_json": {"download_url": "https://example.test/bulletin_context.json"}
            },
        }

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


def test_mcp_surface_contains_expected_agent_facing_tools():
    tools = asyncio.run(server.mcp.list_tools())
    assert [tool.name for tool in tools] == [
        "describe_influenza_dataset",
        "list_influenza_db_cities",
        "export_influenza_db_dataset",
        "export_weather_source_dataset",
        "compare_influenza_age_groups_from_db",
        "generate_br_model_forecast",
        "estimate_br_model_parameters",
        "compute_forecast_shap_explainability",
        "prepare_influenza_bulletin_context",
        "render_influenza_bulletin",
        "run_influenza_forecasting",
        "compare_epidemic_waves",
    ]
    run_tool = next(tool for tool in tools if tool.name == "run_influenza_forecasting")
    assert set(run_tool.parameters["properties"]) == {"session_id", "user_id", "origin_date"}
    assert set(run_tool.parameters["required"]) == {"session_id", "user_id"}
    export_tool = next(tool for tool in tools if tool.name == "export_influenza_db_dataset")
    assert {"session_id", "user_id", "city", "begin_year", "begin_week"}.issubset(
        set(export_tool.parameters["properties"])
    )
    assert set(export_tool.parameters["required"]) == {"session_id", "user_id"}
    weather_tool = next(tool for tool in tools if tool.name == "export_weather_source_dataset")
    assert {"session_id", "user_id", "city", "start_date", "end_date"}.issubset(
        set(weather_tool.parameters["properties"])
    )
    br_forecast_tool = next(tool for tool in tools if tool.name == "generate_br_model_forecast")
    assert {"session_id", "user_id", "city", "forecast_type", "method", "forecast_duration_weeks"}.issubset(
        set(br_forecast_tool.parameters["properties"])
    )
    assert set(br_forecast_tool.parameters["required"]) == {"session_id", "user_id"}
    br_parameters_tool = next(tool for tool in tools if tool.name == "estimate_br_model_parameters")
    assert {"session_id", "user_id", "city", "forecast_type", "method"}.issubset(
        set(br_parameters_tool.parameters["properties"])
    )
    assert set(br_parameters_tool.parameters["required"]) == {"session_id", "user_id"}
    shap_tool = next(tool for tool in tools if tool.name == "compute_forecast_shap_explainability")
    assert {"session_id", "user_id", "max_test_samples", "horizons"}.issubset(
        set(shap_tool.parameters["properties"])
    )
    context_tool = next(tool for tool in tools if tool.name == "prepare_influenza_bulletin_context")
    assert {
        "session_id", "user_id", "city", "season", "forecast_engine", "include_forecast", "include_weather", "include_shap",
        "br_forecast_type", "br_method", "br_forecast_duration_weeks",
    }.issubset(set(context_tool.parameters["properties"]))
    assert set(context_tool.parameters["required"]) == {"session_id", "user_id"}
    render_tool = next(tool for tool in tools if tool.name == "render_influenza_bulletin")
    assert {"session_id", "user_id", "bulletin_context_run_id", "bulletin_markdown", "title"}.issubset(
        set(render_tool.parameters["properties"])
    )
    assert set(render_tool.parameters["required"]) == {
        "session_id", "user_id", "bulletin_context_run_id", "bulletin_markdown"
    }


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


def test_bulletin_context_tool_returns_full_inline_context(monkeypatch):
    class DummyBundle:
        weekly = __import__("pandas").DataFrame(
            {
                "datetime": ["2025-10-06", "2025-10-13"],
                "iso_year": [2025, 2025],
                "iso_week": [41, 42],
                "inc_per_10k": [1.0, 2.0],
            }
        )
        age_groups = __import__("pandas").DataFrame()

    monkeypatch.setattr(server, "_db_request", lambda **kwargs: object())
    monkeypatch.setattr(server, "_influenza_db_settings", lambda: object())
    monkeypatch.setattr(server, "fetch_influenza_db_bundle", lambda *_args: DummyBundle())
    monkeypatch.setattr(
        server,
        "summarize_influenza_db_bundle",
        lambda _bundle: {
            "city": {"slug": "spb", "name_ru": "Санкт-Петербург"},
            "request": {"begin_year": 2025, "begin_week": 41, "end_year": 2025, "end_week": 42},
            "source_url": "https://example.test?auth=%3Credacted%3E",
            "date_range": {"start": "2025-10-06", "end": "2025-10-13"},
        },
    )
    monkeypatch.setattr(
        server,
        "compare_age_groups",
        lambda *_args, **_kwargs: {"season": "2025-2026", "age_group_summary": []},
    )
    monkeypatch.setattr(
        server,
        "compare_recent_epidemic_waves",
        lambda *_args, **_kwargs: {"season_labels": ["2025-2026"], "waves": [], "latest_wave_status": "available"},
    )
    monkeypatch.setattr(server, "_artifact_store", lambda: FakeArtifactStore())

    response = server.prepare_influenza_bulletin_context(
        session_id="session-01",
        user_id="mark",
        city="spb",
        begin_year=2025,
        begin_week=41,
        end_year=2025,
        end_week=42,
        include_weather=False,
        include_forecast=False,
        include_shap=False,
    )

    metadata = response["metadata"]
    assert metadata["result_delivery"]["mode"] == "inline_bulletin_context_plus_s3_artifacts"
    assert metadata["bulletin_context"]["schema_version"] == "epid_forecasting.bulletin_context.v3"
    assert metadata["bulletin_context"]["current_situation"]["latest_week"]["inc_per_10k"] == 2.0
    assert "recommended_prompt" not in metadata
    assert "recommended_prompt" not in metadata["bulletin_context"]



def test_bulletin_context_br_engine_replaces_shap_with_mechanistic_evidence(monkeypatch):
    class DummyBundle:
        weekly = pd.DataFrame(
            {
                "datetime": ["2025-10-06", "2025-10-13"],
                "iso_year": [2025, 2025],
                "iso_week": [41, 42],
                "inc_per_10k": [1.0, 2.0],
            }
        )
        age_groups = pd.DataFrame()
        cases = pd.DataFrame({"placeholder": [1]})

    trajectory = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-10-06", "2025-10-13", "2025-10-20", "2025-10-27"]),
            "group": ["total"] * 4,
            "observed_cases": [10.0, 12.0, None, None],
            "fitted_cases": [9.0, 11.0, 13.0, 14.0],
            "pi80_lower_cases": [8.0, 10.0, 11.0, 12.0],
            "pi80_upper_cases": [10.0, 12.0, 15.0, 16.0],
            "fitted_inc_per_10k": [0.9, 1.1, 1.3, 1.4],
            "pi80_lower_inc_per_10k": [0.8, 1.0, 1.1, 1.2],
            "pi80_upper_inc_per_10k": [1.0, 1.2, 1.5, 1.6],
            "is_forecast": [False, False, True, True],
        }
    )
    dummy_result = SimpleNamespace(
        trajectory=trajectory,
        parameter_samples=pd.DataFrame({"alpha_total": [0.4, 0.42], "beta_total": [0.2, 0.22]}),
        configuration={
            "model_family": "compact_baroyan_rvachev_style_renewal_model",
            "forecast_type": "total",
            "method": "mcmc",
            "forecast_duration_weeks": 2,
        },
        parameter_summary={"best_fit": {"alpha_total": 0.4, "beta_total": 0.2}},
        diagnostics={"r2_observed_vs_fitted": 0.3},
        limitations=["Auxiliary model."],
    )

    monkeypatch.setattr(server, "_db_request", lambda **kwargs: object())
    monkeypatch.setattr(server, "_influenza_db_settings", lambda: object())
    monkeypatch.setattr(server, "fetch_influenza_db_bundle", lambda *_args: DummyBundle())
    monkeypatch.setattr(
        server,
        "summarize_influenza_db_bundle",
        lambda _bundle: {
            "city": {"slug": "spb", "name_ru": "Санкт-Петербург"},
            "request": {"begin_year": 2025, "begin_week": 41, "end_year": 2025, "end_week": 42},
            "source_url": "https://example.test?auth=%3Credacted%3E",
            "date_range": {"start": "2025-10-06", "end": "2025-10-13"},
        },
    )
    monkeypatch.setattr(server, "compare_age_groups", lambda *_args, **_kwargs: {"season": "2025-2026", "age_group_summary": []})
    monkeypatch.setattr(
        server,
        "compare_recent_epidemic_waves",
        lambda *_args, **_kwargs: {"season_labels": ["2025-2026"], "waves": [], "latest_wave_status": "available"},
    )
    monkeypatch.setattr(server, "run_br_calibration", lambda *_args, **_kwargs: dummy_result)
    monkeypatch.setattr(server, "render_br_forecast_figure", lambda *_args, **_kwargs: {"png": b"png", "pdf": b"pdf"})
    monkeypatch.setattr(
        server,
        "render_br_parameter_figures",
        lambda *_args, **_kwargs: {"alpha": {"png": b"png", "pdf": b"pdf"}, "beta": {"png": b"png", "pdf": b"pdf"}},
    )
    monkeypatch.setattr(server, "_artifact_store", lambda: FakeArtifactStore())

    response = server.prepare_influenza_bulletin_context(
        session_id="session-01",
        user_id="mark",
        city="spb",
        begin_year=2025,
        begin_week=41,
        end_year=2025,
        end_week=42,
        forecast_engine="br",
        include_weather=False,
        include_forecast=True,
        include_shap=True,
        br_forecast_duration_weeks=2,
        br_posterior_samples=10,
        br_abc_candidates=10,
    )

    context = response["metadata"]["bulletin_context"]
    assert response["metadata"]["context_summary"]["forecast_engine"] == "br"
    assert context["forecast_model"]["engine"] == "br"
    assert "forecast_explainability" not in context
    assert context["mechanistic_model_interpretation"]["parameter_summary"]["best_fit"]["beta_total"] == 0.2
    assert context["mechanistic_model_interpretation"]["gamma"]["status"] == "not_estimated"
    assert context["short_term_forecast"]["forecast_engine"] == "br"


def test_render_bulletin_tool_delegates_to_renderer_alias(monkeypatch):
    source = {
        "context": {"schema_version": "epid_forecasting.bulletin_context.v3"},
        "weekly": pd.DataFrame({"datetime": ["2026-06-22"], "inc_per_10k": [0.0]}),
        "age_groups": pd.DataFrame(),
        "merged_weekly": None,
        "shap_global_importance": None,
        "br_trajectory": None,
        "br_figures": None,
        "storage_prefix": "mark/session-01/epid_forecasting/bulletin_context/context-run-01",
    }

    class RenderStore:
        def load_bulletin_context_run(self, *, user_id, session_id, run_id):
            assert user_id == "mark"
            assert session_id == "session-01"
            assert run_id == "context-run-01"
            return source

        def save_rendered_bulletin(self, **kwargs):
            assert kwargs["source_context_run_id"] == "context-run-01"
            assert kwargs["bulletin_markdown"] == "# Bulletin"
            assert kwargs["bulletin_html"] == "<html></html>"
            assert kwargs["bulletin_pdf"].startswith(b"%PDF")
            return {
                "run_id": "render-run-01",
                "storage_prefix": "mark/session-01/epid_forecasting/rendered_bulletin/render-run-01",
                "artifacts": {"bulletin_pdf": {"download_url": "https://example.test/bulletin.pdf"}},
            }

    rendered = SimpleNamespace(
        markdown="# Bulletin",
        html="<html></html>",
        pdf=b"%PDF-test",
        figures={},
        manifest={"schema_version": "epid_forecasting.rendered_bulletin.v1"},
    )

    def fake_renderer(**kwargs):
        assert kwargs["context"] is source["context"]
        assert kwargs["writer_markdown"] == "# Authored text"
        assert kwargs["weekly"] is source["weekly"]
        assert kwargs["age_groups"] is source["age_groups"]
        return rendered

    monkeypatch.setattr(server, "_artifact_store", lambda: RenderStore())
    monkeypatch.setattr(server, "render_bulletin_artifacts", fake_renderer)

    response = server.render_influenza_bulletin(
        session_id="session-01",
        user_id="mark",
        bulletin_context_run_id="context-run-01",
        bulletin_markdown="# Authored text",
    )

    assert response["metadata"]["source_bulletin_context"]["run_id"] == "context-run-01"
    assert response["metadata"]["run_id"] == "render-run-01"
    assert response["metadata"]["artifacts"]["bulletin_pdf"]["download_url"].endswith("bulletin.pdf")
