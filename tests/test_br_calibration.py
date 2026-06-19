import numpy as np
import pandas as pd

import epid_forecasting.br_calibration as br


def _synthetic_cases(n_weeks: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2025-09-29", periods=n_weeks, freq="7D")
    total_population = np.full(n_weeks, 1_000_000.0)
    sars_total = np.linspace(1_000, 2_500, n_weeks)
    age0 = sars_total * 0.10
    age1 = sars_total * 0.12
    age2 = sars_total * 0.18
    age3 = sars_total - age0 - age1 - age2
    influenza = np.maximum(10.0, 120.0 + 70.0 * np.sin(np.linspace(0.0, 2.0, n_weeks)))

    frame = pd.DataFrame(
        {
            "datetime": dates,
            "total_population": total_population,
            "sars_total_cases": sars_total,
            "sars_cases_age_group_0": age0,
            "sars_cases_age_group_1": age1,
            "sars_cases_age_group_2": age2,
            "sars_cases_age_group_3": age3,
            "population_age_group_0": np.full(n_weeks, 80_000.0),
            "population_age_group_1": np.full(n_weeks, 110_000.0),
            "population_age_group_2": np.full(n_weeks, 170_000.0),
            "population_age_group_3": np.full(n_weeks, 640_000.0),
        }
    )
    for index, share in enumerate([0.10, 0.25, 0.35, 0.30]):
        frame[f"real_cases_strain_{index}"] = influenza * share
    return frame


def test_br_total_calibration_returns_shared_public_schema(monkeypatch):
    monkeypatch.setattr(
        br,
        "_fit_best_parameters",
        lambda **_kwargs: np.asarray([0.45, 0.20], dtype=float),
    )
    result = br.run_br_calibration(
        _synthetic_cases(),
        config=br.BRCalibrationConfig(
            forecast_type="total",
            method="abc",
            forecast_duration_weeks=2,
            calibration_window_weeks=10,
            posterior_samples=12,
            abc_candidates=32,
            random_state=7,
        ),
    )

    assert result.trajectory["group"].unique().tolist() == ["total"]
    assert result.trajectory["is_forecast"].sum() == 2
    assert {"alpha_total", "beta_total", "calibration_loss"}.issubset(result.parameter_samples.columns)
    assert result.to_public_dict()["forecast_period"]["weeks"] == 2

    figures = br.render_br_forecast_figure(result, language="en")
    assert figures["png"].startswith(b"\x89PNG")
    assert figures["pdf"].startswith(b"%PDF")


def test_br_age_mode_uses_two_age_groups(monkeypatch):
    monkeypatch.setattr(
        br,
        "_fit_best_parameters",
        lambda **_kwargs: np.asarray([0.45, 0.40, 0.20, 0.10, 0.10, 0.18], dtype=float),
    )
    result = br.run_br_calibration(
        _synthetic_cases(),
        config=br.BRCalibrationConfig(
            forecast_type="age",
            method="abc",
            forecast_duration_weeks=1,
            calibration_window_weeks=10,
            posterior_samples=12,
            abc_candidates=32,
            random_state=7,
        ),
    )

    assert set(result.trajectory["group"]) == {"0-14", "15+"}
    assert len(result.parameter_summary["best_fit"]) == 6
    parameter_figures = br.render_br_parameter_figures(result)
    assert parameter_figures["alpha"]["png"].startswith(b"\x89PNG")
    assert parameter_figures["beta"]["pdf"].startswith(b"%PDF")


def test_br_mcmc_marks_optimizer_reference_and_uses_consistent_uncertainty_draws():
    result = br.run_br_calibration(
        _synthetic_cases(),
        config=br.BRCalibrationConfig(
            forecast_type="total",
            method="mcmc",
            forecast_duration_weeks=1,
            calibration_window_weeks=10,
            posterior_samples=12,
            mcmc_burn_in=40,
            mcmc_draws=60,
            random_state=7,
        ),
    )

    reference = result.parameter_samples.loc[
        result.parameter_samples["sample_role"] == "optimizer_best_fit_reference"
    ]
    draws = result.parameter_samples.loc[result.parameter_samples["sample_role"] == "uncertainty_draw"]
    assert len(reference) == 1
    assert len(draws) == 12
    assert reference.iloc[0]["alpha_total"] == result.parameter_summary["optimizer_best_fit"]["alpha_total"]
    assert result.parameter_summary["best_fit"] == result.parameter_summary["optimizer_best_fit"]
    assert "posterior_quantiles" not in result.parameter_summary
    assert result.parameter_summary["uncertainty_samples"]["sample_kind"] == "conditional_gaussian_log1p_pseudo_posterior"
    assert result.diagnostics["uncertainty_sample_loss_median"] < result.diagnostics["optimizer_best_fit_loss"] * 2.0
    assert 0.0 < result.diagnostics["mcmc_acceptance_rate_post_burn"] < 1.0
    assert "optimizer_parameter_bounds" in result.diagnostics
