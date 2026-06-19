"""Mechanistic Baroyan-Rvachev-style calibration and visualization tools.

This module is intentionally independent from the legacy ``Influenza_bulletin``
repository. It consumes normalized cases returned by ``influenza_db.py`` rather
than making its own database request, and it produces reproducible numerical
artifacts suitable for the MCP server's existing S3 storage contract.

The model is a compact renewal-style implementation based on the infectivity
kernel used by the legacy tools. It can be selected explicitly as an alternative forecast engine. It does not
inherit the validation or split-conformal uncertainty guarantees of the default
GBDT workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, dual_annealing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .influenza_db import STRAIN_INDICES

CalibrationMethod = Literal["abc", "mcmc", "annealing", "optuna"]
ForecastType = Literal["total", "age"]

# Legacy BR infectivity weights. A daily incidence signal is convolved with
# these weights to obtain the infectious pressure used by the recurrence.
BR_INFECTIVITY_KERNEL = np.asarray([0.10, 0.10, 1.00, 0.90, 0.55, 0.30, 0.15, 0.05], dtype=float)

PARAMETER_LOWER_BOUND = 0.01
PARAMETER_UPPER_BOUND = 0.99
_PARAMETER_TRANSFORM_EPSILON = 1e-9


@dataclass(frozen=True)
class BRCalibrationConfig:
    """Runtime configuration for one mechanistic calibration run."""

    forecast_type: ForecastType = "total"
    method: CalibrationMethod = "mcmc"
    forecast_duration_weeks: int = 4
    calibration_window_weeks: int | None = 26
    posterior_samples: int = 200
    abc_candidates: int = 3000
    mcmc_burn_in: int = 300
    mcmc_draws: int = 500
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.forecast_type not in {"total", "age"}:
            raise ValueError("forecast_type must be 'total' or 'age'.")
        if self.method not in {"abc", "mcmc", "annealing", "optuna"}:
            raise ValueError("method must be one of: abc, mcmc, annealing, optuna.")
        if not 1 <= self.forecast_duration_weeks <= 16:
            raise ValueError("forecast_duration_weeks must be between 1 and 16.")
        if self.calibration_window_weeks is not None and self.calibration_window_weeks < 8:
            raise ValueError("calibration_window_weeks must be at least 8 when provided.")
        if self.posterior_samples < 10:
            raise ValueError("posterior_samples must be at least 10.")
        if self.abc_candidates < self.posterior_samples:
            raise ValueError("abc_candidates must be greater than or equal to posterior_samples.")
        if self.mcmc_burn_in < 0 or self.mcmc_draws < self.posterior_samples:
            raise ValueError("mcmc_draws must be at least posterior_samples and mcmc_burn_in must be non-negative.")


@dataclass
class BRCalibrationResult:
    """Structured result of a mechanistic calibration run."""

    configuration: dict[str, Any]
    observed: pd.DataFrame
    trajectory: pd.DataFrame
    parameter_samples: pd.DataFrame
    parameter_summary: dict[str, Any]
    diagnostics: dict[str, Any]
    limitations: list[str]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "diagnostics": self.diagnostics,
            "parameter_summary": self.parameter_summary,
            "observed_period": {
                "start": self.observed["datetime"].min().date().isoformat(),
                "end": self.observed["datetime"].max().date().isoformat(),
                "weeks": int(len(self.observed)),
            },
            "forecast_period": {
                "start": self.trajectory.loc[self.trajectory["is_forecast"], "datetime"].min().date().isoformat(),
                "end": self.trajectory.loc[self.trajectory["is_forecast"], "datetime"].max().date().isoformat(),
                "weeks": int(self.trajectory["is_forecast"].sum()),
            },
            "limitations": list(self.limitations),
        }


def _require_case_columns(cases: pd.DataFrame) -> None:
    required = {
        "datetime",
        "total_population",
        "sars_total_cases",
        "sars_cases_age_group_0",
        "sars_cases_age_group_1",
        "sars_cases_age_group_2",
        "sars_cases_age_group_3",
        "population_age_group_0",
        "population_age_group_1",
        "population_age_group_2",
        "population_age_group_3",
        *[f"real_cases_strain_{index}" for index in STRAIN_INDICES],
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Normalized influenza cases are missing required BR columns: {missing}")


def prepare_br_observed_series(
    cases: pd.DataFrame,
    *,
    forecast_type: ForecastType,
    calibration_window_weeks: int | None,
) -> pd.DataFrame:
    """Build a calibrated total or two-group weekly incidence series from DB cases."""

    _require_case_columns(cases)
    frame = cases.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if frame["datetime"].isna().any():
        raise ValueError("Normalized influenza cases contain invalid datetime values.")

    total_influenza_cases = frame[[f"real_cases_strain_{index}" for index in STRAIN_INDICES]].fillna(0.0).sum(axis=1)
    frame["total_influenza_cases"] = total_influenza_cases

    if forecast_type == "total":
        out = frame[["datetime", "total_population", "total_influenza_cases"]].rename(
            columns={"total_population": "population", "total_influenza_cases": "observed_cases"}
        )
        out["group"] = "total"
    else:
        young_ari = frame[["sars_cases_age_group_0", "sars_cases_age_group_1", "sars_cases_age_group_2"]].fillna(0.0).sum(axis=1)
        older_ari = frame["sars_cases_age_group_3"].fillna(0.0)
        total_ari = (young_ari + older_ari).replace(0.0, np.nan)

        young_population = frame[
            ["population_age_group_0", "population_age_group_1", "population_age_group_2"]
        ].fillna(0.0).sum(axis=1)
        older_population = frame["population_age_group_3"].fillna(0.0)

        young = pd.DataFrame(
            {
                "datetime": frame["datetime"],
                "group": "0-14",
                "population": young_population,
                "observed_cases": (young_ari / total_ari * total_influenza_cases).fillna(0.0),
            }
        )
        older = pd.DataFrame(
            {
                "datetime": frame["datetime"],
                "group": "15+",
                "population": older_population,
                "observed_cases": (older_ari / total_ari * total_influenza_cases).fillna(0.0),
            }
        )
        out = pd.concat([young, older], ignore_index=True)

    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    out["observed_cases"] = pd.to_numeric(out["observed_cases"], errors="coerce")
    out = out.dropna(subset=["datetime", "population", "observed_cases"])
    out = out.loc[out["population"] > 0].copy()
    if out.empty:
        raise ValueError("No valid normalized influenza observations are available for BR calibration.")

    if forecast_type == "total":
        out = out.sort_values("datetime").reset_index(drop=True)
    else:
        out = out.sort_values(["datetime", "group"]).reset_index(drop=True)
        expected_groups = {"0-14", "15+"}
        available = set(out["group"])
        if available != expected_groups:
            raise ValueError(f"Age-mode BR calibration requires groups {sorted(expected_groups)}; found {sorted(available)}.")

    if calibration_window_weeks is not None:
        dates = sorted(out["datetime"].drop_duplicates())
        if len(dates) > calibration_window_weeks:
            keep_dates = set(dates[-calibration_window_weeks:])
            out = out.loc[out["datetime"].isin(keep_dates)].copy()

    min_weeks = 8
    observed_weeks = out["datetime"].nunique()
    if observed_weeks < min_weeks:
        raise ValueError(f"BR calibration requires at least {min_weeks} observed weeks; received {observed_weeks}.")

    out["observed_inc_per_10k"] = out["observed_cases"] / out["population"] * 10_000
    return out.reset_index(drop=True)


def _simulate_total(
    *,
    alpha: float,
    beta: float,
    population: float,
    initial_daily_incidence: float,
    total_days: int,
) -> np.ndarray:
    susceptible = np.empty(total_days, dtype=float)
    incidence = np.zeros(total_days, dtype=float)
    susceptible[0] = max(alpha * population, 0.0)
    incidence[0] = min(max(initial_daily_incidence, 0.0), susceptible[0])

    for day in range(total_days - 1):
        infectious_pressure = 0.0
        for lag, weight in enumerate(BR_INFECTIVITY_KERNEL):
            if day - lag >= 0:
                infectious_pressure += incidence[day - lag] * weight
        incidence[day + 1] = min(beta * susceptible[day] * infectious_pressure / population, susceptible[day])
        susceptible[day + 1] = max(susceptible[day] - incidence[day + 1], 0.0)
    return incidence


def _simulate_age(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    populations: np.ndarray,
    initial_daily_incidence: np.ndarray,
    total_days: int,
) -> np.ndarray:
    groups = 2
    susceptible = np.empty((groups, total_days), dtype=float)
    incidence = np.zeros((groups, total_days), dtype=float)
    susceptible[:, 0] = np.maximum(alpha * populations, 0.0)
    incidence[:, 0] = np.minimum(np.maximum(initial_daily_incidence, 0.0), susceptible[:, 0])

    for day in range(total_days - 1):
        pressures = np.zeros(groups, dtype=float)
        for group in range(groups):
            for lag, weight in enumerate(BR_INFECTIVITY_KERNEL):
                if day - lag >= 0:
                    pressures[group] += incidence[group, day - lag] * weight

        for recipient in range(groups):
            force = 0.0
            for source in range(groups):
                force += beta[recipient, source] * pressures[source] / populations[source]
            incidence[recipient, day + 1] = min(susceptible[recipient, day] * force, susceptible[recipient, day])
            susceptible[recipient, day + 1] = max(susceptible[recipient, day] - incidence[recipient, day + 1], 0.0)
    return incidence


def _daily_to_weekly(daily: np.ndarray, weeks: int) -> np.ndarray:
    if daily.ndim == 1:
        return daily[: weeks * 7].reshape(weeks, 7).sum(axis=1)
    return daily[:, : weeks * 7].reshape(daily.shape[0], weeks, 7).sum(axis=2)


def _observed_matrix(observed: pd.DataFrame, forecast_type: ForecastType) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    dates = pd.DatetimeIndex(sorted(observed["datetime"].drop_duplicates()))
    if forecast_type == "total":
        values = observed.sort_values("datetime")["observed_cases"].to_numpy(dtype=float).reshape(1, -1)
        population = np.asarray(
            [float(observed.sort_values("datetime")["population"].median())],
            dtype=float,
        )
        return values, population, ["total"], dates

    pivot = (
        observed.pivot(index="datetime", columns="group", values="observed_cases")
        .reindex(index=dates, columns=["0-14", "15+"])
        .fillna(0.0)
    )
    populations = (
        observed.pivot(index="datetime", columns="group", values="population")
        .reindex(index=dates, columns=["0-14", "15+"])
        .median(axis=0)
        .to_numpy(dtype=float)
    )
    return pivot.to_numpy(dtype=float).T, populations, ["0-14", "15+"], dates


def _parameter_names(forecast_type: ForecastType) -> list[str]:
    if forecast_type == "total":
        return ["alpha_total", "beta_total"]
    return [
        "alpha_0_14",
        "alpha_15_plus",
        "beta_0_14_to_0_14",
        "beta_0_14_to_15_plus",
        "beta_15_plus_to_0_14",
        "beta_15_plus_to_15_plus",
    ]


def _unpack_parameters(vector: np.ndarray, forecast_type: ForecastType) -> tuple[np.ndarray, np.ndarray]:
    if forecast_type == "total":
        return np.asarray([vector[0]], dtype=float), np.asarray([[vector[1]]], dtype=float)
    alpha = np.asarray(vector[:2], dtype=float)
    beta = np.asarray(vector[2:], dtype=float).reshape(2, 2)
    return alpha, beta


def _simulate_weekly_for_vector(
    vector: np.ndarray,
    *,
    observed_values: np.ndarray,
    populations: np.ndarray,
    forecast_type: ForecastType,
    total_weeks: int,
) -> np.ndarray:
    alpha, beta = _unpack_parameters(vector, forecast_type)
    initial_daily = np.maximum(observed_values[:, 0] / 7.0, 1.0)
    total_days = total_weeks * 7
    if forecast_type == "total":
        daily = _simulate_total(
            alpha=float(alpha[0]),
            beta=float(beta[0, 0]),
            population=float(populations[0]),
            initial_daily_incidence=float(initial_daily[0]),
            total_days=total_days,
        )
    else:
        daily = _simulate_age(
            alpha=alpha,
            beta=beta,
            populations=populations,
            initial_daily_incidence=initial_daily,
            total_days=total_days,
        )
    weekly = _daily_to_weekly(daily, total_weeks)
    return weekly.reshape(1, -1) if forecast_type == "total" else weekly


def _log1p_residuals_for_vector(
    vector: np.ndarray,
    *,
    observed_values: np.ndarray,
    populations: np.ndarray,
    forecast_type: ForecastType,
) -> np.ndarray:
    """Return model residuals on the calibration objective scale."""

    predicted = _simulate_weekly_for_vector(
        vector,
        observed_values=observed_values,
        populations=populations,
        forecast_type=forecast_type,
        total_weeks=observed_values.shape[1],
    )
    return np.log1p(predicted) - np.log1p(observed_values)


def _loss_for_vector(
    vector: np.ndarray,
    *,
    observed_values: np.ndarray,
    populations: np.ndarray,
    forecast_type: ForecastType,
) -> float:
    """Mean squared log1p residual used for deterministic calibration."""

    residuals = _log1p_residuals_for_vector(
        vector,
        observed_values=observed_values,
        populations=populations,
        forecast_type=forecast_type,
    )
    return float(np.mean(residuals**2))


def _parameter_bounds(dims: int) -> list[tuple[float, float]]:
    return [(PARAMETER_LOWER_BOUND, PARAMETER_UPPER_BOUND)] * dims


def _to_unconstrained(vector: np.ndarray) -> np.ndarray:
    """Map bounded BR parameters to an unconstrained logit space."""

    scaled = (np.asarray(vector, dtype=float) - PARAMETER_LOWER_BOUND) / (
        PARAMETER_UPPER_BOUND - PARAMETER_LOWER_BOUND
    )
    scaled = np.clip(scaled, _PARAMETER_TRANSFORM_EPSILON, 1.0 - _PARAMETER_TRANSFORM_EPSILON)
    return np.log(scaled) - np.log1p(-scaled)


def _from_unconstrained(vector: np.ndarray) -> np.ndarray:
    """Map an unconstrained vector back to the bounded BR parameter space."""

    z = np.clip(np.asarray(vector, dtype=float), -35.0, 35.0)
    logistic = 1.0 / (1.0 + np.exp(-z))
    return PARAMETER_LOWER_BOUND + (PARAMETER_UPPER_BOUND - PARAMETER_LOWER_BOUND) * logistic


def _log_abs_det_jacobian(unconstrained: np.ndarray) -> float:
    """Jacobian adjustment for a uniform prior on bounded parameters."""

    z = np.clip(np.asarray(unconstrained, dtype=float), -35.0, 35.0)
    logistic = 1.0 / (1.0 + np.exp(-z))
    derivative = (PARAMETER_UPPER_BOUND - PARAMETER_LOWER_BOUND) * logistic * (1.0 - logistic)
    return float(np.log(np.clip(derivative, _PARAMETER_TRANSFORM_EPSILON, None)).sum())


def _estimate_log1p_observation_sigma(*, best_loss: float, n_observations: int, n_parameters: int) -> float:
    """Estimate a plug-in residual scale for the conditional MCMC target."""

    degrees_of_freedom = max(int(n_observations) - int(n_parameters), 1)
    residual_variance = max(float(best_loss) * int(n_observations) / degrees_of_freedom, 1e-6)
    return float(np.sqrt(residual_variance))


def _log_conditional_pseudo_posterior(
    unconstrained: np.ndarray,
    *,
    observed_values: np.ndarray,
    populations: np.ndarray,
    forecast_type: ForecastType,
    observation_sigma: float,
) -> tuple[float, float, np.ndarray]:
    """Evaluate a conditional Gaussian log1p-residual pseudo-posterior.

    The deterministic optimizer and the MCMC sampler therefore use the same
    residual objective.  The only additional assumption is a Gaussian error
    model on log1p weekly estimated influenza cases with a plug-in residual
    scale estimated at the optimizer solution.
    """

    parameters = _from_unconstrained(unconstrained)
    residuals = _log1p_residuals_for_vector(
        parameters,
        observed_values=observed_values,
        populations=populations,
        forecast_type=forecast_type,
    )
    loss = float(np.mean(residuals**2))
    log_likelihood = -0.5 * float(np.sum((residuals / observation_sigma) ** 2))
    # The prior is uniform in the bounded parameter space.  Sampling happens
    # in logit space, so the transformation Jacobian is part of the target.
    return log_likelihood + _log_abs_det_jacobian(unconstrained), loss, parameters


def _fit_best_parameters(
    *,
    observed_values: np.ndarray,
    populations: np.ndarray,
    forecast_type: ForecastType,
    random_state: int,
    method: CalibrationMethod,
) -> np.ndarray:
    dims = 2 if forecast_type == "total" else 6
    bounds = _parameter_bounds(dims)
    objective = lambda vector: _loss_for_vector(
        vector,
        observed_values=observed_values,
        populations=populations,
        forecast_type=forecast_type,
    )

    if method == "annealing":
        result = dual_annealing(
            objective,
            bounds=bounds,
            maxiter=180,
            seed=random_state,
            no_local_search=False,
        )
    else:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=80,
            popsize=12,
            tol=1e-7,
            seed=random_state,
            polish=True,
            workers=1,
            updating="immediate",
        )
    if not result.success:
        # A useful calibrated result is still often available even when SciPy stops
        # on its iteration tolerance instead of emitting the success flag.
        if not np.isfinite(result.fun):
            raise RuntimeError(f"BR optimization failed: {result.message}")
    return np.clip(np.asarray(result.x, dtype=float), PARAMETER_LOWER_BOUND, PARAMETER_UPPER_BOUND)


def _abc_parameter_samples(
    *,
    best: np.ndarray,
    best_loss: float,
    observed_values: np.ndarray,
    populations: np.ndarray,
    forecast_type: ForecastType,
    candidates: int,
    posterior_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build an explicitly labelled ABC-style accepted candidate set.

    This is not labelled as an exact posterior.  The optimizer point is always
    retained as a reference candidate, ensuring the exported sample table and
    its uncertainty rows are coherent with the reported deterministic optimum.
    """

    rng = np.random.default_rng(random_state)
    dims = len(best)
    random_vectors = rng.uniform(
        PARAMETER_LOWER_BOUND,
        PARAMETER_UPPER_BOUND,
        size=(max(candidates - 1, 1), dims),
    )
    vectors = np.vstack([best.reshape(1, -1), random_vectors])
    losses = np.asarray(
        [
            _loss_for_vector(
                vector,
                observed_values=observed_values,
                populations=populations,
                forecast_type=forecast_type,
            )
            for vector in vectors
        ],
        dtype=float,
    )
    losses[0] = float(best_loss)
    order = np.argsort(losses)
    accepted_indices = order[:posterior_samples]
    accepted = vectors[accepted_indices]
    accepted_losses = losses[accepted_indices]
    return accepted, accepted_losses, {
        "sample_kind": "abc_accepted_objective_samples",
        "objective_reference": "mean_squared_log1p_error_on_weekly_estimated_influenza_cases",
        "accepted_from_candidates": int(len(vectors)),
        "acceptance_loss_threshold": float(accepted_losses.max()),
        "contains_optimizer_best_fit": bool(np.any(accepted_indices == 0)),
    }


def _mcmc_parameter_samples(
    *,
    initial: np.ndarray,
    best_loss: float,
    observed_values: np.ndarray,
    populations: np.ndarray,
    forecast_type: ForecastType,
    burn_in: int,
    draws: int,
    posterior_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Draw conditional pseudo-posterior samples with adaptive burn-in only.

    The sampler targets a Gaussian log1p-residual pseudo-posterior centred on
    the same objective used by the deterministic optimizer.  Proposal-scale
    adaptation is confined to burn-in; retained draws use a fixed random-walk
    kernel and are thinned deterministically without replacement.
    """

    rng = np.random.default_rng(random_state)
    n_observations = int(observed_values.size)
    n_parameters = int(len(initial))
    observation_sigma = _estimate_log1p_observation_sigma(
        best_loss=best_loss,
        n_observations=n_observations,
        n_parameters=n_parameters,
    )
    current_z = _to_unconstrained(initial)
    current_log_target, current_loss, _ = _log_conditional_pseudo_posterior(
        current_z,
        observed_values=observed_values,
        populations=populations,
        forecast_type=forecast_type,
        observation_sigma=observation_sigma,
    )

    proposal_scale = 0.20 if n_parameters <= 2 else 0.12
    total_steps = int(burn_in) + int(draws)
    retained_parameters: list[np.ndarray] = []
    retained_losses: list[float] = []
    accepted_total = 0
    accepted_post_burn = 0
    burn_window_accepted = 0
    burn_window_steps = 0

    for step in range(total_steps):
        proposal_z = current_z + rng.normal(0.0, proposal_scale, size=current_z.shape)
        proposal_log_target, proposal_loss, _ = _log_conditional_pseudo_posterior(
            proposal_z,
            observed_values=observed_values,
            populations=populations,
            forecast_type=forecast_type,
            observation_sigma=observation_sigma,
        )
        log_acceptance = proposal_log_target - current_log_target
        accepted = log_acceptance >= 0.0 or np.log(rng.uniform()) < log_acceptance
        if accepted:
            current_z = proposal_z
            current_log_target = proposal_log_target
            current_loss = proposal_loss
            accepted_total += 1
            if step < burn_in:
                burn_window_accepted += 1
            else:
                accepted_post_burn += 1

        if step < burn_in:
            burn_window_steps += 1
            if burn_window_steps == 25:
                acceptance_rate = burn_window_accepted / burn_window_steps
                if acceptance_rate < 0.15:
                    proposal_scale *= 0.75
                elif acceptance_rate > 0.45:
                    proposal_scale *= 1.25
                proposal_scale = float(np.clip(proposal_scale, 0.02, 1.00))
                burn_window_accepted = 0
                burn_window_steps = 0
        else:
            retained_parameters.append(_from_unconstrained(current_z))
            retained_losses.append(float(current_loss))

    samples = np.asarray(retained_parameters, dtype=float)
    sampled_losses = np.asarray(retained_losses, dtype=float)
    if len(samples) < posterior_samples:
        raise RuntimeError("MCMC calibration did not produce enough retained samples.")

    selected_indexes = np.linspace(0, len(samples) - 1, num=posterior_samples, dtype=int)
    selected_samples = samples[selected_indexes]
    selected_losses = sampled_losses[selected_indexes]
    diagnostics = {
        "sample_kind": "conditional_gaussian_log1p_pseudo_posterior",
        "observation_error_model": "gaussian_log1p_residuals_with_plugin_sigma",
        "observation_log1p_sigma": float(observation_sigma),
        "retained_draws_before_thinning": int(len(samples)),
        "selected_draws": int(len(selected_samples)),
        "mcmc_acceptance_rate_total": float(accepted_total / max(total_steps, 1)),
        "mcmc_acceptance_rate_post_burn": float(accepted_post_burn / max(draws, 1)),
        "final_proposal_scale_unconstrained": float(proposal_scale),
        "reference_estimate": "optimizer_best_fit",
    }
    return selected_samples, selected_losses, diagnostics


def _local_perturbation_samples(
    *,
    best: np.ndarray,
    best_loss: float,
    observed_values: np.ndarray,
    populations: np.ndarray,
    forecast_type: ForecastType,
    posterior_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate clearly labelled local objective-neighbourhood samples."""

    rng = np.random.default_rng(random_state)
    n_candidates = max(1000, posterior_samples * 8)
    random_candidates = np.clip(
        best.reshape(1, -1) + rng.normal(0.0, 0.04, size=(n_candidates - 1, len(best))),
        PARAMETER_LOWER_BOUND,
        PARAMETER_UPPER_BOUND,
    )
    candidates = np.vstack([best.reshape(1, -1), random_candidates])
    losses = np.asarray(
        [
            _loss_for_vector(
                vector,
                observed_values=observed_values,
                populations=populations,
                forecast_type=forecast_type,
            )
            for vector in candidates
        ],
        dtype=float,
    )
    losses[0] = float(best_loss)
    indexes = np.argsort(losses)[:posterior_samples]
    return candidates[indexes], losses[indexes], {
        "sample_kind": "local_objective_neighborhood_samples",
        "objective_reference": "mean_squared_log1p_error_on_weekly_estimated_influenza_cases",
        "candidate_count": int(len(candidates)),
        "acceptance_loss_threshold": float(losses[indexes].max()),
        "contains_optimizer_best_fit": bool(np.any(indexes == 0)),
    }


def _calibrate_parameters(
    *,
    observed_values: np.ndarray,
    populations: np.ndarray,
    config: BRCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]]:
    best = _fit_best_parameters(
        observed_values=observed_values,
        populations=populations,
        forecast_type=config.forecast_type,
        random_state=config.random_state,
        method=config.method,
    )
    best_loss = _loss_for_vector(
        best,
        observed_values=observed_values,
        populations=populations,
        forecast_type=config.forecast_type,
    )
    if config.method == "abc":
        samples, losses, sampling_diagnostics = _abc_parameter_samples(
            best=best,
            best_loss=best_loss,
            observed_values=observed_values,
            populations=populations,
            forecast_type=config.forecast_type,
            candidates=config.abc_candidates,
            posterior_samples=config.posterior_samples,
            random_state=config.random_state,
        )
        return best, samples, losses, "abc_rejection_against_calibration_objective", sampling_diagnostics
    if config.method == "mcmc":
        samples, losses, sampling_diagnostics = _mcmc_parameter_samples(
            initial=best,
            best_loss=best_loss,
            observed_values=observed_values,
            populations=populations,
            forecast_type=config.forecast_type,
            burn_in=config.mcmc_burn_in,
            draws=config.mcmc_draws,
            posterior_samples=config.posterior_samples,
            random_state=config.random_state,
        )
        return best, samples, losses, "random_walk_metropolis_conditional_pseudo_posterior", sampling_diagnostics
    samples, losses, sampling_diagnostics = _local_perturbation_samples(
        best=best,
        best_loss=best_loss,
        observed_values=observed_values,
        populations=populations,
        forecast_type=config.forecast_type,
        posterior_samples=config.posterior_samples,
        random_state=config.random_state,
    )
    label = "scipy_dual_annealing_plus_local_objective_neighborhood"
    if config.method == "optuna":
        label = "optuna_compatibility_alias_to_scipy_differential_evolution_plus_local_objective_neighborhood"
    return best, samples, losses, label, sampling_diagnostics


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _optimizer_boundary_flags(parameter_names: list[str], best: np.ndarray, *, tolerance: float = 1e-4) -> dict[str, dict[str, bool]]:
    """Report whether the deterministic optimum is constrained by parameter bounds."""

    return {
        name: {
            "at_lower_bound": bool(float(value) - PARAMETER_LOWER_BOUND <= tolerance),
            "at_upper_bound": bool(PARAMETER_UPPER_BOUND - float(value) <= tolerance),
        }
        for name, value in zip(parameter_names, best)
    }

def run_br_calibration(cases: pd.DataFrame, *, config: BRCalibrationConfig) -> BRCalibrationResult:
    """Calibrate the compact mechanistic model from normalized NII influenza DB cases."""

    observed = prepare_br_observed_series(
        cases,
        forecast_type=config.forecast_type,
        calibration_window_weeks=config.calibration_window_weeks,
    )
    observed_values, populations, groups, dates = _observed_matrix(observed, config.forecast_type)
    n_observed_weeks = observed_values.shape[1]
    total_weeks = n_observed_weeks + config.forecast_duration_weeks

    best, samples, sample_losses, calibration_backend, sampling_diagnostics = _calibrate_parameters(
        observed_values=observed_values,
        populations=populations,
        config=config,
    )
    best_weekly = _simulate_weekly_for_vector(
        best,
        observed_values=observed_values,
        populations=populations,
        forecast_type=config.forecast_type,
        total_weeks=total_weeks,
    )
    sampled_weekly = np.asarray(
        [
            _simulate_weekly_for_vector(
                sample,
                observed_values=observed_values,
                populations=populations,
                forecast_type=config.forecast_type,
                total_weeks=total_weeks,
            )
            for sample in samples
        ],
        dtype=float,
    )
    weekly_step = dates.to_series().diff().dropna().median()
    if pd.isna(weekly_step):
        weekly_step = pd.Timedelta(days=7)
    all_dates = pd.date_range(dates[0], periods=total_weeks, freq=weekly_step)
    # ``sampled_weekly`` has shape [samples, groups, weeks]. The compact
    # DataFrame above is built explicitly so total and age modes share one
    # public schema.
    trajectory_rows = []
    for group_index, group in enumerate(groups):
        lower_series = np.quantile(sampled_weekly[:, group_index, :], 0.10, axis=0)
        upper_series = np.quantile(sampled_weekly[:, group_index, :], 0.90, axis=0)
        observed_series = np.concatenate([observed_values[group_index], np.full(config.forecast_duration_weeks, np.nan)])
        group_frame = pd.DataFrame(
            {
                "datetime": all_dates,
                "group": group,
                "observed_cases": observed_series,
                "fitted_cases": best_weekly[group_index],
                "pi80_lower_cases": lower_series,
                "pi80_upper_cases": upper_series,
                "population": populations[group_index],
                "is_forecast": [False] * n_observed_weeks + [True] * config.forecast_duration_weeks,
            }
        )
        for column in ["observed_cases", "fitted_cases", "pi80_lower_cases", "pi80_upper_cases"]:
            group_frame[column.replace("_cases", "_inc_per_10k")] = group_frame[column] / group_frame["population"] * 10_000
        trajectory_rows.append(group_frame)
    trajectory = pd.concat(trajectory_rows, ignore_index=True)

    parameter_names = _parameter_names(config.forecast_type)
    optimizer_best_loss = float(
        _loss_for_vector(
            best,
            observed_values=observed_values,
            populations=populations,
            forecast_type=config.forecast_type,
        )
    )
    uncertainty_samples = pd.DataFrame(samples, columns=parameter_names)
    uncertainty_samples.insert(0, "sample_id", np.arange(1, len(uncertainty_samples) + 1))
    uncertainty_samples.insert(1, "sample_role", "uncertainty_draw")
    uncertainty_samples["calibration_loss"] = sample_losses

    optimizer_reference = pd.DataFrame([dict(zip(parameter_names, best))])
    optimizer_reference.insert(0, "sample_id", 0)
    optimizer_reference.insert(1, "sample_role", "optimizer_best_fit_reference")
    optimizer_reference["calibration_loss"] = optimizer_best_loss
    parameter_samples = pd.concat([optimizer_reference, uncertainty_samples], ignore_index=True)

    observed_flat = observed_values.reshape(-1)
    fitted_flat = best_weekly[:, :n_observed_weeks].reshape(-1)
    uncertainty_loss_quantiles = _quantile_summary(sample_losses)
    diagnostics = {
        "calibration_backend": calibration_backend,
        "objective": "mean_squared_log1p_error_on_weekly_estimated_influenza_cases",
        "r2_observed_vs_fitted": float(r2_score(observed_flat, fitted_flat)) if len(observed_flat) > 1 else None,
        "mae_cases": float(mean_absolute_error(observed_flat, fitted_flat)),
        "rmse_cases": float(np.sqrt(mean_squared_error(observed_flat, fitted_flat))),
        "optimizer_best_fit_loss": optimizer_best_loss,
        "uncertainty_sample_loss_quantiles": uncertainty_loss_quantiles,
        "uncertainty_sample_loss_median": float(np.median(sample_losses)),
        "uncertainty_sample_count": int(len(uncertainty_samples)),
        "optimizer_reference_row_in_parameter_samples": True,
        "forecast_interval_method": "parameter_sample_quantiles",
        **sampling_diagnostics,
        "observed_weeks": int(n_observed_weeks),
    }
    optimizer_best_fit = {name: float(value) for name, value in zip(parameter_names, best)}
    optimizer_boundary_flags = _optimizer_boundary_flags(parameter_names, best)
    optimizer_at_any_bound = any(
        flag["at_lower_bound"] or flag["at_upper_bound"]
        for flag in optimizer_boundary_flags.values()
    )
    diagnostics["optimizer_parameter_bounds"] = optimizer_boundary_flags
    diagnostics["optimizer_at_any_parameter_bound"] = optimizer_at_any_bound
    parameter_summary = {
        "optimizer_best_fit": optimizer_best_fit,
        "optimizer_parameter_bounds": optimizer_boundary_flags,
        # Retained temporarily for callers that already read the original field.
        "best_fit": optimizer_best_fit,
        "uncertainty_samples": {
            "sample_kind": sampling_diagnostics["sample_kind"],
            "reference_estimate": "optimizer_best_fit",
            "parameter_quantiles": {
                name: _quantile_summary(uncertainty_samples[name].to_numpy(dtype=float))
                for name in parameter_names
            },
            "calibration_loss_quantiles": uncertainty_loss_quantiles,
            "sample_count": int(len(uncertainty_samples)),
            "optimizer_reference_row_in_artifact": True,
        },
    }
    limitations = [
        "This caller-selected mechanistic forecast does not inherit the validation or split-conformal uncertainty guarantees of the default GBDT workflow.",
        "Alpha and beta are fitted model parameters; they should not be interpreted as directly observed biological quantities.",
        "Age-mode influenza cases are allocated from estimated total influenza cases using the observed ARI age composition.",
        "The displayed forecast interval reflects quantiles across the reported calibration sample set, not the GBDT conformal interval.",
    ]
    if sampling_diagnostics["sample_kind"] == "conditional_gaussian_log1p_pseudo_posterior":
        limitations.append(
            "For MCMC, uncertainty draws target a conditional Gaussian log1p-residual pseudo-posterior with a plug-in residual scale; they are conditional model-based uncertainty, not a full external validation guarantee."
        )
    else:
        limitations.append(
            "For non-MCMC methods, uncertainty draws are objective-based calibration samples and should not be interpreted as a Bayesian posterior."
        )
    if optimizer_at_any_bound:
        limitations.append(
            "At least one optimizer parameter reached a configured calibration bound; this indicates a constrained or weakly identified fit and should be interpreted cautiously."
        )
    return BRCalibrationResult(
        configuration={
            "model_family": "compact_baroyan_rvachev_style_renewal_model",
            "forecast_type": config.forecast_type,
            "method": config.method,
            "forecast_duration_weeks": config.forecast_duration_weeks,
            "calibration_window_weeks": config.calibration_window_weeks,
            "posterior_samples": config.posterior_samples,
            "random_state": config.random_state,
        },
        observed=observed,
        trajectory=trajectory,
        parameter_samples=parameter_samples,
        parameter_summary=parameter_summary,
        diagnostics=diagnostics,
        limitations=limitations,
    )


def _render_bytes(fig: plt.Figure, *, fmt: str, dpi: int = 180) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def render_br_forecast_figure(result: BRCalibrationResult, *, language: Literal["ru", "en"]) -> dict[str, bytes]:
    """Render a fitted and forecast trajectory figure as PNG and PDF bytes."""

    trajectory = result.trajectory.copy()
    groups = trajectory["group"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, max(3.5, 3.2 * len(groups))), sharex=True)
    if len(groups) == 1:
        axes = [axes]

    sample_kind = str(result.diagnostics.get("sample_kind", "calibration_parameter_samples"))
    interval_label = (
        "80% conditional calibration-sample interval"
        if sample_kind == "conditional_gaussian_log1p_pseudo_posterior"
        else "80% calibration-sample interval"
    )
    labels = {
        "ru": {
            "observed": "Наблюдаемые оценки случаев",
            "fitted": "Калиброванная модель",
            "interval": "80% интервал по калибровочным выборкам",
            "forecast": "Прогнозный период",
            "ylabel": "Заболеваемость на 10 тыс.",
            "title": "Механистическая калибровка и прогноз гриппа",
        },
        "en": {
            "observed": "Observed estimated cases",
            "fitted": "Calibrated model",
            "interval": interval_label,
            "forecast": "Forecast period",
            "ylabel": "Incidence per 10,000",
            "title": "Mechanistic influenza calibration and forecast",
        },
    }[language]

    for axis, group in zip(axes, groups):
        frame = trajectory.loc[trajectory["group"] == group].sort_values("datetime")
        axis.fill_between(
            frame["datetime"],
            frame["pi80_lower_inc_per_10k"],
            frame["pi80_upper_inc_per_10k"],
            alpha=0.20,
            label=labels["interval"],
        )
        axis.plot(frame["datetime"], frame["fitted_inc_per_10k"], linewidth=1.8, label=labels["fitted"])
        observed = frame.dropna(subset=["observed_inc_per_10k"])
        axis.scatter(observed["datetime"], observed["observed_inc_per_10k"], s=16, label=labels["observed"])
        forecast_start = frame.loc[frame["is_forecast"], "datetime"].min()
        axis.axvline(forecast_start, linestyle="--", linewidth=1.0, label=labels["forecast"])
        axis.set_ylabel(labels["ylabel"])
        axis.set_title(f"{group}")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    axes[0].set_title(labels["title"] + f" — {groups[0]}" if len(groups) == 1 else labels["title"])
    fig.autofmt_xdate()
    fig.tight_layout()

    png = BytesIO()
    fig.savefig(png, format="png", dpi=180, bbox_inches="tight")
    pdf = BytesIO()
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return {"png": png.getvalue(), "pdf": pdf.getvalue()}


def render_br_parameter_figures(result: BRCalibrationResult) -> dict[str, dict[str, bytes]]:
    """Render alpha and beta parameter-sample distributions as PNG and PDF bytes."""

    samples = result.parameter_samples.copy()
    if "sample_role" in samples.columns:
        uncertainty_samples = samples.loc[samples["sample_role"] == "uncertainty_draw"].copy()
    else:
        uncertainty_samples = samples.copy()
    parameter_columns = [
        column
        for column in uncertainty_samples.columns
        if column not in {"sample_id", "sample_role", "calibration_loss"}
    ]
    alpha_columns = [column for column in parameter_columns if column.startswith("alpha_")]
    beta_columns = [column for column in parameter_columns if column.startswith("beta_")]
    optimizer_best_fit = result.parameter_summary.get("optimizer_best_fit") or result.parameter_summary.get("best_fit", {})

    def plot_columns(columns: list[str], title: str) -> dict[str, bytes]:
        n = len(columns)
        fig, axes = plt.subplots(1, n, figsize=(max(4.5, 3.6 * n), 3.6), squeeze=False)
        for axis, column in zip(axes.ravel(), columns):
            axis.hist(
                uncertainty_samples[column].to_numpy(dtype=float),
                bins=min(30, max(10, len(uncertainty_samples) // 5)),
                density=True,
                label="uncertainty draws",
            )
            if column in optimizer_best_fit:
                axis.axvline(float(optimizer_best_fit[column]), linestyle="--", linewidth=1.4, label="optimizer best fit")
            axis.set_title(column)
            axis.set_xlabel("value")
            axis.set_ylabel("density")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        png = BytesIO()
        fig.savefig(png, format="png", dpi=180, bbox_inches="tight")
        pdf = BytesIO()
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return {"png": png.getvalue(), "pdf": pdf.getvalue()}

    return {
        "alpha": plot_columns(alpha_columns, "Alpha parameter sample distribution"),
        "beta": plot_columns(beta_columns, "Beta parameter sample distribution"),
    }
