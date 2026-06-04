"""Model fitting, evaluation, and conformal interval utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def fit_models_hist_gbdt(
    x_train: np.ndarray,
    y_train_list: list[np.ndarray],
    *,
    random_state: int = 42,
) -> list[HistGradientBoostingRegressor]:
    """Fit one HistGradientBoostingRegressor per forecast horizon."""
    models: list[HistGradientBoostingRegressor] = []
    for horizon, y_train in enumerate(y_train_list, start=1):
        model = HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=None,
            max_leaf_nodes=None,
            learning_rate=0.01,
            max_iter=8000,
            min_samples_leaf=40,
            l2_regularization=5.0,
            early_stopping=True,
            validation_fraction=None,
            random_state=random_state + horizon,
        )
        model.fit(x_train, y_train)
        models.append(model)
    return models


def evaluate_multistep(y_true_mat: np.ndarray, y_pred_mat: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    horizons = y_true_mat.shape[1]
    rows: list[dict[str, float | int]] = []
    for index in range(horizons):
        y_true = y_true_mat[:, index]
        y_pred = y_pred_mat[:, index]
        rows.append(
            {
                "horizon_weeks": index + 1,
                "r2": float(r2_score(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
            }
        )
    overall = {
        "r2_overall": float(r2_score(y_true_mat.reshape(-1), y_pred_mat.reshape(-1))),
        "rmse_overall": float(np.sqrt(mean_squared_error(y_true_mat.reshape(-1), y_pred_mat.reshape(-1)))),
        "mae_overall": float(mean_absolute_error(y_true_mat.reshape(-1), y_pred_mat.reshape(-1))),
    }
    return pd.DataFrame(rows), overall


def conformal_radius_from_abs_errors(abs_errors: np.ndarray, *, alpha: float = 0.20) -> float:
    errors = np.asarray(abs_errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    n = len(errors)
    if n == 0:
        return float("nan")
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(max(q_level, 0.0), 1.0)
    return float(np.quantile(errors, q_level, method="higher"))


def evaluate_intervals(y_true_mat: np.ndarray, lo_mat: np.ndarray, hi_mat: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | int]] = []
    for index in range(y_true_mat.shape[1]):
        y_true = y_true_mat[:, index]
        lo = lo_mat[:, index]
        hi = hi_mat[:, index]
        rows.append(
            {
                "horizon_weeks": index + 1,
                "coverage": float(((y_true >= lo) & (y_true <= hi)).mean()),
                "avg_width": float(np.mean(hi - lo)),
            }
        )
    overall = {
        "coverage_mean": float(np.mean([row["coverage"] for row in rows])),
        "avg_width_mean": float(np.mean([row["avg_width"] for row in rows])),
    }
    return pd.DataFrame(rows), overall


def predict_matrix(models: list[HistGradientBoostingRegressor], x: np.ndarray) -> np.ndarray:
    pred = np.zeros((x.shape[0], len(models)), dtype=float)
    for horizon, model in enumerate(models, start=1):
        pred[:, horizon - 1] = model.predict(x)
    return pred


def transform_target(y: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "none":
        return y
    if target_transform == "log1p":
        return np.log1p(y)
    raise ValueError("target_transform must be either 'none' or 'log1p'.")


def inverse_transform_target(z: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "none":
        return z
    if target_transform == "log1p":
        return np.expm1(z)
    raise ValueError("target_transform must be either 'none' or 'log1p'.")
