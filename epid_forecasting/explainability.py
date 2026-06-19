"""SHAP explainability for the fixed direct multi-horizon forecast workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .features import feature_group
from .modeling import inverse_transform_target, predict_matrix
from .service import TrainingState, _jsonable


class ShapExplainabilityError(RuntimeError):
    """Raised when SHAP explainability cannot be computed."""


@dataclass(frozen=True)
class ShapExplainabilityResult:
    """Tabular and compact SHAP outputs."""

    global_importance: pd.DataFrame
    local_values: pd.DataFrame
    worst_cases: pd.DataFrame
    summary: dict[str, Any]

    def to_public_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "method": "SHAP",
                "model_type": "HistGradientBoostingRegressor",
                "summary": self.summary,
                "tables": {
                    "global_importance_rows": len(self.global_importance),
                    "local_values_rows": len(self.local_values),
                    "worst_cases_rows": len(self.worst_cases),
                },
            }
        )


def _select_horizons(state: TrainingState, horizons: Iterable[int] | None) -> list[int]:
    if horizons is None:
        return list(range(1, state.config.horizon_weeks + 1))
    selected = sorted({int(item) for item in horizons})
    if not selected:
        raise ValueError("horizons must not be empty.")
    if selected[0] < 1 or selected[-1] > state.config.horizon_weeks:
        raise ValueError(f"horizons must be between 1 and {state.config.horizon_weeks}.")
    return selected


def _sample_test_rows(state: TrainingState, max_test_samples: int | None, random_state: int) -> pd.DataFrame:
    test = state.data_valid.loc[state.test_mask].copy().reset_index(drop=True)
    if max_test_samples is None or len(test) <= int(max_test_samples):
        return test
    return test.sample(n=int(max_test_samples), random_state=int(random_state)).sort_index().reset_index(drop=True)


def _compute_shap_values(model: Any, x_sample: np.ndarray, x_background: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import shap  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on runtime environment.
        raise ShapExplainabilityError(
            "Package shap is required for compute_forecast_shap_explainability. Install it with `uv add shap`."
        ) from exc

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_sample)
        return np.asarray(values, dtype=float), "tree"
    except Exception:
        # Compatibility fallback used by the source AI4Epi design: explain model.predict
        # with a permutation SHAP estimator when TreeExplainer is unavailable for the
        # installed sklearn/shap combination.
        explainer = shap.Explainer(model.predict, x_background, algorithm="permutation")
        explanation = explainer(x_sample, max_evals=2 * x_sample.shape[1] + 1)
        return np.asarray(explanation.values, dtype=float), "permutation"


def _prediction_and_actual_matrices(state: TrainingState, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = rows[state.feature_cols].to_numpy(dtype=float)
    pred_raw = predict_matrix(state.evaluation_models, x)
    prediction = np.clip(inverse_transform_target(pred_raw, state.config.target_transform), 0.0, None)
    actual = np.zeros((len(rows), state.config.horizon_weeks), dtype=float)
    for horizon in range(1, state.config.horizon_weeks + 1):
        actual[:, horizon - 1] = rows[f"y_h{horizon}"].to_numpy(dtype=float)
    return prediction, actual


def compute_forecast_shap_explainability(
    state: TrainingState,
    *,
    horizons: Iterable[int] | None = None,
    max_test_samples: int | None = 64,
    background_size: int = 128,
    top_features_per_horizon: int = 8,
    worst_cases_per_horizon: int = 5,
    random_state: int = 42,
) -> ShapExplainabilityResult:
    """Compute SHAP global/local explanations for evaluation forecast models."""
    if max_test_samples is not None and int(max_test_samples) <= 0:
        raise ValueError("max_test_samples must be positive or None.")
    if background_size <= 0:
        raise ValueError("background_size must be positive.")
    selected_horizons = _select_horizons(state, horizons)
    rows = _sample_test_rows(state, max_test_samples, random_state)
    if rows.empty:
        raise ShapExplainabilityError("No test rows are available for SHAP explainability.")

    background_source = state.data_valid.loc[state.train_mask, state.feature_cols]
    background = background_source.sample(
        n=min(int(background_size), len(background_source)), random_state=int(random_state)
    ).to_numpy(dtype=float)
    x_sample = rows[state.feature_cols].to_numpy(dtype=float)
    predictions, actuals = _prediction_and_actual_matrices(state, rows)

    global_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    explainers_used: dict[str, str] = {}
    feature_groups = {name: feature_group(name) for name in state.feature_cols}

    for horizon in selected_horizons:
        model = state.evaluation_models[horizon - 1]
        shap_values, explainer_kind = _compute_shap_values(model, x_sample, background)
        explainers_used[str(horizon)] = explainer_kind
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_value = shap_values.mean(axis=0)
        order = np.argsort(-mean_abs)
        for rank, feature_index in enumerate(order, start=1):
            feature = state.feature_cols[int(feature_index)]
            signed = float(mean_value[int(feature_index)])
            global_rows.append(
                {
                    "horizon_weeks": int(horizon),
                    "feature": feature,
                    "feature_group": feature_groups[feature],
                    "rank": int(rank),
                    "mean_abs_shap": float(mean_abs[int(feature_index)]),
                    "mean_shap": signed,
                    "direction": "increases_prediction" if signed > 0 else "decreases_prediction" if signed < 0 else "neutral",
                    "explainer": explainer_kind,
                }
            )
        for row_idx in range(len(rows)):
            row_date = pd.Timestamp(rows.loc[row_idx, "datetime"]).date().isoformat()
            for feature_index, feature in enumerate(state.feature_cols):
                local_rows.append(
                    {
                        "sample_index": int(row_idx),
                        "origin_date": row_date,
                        "horizon_weeks": int(horizon),
                        "feature": feature,
                        "feature_group": feature_groups[feature],
                        "feature_value": float(x_sample[row_idx, feature_index]),
                        "shap_value": float(shap_values[row_idx, feature_index]),
                        "prediction": float(predictions[row_idx, horizon - 1]),
                        "actual": float(actuals[row_idx, horizon - 1]),
                        "abs_error": float(abs(actuals[row_idx, horizon - 1] - predictions[row_idx, horizon - 1])),
                    }
                )

    global_importance = pd.DataFrame(global_rows).sort_values(["horizon_weeks", "rank"]).reset_index(drop=True)
    local_values = pd.DataFrame(local_rows)

    worst_rows: list[pd.DataFrame] = []
    sample_base = rows[["datetime"]].copy().reset_index(drop=True)
    for horizon in selected_horizons:
        temp = sample_base.copy()
        temp["horizon_weeks"] = int(horizon)
        temp["prediction"] = predictions[:, horizon - 1]
        temp["actual"] = actuals[:, horizon - 1]
        temp["abs_error"] = np.abs(temp["actual"] - temp["prediction"])
        temp["datetime"] = pd.to_datetime(temp["datetime"]).dt.date.astype(str)
        worst_rows.append(temp.sort_values("abs_error", ascending=False).head(int(worst_cases_per_horizon)))
    worst_cases = pd.concat(worst_rows, ignore_index=True) if worst_rows else pd.DataFrame()

    by_horizon: dict[str, list[dict[str, Any]]] = {}
    key_parts: list[str] = []
    for horizon in selected_horizons:
        top = global_importance.loc[global_importance["horizon_weeks"] == horizon].head(int(top_features_per_horizon))
        rows_for_h = _jsonable(top.to_dict(orient="records"))
        by_horizon[str(horizon)] = rows_for_h
        if rows_for_h:
            key_parts.append(f"h{horizon}: {rows_for_h[0]['feature']} ({rows_for_h[0]['feature_group']})")
    summary = _jsonable(
        {
            "method": "SHAP",
            "model_type": "HistGradientBoostingRegressor",
            "explainer_by_horizon": explainers_used,
            "horizons": selected_horizons,
            "sample_rows": int(len(rows)),
            "background_rows": int(len(background)),
            "top_features_per_horizon": int(top_features_per_horizon),
            "key_insight": "Top SHAP drivers by horizon: " + "; ".join(key_parts) if key_parts else "No SHAP drivers computed.",
            "by_horizon": by_horizon,
        }
    )
    return ShapExplainabilityResult(
        global_importance=global_importance,
        local_values=local_values,
        worst_cases=worst_cases,
        summary=summary,
    )
