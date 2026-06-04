"""Application service behind the EpidForecasting MCP tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_PATH, ForecastConfig
from .features import build_supervised, feature_group
from .storage import S3ArtifactStorage
from .modeling import (
    conformal_radius_from_abs_errors,
    evaluate_intervals,
    evaluate_multistep,
    fit_models_hist_gbdt,
    inverse_transform_target,
    predict_matrix,
    transform_target,
)

REQUIRED_COLUMNS = {
    "datetime",
    "iso_year",
    "iso_week",
    "total_population",
    "total_cases_formula",
    "inc_per_10k",
    "temp_mean",
    "temp_max",
    "temp_min",
    "rh_mean",
    "rh_max",
    "rh_min",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray, pd.DataFrame, pd.Series)) else False:
        return None
    return value


@dataclass
class TrainingState:
    config: ForecastConfig
    data: pd.DataFrame
    supervised_data: pd.DataFrame
    data_valid: pd.DataFrame
    feature_cols: list[str]
    train_mask: np.ndarray
    calib_mask: np.ndarray
    test_mask: np.ndarray
    models_train_only: list[HistGradientBoostingRegressor]
    models_prod: list[HistGradientBoostingRegressor]
    conformal_radii: np.ndarray
    per_h_metrics: pd.DataFrame
    overall_metrics: dict[str, float]
    per_h_interval_metrics: pd.DataFrame
    overall_interval_metrics: dict[str, float]
    per_h_metrics_refit: pd.DataFrame
    overall_metrics_refit: dict[str, float]
    per_h_interval_metrics_refit: pd.DataFrame
    overall_interval_metrics_refit: dict[str, float]
    y_test_mat: np.ndarray
    y_test_pred_mat: np.ndarray
    y_test_pred_refit_mat: np.ndarray
    test_lo_mat: np.ndarray
    test_hi_mat: np.ndarray
    test_lo_refit_mat: np.ndarray
    test_hi_refit_mat: np.ndarray


class EpidForecastingService:
    """Service that owns dataset loading, model training, forecasting, and exports."""

    def __init__(self, data_path: str | Path = DEFAULT_DATA_PATH, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> None:
        self.data_path = Path(data_path)
        self.artifact_dir = Path(artifact_dir)
        self.state: TrainingState | None = None
        self._raw_data: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        self._raw_data = df.copy()
        return df

    def describe_dataset(self) -> dict[str, Any]:
        df = self._raw_data.copy() if self._raw_data is not None else self.load_data()
        null_counts = {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}
        numeric_summary = df.describe(include=[np.number]).T.reset_index().rename(columns={"index": "column"})
        return _jsonable(
            {
                "rows": len(df),
                "columns": list(df.columns),
                "date_min": df["datetime"].min().date().isoformat(),
                "date_max": df["datetime"].max().date().isoformat(),
                "iso_year_min": int(df["iso_year"].min()),
                "iso_year_max": int(df["iso_year"].max()),
                "target_variable": "inc_per_10k",
                "target_description": "Weekly influenza morbidity per 10,000 population.",
                "static_dataset_path": str(self.data_path),
                "missing_values": null_counts,
                "numeric_summary": numeric_summary,
            }
        )

    def _prepare_supervised(self, config: ForecastConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
        df = self.load_data()
        supervised, feature_cols = build_supervised(
            df,
            target_col=config.target_col,
            temp_cols=config.temp_cols,
            horizon_weeks=config.horizon_weeks,
            y_lags=config.y_lags,
            temp_lags=config.temp_lags,
            y_roll_windows=config.y_roll_windows,
            temp_roll_windows=config.temp_roll_windows,
            fourier_k=config.fourier_k,
            growth_lags=config.growth_lags,
        )

        valid_mask = np.ones(len(supervised), dtype=bool)
        for horizon in range(1, config.horizon_weeks + 1):
            valid_mask &= supervised[f"y_h{horizon}"].notna().to_numpy()
        valid_mask &= supervised[feature_cols].notna().all(axis=1).to_numpy()
        data_valid = supervised.loc[valid_mask].copy().reset_index(drop=True)

        n_valid = len(data_valid)
        if n_valid <= config.test_weeks + config.calib_weeks + 20:
            raise ValueError(
                f"Insufficient valid rows: n={n_valid}; need substantially more than "
                f"test_weeks + calib_weeks = {config.test_weeks + config.calib_weeks}."
            )

        train_end = n_valid - (config.calib_weeks + config.test_weeks)
        calib_end = n_valid - config.test_weeks

        train_mask = np.zeros(n_valid, dtype=bool)
        calib_mask = np.zeros(n_valid, dtype=bool)
        test_mask = np.zeros(n_valid, dtype=bool)
        train_mask[:train_end] = True
        calib_mask[train_end:calib_end] = True
        test_mask[calib_end:] = True

        return supervised, data_valid, feature_cols, train_mask, calib_mask, test_mask

    def train_forecast_models(
        self,
        *,
        test_weeks: int = 52,
        calib_weeks: int = 52,
        alpha: float = 0.20,
        target_transform: str = "none",
        persist_artifacts: bool = True,
    ) -> dict[str, Any]:
        if test_weeks <= 0 or calib_weeks <= 0:
            raise ValueError("test_weeks and calib_weeks must be positive integers.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in the open interval (0, 1).")
        if target_transform not in {"none", "log1p"}:
            raise ValueError("target_transform must be either 'none' or 'log1p'.")

        config = ForecastConfig(test_weeks=test_weeks, calib_weeks=calib_weeks, alpha=alpha, target_transform=target_transform)
        supervised, data_valid, feature_cols, train_mask, calib_mask, test_mask = self._prepare_supervised(config)

        x = data_valid[feature_cols].to_numpy(dtype=float)
        x_train = x[train_mask]
        x_cal = x[calib_mask]
        x_test = x[test_mask]

        y_train_list: list[np.ndarray] = []
        y_prod_list: list[np.ndarray] = []
        y_cal_mat = np.zeros((int(calib_mask.sum()), config.horizon_weeks), dtype=float)
        y_test_mat = np.zeros((int(test_mask.sum()), config.horizon_weeks), dtype=float)

        prod_train_mask = train_mask | calib_mask
        for horizon in range(1, config.horizon_weeks + 1):
            y_h = data_valid[f"y_h{horizon}"].to_numpy(dtype=float)
            y_train_list.append(transform_target(y_h[train_mask], target_transform))
            y_prod_list.append(transform_target(y_h[prod_train_mask], target_transform))
            y_cal_mat[:, horizon - 1] = y_h[calib_mask]
            y_test_mat[:, horizon - 1] = y_h[test_mask]

        models_train_only = fit_models_hist_gbdt(x_train, y_train_list, random_state=config.point_random_state)

        z_cal_pred_mat = predict_matrix(models_train_only, x_cal)
        z_test_pred_mat = predict_matrix(models_train_only, x_test)
        y_cal_pred_mat = np.clip(inverse_transform_target(z_cal_pred_mat, target_transform), 0.0, None)
        y_test_pred_mat = np.clip(inverse_transform_target(z_test_pred_mat, target_transform), 0.0, None)

        per_h_metrics, overall_metrics = evaluate_multistep(y_test_mat, y_test_pred_mat)
        abs_err_cal = np.abs(y_cal_mat - y_cal_pred_mat)
        conformal_radii = np.array(
            [conformal_radius_from_abs_errors(abs_err_cal[:, idx], alpha=alpha) for idx in range(config.horizon_weeks)],
            dtype=float,
        )

        test_lo_mat = np.clip(y_test_pred_mat - conformal_radii, 0.0, None)
        test_hi_mat = np.clip(y_test_pred_mat + conformal_radii, 0.0, None)
        test_lo_mat, test_hi_mat = np.minimum(test_lo_mat, test_hi_mat), np.maximum(test_lo_mat, test_hi_mat)
        per_h_interval_metrics, overall_interval_metrics = evaluate_intervals(y_test_mat, test_lo_mat, test_hi_mat)

        models_prod = fit_models_hist_gbdt(x[prod_train_mask], y_prod_list, random_state=config.prod_random_state)
        z_test_pred_refit_mat = predict_matrix(models_prod, x_test)
        y_test_pred_refit_mat = np.clip(inverse_transform_target(z_test_pred_refit_mat, target_transform), 0.0, None)
        per_h_metrics_refit, overall_metrics_refit = evaluate_multistep(y_test_mat, y_test_pred_refit_mat)

        test_lo_refit_mat = np.clip(y_test_pred_refit_mat - conformal_radii, 0.0, None)
        test_hi_refit_mat = np.clip(y_test_pred_refit_mat + conformal_radii, 0.0, None)
        test_lo_refit_mat, test_hi_refit_mat = np.minimum(test_lo_refit_mat, test_hi_refit_mat), np.maximum(test_lo_refit_mat, test_hi_refit_mat)
        per_h_interval_metrics_refit, overall_interval_metrics_refit = evaluate_intervals(y_test_mat, test_lo_refit_mat, test_hi_refit_mat)

        self.state = TrainingState(
            config=config,
            data=self._raw_data.copy() if self._raw_data is not None else self.load_data(),
            supervised_data=supervised,
            data_valid=data_valid,
            feature_cols=feature_cols,
            train_mask=train_mask,
            calib_mask=calib_mask,
            test_mask=test_mask,
            models_train_only=models_train_only,
            models_prod=models_prod,
            conformal_radii=conformal_radii,
            per_h_metrics=per_h_metrics,
            overall_metrics=overall_metrics,
            per_h_interval_metrics=per_h_interval_metrics,
            overall_interval_metrics=overall_interval_metrics,
            per_h_metrics_refit=per_h_metrics_refit,
            overall_metrics_refit=overall_metrics_refit,
            per_h_interval_metrics_refit=per_h_interval_metrics_refit,
            overall_interval_metrics_refit=overall_interval_metrics_refit,
            y_test_mat=y_test_mat,
            y_test_pred_mat=y_test_pred_mat,
            y_test_pred_refit_mat=y_test_pred_refit_mat,
            test_lo_mat=test_lo_mat,
            test_hi_mat=test_hi_mat,
            test_lo_refit_mat=test_lo_refit_mat,
            test_hi_refit_mat=test_hi_refit_mat,
        )

        artifact_path = None
        if persist_artifacts:
            artifact_path = self.save_artifacts()

        result = {
            "training_status": "trained",
            "config": asdict(config),
            "split": self._split_summary(self.state),
            "n_features": len(feature_cols),
            "conformal_interval": {
                "type": "split_conformal_absolute_residual",
                "alpha": alpha,
                "nominal_coverage": 1 - alpha,
                "radii_by_horizon": [float(x) for x in conformal_radii],
            },
            "point_metrics_train_only": {
                "per_horizon": per_h_metrics,
                "overall": overall_metrics,
            },
            "interval_metrics_train_only": {
                "per_horizon": per_h_interval_metrics,
                "overall": overall_interval_metrics,
            },
            "point_metrics_refit_train_plus_calib": {
                "per_horizon": per_h_metrics_refit,
                "overall": overall_metrics_refit,
            },
            "interval_metrics_refit_train_plus_calib": {
                "per_horizon": per_h_interval_metrics_refit,
                "overall": overall_interval_metrics_refit,
            },
            "artifact_path": str(artifact_path) if artifact_path else None,
        }
        return _jsonable(result)

    def save_artifacts(self) -> Path:
        state = self._require_state()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifact_dir / "epid_forecasting_model_bundle.joblib"
        joblib.dump(
            {
                "config": state.config,
                "feature_cols": state.feature_cols,
                "models_train_only": state.models_train_only,
                "models_prod": state.models_prod,
                "conformal_radii": state.conformal_radii,
                "model_registry": self.get_model_registry(),
            },
            artifact_path,
        )
        return artifact_path

    def _require_state(self) -> TrainingState:
        if self.state is None:
            raise RuntimeError("Models are not trained. Call train_forecast_models first.")
        return self.state

    @staticmethod
    def _split_summary(state: TrainingState) -> dict[str, Any]:
        data_valid = state.data_valid
        return {
            "train_start": data_valid.loc[state.train_mask, "datetime"].min().date().isoformat(),
            "train_end": data_valid.loc[state.train_mask, "datetime"].max().date().isoformat(),
            "calib_start": data_valid.loc[state.calib_mask, "datetime"].min().date().isoformat(),
            "calib_end": data_valid.loc[state.calib_mask, "datetime"].max().date().isoformat(),
            "test_start": data_valid.loc[state.test_mask, "datetime"].min().date().isoformat(),
            "test_end": data_valid.loc[state.test_mask, "datetime"].max().date().isoformat(),
            "n_train": int(state.train_mask.sum()),
            "n_calib": int(state.calib_mask.sum()),
            "n_test": int(state.test_mask.sum()),
            "n_valid": int(len(data_valid)),
        }

    def get_feature_schema(self) -> dict[str, Any]:
        if self.state is None:
            config = ForecastConfig()
            _, data_valid, feature_cols, _, _, _ = self._prepare_supervised(config)
            schema_basis = "default_feature_pipeline"
        else:
            data_valid = self.state.data_valid
            feature_cols = self.state.feature_cols
            schema_basis = "trained_model_state"

        rows = []
        for index, col in enumerate(feature_cols):
            rows.append(
                {
                    "feature_index": index,
                    "feature_name": col,
                    "group": feature_group(col),
                    "dtype": str(data_valid[col].dtype),
                    "missing_after_valid_filter": int(data_valid[col].isna().sum()),
                }
            )
        return _jsonable({"schema_basis": schema_basis, "n_features": len(feature_cols), "features": rows})

    def get_model_registry(self) -> dict[str, Any]:
        state = self._require_state()
        models = []
        for horizon, model in enumerate(state.models_prod, start=1):
            params = model.get_params()
            models.append(
                {
                    "id": f"point_forecast_h{horizon}",
                    "family": "HistGradientBoostingRegressor",
                    "forecast_type": "point",
                    "output_type": "single_value",
                    "target_variable": state.config.target_col,
                    "target_time_offset": horizon,
                    "target_time_unit": "week",
                    "predicts": f"{state.config.target_col} at t+{horizon}",
                    "training_strategy": "direct",
                    "loss": params.get("loss"),
                    "purpose": "point_forecast",
                    "hyperparameters": {
                        "learning_rate": params.get("learning_rate"),
                        "max_iter": params.get("max_iter"),
                        "min_samples_leaf": params.get("min_samples_leaf"),
                        "l2_regularization": params.get("l2_regularization"),
                        "early_stopping": params.get("early_stopping"),
                        "max_depth": params.get("max_depth"),
                        "max_leaf_nodes": params.get("max_leaf_nodes"),
                    },
                }
            )
        return _jsonable(
            {
                "model_specification_id": "gbdt_influenza_forecast_4w_conformal",
                "task": f"forecast weekly {state.config.target_col} for 1-{state.config.horizon_weeks} weeks ahead",
                "forecast_design": {
                    "strategy": "direct_multi_step",
                    "per_model_output": "single_value",
                    "semantic_rule": "each model predicts exactly one target value for one specific future offset t+h",
                },
                "training_scheme": {
                    "strategy": "direct_multi_step",
                    "test_split": "time_based_holdout",
                    "test_weeks": state.config.test_weeks,
                    "calib_weeks": state.config.calib_weeks,
                    "prod_refit": "train_plus_calib",
                    "interval_method": "split_conformal_absolute_residual",
                    "alpha": state.config.alpha,
                    "nominal_interval_coverage": 1 - state.config.alpha,
                },
                "feature_groups": sorted({feature_group(col) for col in state.feature_cols}),
                "models": models,
            }
        )

    def _resolve_origin_row(self, origin_date: str | None) -> tuple[pd.Timestamp, pd.Series]:
        state = self._require_state()
        supervised = state.supervised_data.copy()
        feature_ready_mask = supervised[state.feature_cols].notna().all(axis=1).to_numpy()
        feature_ready = supervised.loc[feature_ready_mask].copy()
        if origin_date is None:
            row = feature_ready.iloc[-1]
            return pd.to_datetime(row["datetime"]), row

        parsed = pd.to_datetime(origin_date)
        matches = feature_ready.loc[pd.to_datetime(feature_ready["datetime"]) == parsed]
        if matches.empty:
            available_min = pd.to_datetime(feature_ready["datetime"]).min().date().isoformat()
            available_max = pd.to_datetime(feature_ready["datetime"]).max().date().isoformat()
            raise ValueError(
                f"origin_date must match a dataset week with complete features. "
                f"Available range: {available_min} to {available_max}. Received: {parsed.date().isoformat()}."
            )
        row = matches.iloc[-1]
        return pd.to_datetime(row["datetime"]), row

    def forecast_next_4_weeks(self, *, origin_date: str | None = None) -> dict[str, Any]:
        state = self._require_state()
        resolved_origin, row = self._resolve_origin_row(origin_date)
        x0 = row[state.feature_cols].to_numpy(dtype=float).reshape(1, -1)
        z_hat = np.array([model.predict(x0)[0] for model in state.models_prod], dtype=float)
        y_hat = np.clip(inverse_transform_target(z_hat, state.config.target_transform), 0.0, None)
        y_lo = np.clip(y_hat - state.conformal_radii, 0.0, None)
        y_hi = np.clip(y_hat + state.conformal_radii, 0.0, None)
        y_lo, y_hi = np.minimum(y_lo, y_hi), np.maximum(y_lo, y_hi)

        step = state.data["datetime"].diff().dropna().median()
        if pd.isna(step):
            step = pd.Timedelta(days=7)
        future_dates = pd.date_range(resolved_origin + step, periods=state.config.horizon_weeks, freq=step)

        forecast = pd.DataFrame(
            {
                "origin_date": [resolved_origin.date().isoformat()] * state.config.horizon_weeks,
                "target_date": [date.date().isoformat() for date in future_dates],
                "horizon_weeks": np.arange(1, state.config.horizon_weeks + 1),
                "inc_per_10k_pred": y_hat,
                "pi80_lower": y_lo,
                "pi80_upper": y_hi,
                "pi80_width": y_hi - y_lo,
            }
        )
        return _jsonable(
            {
                "origin_date": resolved_origin.date().isoformat(),
                "forecast": forecast,
                "interval": {
                    "method": "split_conformal_absolute_residual",
                    "alpha": state.config.alpha,
                    "nominal_coverage": 1 - state.config.alpha,
                    "lower_column": "pi80_lower",
                    "upper_column": "pi80_upper",
                },
            }
        )

    def backtest_forecast_models(self) -> dict[str, Any]:
        state = self._require_state()
        return _jsonable(
            {
                "split": self._split_summary(state),
                "point_metrics_train_only": {
                    "per_horizon": state.per_h_metrics,
                    "overall": state.overall_metrics,
                },
                "interval_metrics_train_only": {
                    "per_horizon": state.per_h_interval_metrics,
                    "overall": state.overall_interval_metrics,
                },
                "point_metrics_refit_train_plus_calib": {
                    "per_horizon": state.per_h_metrics_refit,
                    "overall": state.overall_metrics_refit,
                },
                "interval_metrics_refit_train_plus_calib": {
                    "per_horizon": state.per_h_interval_metrics_refit,
                    "overall": state.overall_interval_metrics_refit,
                },
                "conformal_radii": [float(x) for x in state.conformal_radii],
            }
        )

    @staticmethod
    def _validate_storage_component(value: str, field_name: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must be a non-empty string.")
        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError(f"{field_name} must be a single S3 key component without path separators.")
        if any(ord(character) < 32 for character in normalized):
            raise ValueError(f"{field_name} must not contain control characters.")
        return normalized

    def export_forecast_results(
        self,
        *,
        user_id: str,
        session_id: str,
        origin_date: str | None = None,
        history_tail_n: int = 40,
        storage: S3ArtifactStorage | None = None,
    ) -> dict[str, Any]:
        """Publish CSV/JSON forecast artifacts to S3-compatible object storage."""
        state = self._require_state()
        if history_tail_n <= 0:
            raise ValueError("history_tail_n must be a positive integer.")

        validated_user_id = self._validate_storage_component(user_id, "user_id")
        validated_session_id = self._validate_storage_component(session_id, "session_id")
        artifact_storage = storage or S3ArtifactStorage.from_environment()
        run_id = str(uuid.uuid4())
        object_prefix = f"{validated_user_id}/{validated_session_id}/epid_forecasting/{run_id}"

        with TemporaryDirectory(prefix="epid_forecasting_export_") as temporary_directory:
            out_dir = Path(temporary_directory)
            split = self._split_summary(state)
            metrics_per_h = state.per_h_metrics_refit.copy()
            for key, value in split.items():
                metrics_per_h[key] = value
            metrics_per_h["n_features"] = len(state.feature_cols)
            metrics_per_h["target_transform"] = state.config.target_transform
            metrics_per_h["section"] = "per_horizon_refit_train_plus_calib"
            metrics_overall = pd.DataFrame(
                [
                    {
                        "section": "overall_refit_train_plus_calib",
                        "horizon_weeks": "all",
                        "r2": state.overall_metrics_refit["r2_overall"],
                        "rmse": state.overall_metrics_refit["rmse_overall"],
                        "mae": state.overall_metrics_refit["mae_overall"],
                        **split,
                        "n_features": len(state.feature_cols),
                        "target_transform": state.config.target_transform,
                    }
                ]
            )
            all_metric_cols = sorted(set(metrics_per_h.columns) | set(metrics_overall.columns))
            metrics_summary = pd.concat(
                [metrics_per_h.reindex(columns=all_metric_cols), metrics_overall.reindex(columns=all_metric_cols)],
                ignore_index=True,
            )
            metrics_summary_path = out_dir / "metrics_summary.csv"
            metrics_summary.to_csv(metrics_summary_path, index=False, encoding="utf-8")

            test_origin_dates = pd.to_datetime(state.data_valid.loc[state.test_mask, "datetime"]).reset_index(drop=True)
            test_predictions = pd.DataFrame({"origin_date": test_origin_dates.dt.date.astype(str)})
            for horizon in range(1, state.config.horizon_weeks + 1):
                test_predictions[f"target_date_h{horizon}"] = (
                    test_origin_dates + pd.to_timedelta(7 * horizon, unit="D")
                ).dt.date.astype(str)
                test_predictions[f"y_true_h{horizon}"] = state.y_test_mat[:, horizon - 1]
                test_predictions[f"y_pred_h{horizon}"] = state.y_test_pred_refit_mat[:, horizon - 1]
                test_predictions[f"abs_error_h{horizon}"] = np.abs(
                    state.y_test_mat[:, horizon - 1] - state.y_test_pred_refit_mat[:, horizon - 1]
                )
                test_predictions[f"pi80_lower_h{horizon}"] = state.test_lo_refit_mat[:, horizon - 1]
                test_predictions[f"pi80_upper_h{horizon}"] = state.test_hi_refit_mat[:, horizon - 1]
            test_predictions_path = out_dir / "test_predictions.csv"
            test_predictions.to_csv(test_predictions_path, index=False, encoding="utf-8")

            forecast_result = self.forecast_next_4_weeks(origin_date=origin_date)
            forecast_next = pd.DataFrame(forecast_result["forecast"])
            forecast_next_path = out_dir / "forecast_next_4w.csv"
            forecast_next.to_csv(forecast_next_path, index=False, encoding="utf-8")

            feature_list = pd.DataFrame(
                {
                    "feature_name": state.feature_cols,
                    "feature_index": np.arange(len(state.feature_cols)),
                    "feature_group": [feature_group(col) for col in state.feature_cols],
                }
            )
            feature_list_path = out_dir / "feature_list.csv"
            feature_list.to_csv(feature_list_path, index=False, encoding="utf-8")

            origin_ts = pd.to_datetime(forecast_result["origin_date"])
            history_tail = (
                state.data.loc[pd.to_datetime(state.data["datetime"]) <= origin_ts, ["datetime", state.config.target_col]]
                .copy()
                .sort_values("datetime")
                .tail(history_tail_n)
                .reset_index(drop=True)
            )
            hist_out = pd.DataFrame(
                {
                    "date": pd.to_datetime(history_tail["datetime"]).dt.date.astype(str),
                    "value": history_tail[state.config.target_col].astype(float),
                    "row_type": "history",
                    "origin_date": None,
                    "horizon_weeks": None,
                    "pi80_lower": np.nan,
                    "pi80_upper": np.nan,
                }
            )
            forecast_tail_out = pd.DataFrame(
                {
                    "date": forecast_next["target_date"],
                    "value": forecast_next["inc_per_10k_pred"].astype(float),
                    "row_type": "forecast",
                    "origin_date": forecast_next["origin_date"],
                    "horizon_weeks": forecast_next["horizon_weeks"],
                    "pi80_lower": forecast_next["pi80_lower"],
                    "pi80_upper": forecast_next["pi80_upper"],
                }
            )
            history_plus_forecast = pd.concat([hist_out, forecast_tail_out], ignore_index=True)
            history_plus_forecast_path = out_dir / "history_plus_forecast_40.csv"
            history_plus_forecast.to_csv(history_plus_forecast_path, index=False, encoding="utf-8")

            registry_path = out_dir / "model_registry.json"
            import json
            registry_path.write_text(
                json.dumps(_jsonable(self.get_model_registry()), ensure_ascii=False, indent=2), encoding="utf-8"
            )

            export_files = {
                "metrics": metrics_summary_path,
                "test_predictions": test_predictions_path,
                "forecast": forecast_next_path,
                "feature_list": feature_list_path,
                "history_plus_forecast": history_plus_forecast_path,
                "model_registry": registry_path,
            }
            artifacts = {
                artifact_name: artifact_storage.upload_file(file_path, f"{object_prefix}/{file_path.name}")
                for artifact_name, file_path in export_files.items()
            }

        return _jsonable(
            {
                "storage": "s3",
                "run_id": run_id,
                "forecast_origin_date": forecast_result["origin_date"],
                "artifacts": artifacts,
            }
        )

