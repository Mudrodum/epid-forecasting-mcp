"""Fixed analytical workflow behind the compact influenza forecasting MCP tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import DEFAULT_DATA_PATH, ForecastConfig
from .features import build_supervised
from .seasonal_analysis import compare_recent_epidemic_waves
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if not isinstance(value, (list, tuple, dict, np.ndarray, pd.DataFrame, pd.Series)) and pd.isna(value):
        return None
    return value


@dataclass
class TrainingState:
    """Cached result of the fixed model-fitting and conformal-calibration stage."""

    config: ForecastConfig
    data: pd.DataFrame
    supervised_data: pd.DataFrame
    data_valid: pd.DataFrame
    feature_cols: list[str]
    train_mask: np.ndarray
    calib_mask: np.ndarray
    test_mask: np.ndarray
    evaluation_models: list[HistGradientBoostingRegressor]
    production_models: list[HistGradientBoostingRegressor]
    conformal_radii: np.ndarray
    per_h_metrics: pd.DataFrame
    overall_metrics: dict[str, float]
    per_h_interval_metrics: pd.DataFrame
    overall_interval_metrics: dict[str, float]


class EpidForecastingService:
    """Runs one fixed, agent-facing influenza forecasting workflow."""

    def __init__(self, data_path: str | Path = DEFAULT_DATA_PATH) -> None:
        self.data_path = Path(data_path)
        self.state: TrainingState | None = None
        self._raw_data: pd.DataFrame | None = None

    @staticmethod
    def _validate_data_frame(frame: pd.DataFrame, *, source_name: str = "dataset") -> pd.DataFrame:
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{source_name} is missing required columns: {missing}")
        df = frame.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if df["datetime"].isna().any():
            raise ValueError(f"{source_name} contains invalid datetime values.")
        for column in sorted(REQUIRED_COLUMNS - {"datetime"}):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        required_without_optional_humidity_extremes = sorted(REQUIRED_COLUMNS - {"rh_min", "rh_max"})
        if df[required_without_optional_humidity_extremes].isna().any().any():
            bad_columns = [
                col for col in required_without_optional_humidity_extremes if df[col].isna().any()
            ]
            raise ValueError(f"{source_name} contains NaN values in required columns: {bad_columns}")
        return df.sort_values("datetime").reset_index(drop=True)

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        df = self._validate_data_frame(df, source_name="dataset")
        self._raw_data = df.copy()
        return df

    def describe_dataset(self) -> dict[str, Any]:
        df = self._raw_data.copy() if self._raw_data is not None else self.load_data()
        null_counts = {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}
        return _jsonable(
            {
                "rows": len(df),
                "date_min": df["datetime"].min().date().isoformat(),
                "date_max": df["datetime"].max().date().isoformat(),
                "target_variable": "inc_per_10k",
                "target_description": "Weekly influenza morbidity per 10,000 population.",
                "missing_values": null_counts,
            }
        )

    def _prepare_supervised(
        self, config: ForecastConfig, data: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
        df = self.load_data() if data is None else self._validate_data_frame(data, source_name="provided weekly data")
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

    def _fit_fixed_workflow(self, data: pd.DataFrame | None = None) -> TrainingState:
        """Fit the pre-configured model once and calibrate its prediction intervals."""
        config = ForecastConfig()
        validated_data = self.load_data() if data is None else self._validate_data_frame(data, source_name="provided weekly data")
        supervised, data_valid, feature_cols, train_mask, calib_mask, test_mask = self._prepare_supervised(config, data=validated_data)

        x = data_valid[feature_cols].to_numpy(dtype=float)
        x_train = x[train_mask]
        x_calib = x[calib_mask]
        x_test = x[test_mask]
        y_train_list: list[np.ndarray] = []
        y_calib = np.zeros((int(calib_mask.sum()), config.horizon_weeks), dtype=float)
        y_test = np.zeros((int(test_mask.sum()), config.horizon_weeks), dtype=float)
        for horizon in range(1, config.horizon_weeks + 1):
            y_h = data_valid[f"y_h{horizon}"].to_numpy(dtype=float)
            y_train_list.append(transform_target(y_h[train_mask], config.target_transform))
            y_calib[:, horizon - 1] = y_h[calib_mask]
            y_test[:, horizon - 1] = y_h[test_mask]

        evaluation_models = fit_models_hist_gbdt(x_train, y_train_list, random_state=config.point_random_state)
        y_calib_pred = np.clip(
            inverse_transform_target(predict_matrix(evaluation_models, x_calib), config.target_transform), 0.0, None
        )
        y_test_pred = np.clip(
            inverse_transform_target(predict_matrix(evaluation_models, x_test), config.target_transform), 0.0, None
        )
        conformal_radii = np.array(
            [
                conformal_radius_from_abs_errors(np.abs(y_calib[:, idx] - y_calib_pred[:, idx]), alpha=config.alpha)
                for idx in range(config.horizon_weeks)
            ],
            dtype=float,
        )
        y_test_lower = np.clip(y_test_pred - conformal_radii, 0.0, None)
        y_test_upper = np.clip(y_test_pred + conformal_radii, 0.0, None)
        point_per_h, point_overall = evaluate_multistep(y_test, y_test_pred)
        interval_per_h, interval_overall = evaluate_intervals(y_test, y_test_lower, y_test_upper)

        # Point forecasts for the future use every labelled feature-complete observation.
        # The interval radii are reported as calibration-derived uncertainty bounds for
        # this production refit; nominal split-conformal coverage is asserted only for
        # the untouched holdout evaluation above.
        y_production_list: list[np.ndarray] = []
        for horizon in range(1, config.horizon_weeks + 1):
            y_h = data_valid[f"y_h{horizon}"].to_numpy(dtype=float)
            y_production_list.append(transform_target(y_h, config.target_transform))
        production_models = fit_models_hist_gbdt(x, y_production_list, random_state=config.prod_random_state)

        state = TrainingState(
            config=config,
            data=validated_data.copy(),
            supervised_data=supervised,
            data_valid=data_valid,
            feature_cols=feature_cols,
            train_mask=train_mask,
            calib_mask=calib_mask,
            test_mask=test_mask,
            evaluation_models=evaluation_models,
            production_models=production_models,
            conformal_radii=conformal_radii,
            per_h_metrics=point_per_h,
            overall_metrics=point_overall,
            per_h_interval_metrics=interval_per_h,
            overall_interval_metrics=interval_overall,
        )
        if data is None:
            self.state = state
        return state

    def _ensure_state(self) -> TrainingState:
        return self.state if self.state is not None else self._fit_fixed_workflow()

    def fit_forecasting_state_for_frame(self, frame: pd.DataFrame) -> TrainingState:
        """Fit the fixed workflow on an externally supplied merged weekly table."""
        return self._fit_fixed_workflow(data=frame)

    @staticmethod
    def _split_summary(state: TrainingState) -> dict[str, Any]:
        data_valid = state.data_valid
        return {
            "train_start": data_valid.loc[state.train_mask, "datetime"].min().date().isoformat(),
            "train_end": data_valid.loc[state.train_mask, "datetime"].max().date().isoformat(),
            "calibration_start": data_valid.loc[state.calib_mask, "datetime"].min().date().isoformat(),
            "calibration_end": data_valid.loc[state.calib_mask, "datetime"].max().date().isoformat(),
            "test_start": data_valid.loc[state.test_mask, "datetime"].min().date().isoformat(),
            "test_end": data_valid.loc[state.test_mask, "datetime"].max().date().isoformat(),
            "n_train": int(state.train_mask.sum()),
            "n_calibration": int(state.calib_mask.sum()),
            "n_test": int(state.test_mask.sum()),
        }

    def _resolve_origin_row(self, state: TrainingState, origin_date: str | None) -> tuple[pd.Timestamp, pd.Series]:
        feature_ready = state.supervised_data.loc[
            state.supervised_data[state.feature_cols].notna().all(axis=1)
        ].copy()
        if origin_date is None:
            row = feature_ready.iloc[-1]
            return pd.to_datetime(row["datetime"]), row
        parsed = pd.to_datetime(origin_date)
        matches = feature_ready.loc[pd.to_datetime(feature_ready["datetime"]) == parsed]
        if matches.empty:
            available_min = pd.to_datetime(feature_ready["datetime"]).min().date().isoformat()
            available_max = pd.to_datetime(feature_ready["datetime"]).max().date().isoformat()
            raise ValueError(
                "origin_date must match a dataset week with complete features. "
                f"Available range: {available_min} to {available_max}. Received: {parsed.date().isoformat()}."
            )
        row = matches.iloc[-1]
        return pd.to_datetime(row["datetime"]), row

    def _forecast_next_4_weeks(self, state: TrainingState, origin_date: str | None) -> dict[str, Any]:
        resolved_origin, row = self._resolve_origin_row(state, origin_date)
        x0 = row[state.feature_cols].to_numpy(dtype=float).reshape(1, -1)
        transformed_pred = np.array([model.predict(x0)[0] for model in state.production_models], dtype=float)
        prediction = np.clip(inverse_transform_target(transformed_pred, state.config.target_transform), 0.0, None)
        lower = np.clip(prediction - state.conformal_radii, 0.0, None)
        upper = np.clip(prediction + state.conformal_radii, 0.0, None)
        step = state.data["datetime"].diff().dropna().median()
        if pd.isna(step):
            step = pd.Timedelta(days=7)
        future_dates = pd.date_range(resolved_origin + step, periods=state.config.horizon_weeks, freq=step)
        forecast = pd.DataFrame(
            {
                "origin_date": [resolved_origin.date().isoformat()] * state.config.horizon_weeks,
                "target_date": [date.date().isoformat() for date in future_dates],
                "horizon_weeks": np.arange(1, state.config.horizon_weeks + 1),
                "inc_per_10k_prediction": prediction,
                "pi80_lower": lower,
                "pi80_upper": upper,
            }
        )
        return _jsonable({"origin_date": resolved_origin.date().isoformat(), "forecast": forecast})

    def _forecast_result_from_state(self, state: TrainingState, *, origin_date: str | None = None) -> dict[str, Any]:
        """Build compact inline forecast results from an already fitted state."""
        forecast = self._forecast_next_4_weeks(state, origin_date)
        config = asdict(state.config)
        return _jsonable(
            {
                "task": "four-week influenza incidence forecasting",
                "target_variable": state.config.target_col,
                "forecast_horizon_weeks": state.config.horizon_weeks,
                "forecast_origin_date": forecast["origin_date"],
                "fixed_configuration": {
                    "forecast_strategy": "direct_multi_step",
                    "model_family": "HistGradientBoostingRegressor",
                    "target_transform": state.config.target_transform,
                    "holdout_interval_method": "split_conformal_absolute_residual",
                    "holdout_nominal_interval_coverage": 1 - state.config.alpha,
                    "production_forecast_fit": "all_labelled_feature_complete_observations",
                    "test_weeks": state.config.test_weeks,
                    "calibration_weeks": state.config.calib_weeks,
                    "n_features": len(state.feature_cols),
                },
                "holdout_evaluation": {
                    "split": self._split_summary(state),
                    "point_metrics": {
                        "per_horizon": state.per_h_metrics,
                        "overall": state.overall_metrics,
                    },
                    "interval_metrics": {
                        "per_horizon": state.per_h_interval_metrics,
                        "overall": state.overall_interval_metrics,
                    },
                },
                "forecast": forecast["forecast"],
                "forecast_uncertainty_bounds": {
                    "method": "calibration_residual_bounds_transferred_to_production_refit",
                    "lower_column": "pi80_lower",
                    "upper_column": "pi80_upper",
                    "interpretation": (
                        "Bounds use conformal radii calibrated for the holdout-evaluated model. "
                        "Nominal split-conformal coverage is reported only for holdout evaluation, "
                        "not asserted for the all-labelled-data production refit."
                    ),
                },
            }
        )

    def run_influenza_forecasting(self, *, origin_date: str | None = None) -> dict[str, Any]:
        """Execute the fixed public forecasting workflow and return compact inline results."""
        return self._forecast_result_from_state(self._ensure_state(), origin_date=origin_date)

    def run_influenza_forecasting_for_frame(
        self, frame: pd.DataFrame, *, origin_date: str | None = None
    ) -> tuple[dict[str, Any], TrainingState]:
        """Fit and forecast from a supplied merged weekly influenza/weather table."""
        state = self.fit_forecasting_state_for_frame(frame)
        return self._forecast_result_from_state(state, origin_date=origin_date), state

    def compare_epidemic_waves(
        self, *, season_start_week: int = 40, smooth_window: int = 3, n_last_seasons: int = 3
    ) -> dict[str, Any]:
        """Compare recent epidemic waves in the aggregate weekly incidence series."""
        df = self._raw_data.copy() if self._raw_data is not None else self.load_data()
        return _jsonable(
            compare_recent_epidemic_waves(
                df,
                season_start_week=season_start_week,
                smooth_window=smooth_window,
                n_last_seasons=n_last_seasons,
                target_col="inc_per_10k",
            )
        )


