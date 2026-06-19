"""S3-compatible artifact persistence for forecasting MCP results."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import boto3
import pandas as pd
from botocore.client import Config

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class S3StorageSettings:
    """Configuration for server-side S3 artifact storage."""

    endpoint_url: str
    access_key: str
    secret_key: str
    bucket_name: str
    presigned_expiration_seconds: int = 3600

    @classmethod
    def from_env(cls) -> "S3StorageSettings":
        """Read storage settings using the CoScientist dataset-server variable names.

        S3-prefixed aliases are accepted to remain compatible with the chemical MCP
        server naming convention already present in the repository.
        """
        endpoint_url = os.getenv("ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
        access_key = os.getenv("ACCESS_KEY") or os.getenv("S3_ACCESS_KEY")
        secret_key = os.getenv("SECRET_KEY") or os.getenv("S3_SECRET_KEY")
        bucket_name = os.getenv("BUCKET_NAME") or os.getenv("S3_BUCKET_NAME")
        missing = [
            name
            for name, value in {
                "ENDPOINT_URL": endpoint_url,
                "ACCESS_KEY": access_key,
                "SECRET_KEY": secret_key,
                "BUCKET_NAME": bucket_name,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                "S3 artifact storage is required for run_influenza_forecasting. "
                "Missing environment variables: " + ", ".join(missing)
            )
        expiration_text = os.getenv("PRESIGNED_URL_EXPIRATION_SECONDS", "3600")
        try:
            expiration = int(expiration_text)
        except ValueError as exc:
            raise RuntimeError("PRESIGNED_URL_EXPIRATION_SECONDS must be an integer.") from exc
        if expiration <= 0:
            raise RuntimeError("PRESIGNED_URL_EXPIRATION_SECONDS must be positive.")
        return cls(
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket_name,
            presigned_expiration_seconds=expiration,
        )


class S3ForecastArtifactStore:
    """Persist forecasting outputs in S3 and return temporary download URLs."""

    def __init__(self, settings: S3StorageSettings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.endpoint_url,
                aws_access_key_id=self.settings.access_key,
                aws_secret_access_key=self.settings.secret_key,
                config=Config(signature_version="s3v4"),
            )
        return self._client

    @staticmethod
    def _validate_identifier(value: str, name: str) -> str:
        if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
            raise ValueError(
                f"{name} must contain only letters, digits, dot, underscore, or hyphen "
                "and must be 1 to 128 characters long."
            )
        return value

    def _put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.client.upload_fileobj(
            BytesIO(data),
            self.settings.bucket_name,
            key,
            ExtraArgs={"ContentType": content_type},
        )

    def _presigned_url(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.bucket_name, "Key": key},
            ExpiresIn=self.settings.presigned_expiration_seconds,
        )

    def _upload_artifacts(self, prefix: str, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
        for artifact in artifacts.values():
            self._put_bytes(artifact["key"], artifact["bytes"], artifact["content_type"])
        return {
            "storage_prefix": prefix,
            "download_access": "presigned_urls",
            "presigned_url_expiration_seconds": self.settings.presigned_expiration_seconds,
            "artifacts": {
                name: {
                    "s3_uri": f"s3://{self.settings.bucket_name}/{artifact['key']}",
                    "download_url": self._presigned_url(artifact["key"]),
                    "content_type": artifact["content_type"],
                }
                for name, artifact in artifacts.items()
            },
        }

    def save_forecasting_run(
        self,
        *,
        result: dict[str, Any],
        user_id: str,
        session_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload forecast, metrics and run summary under a session-scoped S3 prefix."""
        user_id = self._validate_identifier(user_id, "user_id")
        session_id = self._validate_identifier(session_id, "session_id")
        run_id = run_id or str(uuid.uuid4())
        self._validate_identifier(run_id, "run_id")
        prefix = f"{user_id}/{session_id}/epid_forecasting/{run_id}"

        forecast_key = f"{prefix}/forecast.csv"
        metrics_key = f"{prefix}/metrics.json"
        summary_key = f"{prefix}/run_summary.json"

        forecast_csv = pd.DataFrame(result["forecast"]).to_csv(index=False).encode("utf-8")
        metrics_payload = {
            "task": result["task"],
            "target_variable": result["target_variable"],
            "forecast_origin_date": result["forecast_origin_date"],
            "holdout_evaluation": result["holdout_evaluation"],
        }
        summary_payload = {
            "task": result["task"],
            "target_variable": result["target_variable"],
            "forecast_horizon_weeks": result["forecast_horizon_weeks"],
            "forecast_origin_date": result["forecast_origin_date"],
            "fixed_configuration": result["fixed_configuration"],
            "forecast_uncertainty_bounds": result["forecast_uncertainty_bounds"],
            "artifacts": {
                "forecast_s3_path": f"s3://{self.settings.bucket_name}/{forecast_key}",
                "metrics_s3_path": f"s3://{self.settings.bucket_name}/{metrics_key}",
                "summary_s3_path": f"s3://{self.settings.bucket_name}/{summary_key}",
            },
        }

        self._put_bytes(forecast_key, forecast_csv, "text/csv")
        self._put_bytes(
            metrics_key,
            json.dumps(metrics_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            "application/json",
        )
        self._put_bytes(
            summary_key,
            json.dumps(summary_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            "application/json",
        )

        return {
            "run_id": run_id,
            "storage_prefix": prefix,
            "download_access": "presigned_urls",
            "presigned_url_expiration_seconds": self.settings.presigned_expiration_seconds,
            "artifacts": {
                "forecast": {
                    "s3_uri": f"s3://{self.settings.bucket_name}/{forecast_key}",
                    "download_url": self._presigned_url(forecast_key),
                    "content_type": "text/csv",
                },
                "metrics": {
                    "s3_uri": f"s3://{self.settings.bucket_name}/{metrics_key}",
                    "download_url": self._presigned_url(metrics_key),
                    "content_type": "application/json",
                },
                "run_summary": {
                    "s3_uri": f"s3://{self.settings.bucket_name}/{summary_key}",
                    "download_url": self._presigned_url(summary_key),
                    "content_type": "application/json",
                },
            },
        }

    def save_influenza_db_dataset(
        self,
        *,
        weekly: pd.DataFrame,
        cases: pd.DataFrame,
        age_groups: pd.DataFrame,
        summary: dict[str, Any],
        user_id: str,
        session_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload normalized influenza DB tables under a session-scoped S3 prefix."""
        user_id = self._validate_identifier(user_id, "user_id")
        session_id = self._validate_identifier(session_id, "session_id")
        run_id = run_id or str(uuid.uuid4())
        self._validate_identifier(run_id, "run_id")
        prefix = f"{user_id}/{session_id}/epid_forecasting/influenza_db/{run_id}"

        artifacts = {
            "weekly": {
                "key": f"{prefix}/weekly.csv",
                "bytes": weekly.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "cases": {
                "key": f"{prefix}/cases.csv",
                "bytes": cases.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "age_groups": {
                "key": f"{prefix}/age_groups.csv",
                "bytes": age_groups.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "summary": {
                "key": f"{prefix}/summary.json",
                "bytes": json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
                "content_type": "application/json",
            },
        }

        for artifact in artifacts.values():
            self._put_bytes(artifact["key"], artifact["bytes"], artifact["content_type"])

        return {
            "run_id": run_id,
            "storage_prefix": prefix,
            "download_access": "presigned_urls",
            "presigned_url_expiration_seconds": self.settings.presigned_expiration_seconds,
            "artifacts": {
                name: {
                    "s3_uri": f"s3://{self.settings.bucket_name}/{artifact['key']}",
                    "download_url": self._presigned_url(artifact["key"]),
                    "content_type": artifact["content_type"],
                }
                for name, artifact in artifacts.items()
            },
        }

    def save_weather_dataset(
        self,
        *,
        hourly: pd.DataFrame,
        weekly: pd.DataFrame,
        location: dict[str, Any],
        summary: dict[str, Any],
        user_id: str,
        session_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload Open-Meteo hourly/weekly weather artifacts."""
        user_id = self._validate_identifier(user_id, "user_id")
        session_id = self._validate_identifier(session_id, "session_id")
        run_id = run_id or str(uuid.uuid4())
        self._validate_identifier(run_id, "run_id")
        prefix = f"{user_id}/{session_id}/epid_forecasting/weather/{run_id}"
        artifacts = {
            "weather_hourly": {
                "key": f"{prefix}/weather_hourly.csv",
                "bytes": hourly.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "weather_weekly": {
                "key": f"{prefix}/weather_weekly.csv",
                "bytes": weekly.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "weather_location": {
                "key": f"{prefix}/weather_location.json",
                "bytes": json.dumps(location, ensure_ascii=False, indent=2).encode("utf-8"),
                "content_type": "application/json",
            },
            "weather_summary": {
                "key": f"{prefix}/weather_summary.json",
                "bytes": json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
                "content_type": "application/json",
            },
        }
        metadata = self._upload_artifacts(prefix, artifacts)
        return {"run_id": run_id, **metadata}

    def save_shap_explainability(
        self,
        *,
        global_importance: pd.DataFrame,
        local_values: pd.DataFrame,
        worst_cases: pd.DataFrame,
        summary: dict[str, Any],
        user_id: str,
        session_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload SHAP explainability tables and summary."""
        user_id = self._validate_identifier(user_id, "user_id")
        session_id = self._validate_identifier(session_id, "session_id")
        run_id = run_id or str(uuid.uuid4())
        self._validate_identifier(run_id, "run_id")
        prefix = f"{user_id}/{session_id}/epid_forecasting/shap/{run_id}"
        artifacts = {
            "shap_global_importance": {
                "key": f"{prefix}/shap_global_importance.csv",
                "bytes": global_importance.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "shap_local_values": {
                "key": f"{prefix}/shap_local_values.csv",
                "bytes": local_values.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "shap_worst_cases": {
                "key": f"{prefix}/shap_worst_cases.csv",
                "bytes": worst_cases.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "shap_summary": {
                "key": f"{prefix}/shap_summary.json",
                "bytes": json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
                "content_type": "application/json",
            },
        }
        metadata = self._upload_artifacts(prefix, artifacts)
        return {"run_id": run_id, **metadata}

    def save_br_calibration_run(
        self,
        *,
        kind: str,
        trajectory: pd.DataFrame,
        parameter_samples: pd.DataFrame,
        parameter_summary: dict[str, Any],
        diagnostics: dict[str, Any],
        configuration: dict[str, Any],
        limitations: list[str],
        figures: dict[str, dict[str, bytes]],
        user_id: str,
        session_id: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload mechanistic BR-calibration tables, summaries, and figures."""

        user_id = self._validate_identifier(user_id, "user_id")
        session_id = self._validate_identifier(session_id, "session_id")
        kind = self._validate_identifier(kind, "kind")
        run_id = run_id or str(uuid.uuid4())
        self._validate_identifier(run_id, "run_id")
        prefix = f"{user_id}/{session_id}/epid_forecasting/br_calibration/{kind}/{run_id}"

        summary_payload = {
            "configuration": configuration,
            "parameter_summary": parameter_summary,
            "diagnostics": diagnostics,
            "limitations": limitations,
        }
        artifacts: dict[str, dict[str, Any]] = {
            "trajectory": {
                "key": f"{prefix}/trajectory.csv",
                "bytes": trajectory.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "parameter_samples": {
                "key": f"{prefix}/parameter_samples.csv",
                "bytes": parameter_samples.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "run_summary": {
                "key": f"{prefix}/run_summary.json",
                "bytes": json.dumps(summary_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                "content_type": "application/json",
            },
        }
        for figure_name, formats in figures.items():
            for extension, payload in formats.items():
                if extension not in {"png", "pdf"}:
                    raise ValueError(f"Unsupported BR figure extension: {extension}.")
                artifacts[f"{figure_name}_{extension}"] = {
                    "key": f"{prefix}/{figure_name}.{extension}",
                    "bytes": payload,
                    "content_type": "image/png" if extension == "png" else "application/pdf",
                }

        metadata = self._upload_artifacts(prefix, artifacts)
        return {"run_id": run_id, **metadata}

    def save_bulletin_context(
        self,
        *,
        context: dict[str, Any],
        markdown: str,
        weekly: pd.DataFrame,
        age_groups: pd.DataFrame,
        user_id: str,
        session_id: str,
        run_id: str | None = None,
        weather_hourly: pd.DataFrame | None = None,
        weather_weekly: pd.DataFrame | None = None,
        merged_weekly: pd.DataFrame | None = None,
        shap_global_importance: pd.DataFrame | None = None,
        shap_local_values: pd.DataFrame | None = None,
        shap_worst_cases: pd.DataFrame | None = None,
        br_trajectory: pd.DataFrame | None = None,
        br_parameter_samples: pd.DataFrame | None = None,
        br_summary: dict[str, Any] | None = None,
        br_figures: dict[str, dict[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        """Upload a structured bulletin evidence packet and supporting tables."""
        user_id = self._validate_identifier(user_id, "user_id")
        session_id = self._validate_identifier(session_id, "session_id")
        run_id = run_id or str(uuid.uuid4())
        self._validate_identifier(run_id, "run_id")
        prefix = f"{user_id}/{session_id}/epid_forecasting/bulletin_context/{run_id}"

        artifacts = {
            "bulletin_context_json": {
                "key": f"{prefix}/bulletin_context.json",
                "bytes": json.dumps(context, ensure_ascii=False, indent=2).encode("utf-8"),
                "content_type": "application/json",
            },
            "bulletin_context_markdown": {
                "key": f"{prefix}/bulletin_context.md",
                "bytes": markdown.encode("utf-8"),
                "content_type": "text/markdown",
            },
            "weekly": {
                "key": f"{prefix}/weekly.csv",
                "bytes": weekly.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
            "age_groups": {
                "key": f"{prefix}/age_groups.csv",
                "bytes": age_groups.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            },
        }
        if weather_hourly is not None:
            artifacts["weather_hourly"] = {
                "key": f"{prefix}/weather_hourly.csv",
                "bytes": weather_hourly.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            }
        if weather_weekly is not None:
            artifacts["weather_weekly"] = {
                "key": f"{prefix}/weather_weekly.csv",
                "bytes": weather_weekly.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            }
        if merged_weekly is not None:
            artifacts["merged_weekly"] = {
                "key": f"{prefix}/merged_weekly.csv",
                "bytes": merged_weekly.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            }
        if shap_global_importance is not None:
            artifacts["shap_global_importance"] = {
                "key": f"{prefix}/shap_global_importance.csv",
                "bytes": shap_global_importance.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            }
        if shap_local_values is not None:
            artifacts["shap_local_values"] = {
                "key": f"{prefix}/shap_local_values.csv",
                "bytes": shap_local_values.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            }
        if shap_worst_cases is not None:
            artifacts["shap_worst_cases"] = {
                "key": f"{prefix}/shap_worst_cases.csv",
                "bytes": shap_worst_cases.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            }
        if br_trajectory is not None:
            artifacts["br_trajectory"] = {
                "key": f"{prefix}/br_trajectory.csv",
                "bytes": br_trajectory.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            }
        if br_parameter_samples is not None:
            artifacts["br_parameter_samples"] = {
                "key": f"{prefix}/br_parameter_samples.csv",
                "bytes": br_parameter_samples.to_csv(index=False).encode("utf-8"),
                "content_type": "text/csv",
            }
        if br_summary is not None:
            artifacts["br_summary"] = {
                "key": f"{prefix}/br_summary.json",
                "bytes": json.dumps(br_summary, ensure_ascii=False, indent=2).encode("utf-8"),
                "content_type": "application/json",
            }
        for figure_name, formats in (br_figures or {}).items():
            for extension, payload in formats.items():
                if extension not in {"png", "pdf"}:
                    raise ValueError(f"Unsupported bulletin BR figure extension: {extension}.")
                artifacts[f"{figure_name}_{extension}"] = {
                    "key": f"{prefix}/{figure_name}.{extension}",
                    "bytes": payload,
                    "content_type": "image/png" if extension == "png" else "application/pdf",
                }

        for artifact in artifacts.values():
            self._put_bytes(artifact["key"], artifact["bytes"], artifact["content_type"])

        return {
            "run_id": run_id,
            "storage_prefix": prefix,
            "download_access": "presigned_urls",
            "presigned_url_expiration_seconds": self.settings.presigned_expiration_seconds,
            "artifacts": {
                name: {
                    "s3_uri": f"s3://{self.settings.bucket_name}/{artifact['key']}",
                    "download_url": self._presigned_url(artifact["key"]),
                    "content_type": artifact["content_type"],
                }
                for name, artifact in artifacts.items()
            },
        }

