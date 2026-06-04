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
