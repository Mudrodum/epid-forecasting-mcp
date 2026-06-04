"""S3-compatible object storage for exported forecasting artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3


class S3ConfigurationError(RuntimeError):
    """Raised when the S3 artifact storage configuration is incomplete."""


@dataclass(frozen=True)
class S3StorageConfig:
    """Configuration required to publish artifacts to S3-compatible storage."""

    endpoint_url: str
    bucket_name: str
    access_key: str
    secret_key: str

    @classmethod
    def from_environment(cls) -> "S3StorageConfig":
        names = {
            "endpoint_url": "S3_ENDPOINT_URL",
            "bucket_name": "S3_BUCKET_NAME",
            "access_key": "S3_ACCESS_KEY",
            "secret_key": "S3_SECRET_KEY",
        }
        values = {field: os.getenv(env_name, "").strip() for field, env_name in names.items()}
        missing = [env_name for field, env_name in names.items() if not values[field]]
        if missing:
            raise S3ConfigurationError(
                "S3 export requires configured environment variables: " + ", ".join(missing)
            )
        return cls(**values)


class S3ArtifactStorage:
    """Upload materialized export files and return stable S3 URIs."""

    def __init__(self, config: S3StorageConfig, *, client: Any | None = None) -> None:
        self.config = config
        self._client = client or boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
        )

    @classmethod
    def from_environment(cls) -> "S3ArtifactStorage":
        return cls(S3StorageConfig.from_environment())

    def upload_file(self, file_path: Path, object_key: str) -> str:
        """Upload a local binary file to the configured bucket and return its S3 URI."""
        with file_path.open("rb") as file_obj:
            self._client.upload_fileobj(file_obj, self.config.bucket_name, object_key)
        return f"s3://{self.config.bucket_name}/{object_key}"
