# s3_service.py
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config


class S3BucketService:
    """Работа с S3 хранилищем для прогнозов гриппа"""
    
    def __init__(
            self,
            endpoint: str,
            access_key: str,
            secret_key: str,
            bucket_name: str = "forecasts",
    ) -> None:
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key

    def create_s3_client(self) -> boto3.client:
        """Создает клиент для работы с S3"""
        client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
        )
        return client
    
    def upload_bytes(self, prefix: str, source_file_name: str, data: bytes) -> str:
        """Загружает байты в S3"""
        client = self.create_s3_client()
        destination_path = (Path(prefix, source_file_name)).as_posix()
        buffer = BytesIO(data)
        client.upload_fileobj(buffer, self.bucket_name, destination_path)
        return destination_path
    
    def upload_file(self, prefix: str, source_file_name: str, local_path: str) -> str:
        """Загружает файл из локальной файловой системы в S3"""
        client = self.create_s3_client()
        destination_path = (Path(prefix, source_file_name)).as_posix()
        client.upload_file(local_path, self.bucket_name, destination_path)
        return destination_path
    
    def generate_presigned_url(self, s3_key: str, method: str = 'get_object', expiration: int = 3600) -> str:
        """Генерирует временную ссылку на файл в S3"""
        client = self.create_s3_client()
        return client.generate_presigned_url(
            method,
            Params={'Bucket': self.bucket_name, 'Key': s3_key},
            ExpiresIn=expiration
        )
    
    def list_objects(self, prefix: str) -> list[str]:
        """Список всех объектов в S3 с заданным префиксом"""
        client = self.create_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
        
        storage_content: list[str] = []
        for page in page_iterator:
            contents = page.get("Contents", [])
            for item in contents:
                storage_content.append(item["Key"])
        
        return storage_content