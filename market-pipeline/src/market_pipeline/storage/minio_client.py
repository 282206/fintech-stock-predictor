"""MinIO (S3-compatible) client for reading/writing Parquet to the bronze layer."""

from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
from minio import Minio
from minio.error import S3Error

from market_pipeline.config import MinioConfig

logger = logging.getLogger("market_pipeline.storage")


def get_minio_client(cfg: MinioConfig) -> Minio:
    return Minio(
        cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("Created bucket: %s", bucket)


def bronze_object_key(prefix: str, run_date: date, ticker: str) -> str:
    """Return the MinIO object key for a bronze parquet file."""
    return f"{prefix}/run_date={run_date.isoformat()}/ticker={ticker}/data.parquet"


def write_parquet(
    client: Minio,
    bucket: str,
    object_key: str,
    df: pd.DataFrame,
) -> None:
    """Serialise DataFrame to Parquet and upload to MinIO."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    size = buf.getbuffer().nbytes
    client.put_object(
        bucket,
        object_key,
        data=buf,
        length=size,
        content_type="application/octet-stream",
    )
    logger.info("Uploaded %s (%d bytes)", object_key, size)


def read_parquet(
    client: Minio,
    bucket: str,
    object_key: str,
) -> pd.DataFrame:
    """Download a Parquet object from MinIO and return as DataFrame."""
    try:
        response = client.get_object(bucket, object_key)
        buf = io.BytesIO(response.read())
        return pd.read_parquet(buf, engine="pyarrow")
    except S3Error as exc:
        logger.error("Failed to read %s: %s", object_key, exc)
        raise


def list_bronze_keys(
    client: Minio,
    bucket: str,
    prefix: str,
    run_date: date,
) -> list[str]:
    """List all bronze object keys for a given run date."""
    date_prefix = f"{prefix}/run_date={run_date.isoformat()}/"
    objects = client.list_objects(bucket, prefix=date_prefix, recursive=True)
    return [obj.object_name for obj in objects]
