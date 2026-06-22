"""Pipeline configuration loaded from environment variables and YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExtractConfig:
    max_retries: int = 3
    backoff_base_seconds: float = 2.0


@dataclass
class QualityConfig:
    max_null_pct: float = 0.05
    min_rows_per_ticker: int = 1


@dataclass
class PipelineConfig:
    tickers: list[str] = field(default_factory=list)
    lookback_days: int = 90
    interval: str = "1d"
    bucket: str = "market-data"
    bronze_prefix: str = "bronze/ohlcv"
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)


@dataclass
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    db: str = "market_data"
    user: str = "market_user"
    password: str = "market_pass"

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@dataclass
class MinioConfig:
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "market-data"
    secure: bool = False


def load_pipeline_config(path: str | None = None) -> PipelineConfig:
    """Load pipeline.yml and return a PipelineConfig."""
    config_path = path or os.getenv("PIPELINE_CONFIG_PATH", "config/pipeline.yml")
    raw: dict = {}
    p = Path(config_path)
    if p.exists():
        with p.open() as fh:
            raw = yaml.safe_load(fh) or {}

    extract_raw = raw.get("extract", {})
    quality_raw = raw.get("quality", {})

    return PipelineConfig(
        tickers=raw.get("tickers", ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "SPY", "QQQ"]),
        lookback_days=int(raw.get("lookback_days", 90)),
        interval=raw.get("interval", "1d"),
        bucket=raw.get("bucket", "market-data"),
        bronze_prefix=raw.get("bronze_prefix", "bronze/ohlcv"),
        extract=ExtractConfig(
            max_retries=int(extract_raw.get("max_retries", 3)),
            backoff_base_seconds=float(extract_raw.get("backoff_base_seconds", 2.0)),
        ),
        quality=QualityConfig(
            max_null_pct=float(quality_raw.get("max_null_pct", 0.05)),
            min_rows_per_ticker=int(quality_raw.get("min_rows_per_ticker", 1)),
        ),
    )


def load_postgres_config() -> PostgresConfig:
    return PostgresConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        db=os.getenv("POSTGRES_DB", "market_data"),
        user=os.getenv("POSTGRES_USER", "market_user"),
        password=os.getenv("POSTGRES_PASSWORD", "market_pass"),
    )


def load_minio_config() -> MinioConfig:
    return MinioConfig(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        bucket=os.getenv("MINIO_BUCKET", "market-data"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )
