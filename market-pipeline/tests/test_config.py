"""Tests for config loading."""

import os
from pathlib import Path
import pytest

from market_pipeline.config import (
    load_pipeline_config,
    load_postgres_config,
    load_minio_config,
)


def test_load_pipeline_config_defaults(tmp_path):
    """Without a config file we should get sensible defaults."""
    cfg = load_pipeline_config(path=str(tmp_path / "nonexistent.yml"))
    assert isinstance(cfg.tickers, list)
    assert len(cfg.tickers) > 0
    assert cfg.lookback_days > 0
    assert cfg.interval == "1d"


def test_load_pipeline_config_from_file(tmp_path):
    yml = tmp_path / "pipeline.yml"
    yml.write_text(
        "tickers:\n  - AAPL\n  - GOOG\nlookback_days: 30\ninterval: '1h'\n"
    )
    cfg = load_pipeline_config(path=str(yml))
    assert cfg.tickers == ["AAPL", "GOOG"]
    assert cfg.lookback_days == 30
    assert cfg.interval == "1h"


def test_postgres_config_from_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "myhost")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("POSTGRES_USER", "user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pass")
    cfg = load_postgres_config()
    assert cfg.host == "myhost"
    assert cfg.port == 5433
    assert "testdb" in cfg.dsn


def test_minio_config_from_env(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "ak")
    monkeypatch.setenv("MINIO_SECRET_KEY", "sk")
    cfg = load_minio_config()
    assert cfg.endpoint == "minio:9000"
    assert cfg.secure is False
