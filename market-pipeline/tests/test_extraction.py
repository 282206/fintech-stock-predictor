"""Tests for yfinance extraction (uses mocks — no network required)."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from market_pipeline.config import ExtractConfig
from market_pipeline.extract.yfinance_client import extract_ticker, extract_all_tickers


def _fake_download(*args, **kwargs):
    """Return a minimal OHLCV DataFrame mimicking yfinance output."""
    return pd.DataFrame({
        "Date":   pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "Open":   [150.0, 151.0],
        "High":   [155.0, 156.0],
        "Low":    [149.0, 150.0],
        "Close":  [153.0, 154.0],
        "Adj Close": [153.0, 154.0],
        "Volume": [1_000_000, 1_100_000],
    }).set_index("Date")


@patch("market_pipeline.extract.yfinance_client.yf.download", side_effect=_fake_download)
def test_extract_ticker_success(mock_dl):
    cfg = ExtractConfig(max_retries=1, backoff_base_seconds=0)
    result = extract_ticker("AAPL", date(2024, 1, 1), date(2024, 1, 5), "1d", cfg)
    assert result.success
    assert result.df is not None
    assert len(result.df) == 2
    assert "ticker" in result.df.columns
    assert result.df["ticker"].iloc[0] == "AAPL"


@patch("market_pipeline.extract.yfinance_client.yf.download", side_effect=Exception("network error"))
def test_extract_ticker_failure(mock_dl):
    cfg = ExtractConfig(max_retries=2, backoff_base_seconds=0)
    result = extract_ticker("BAD", date(2024, 1, 1), date(2024, 1, 5), "1d", cfg)
    assert not result.success
    assert result.df is None
    assert "network error" in result.error


@patch("market_pipeline.extract.yfinance_client.yf.download", side_effect=_fake_download)
def test_extract_all_partial_failure(mock_dl):
    """One bad ticker should not stop the others."""
    cfg = ExtractConfig(max_retries=1, backoff_base_seconds=0)

    def side_effect(ticker, **kwargs):
        if ticker == "FAIL":
            raise ValueError("bad ticker")
        return _fake_download()

    mock_dl.side_effect = side_effect
    results = extract_all_tickers(["AAPL", "FAIL", "MSFT"], 30, "1d", cfg)
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    assert len(successes) == 2
    assert len(failures) == 1
    assert failures[0].ticker == "FAIL"
