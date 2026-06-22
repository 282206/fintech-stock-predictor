"""Tests for gold-layer metric computation."""

import pandas as pd
import pytest

from market_pipeline.metrics.daily_metrics import compute_daily_metrics


def _ohlcv(closes):
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({
        "price_date": dates,
        "open":  closes,
        "high":  [c + 1 for c in closes],
        "low":   [c - 1 for c in closes],
        "close": closes,
        "adj_close": closes,
        "volume": [1_000_000] * len(closes),
    })


def test_daily_return_columns_present():
    df = _ohlcv([100, 101, 102, 103, 104])
    result = compute_daily_metrics(df)
    assert set(result.columns) >= {"price_date", "daily_return", "rolling_7d_close", "rolling_20d_close", "rolling_20d_vol"}


def test_daily_return_values():
    df = _ohlcv([100.0, 110.0, 121.0])
    result = compute_daily_metrics(df)
    # Second row: (110 - 100) / 100 = 0.1
    assert abs(result.iloc[1]["daily_return"] - 0.1) < 1e-9
    # Third row: (121 - 110) / 110 ≈ 0.1
    assert abs(result.iloc[2]["daily_return"] - 0.1) < 1e-9


def test_rolling_7d_close():
    closes = list(range(1, 21))  # 1..20
    df = _ohlcv(closes)
    result = compute_daily_metrics(df)
    # With min_periods=1, first value should equal close[0]
    assert result.iloc[0]["rolling_7d_close"] == 1.0
    # 8th row: mean of 2..8 = 5.0
    assert abs(result.iloc[7]["rolling_7d_close"] - 5.0) < 1e-9


def test_volatility_nan_for_single_row():
    df = _ohlcv([100.0])
    result = compute_daily_metrics(df)
    # Only 1 row → rolling_20d_vol must be NaN
    assert pd.isna(result.iloc[0]["rolling_20d_vol"])


def test_compute_all_metrics_skips_bad_ticker():
    from market_pipeline.metrics.daily_metrics import compute_all_metrics
    frames = {
        "AAPL": _ohlcv([100, 101, 102]),
        "BAD": pd.DataFrame(),  # empty → should not crash compute_all
    }
    results = compute_all_metrics(frames)
    assert "AAPL" in results
    # BAD may or may not be in results depending on error handling, but should not raise
