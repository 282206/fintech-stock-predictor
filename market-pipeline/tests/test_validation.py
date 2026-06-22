"""Tests for pandera schema validation."""

import pandas as pd
import pytest

from market_pipeline.validation.schemas import validate_bronze, validate_silver, null_pct_check


def _good_df():
    return pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "price_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "open":   [150.0, 151.0],
        "high":   [155.0, 156.0],
        "low":    [149.0, 150.0],
        "close":  [153.0, 154.0],
        "adj_close": [153.0, 154.0],
        "volume": [1_000_000.0, 1_100_000.0],
    })


def test_bronze_valid():
    passed, err = validate_bronze(_good_df())
    assert passed, err


def test_bronze_negative_price_fails():
    df = _good_df()
    df.loc[0, "close"] = -1.0
    passed, _ = validate_bronze(df)
    assert not passed


def test_bronze_high_less_than_low_fails():
    df = _good_df()
    df.loc[0, "high"] = 140.0  # below low=149
    passed, _ = validate_bronze(df)
    assert not passed


def test_silver_valid():
    passed, err = validate_silver(_good_df())
    assert passed, err


def test_silver_null_close_fails():
    df = _good_df()
    df.loc[0, "close"] = None
    passed, _ = validate_silver(df)
    assert not passed


def test_null_pct_check_pass():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, None]})
    ok, pct = null_pct_check(df, "close", max_pct=0.3)
    assert ok
    assert abs(pct - 0.2) < 1e-9


def test_null_pct_check_fail():
    df = pd.DataFrame({"close": [None, None, 3.0]})
    ok, pct = null_pct_check(df, "close", max_pct=0.1)
    assert not ok
