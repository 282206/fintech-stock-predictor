"""Compute gold-layer metrics from cleaned OHLCV DataFrames."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("market_pipeline.metrics")


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a cleaned OHLCV DataFrame (one ticker), compute:
    - daily_return: percentage change in close price
    - rolling_7d_close: 7-day rolling mean of close
    - rolling_20d_close: 20-day rolling mean of close
    - rolling_20d_vol: 20-day rolling std of daily returns (annualised)
    """
    df = df.sort_values("price_date").copy()

    df["daily_return"] = df["close"].pct_change()
    df["rolling_7d_close"] = df["close"].rolling(window=7, min_periods=1).mean()
    df["rolling_20d_close"] = df["close"].rolling(window=20, min_periods=1).mean()
    # Annualised volatility: std of daily returns * sqrt(252)
    df["rolling_20d_vol"] = (
        df["daily_return"].rolling(window=20, min_periods=2).std() * (252 ** 0.5)
    )

    result = df[
        ["price_date", "daily_return", "rolling_7d_close", "rolling_20d_close", "rolling_20d_vol"]
    ].copy()
    logger.debug("Computed metrics for %d rows", len(result))
    return result


def compute_all_metrics(
    clean_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Run metric computation for every ticker."""
    metrics: dict[str, pd.DataFrame] = {}
    for ticker, df in clean_frames.items():
        try:
            metrics[ticker] = compute_daily_metrics(df)
        except Exception as exc:
            logger.error("Metric computation failed for %s: %s", ticker, exc)
    return metrics
