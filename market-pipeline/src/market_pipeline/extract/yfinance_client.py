"""yfinance extraction with retries and partial-failure handling."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from market_pipeline.config import ExtractConfig

logger = logging.getLogger("market_pipeline.extract")


@dataclass
class TickerResult:
    ticker: str
    df: pd.DataFrame | None
    success: bool
    error: str | None = None


def _fetch_one(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """Download OHLCV for a single ticker via yfinance."""
    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw = raw.reset_index()
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    # Normalise the date column name (yfinance may return "Date" or "Datetime")
    for candidate in ("date", "datetime"):
        if candidate in raw.columns:
            raw = raw.rename(columns={candidate: "price_date"})
            break

    raw["ticker"] = ticker
    raw["interval"] = interval
    return raw


def extract_ticker(
    ticker: str,
    start: date,
    end: date,
    interval: str,
    cfg: ExtractConfig,
) -> TickerResult:
    """Fetch one ticker with exponential-backoff retries."""
    last_exc: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            df = _fetch_one(ticker, start, end, interval)
            logger.info("Extracted %d rows for %s", len(df), ticker)
            return TickerResult(ticker=ticker, df=df, success=True)
        except Exception as exc:
            last_exc = exc
            wait = cfg.backoff_base_seconds * (2 ** attempt)
            logger.warning(
                "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                attempt + 1,
                cfg.max_retries,
                ticker,
                exc,
                wait,
            )
            time.sleep(wait)

    logger.error("All retries exhausted for %s: %s", ticker, last_exc)
    return TickerResult(ticker=ticker, df=None, success=False, error=str(last_exc))


def extract_all_tickers(
    tickers: list[str],
    lookback_days: int,
    interval: str,
    cfg: ExtractConfig,
) -> list[TickerResult]:
    """Extract OHLCV for every ticker; continue even if some fail."""
    end = date.today()
    start = end - timedelta(days=lookback_days)
    results: list[TickerResult] = []
    for ticker in tickers:
        result = extract_ticker(ticker, start, end, interval, cfg)
        results.append(result)
    return results
