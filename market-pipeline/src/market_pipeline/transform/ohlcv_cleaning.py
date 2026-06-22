"""Bronze → Silver transformation: normalise, cast, deduplicate, validate."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("market_pipeline.transform")

# Map known yfinance column name variations to canonical names
_COLUMN_MAP = {
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "date": "price_date",
    "datetime": "price_date",
}


def clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean a raw bronze DataFrame for one ticker.

    Steps:
    1. Normalise column names.
    2. Ensure required columns exist.
    3. Type-cast date and numeric columns.
    4. Drop fully null rows and duplicates.
    5. Handle out-of-range values (negatives → NaN).
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=_COLUMN_MAP)

    # Add ticker if not present
    if "ticker" not in df.columns:
        df["ticker"] = ticker

    # Ensure price_date
    if "price_date" not in df.columns:
        raise ValueError(f"No price_date column in data for {ticker}")

    df["price_date"] = pd.to_datetime(df["price_date"], utc=False, errors="coerce")
    df = df.dropna(subset=["price_date"])

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Negative prices are invalid
            if col != "volume":
                df.loc[df[col] < 0, col] = None

    # Ensure adj_close column exists
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Drop duplicate (ticker, price_date) rows, keep last
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "price_date"], keep="last")
    if before != len(df):
        logger.debug("Dropped %d duplicate rows for %s", before - len(df), ticker)

    # Drop rows where all price columns are null
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    df = df.dropna(subset=price_cols, how="all")

    df = df.sort_values("price_date").reset_index(drop=True)
    logger.info("Cleaned %d rows for %s", len(df), ticker)
    return df


def clean_all(raw_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean a mapping of ticker → raw DataFrame."""
    cleaned: dict[str, pd.DataFrame] = {}
    for ticker, df in raw_frames.items():
        try:
            cleaned[ticker] = clean_ohlcv(df, ticker)
        except Exception as exc:
            logger.error("Failed to clean %s: %s", ticker, exc)
    return cleaned
