"""Load cleaned DataFrames into Postgres using explicit upsert SQL."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from market_pipeline.config import PostgresConfig
from market_pipeline.db.connection import get_connection

logger = logging.getLogger("market_pipeline.db.load")


def upsert_instrument(conn: psycopg.Connection, ticker: str) -> int:
    """Insert instrument if missing; return its instrument_id."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO market.dim_instrument (ticker, updated_at)
            VALUES (%s, NOW())
            ON CONFLICT (ticker) DO UPDATE SET updated_at = NOW()
            RETURNING instrument_id
            """,
            (ticker,),
        )
        row = cur.fetchone()
    return row["instrument_id"]


def upsert_ohlcv_rows(
    conn: psycopg.Connection,
    instrument_id: int,
    df: pd.DataFrame,
    interval: str,
) -> int:
    """Bulk upsert OHLCV rows for one instrument. Returns rows inserted/updated."""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                instrument_id,
                r["price_date"].date() if hasattr(r["price_date"], "date") else r["price_date"],
                float(r["open"]) if pd.notna(r.get("open")) else None,
                float(r["high"]) if pd.notna(r.get("high")) else None,
                float(r["low"]) if pd.notna(r.get("low")) else None,
                float(r["close"]) if pd.notna(r.get("close")) else None,
                float(r["adj_close"]) if pd.notna(r.get("adj_close")) else None,
                int(r["volume"]) if pd.notna(r.get("volume")) else None,
                interval,
            )
        )

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO market.fact_ohlcv
                (instrument_id, price_date, open, high, low, close,
                 adj_close, volume, interval, ingested_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (instrument_id, price_date, interval)
            DO UPDATE SET
                open       = EXCLUDED.open,
                high       = EXCLUDED.high,
                low        = EXCLUDED.low,
                close      = EXCLUDED.close,
                adj_close  = EXCLUDED.adj_close,
                volume     = EXCLUDED.volume,
                ingested_at = NOW()
            """,
            rows,
        )
    return len(rows)


def load_silver_to_postgres(
    cfg: PostgresConfig,
    clean_frames: dict[str, pd.DataFrame],
    interval: str,
) -> dict[str, int]:
    """Load all cleaned DataFrames into Postgres. Returns {ticker: rows_loaded}."""
    counts: dict[str, int] = {}
    with get_connection(cfg) as conn:
        for ticker, df in clean_frames.items():
            try:
                instrument_id = upsert_instrument(conn, ticker)
                n = upsert_ohlcv_rows(conn, instrument_id, df, interval)
                conn.commit()
                counts[ticker] = n
                logger.info("Loaded %d rows for %s (id=%d)", n, ticker, instrument_id)
            except Exception as exc:
                conn.rollback()
                logger.error("Failed to load %s: %s", ticker, exc)
                counts[ticker] = 0
    return counts


def upsert_daily_metrics_rows(
    conn: psycopg.Connection,
    instrument_id: int,
    df: pd.DataFrame,
) -> int:
    """Upsert computed daily metrics for one instrument."""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                instrument_id,
                r["price_date"].date() if hasattr(r["price_date"], "date") else r["price_date"],
                float(r["daily_return"]) if pd.notna(r.get("daily_return")) else None,
                float(r["rolling_7d_close"]) if pd.notna(r.get("rolling_7d_close")) else None,
                float(r["rolling_20d_close"]) if pd.notna(r.get("rolling_20d_close")) else None,
                float(r["rolling_20d_vol"]) if pd.notna(r.get("rolling_20d_vol")) else None,
            )
        )

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO market.fact_daily_metrics
                (instrument_id, price_date, daily_return,
                 rolling_7d_close, rolling_20d_close, rolling_20d_vol)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (instrument_id, price_date)
            DO UPDATE SET
                daily_return      = EXCLUDED.daily_return,
                rolling_7d_close  = EXCLUDED.rolling_7d_close,
                rolling_20d_close = EXCLUDED.rolling_20d_close,
                rolling_20d_vol   = EXCLUDED.rolling_20d_vol,
                created_at        = NOW()
            """,
            rows,
        )
    return len(rows)
