"""Read-side queries for the FastAPI serving layer."""

from __future__ import annotations

from datetime import date

import psycopg
from psycopg.rows import dict_row


def get_instruments(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM market.dim_instrument WHERE is_active ORDER BY ticker"
        )
        return cur.fetchall()


def get_prices(
    conn: psycopg.Connection,
    ticker: str,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 500,
) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT f.price_date, f.open, f.high, f.low, f.close,
                   f.adj_close, f.volume, f.interval, f.ingested_at
            FROM market.fact_ohlcv f
            JOIN market.dim_instrument d USING (instrument_id)
            WHERE d.ticker = %s
              AND (%s IS NULL OR f.price_date >= %s)
              AND (%s IS NULL OR f.price_date <= %s)
            ORDER BY f.price_date DESC
            LIMIT %s
            """,
            (ticker, start_date, start_date, end_date, end_date, limit),
        )
        return cur.fetchall()


def get_metrics(
    conn: psycopg.Connection,
    ticker: str,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 500,
) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT m.price_date, m.daily_return, m.rolling_7d_close,
                   m.rolling_20d_close, m.rolling_20d_vol
            FROM market.fact_daily_metrics m
            JOIN market.dim_instrument d USING (instrument_id)
            WHERE d.ticker = %s
              AND (%s IS NULL OR m.price_date >= %s)
              AND (%s IS NULL OR m.price_date <= %s)
            ORDER BY m.price_date DESC
            LIMIT %s
            """,
            (ticker, start_date, start_date, end_date, end_date, limit),
        )
        return cur.fetchall()


def get_pipeline_runs(conn: psycopg.Connection, limit: int = 50) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM market.pipeline_run_log ORDER BY started_at DESC LIMIT %s",
            (limit,),
        )
        return cur.fetchall()


def get_dq_checks(
    conn: psycopg.Connection,
    run_id: str | None = None,
    limit: int = 200,
) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT * FROM market.data_quality_check_log
            WHERE (%s IS NULL OR run_id = %s)
            ORDER BY checked_at DESC
            LIMIT %s
            """,
            (run_id, run_id, limit),
        )
        return cur.fetchall()
