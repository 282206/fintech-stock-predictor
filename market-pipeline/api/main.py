"""FastAPI serving layer — reads only from Postgres, never calls yfinance."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Depends
import psycopg
from psycopg.rows import dict_row

from market_pipeline.config import load_postgres_config
from market_pipeline.db.queries import (
    get_instruments,
    get_prices,
    get_metrics,
    get_pipeline_runs,
    get_dq_checks,
)

app = FastAPI(
    title="Market Data Pipeline API",
    description="Serves final OHLCV data and pipeline observability from Postgres.",
    version="1.0.0",
)

_pg_cfg = load_postgres_config()


def get_db() -> psycopg.Connection:
    conn = psycopg.connect(_pg_cfg.dsn, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    try:
        conn = psycopg.connect(_pg_cfg.dsn, row_factory=dict_row)
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "ok", "db": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unreachable: {exc}")


@app.get("/instruments")
def instruments(db: psycopg.Connection = Depends(get_db)) -> list[dict[str, Any]]:
    """List all active instruments."""
    return get_instruments(db)


@app.get("/prices/{ticker}")
def prices(
    ticker: str,
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    db: psycopg.Connection = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return OHLCV price history for a ticker."""
    rows = get_prices(db, ticker.upper(), start_date, end_date, limit)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No price data found for {ticker.upper()}")
    return rows


@app.get("/metrics/{ticker}")
def metrics(
    ticker: str,
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    db: psycopg.Connection = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return computed daily metrics for a ticker."""
    rows = get_metrics(db, ticker.upper(), start_date, end_date, limit)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No metrics found for {ticker.upper()}")
    return rows


@app.get("/pipeline-runs")
def pipeline_runs(
    limit: int = Query(default=50, ge=1, le=500),
    db: psycopg.Connection = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return recent pipeline run logs."""
    return get_pipeline_runs(db, limit)


@app.get("/data-quality")
def data_quality(
    run_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    db: psycopg.Connection = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return data quality check logs, optionally filtered by run_id."""
    return get_dq_checks(db, run_id, limit)
