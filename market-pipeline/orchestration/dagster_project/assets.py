"""Dagster software-defined assets for the market data pipeline."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from dagster import (
    AssetExecutionContext,
    asset,
    RetryPolicy,
    Backoff,
)

from market_pipeline.config import (
    load_pipeline_config,
    load_postgres_config,
    load_minio_config,
)
from market_pipeline.extract.yfinance_client import extract_all_tickers
from market_pipeline.storage.minio_client import (
    get_minio_client,
    ensure_bucket,
    bronze_object_key,
    write_parquet,
    read_parquet,
    list_bronze_keys,
)
from market_pipeline.transform.ohlcv_cleaning import clean_ohlcv
from market_pipeline.validation.schemas import validate_bronze, validate_silver, null_pct_check
from market_pipeline.db.load import load_silver_to_postgres, upsert_daily_metrics_rows
from market_pipeline.db.connection import get_connection
from market_pipeline.metrics.daily_metrics import compute_all_metrics
from market_pipeline.observability.run_logger import PipelineRunContext, log_dq_check

logger = logging.getLogger("market_pipeline.assets")


@asset(
    group_name="bronze",
    retry_policy=RetryPolicy(max_retries=2, delay=30, backoff=Backoff.EXPONENTIAL),
)
def bronze_ohlcv(context: AssetExecutionContext) -> dict:
    """
    Extract OHLCV data from yfinance and store raw Parquet files in MinIO.
    Returns metadata about the extraction.
    """
    pipeline_cfg = load_pipeline_config()
    minio_cfg = load_minio_config()
    pg_cfg = load_postgres_config()

    run_date = date.today()
    client = get_minio_client(minio_cfg)
    ensure_bucket(client, minio_cfg.bucket)

    results = extract_all_tickers(
        pipeline_cfg.tickers,
        pipeline_cfg.lookback_days,
        pipeline_cfg.interval,
        pipeline_cfg.extract,
    )

    succeeded, failed = [], []
    rows_extracted = 0

    for result in results:
        if not result.success or result.df is None:
            failed.append(result.ticker)
            context.log.warning("Extraction failed for %s: %s", result.ticker, result.error)
            continue

        # Tag with metadata before storing
        df = result.df.copy()
        df["run_date"] = run_date.isoformat()
        df["source"] = "yfinance"

        key = bronze_object_key(pipeline_cfg.bronze_prefix, run_date, result.ticker)
        write_parquet(client, minio_cfg.bucket, key, df)
        succeeded.append(result.ticker)
        rows_extracted += len(df)
        context.log.info("Bronze: stored %d rows for %s at %s", len(df), result.ticker, key)

    context.add_output_metadata({
        "run_date": run_date.isoformat(),
        "tickers_succeeded": len(succeeded),
        "tickers_failed": len(failed),
        "rows_extracted": rows_extracted,
        "failed_tickers": failed,
    })

    return {
        "run_date": run_date,
        "succeeded": succeeded,
        "failed": failed,
        "rows_extracted": rows_extracted,
    }


@asset(
    group_name="silver",
    deps=[bronze_ohlcv],
)
def silver_ohlcv(context: AssetExecutionContext, bronze_ohlcv: dict) -> dict:
    """
    Read bronze Parquet files from MinIO, clean and validate,
    then load into Postgres fact_ohlcv table.
    """
    pipeline_cfg = load_pipeline_config()
    minio_cfg = load_minio_config()
    pg_cfg = load_postgres_config()

    run_date: date = bronze_ohlcv["run_date"]
    succeeded_tickers: list[str] = bronze_ohlcv["succeeded"]

    client = get_minio_client(minio_cfg)

    clean_frames: dict[str, pd.DataFrame] = {}
    dq_results: list[dict] = []

    for ticker in succeeded_tickers:
        key = bronze_object_key(pipeline_cfg.bronze_prefix, run_date, ticker)
        try:
            raw_df = read_parquet(client, minio_cfg.bucket, key)
        except Exception as exc:
            context.log.error("Could not read bronze for %s: %s", ticker, exc)
            continue

        # Bronze validation
        passed, err = validate_bronze(raw_df)
        dq_results.append({
            "ticker": ticker,
            "check": "bronze_schema",
            "passed": passed,
            "detail": err,
        })
        if not passed:
            context.log.warning("Bronze schema check failed for %s: %s", ticker, err)

        # Null pct check on close column
        if "close" in raw_df.columns:
            null_ok, null_pct = null_pct_check(raw_df, "close", pipeline_cfg.quality.max_null_pct)
            dq_results.append({
                "ticker": ticker,
                "check": "close_null_pct",
                "passed": null_ok,
                "detail": f"null_pct={null_pct:.3f}",
            })

        try:
            cleaned = clean_ohlcv(raw_df, ticker)
        except Exception as exc:
            context.log.error("Clean failed for %s: %s", ticker, exc)
            continue

        # Silver validation
        passed_silver, err_silver = validate_silver(cleaned)
        dq_results.append({
            "ticker": ticker,
            "check": "silver_schema",
            "passed": passed_silver,
            "detail": err_silver,
        })

        if len(cleaned) < pipeline_cfg.quality.min_rows_per_ticker:
            context.log.warning("Too few rows for %s after cleaning: %d", ticker, len(cleaned))
        else:
            clean_frames[ticker] = cleaned

    # Load to Postgres
    counts = load_silver_to_postgres(pg_cfg, clean_frames, pipeline_cfg.interval)
    rows_loaded = sum(counts.values())

    context.add_output_metadata({
        "tickers_cleaned": len(clean_frames),
        "rows_loaded": rows_loaded,
        "dq_checks": len(dq_results),
        "dq_failures": sum(1 for r in dq_results if not r["passed"]),
    })

    return {
        "clean_frames": clean_frames,
        "rows_loaded": rows_loaded,
        "dq_results": dq_results,
    }


@asset(
    group_name="gold",
    deps=[silver_ohlcv],
)
def gold_metrics(context: AssetExecutionContext, silver_ohlcv: dict) -> dict:
    """
    Compute daily metrics (returns, rolling averages, volatility)
    and load into fact_daily_metrics.
    """
    pg_cfg = load_postgres_config()
    clean_frames: dict[str, pd.DataFrame] = silver_ohlcv["clean_frames"]

    metrics_frames = compute_all_metrics(clean_frames)
    total_rows = 0

    with get_connection(pg_cfg) as conn:
        for ticker, mdf in metrics_frames.items():
            # look up instrument_id
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT instrument_id FROM market.dim_instrument WHERE ticker = %s",
                    (ticker,),
                )
                row = cur.fetchone()
            if not row:
                context.log.warning("No instrument_id found for %s — skipping metrics", ticker)
                continue
            instrument_id = row["instrument_id"]
            n = upsert_daily_metrics_rows(conn, instrument_id, mdf)
            conn.commit()
            total_rows += n
            context.log.info("Gold metrics: %d rows for %s", n, ticker)

    context.add_output_metadata({"total_metric_rows": total_rows, "tickers": list(metrics_frames)})
    return {"rows": total_rows}


@asset(
    group_name="observability",
    deps=[bronze_ohlcv, silver_ohlcv, gold_metrics],
)
def pipeline_run_summary(
    context: AssetExecutionContext,
    bronze_ohlcv: dict,
    silver_ohlcv: dict,
    gold_metrics: dict,
) -> None:
    """
    Write final pipeline run log and all DQ results to Postgres.
    """
    import uuid, json
    from market_pipeline.observability.run_logger import start_run, finish_run, log_dq_check
    pg_cfg = load_postgres_config()

    run_id = str(uuid.uuid4())
    with get_connection(pg_cfg) as conn:
        start_run(conn, run_id, "market_ohlcv_pipeline")

        for dq in silver_ohlcv.get("dq_results", []):
            status = "passed" if dq["passed"] else "failed"
            log_dq_check(
                conn,
                run_id,
                check_name=f"{dq['ticker']}_{dq['check']}",
                table_name="fact_ohlcv",
                status=status,
                details={"detail": dq.get("detail"), "ticker": dq["ticker"]},
            )

        finish_run(
            conn,
            run_id,
            status="success",
            rows_extracted=bronze_ohlcv.get("rows_extracted", 0),
            rows_loaded=silver_ohlcv.get("rows_loaded", 0),
            rows_failed=len(bronze_ohlcv.get("failed", [])),
        )

    context.log.info("Pipeline run logged: %s", run_id)
