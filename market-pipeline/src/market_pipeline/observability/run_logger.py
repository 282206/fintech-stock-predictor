"""Log pipeline run metadata and data quality results to Postgres."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import psycopg

from market_pipeline.config import PostgresConfig
from market_pipeline.db.connection import get_connection

logger = logging.getLogger("market_pipeline.observability")


def new_run_id() -> str:
    return str(uuid.uuid4())


def start_run(
    conn: psycopg.Connection,
    run_id: str,
    pipeline_name: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO market.pipeline_run_log
                (run_id, pipeline_name, status, started_at)
            VALUES (%s, %s, 'running', NOW())
            ON CONFLICT (run_id) DO NOTHING
            """,
            (run_id, pipeline_name),
        )
    conn.commit()


def finish_run(
    conn: psycopg.Connection,
    run_id: str,
    status: str,
    rows_extracted: int = 0,
    rows_loaded: int = 0,
    rows_failed: int = 0,
    error_message: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE market.pipeline_run_log SET
                status           = %s,
                completed_at     = NOW(),
                duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at)),
                rows_extracted   = %s,
                rows_loaded      = %s,
                rows_failed      = %s,
                error_message    = %s
            WHERE run_id = %s
            """,
            (status, rows_extracted, rows_loaded, rows_failed, error_message, run_id),
        )
    conn.commit()


def log_dq_check(
    conn: psycopg.Connection,
    run_id: str,
    check_name: str,
    table_name: str,
    status: str,
    details: dict | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO market.data_quality_check_log
                (run_id, check_name, table_name, status, details)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, check_name, table_name, status, json.dumps(details or {})),
        )
    conn.commit()
    logger.info("DQ check [%s] %s: %s", check_name, table_name, status)


class PipelineRunContext:
    """Context manager that auto-logs run start/finish to Postgres."""

    def __init__(self, cfg: PostgresConfig, pipeline_name: str) -> None:
        self.cfg = cfg
        self.pipeline_name = pipeline_name
        self.run_id = new_run_id()
        self._conn: psycopg.Connection | None = None
        self.rows_extracted = 0
        self.rows_loaded = 0
        self.rows_failed = 0

    def __enter__(self) -> "PipelineRunContext":
        self._conn = get_connection(self.cfg)
        start_run(self._conn, self.run_id, self.pipeline_name)
        logger.info("Pipeline run started: %s", self.run_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        status = "failed" if exc_type else "success"
        error_msg = str(exc_val) if exc_val else None
        if self._conn:
            finish_run(
                self._conn,
                self.run_id,
                status,
                self.rows_extracted,
                self.rows_loaded,
                self.rows_failed,
                error_msg,
            )
            self._conn.close()
        logger.info("Pipeline run finished: %s [%s]", self.run_id, status)
        return False  # don't suppress exceptions

    def log_dq(self, check_name: str, table_name: str, status: str, details: dict | None = None) -> None:
        if self._conn:
            log_dq_check(self._conn, self.run_id, check_name, table_name, status, details)
