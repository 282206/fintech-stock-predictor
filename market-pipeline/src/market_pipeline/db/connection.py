"""Database connection factory using psycopg."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row

from market_pipeline.config import PostgresConfig

logger = logging.getLogger("market_pipeline.db")


def get_connection(cfg: PostgresConfig) -> psycopg.Connection:
    return psycopg.connect(cfg.dsn, row_factory=dict_row)


@contextmanager
def transaction(cfg: PostgresConfig) -> Generator[psycopg.Cursor, None, None]:
    """Context manager that yields a cursor inside a committed transaction."""
    with get_connection(cfg) as conn:
        with conn.cursor() as cur:
            yield cur
        conn.commit()


def init_schema(cfg: PostgresConfig, sql_dir: str = "sql") -> None:
    """Run all SQL init scripts in order."""
    import os
    from pathlib import Path

    scripts = sorted(Path(sql_dir).glob("*.sql"))
    with get_connection(cfg) as conn:
        for script in scripts:
            logger.info("Running %s", script.name)
            conn.execute(script.read_text())
        conn.commit()
    logger.info("Schema initialised from %d scripts", len(scripts))
