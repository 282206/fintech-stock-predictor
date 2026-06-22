"""Dagster Definitions — entry point for the pipeline."""

from dagster import Definitions

from orchestration.dagster_project.assets import (
    bronze_ohlcv,
    silver_ohlcv,
    gold_metrics,
    pipeline_run_summary,
)
from orchestration.dagster_project.jobs import market_pipeline_job
from orchestration.dagster_project.schedules import daily_schedule

defs = Definitions(
    assets=[bronze_ohlcv, silver_ohlcv, gold_metrics, pipeline_run_summary],
    jobs=[market_pipeline_job],
    schedules=[daily_schedule],
)
