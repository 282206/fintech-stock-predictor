"""Dagster schedule definitions — daily pipeline run."""

from dagster import ScheduleDefinition
from orchestration.dagster_project.jobs import market_pipeline_job

daily_schedule = ScheduleDefinition(
    job=market_pipeline_job,
    cron_schedule="0 6 * * *",   # 06:00 UTC every day
    name="daily_market_pipeline",
    description="Ingest OHLCV data daily at 06:00 UTC",
)
