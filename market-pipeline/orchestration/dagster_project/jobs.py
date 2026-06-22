"""Dagster job definitions."""

from dagster import define_asset_job, AssetSelection

market_pipeline_job = define_asset_job(
    name="market_pipeline_job",
    selection=AssetSelection.all(),
    description="Full medallion pipeline: extract → validate → silver → gold → observability",
)
