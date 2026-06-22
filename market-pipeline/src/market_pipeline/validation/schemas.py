"""Pandera schemas for data quality validation."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check


# Bronze-layer schema: minimal requirements on raw data
bronze_schema = DataFrameSchema(
    columns={
        "ticker": Column(str, nullable=False),
        "price_date": Column("datetime64[ns]", nullable=False, coerce=True),
        "open": Column(float, checks=Check.ge(0), nullable=True, coerce=True),
        "high": Column(float, checks=Check.ge(0), nullable=True, coerce=True),
        "low": Column(float, checks=Check.ge(0), nullable=True, coerce=True),
        "close": Column(float, checks=Check.ge(0), nullable=True, coerce=True),
        "volume": Column(float, checks=Check.ge(0), nullable=True, coerce=True),
    },
    checks=[
        Check(
            lambda df: (df["high"] >= df["low"]).all(),
            error="high must be >= low",
        ),
    ],
    coerce=True,
)

# Silver-layer schema: stricter — all price columns non-null
silver_schema = DataFrameSchema(
    columns={
        "ticker": Column(str, nullable=False),
        "price_date": Column("datetime64[ns]", nullable=False, coerce=True),
        "open": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "high": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "low": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "close": Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "adj_close": Column(float, checks=Check.ge(0), nullable=True, coerce=True),
        "volume": Column(float, checks=Check.ge(0), nullable=True, coerce=True),
    },
    checks=[
        Check(
            lambda df: (df["high"] >= df["low"]).all(),
            error="high must be >= low",
        ),
        Check(
            lambda df: (df["close"] >= df["low"]).all(),
            error="close must be >= low",
        ),
        Check(
            lambda df: (df["close"] <= df["high"]).all(),
            error="close must be <= high",
        ),
    ],
    coerce=True,
)


def validate_bronze(df) -> tuple[bool, str | None]:
    """Run bronze schema validation. Returns (passed, error_message)."""
    try:
        bronze_schema.validate(df, lazy=True)
        return True, None
    except pa.errors.SchemaErrors as exc:
        return False, str(exc.failure_cases.to_dict())


def validate_silver(df) -> tuple[bool, str | None]:
    """Run silver schema validation. Returns (passed, error_message)."""
    try:
        silver_schema.validate(df, lazy=True)
        return True, None
    except pa.errors.SchemaErrors as exc:
        return False, str(exc.failure_cases.to_dict())


def null_pct_check(df, col: str, max_pct: float) -> tuple[bool, float]:
    """Check that null percentage in a column stays below threshold."""
    pct = df[col].isna().mean()
    return pct <= max_pct, float(pct)
