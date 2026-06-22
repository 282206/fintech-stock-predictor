# Market Data Engineering Pipeline

An end-to-end, containerised data engineering pipeline that ingests live OHLCV
financial market data, processes it through a **medallion architecture**
(Bronze → Silver → Gold), and serves the results via a FastAPI REST API.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Docker Compose Network                           │
│                                                                         │
│  ┌──────────┐    ┌─────────────────────────────────────────────────┐   │
│  │ yfinance │───▶│             Dagster Pipeline                    │   │
│  │  (API)   │    │                                                 │   │
│  └──────────┘    │  bronze_ohlcv ──▶ silver_ohlcv ──▶ gold_metrics│   │
│                  │        │               │                ▼        │   │
│                  │        ▼               ▼         pipeline_run_  │   │
│                  │    [MinIO]        [Postgres]       summary       │   │
│                  │  Parquet files   fact_ohlcv                      │   │
│                  │  bronze layer    fact_daily_metrics              │   │
│                  └─────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐   │
│  │  Postgres │   │    MinIO     │   │ Dagster  │   │   FastAPI    │   │
│  │  :5432    │   │  :9000/:9001 │   │  :3000   │   │   :8000      │   │
│  └───────────┘   └──────────────┘   └──────────┘   └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Medallion Architecture

| Layer | Storage | Description |
|-------|---------|-------------|
| **Bronze** | MinIO (Parquet) | Raw API output, partitioned by `run_date` and `ticker`. Minimal transformation. |
| **Silver** | Postgres `fact_ohlcv` | Cleaned, validated, deduplicated OHLCV rows loaded via explicit SQL upserts. |
| **Gold** | Postgres `fact_daily_metrics` | Computed analytics: daily returns, rolling averages (7d/20d), annualised volatility. |

---

## Service Overview

| Service | Port | Purpose |
|---------|------|---------|
| `postgres` | 5432 | Relational store for silver and gold layers, plus observability tables |
| `minio` | 9000 / 9001 | S3-compatible object store for bronze Parquet files |
| `dagster-webserver` | 3000 | Dagster UI — trigger runs, view asset graph, inspect logs |
| `dagster-daemon` | — | Runs scheduled jobs and sensors |
| `api` | 8000 | FastAPI serving layer (docs at `/docs`) |

---

## Database Schema

```sql
market.dim_instrument        -- ticker dimension (AAPL, MSFT, ...)
market.fact_ohlcv            -- silver: daily OHLCV rows, PK (instrument_id, price_date, interval)
market.fact_daily_metrics    -- gold: returns + rolling averages, PK (instrument_id, price_date)
market.pipeline_run_log      -- observability: every pipeline run's metadata
market.data_quality_check_log -- observability: per-check DQ results as JSONB
```

Key design decisions:
- Explicit upsert SQL (`ON CONFLICT DO UPDATE`) — no blind `to_sql` dumps
- `fact_ohlcv` supports multiple intervals (`1d`, `1h`, etc.) via composite PK
- `data_quality_check_log.details` uses JSONB for flexible check payloads
- Indexes on `ticker`, `price_date`, and `(instrument_id, price_date)` for API query patterns

---

## How to Run Locally

### Prerequisites
- Docker + Docker Compose v2
- (Optional for local tests) Python 3.11+

### 1. Clone and configure

```bash
cd market-pipeline
cp .env.example .env    # defaults work out-of-the-box
```

### 2. Start all services

```bash
make up
```

This starts Postgres, MinIO, Dagster (webserver + daemon), and FastAPI.
The SQL schema is automatically applied by Postgres on first start via `initdb`.

Wait ~30 seconds for all health checks to pass.

### 3. Trigger the pipeline

**Option A — Dagster UI (recommended)**
```
Open http://localhost:3000
→ Assets → Materialize All
```

**Option B — CLI**
```bash
make run-pipeline
```

### 4. Query the API

```bash
# Health check
curl http://localhost:8000/health

# List instruments
curl http://localhost:8000/instruments

# AAPL price history (last 90 days)
curl "http://localhost:8000/prices/AAPL"

# AAPL with date filter
curl "http://localhost:8000/prices/AAPL?start_date=2024-01-01&end_date=2024-06-30"

# Daily metrics
curl "http://localhost:8000/metrics/AAPL"

# Pipeline run history
curl "http://localhost:8000/pipeline-runs"

# Data quality check log
curl "http://localhost:8000/data-quality"
```

Full Swagger UI: `http://localhost:8000/docs`

### 5. Explore MinIO (bronze layer)

Open `http://localhost:9001` — login with `minioadmin` / `minioadmin`.

Bronze files land at:
```
market-data/bronze/ohlcv/run_date=YYYY-MM-DD/ticker=AAPL/data.parquet
```

### 6. Query Postgres directly

```bash
make psql
```

```sql
SELECT ticker, price_date, close FROM market.fact_ohlcv
JOIN market.dim_instrument USING (instrument_id)
WHERE ticker = 'AAPL' ORDER BY price_date DESC LIMIT 10;

SELECT * FROM market.fact_daily_metrics
JOIN market.dim_instrument USING (instrument_id)
WHERE ticker = 'NVDA' ORDER BY price_date DESC LIMIT 20;
```

---

## How to Run Tests

```bash
# Install dependencies locally
pip install -r requirements.txt

# Run test suite
make test
```

Tests cover:
- Config loading from YAML and environment variables
- Pandera schema validation (bronze and silver)
- Gold metric computation (daily return, rolling windows, volatility)
- yfinance extraction with network-free mocks
- FastAPI endpoint responses with mocked database

---

## How Data Quality Works

1. **Bronze validation** (pandera): column presence, non-negative prices, `high >= low`
2. **Silver validation** (pandera): stricter — no null prices, `close` between `low` and `high`
3. **Null pct checks**: configurable threshold (default 5%) per key column
4. **Row count check**: each ticker must produce at least `min_rows_per_ticker` rows
5. All results logged to `market.data_quality_check_log` with JSONB detail payload
6. Queryable via `GET /data-quality`

---

## How Observability Works

Every pipeline run logs to `market.pipeline_run_log`:
- `run_id` (UUID)
- `status`: `running → success | failed`
- `started_at / completed_at / duration_seconds`
- `rows_extracted / rows_loaded / rows_failed`
- `error_message` (if failed)

Query via:
```bash
curl http://localhost:8000/pipeline-runs
```

Or directly in psql:
```sql
SELECT * FROM market.pipeline_run_log ORDER BY started_at DESC;
```

---

## Configuring Tickers and Schedule

Edit `config/pipeline.yml`:

```yaml
tickers:
  - AAPL
  - MSFT
  - NVDA
  - AMZN
  - META
  - TSLA
  - SPY
  - QQQ

lookback_days: 90
interval: "1d"
```

The Dagster schedule runs daily at 06:00 UTC. Modify in
`orchestration/dagster_project/schedules.py`.

---

## Project Structure

```
market-pipeline/
├── config/pipeline.yml         # Ticker list, intervals, DQ thresholds
├── sql/                        # Schema init scripts (run by Postgres on startup)
│   ├── 001_create_schema.sql
│   ├── 002_create_tables.sql
│   └── 003_create_indexes.sql
├── src/market_pipeline/        # Core Python package
│   ├── config.py               # Config loading (YAML + env vars)
│   ├── logging_config.py
│   ├── extract/yfinance_client.py   # Extraction with retries
│   ├── storage/minio_client.py      # MinIO read/write
│   ├── validation/schemas.py        # Pandera schemas
│   ├── transform/ohlcv_cleaning.py  # Bronze → Silver cleaning
│   ├── db/
│   │   ├── connection.py       # psycopg connection factory
│   │   ├── load.py             # Upsert logic
│   │   └── queries.py          # Read-side queries for API
│   ├── metrics/daily_metrics.py     # Gold metric computation
│   └── observability/run_logger.py  # Pipeline run + DQ logging
├── orchestration/dagster_project/
│   ├── assets.py               # Software-defined assets
│   ├── jobs.py
│   ├── schedules.py
│   └── definitions.py          # Dagster entry point
├── api/main.py                 # FastAPI serving layer
├── tests/                      # pytest test suite
├── docker-compose.yml
├── Dockerfile
├── workspace.yaml              # Dagster workspace config
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

## Resume Bullet Suggestions

- Built an end-to-end containerised market data pipeline using Docker Compose, orchestrated with Dagster across bronze/silver/gold medallion layers
- Ingested live OHLCV data via yfinance with exponential-backoff retries and partial-failure tolerance; stored raw Parquet in MinIO (S3-compatible) partitioned by run date and ticker
- Designed normalised Postgres schema (fact/dimension tables, composite PKs, foreign keys, indexes) with explicit upsert SQL — no ORM blind writes
- Implemented data quality checks using pandera (schema validation, null pct, range checks) with structured results persisted to a DQ log table queryable via API
- Computed gold-layer analytics (daily returns, 7d/20d rolling close, annualised volatility) as derived Postgres tables served through a FastAPI REST API
- Added pipeline observability: every run logs status, row counts, duration, and error messages to Postgres; monitored via `/pipeline-runs` endpoint
