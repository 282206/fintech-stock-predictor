-- ── Dimension: instruments ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market.dim_instrument (
    instrument_id   SERIAL PRIMARY KEY,
    ticker          VARCHAR(20)  UNIQUE NOT NULL,
    name            VARCHAR(255),
    asset_class     VARCHAR(50)  DEFAULT 'equity',
    exchange        VARCHAR(50),
    currency        VARCHAR(10)  DEFAULT 'USD',
    is_active       BOOLEAN      DEFAULT TRUE,
    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  DEFAULT NOW()
);

-- ── Fact: raw OHLCV prices ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market.fact_ohlcv (
    instrument_id   INTEGER     NOT NULL REFERENCES market.dim_instrument(instrument_id),
    price_date      DATE        NOT NULL,
    open            NUMERIC(18,6),
    high            NUMERIC(18,6),
    low             NUMERIC(18,6),
    close           NUMERIC(18,6),
    adj_close       NUMERIC(18,6),
    volume          BIGINT,
    source          VARCHAR(50)  DEFAULT 'yfinance',
    interval        VARCHAR(10)  DEFAULT '1d',
    ingested_at     TIMESTAMPTZ  DEFAULT NOW(),
    PRIMARY KEY (instrument_id, price_date, interval)
);

-- ── Fact: computed daily metrics ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market.fact_daily_metrics (
    instrument_id       INTEGER     NOT NULL REFERENCES market.dim_instrument(instrument_id),
    price_date          DATE        NOT NULL,
    daily_return        NUMERIC(18,8),
    rolling_7d_close    NUMERIC(18,6),
    rolling_20d_close   NUMERIC(18,6),
    rolling_20d_vol     NUMERIC(18,8),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (instrument_id, price_date)
);

-- ── Observability: pipeline run log ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market.pipeline_run_log (
    run_id            VARCHAR(64)  PRIMARY KEY,
    pipeline_name     VARCHAR(100) NOT NULL,
    status            VARCHAR(20)  NOT NULL,   -- running | success | failed
    started_at        TIMESTAMPTZ  NOT NULL,
    completed_at      TIMESTAMPTZ,
    duration_seconds  NUMERIC(10,2),
    rows_extracted    INTEGER      DEFAULT 0,
    rows_loaded       INTEGER      DEFAULT 0,
    rows_failed       INTEGER      DEFAULT 0,
    error_message     TEXT
);

-- ── Observability: data quality check log ────────────────────────────────────
CREATE TABLE IF NOT EXISTS market.data_quality_check_log (
    check_id    SERIAL       PRIMARY KEY,
    run_id      VARCHAR(64)  NOT NULL,
    check_name  VARCHAR(100) NOT NULL,
    table_name  VARCHAR(100),
    status      VARCHAR(20)  NOT NULL,   -- passed | failed | warning
    checked_at  TIMESTAMPTZ  DEFAULT NOW(),
    details     JSONB
);
