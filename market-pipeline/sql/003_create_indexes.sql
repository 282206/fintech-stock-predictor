-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_fact_ohlcv_price_date
    ON market.fact_ohlcv (price_date);

CREATE INDEX IF NOT EXISTS idx_fact_ohlcv_instrument_date
    ON market.fact_ohlcv (instrument_id, price_date);

CREATE INDEX IF NOT EXISTS idx_fact_metrics_instrument_date
    ON market.fact_daily_metrics (instrument_id, price_date);

CREATE INDEX IF NOT EXISTS idx_dim_instrument_ticker
    ON market.dim_instrument (ticker);

CREATE INDEX IF NOT EXISTS idx_dq_run_id
    ON market.data_quality_check_log (run_id);
