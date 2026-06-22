"""API endpoint tests with mocked database."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_client(mock_conn: MagicMock) -> TestClient:
    """Create a TestClient with the DB dependency overridden."""
    from api.main import app, get_db

    app.dependency_overrides[get_db] = lambda: mock_conn
    client = TestClient(app, raise_server_exceptions=True)
    return client


def test_health_endpoint_db_down():
    """Health endpoint returns 503 when DB is unreachable."""
    with patch("api.main.psycopg.connect", side_effect=Exception("connection refused")):
        from api.main import app
        c = TestClient(app)
        resp = c.get("/health")
        assert resp.status_code == 503


def test_instruments_empty():
    mock_conn = MagicMock()
    c = _make_client(mock_conn)
    with patch("api.main.get_instruments", return_value=[]):
        resp = c.get("/instruments")
    assert resp.status_code == 200
    assert resp.json() == []


def test_prices_not_found():
    mock_conn = MagicMock()
    c = _make_client(mock_conn)
    with patch("api.main.get_prices", return_value=[]):
        resp = c.get("/prices/AAPL")
    assert resp.status_code == 404


def test_prices_found():
    mock_conn = MagicMock()
    c = _make_client(mock_conn)
    fake_row = {
        "price_date": "2024-01-02",
        "open": 150.0, "high": 155.0, "low": 149.0,
        "close": 153.0, "adj_close": 153.0,
        "volume": 1000000, "interval": "1d", "ingested_at": "2024-01-02T10:00:00",
    }
    with patch("api.main.get_prices", return_value=[fake_row]):
        resp = c.get("/prices/AAPL")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_pipeline_runs_empty():
    mock_conn = MagicMock()
    c = _make_client(mock_conn)
    with patch("api.main.get_pipeline_runs", return_value=[]):
        resp = c.get("/pipeline-runs")
    assert resp.status_code == 200
    assert resp.json() == []
