from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.cache import SnapshotStore
from app.data import FetchResult
from app.main import app, reset_memo


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    reset_memo()
    app.state.store = SnapshotStore(tmp_path / "cache.db")
    with TestClient(app) as c:
        yield c
    reset_memo()


def _set_fetcher(results: dict[str, FetchResult] | Exception) -> None:
    async def fetcher() -> dict[str, FetchResult]:
        if isinstance(results, Exception):
            raise results
        return results

    app.state.fetcher = fetcher


def test_regime_payload_shape(client: TestClient, calm_market: dict[str, FetchResult]) -> None:
    _set_fetcher(calm_market)
    body = client.get("/api/v1/regime").json()
    for key in ("phase", "risk_level", "liquidity_squeeze", "semi_rsi", "tnx_yield", "actionable_summary"):
        assert key in body  # original spec contract preserved
    assert body["stale"] is False
    assert body["liquidity_squeeze"] is False
    assert body["tnx_yield"] == pytest.approx(4.0, abs=0.05)
    assert len(body["rules"]) == 4 and len(body["telemetry"]) == 7


def test_serves_stale_snapshot_on_outage(client: TestClient, calm_market: dict[str, FetchResult]) -> None:
    _set_fetcher(calm_market)
    live = client.get("/api/v1/regime").json()
    _set_fetcher(RuntimeError("yahoo down"))
    stale = client.get("/api/v1/regime?refresh=true").json()
    assert stale["stale"] is True and "yahoo down" in stale["stale_reason"]
    assert stale["phase"] == live["phase"]


def test_all_critical_feeds_missing_uses_snapshot(client: TestClient, calm_market: dict[str, FetchResult]) -> None:
    _set_fetcher(calm_market)
    client.get("/api/v1/regime")
    empty = {k: FetchResult(k, k, "yahoo", None, "down") for k in calm_market}
    _set_fetcher(empty)
    body = client.get("/api/v1/regime?refresh=true").json()
    assert body["stale"] is True and "critical" in body["stale_reason"]


def test_503_when_no_data_and_no_snapshot(client: TestClient) -> None:
    _set_fetcher(RuntimeError("everything down"))
    response = client.get("/api/v1/regime")
    assert response.status_code == 503


def test_memo_avoids_refetch(client: TestClient, calm_market: dict[str, FetchResult]) -> None:
    calls = 0

    async def counting() -> dict[str, FetchResult]:
        nonlocal calls
        calls += 1
        return calm_market

    app.state.fetcher = counting
    client.get("/api/v1/regime")
    client.get("/api/v1/regime")
    assert calls == 1
