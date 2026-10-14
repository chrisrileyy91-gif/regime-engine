"""FastAPI entrypoint: `uvicorn app.main:app --reload --port 8000`."""

from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI, HTTPException

from app.cache import SnapshotStore
from app.config import CACHE_TTL_SECONDS
from app.data import fetch_all
from app.models import RegimeStatus
from app.service import EngineUnavailable, Fetcher, compute_regime

app = FastAPI(title="Market Regime Engine", version="0.1.0")
app.state.store = SnapshotStore()
app.state.fetcher = fetch_all

_lock = asyncio.Lock()
_memo: dict[str, tuple[float, RegimeStatus]] = {}


def reset_memo() -> None:
    _memo.clear()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/regime", response_model=RegimeStatus)
async def evaluate_market_regime(refresh: bool = False) -> RegimeStatus:
    """Classify the current regime. Cached for CACHE_TTL_SECONDS to spare upstream feeds."""
    async with _lock:
        hit = _memo.get("regime")
        if hit is not None and not refresh and time.monotonic() - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]
        store: SnapshotStore = app.state.store
        fetcher: Fetcher = app.state.fetcher
        try:
            status = await compute_regime(store, fetcher)
        except EngineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not status.stale:
            _memo["regime"] = (time.monotonic(), status)
        return status
