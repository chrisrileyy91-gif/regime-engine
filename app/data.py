"""Async data layer. Every request goes through httpx (CLAUDE.md contract).

yfinance was dropped: it is synchronous (blocks the event loop inside an
`async def`), and its multi-ticker download aligns US and Korean calendars on
one index, so `iloc[-1]` can land on a NaN row. Each symbol is fetched and
cleaned independently here.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

from app.config import (
    FRED_SERIES,
    HISTORY_RANGE,
    HTTP_TIMEOUT_SECONDS,
    USER_AGENT,
    YAHOO_SYMBOLS,
)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class DataError(Exception):
    pass


@dataclass
class FetchResult:
    key: str
    symbol: str
    source: str
    series: pd.Series | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.series is not None and not self.series.empty

    @property
    def last_date(self) -> str | None:
        if not self.ok or self.series is None:
            return None
        return str(self.series.index[-1].date())


def parse_yahoo_chart(payload: dict[str, Any]) -> pd.Series:
    chart = payload.get("chart") or {}
    results = chart.get("result")
    if not results:
        raise DataError(f"Yahoo returned no result: {chart.get('error')}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators") or {}).get("quote") or [{}]
    closes = quotes[0].get("close") or []
    if not timestamps or len(timestamps) != len(closes):
        raise DataError("Yahoo payload missing timestamps/closes")

    gmtoffset = int((result.get("meta") or {}).get("gmtoffset") or 0)
    local_seconds = np.asarray(timestamps, dtype="int64") + gmtoffset
    index = pd.to_datetime(local_seconds, unit="s").normalize()
    values = pd.to_numeric(pd.Series(closes, dtype="object"), errors="coerce").to_numpy(dtype=float)
    series = pd.Series(values, index=index, dtype=float).dropna()
    # Yahoo sometimes appends a live intraday bar that duplicates today's date.
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise DataError("Yahoo series empty after cleaning")
    return series


def parse_fred_csv(text: str) -> pd.Series:
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    if frame.shape[1] < 2:
        raise DataError("FRED CSV has fewer than two columns")
    dates = pd.to_datetime(frame.iloc[:, 0])
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    series = pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float).dropna().sort_index()
    if series.empty:
        raise DataError("FRED series empty after cleaning")
    return series


async def fetch_yahoo(client: httpx.AsyncClient, key: str, symbol: str) -> FetchResult:
    url = YAHOO_CHART_URL.format(symbol=quote(symbol, safe=""))
    try:
        response = await client.get(url, params={"range": HISTORY_RANGE, "interval": "1d"})
        response.raise_for_status()
        return FetchResult(key, symbol, "yahoo", parse_yahoo_chart(response.json()))
    except Exception as exc:  # never crash on a single feed
        return FetchResult(key, symbol, "yahoo", None, f"{type(exc).__name__}: {exc}")


async def fetch_fred(client: httpx.AsyncClient, key: str, series_id: str) -> FetchResult:
    start = (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    try:
        response = await client.get(FRED_CSV_URL, params={"id": series_id, "cosd": start})
        response.raise_for_status()
        return FetchResult(key, series_id, "fred", parse_fred_csv(response.text))
    except Exception as exc:
        return FetchResult(key, series_id, "fred", None, f"{type(exc).__name__}: {exc}")


async def fetch_all(client: httpx.AsyncClient | None = None) -> dict[str, FetchResult]:
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
    try:
        tasks = [fetch_yahoo(client, k, s) for k, s in YAHOO_SYMBOLS.items()]
        tasks += [fetch_fred(client, k, s) for k, s in FRED_SERIES.items()]
        results = await asyncio.gather(*tasks)
    finally:
        if owns_client:
            await client.aclose()
    return {r.key: r for r in results}
