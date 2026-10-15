"""Async data layer: the record first, independent checks second.

Every request goes through httpx with retries. Each source is fetched and parsed on its own;
one failing source never takes down another. Results are keyed "<input>@<source>" (see
config.INPUTS); verify.py decides what is trustworthy.

Same-day sources, so the 16:25 read is not a session behind:
  Cboe (10Y TNX, 3M IRX, VIX: history + delayed quote whose close is official after the
  session) · Alpaca consolidated tape (SOXX, IGV) · NY Fed (EFFR) · EODHD (KOSPI and
  Samsung end-of-day, posted 2-3h after the Korean close; live FX for the dollar index)
Lagging cross-checks: U.S. Treasury par curve, ECB reference rates, KRX Open API.
Same-day cross-check: Yahoo.
"""

from __future__ import annotations

import asyncio
import io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from app.config import (
    ALPACA_BARS_URL,
    ALPACA_SIP_MIN_AGE_MINUTES,
    CBOE_HISTORY_URL,
    CBOE_INDEXES,
    CBOE_QUOTE_URL,
    CBOE_TALLY,
    CBOE_VIX_CSV_URL,
    CYCLE_PRICE_TICKERS,
    DXY_CONSTANT,
    DXY_WEIGHTS,
    ECB_90D_URL,
    ECB_DAILY_URL,
    EODHD_EOD_SYMBOLS,
    EODHD_EOD_URL,
    EODHD_FX,
    EODHD_HISTORY_DAYS,
    EODHD_LIVE_URL,
    HISTORY_YEARS,
    HTTP_ATTEMPTS,
    HTTP_BACKOFF_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    KRX_BASE_URL,
    KRX_HISTORY_SESSIONS,
    KRX_MAX_CALLS_PER_RUN,
    NYFED_EFFR_URL,
    SECRET_ENV_VARS,
    TALLY_TICKERS,
    TREASURY_CSV_URL,
    US_SESSION_DONE_NY,
    USER_AGENT,
    YAHOO_HOSTS,
    YAHOO_SYMBOLS,
)

NY = ZoneInfo("America/New_York")
RETRY_STATUS = {429, 500, 502, 503, 504}
ECB_NS = {"e": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}
KrxCache = dict[str, dict[str, float | None]]  # input -> {YYYYMMDD: close or None (no session)}


class DataError(Exception):
    pass


@dataclass
class FetchResult:
    key: str  # "<input>@<source>"
    symbol: str
    source: str
    series: pd.Series | None
    error: str | None = None
    details: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.series is not None and not self.series.empty

    @property
    def last_date(self) -> str | None:
        if not self.ok or self.series is None:
            return None
        return str(self.series.index[-1].date())


def _clean(series: pd.Series) -> pd.Series:
    series = series.dropna()
    series.index = pd.DatetimeIndex(series.index).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise DataError("series empty after cleaning")
    return series.astype(float)


def redact(text: str) -> str:
    """Strip secret values from anything that may be published (errors include URLs)."""
    for var in SECRET_ENV_VARS:
        value = os.environ.get(var)
        if value and len(value) >= 6:
            text = text.replace(value, "***")
    return text


def _error(exc: Exception) -> str:
    return redact(f"{type(exc).__name__}: {exc}")


def bill_bond_equivalent(discount_pct: float, days: int = 91) -> float:
    """13-week bill: discount rate -> bond-equivalent yield (the basis Treasury's curve uses).

    BEY = 365 * d / (360 - days * d), with d as a decimal.
    """
    d = discount_pct / 100.0
    return 100.0 * 365.0 * d / (360.0 - days * d)


async def get_with_retry(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
    """GET with retries on 429/5xx/transport errors. Raises the last error."""
    last_exc: Exception | None = None
    for attempt in range(HTTP_ATTEMPTS):
        try:
            response = await client.get(url, **kwargs)
            if response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                return response
            last_exc = httpx.HTTPStatusError(
                f"HTTP {response.status_code}", request=response.request, response=response
            )
        except httpx.TransportError as exc:
            last_exc = exc
        if attempt < HTTP_ATTEMPTS - 1:
            await asyncio.sleep(HTTP_BACKOFF_SECONDS[min(attempt, len(HTTP_BACKOFF_SECONDS) - 1)])
    assert last_exc is not None
    raise last_exc


# --- Parsers (pure) -------------------------------------------------------------------


def parse_yahoo_chart(payload: dict[str, Any]) -> tuple[pd.Series, bool]:
    """Daily closes by exchange-local date, plus whether the live quote was appended."""
    chart = payload.get("chart") or {}
    results = chart.get("result")
    if not results:
        raise DataError(f"Yahoo returned no result: {chart.get('error')}")
    result = results[0]
    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    closes = ((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    if not timestamps or len(timestamps) != len(closes):
        raise DataError("Yahoo payload missing timestamps/closes")
    gmtoffset = int(meta.get("gmtoffset") or 0)
    index = pd.to_datetime(np.asarray(timestamps, dtype="int64") + gmtoffset, unit="s")
    values = pd.to_numeric(pd.Series(closes, dtype="object"), errors="coerce").to_numpy(dtype=float)
    series = _clean(pd.Series(values, index=index))

    appended = False
    price, when = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
    if isinstance(price, (int, float)) and isinstance(when, (int, float)) and price > 0:
        quote_day = pd.to_datetime(int(when) + gmtoffset, unit="s").normalize()
        if quote_day > series.index[-1]:
            series = pd.concat([series, pd.Series([float(price)], index=[quote_day])])
            appended = True
    return series, appended


def parse_cboe_history(payload: dict[str, Any], scale: float) -> pd.Series:
    rows = payload.get("data") or []
    if not rows:
        raise DataError("Cboe history returned no rows")
    index = pd.to_datetime([r["date"] for r in rows])
    values = pd.to_numeric(pd.Series([r.get("close") for r in rows], dtype="object"), errors="coerce")
    return _clean(pd.Series(values.to_numpy(dtype=float) / scale, index=index))


def parse_cboe_highs(payload: dict[str, Any], scale: float) -> pd.Series:
    """Each session's intraday high from Cboe's daily history."""
    rows = payload.get("data") or []
    if not rows:
        raise DataError("Cboe history returned no rows")
    index = pd.to_datetime([r["date"] for r in rows])
    values = pd.to_numeric(pd.Series([r.get("high") for r in rows], dtype="object"), errors="coerce")
    return _clean(pd.Series(values.to_numpy(dtype=float) / scale, index=index))


def parse_cboe_quote(payload: dict[str, Any], scale: float, cutoff_ny: str) -> tuple[pd.Timestamp, float] | None:
    """The quote's close, dated in New York, once the session is over; otherwise None.

    `last_trade_time` comes with an offset (rates, Central time) or without one (VIX, which
    Cboe stamps in New York time). A quote before `cutoff_ny` is intraday or an overnight
    print and is never treated as a close.
    """
    data = payload.get("data") or {}
    raw_time, close = data.get("last_trade_time"), data.get("close")
    if not raw_time or not isinstance(close, (int, float)) or close <= 0:
        return None
    stamp = pd.Timestamp(raw_time)
    stamp = stamp.tz_localize(NY) if stamp.tzinfo is None else stamp.tz_convert(NY)
    hour, minute = (int(x) for x in cutoff_ny.split(":"))
    if (stamp.hour, stamp.minute) < (hour, minute):
        return None
    return pd.Timestamp(stamp.date()), float(close) / scale


def parse_eodhd_eod(rows: list[dict[str, Any]]) -> pd.Series:
    if not isinstance(rows, list) or not rows:
        raise DataError("EODHD returned no rows")
    index = pd.to_datetime([r["date"] for r in rows])
    values = pd.to_numeric(pd.Series([r.get("close") for r in rows], dtype="object"), errors="coerce")
    return _clean(pd.Series(values.to_numpy(dtype=float), index=index))


def parse_eodhd_dxy(payload: Any) -> tuple[pd.Timestamp, float]:
    """DXY from EODHD live FX quotes with ICE's formula, dated by the oldest quote (New York)."""
    rows = payload if isinstance(payload, list) else [payload]
    by_code = {str(r.get("code", "")).upper(): r for r in rows if isinstance(r, dict)}
    value, stamps = DXY_CONSTANT, []
    for pair, code in EODHD_FX.items():
        row = by_code.get(code)
        price = row.get("close") if row else None
        if not isinstance(price, (int, float)) or price <= 0:
            raise DataError(f"EODHD has no usable quote for {code}")
        value *= float(price) ** DXY_WEIGHTS[pair]
        assert row is not None
        stamps.append(int(row.get("timestamp") or 0))
    if min(stamps) <= 0:
        raise DataError("EODHD FX quote missing a timestamp")
    oldest = pd.Timestamp(min(stamps), unit="s", tz="UTC").tz_convert(NY)
    return pd.Timestamp(oldest.date()), value


def parse_treasury_csv(texts: list[str]) -> dict[str, pd.Series]:
    frames = [pd.read_csv(io.StringIO(t)) for t in texts if t.strip()]
    if not frames:
        raise DataError("Treasury returned no CSV")
    frame = pd.concat(frames, ignore_index=True)
    for column in ("Date", "10 Yr", "3 Mo"):
        if column not in frame.columns:
            raise DataError(f"Treasury CSV missing column {column!r}")
    index = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    out = {}
    for key, column in (("y10@treasury", "10 Yr"), ("t3m@treasury", "3 Mo")):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        out[key] = _clean(pd.Series(values, index=index))
    return out


def parse_cboe_vix(text: str) -> pd.Series:
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [c.strip().upper() for c in frame.columns]
    if "DATE" not in frame.columns or "CLOSE" not in frame.columns:
        raise DataError("Cboe CSV missing DATE/CLOSE")
    index = pd.to_datetime(frame["DATE"], format="%m/%d/%Y")
    values = pd.to_numeric(frame["CLOSE"], errors="coerce").to_numpy(dtype=float)
    return _clean(pd.Series(values, index=index))


def parse_nyfed_effr(payload: dict[str, Any]) -> pd.Series:
    rows = [r for r in payload.get("refRates") or [] if r.get("type") == "EFFR"]
    if not rows:
        raise DataError("NY Fed returned no EFFR rows")
    rows.sort(key=lambda r: r["effectiveDate"])
    latest = rows[-1]
    low, high = latest.get("targetRateFrom"), latest.get("targetRateTo")
    rate = float(latest["percentRate"])
    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and not low - 0.25 <= rate <= high + 0.25:
        raise DataError(f"EFFR {rate} is inconsistent with the target range {low}-{high}")
    index = pd.to_datetime([r["effectiveDate"] for r in rows])
    return _clean(pd.Series([float(r["percentRate"]) for r in rows], index=index))


def parse_ecb_dxy(xml_texts: list[str]) -> pd.Series:
    """DXY by ECB reference date, computed with ICE's published formula."""
    by_date: dict[str, float] = {}
    for text in xml_texts:
        root = ET.fromstring(text)
        for day in root.iterfind(".//e:Cube[@time]", ECB_NS):
            rates = {c.get("currency"): float(c.get("rate", "nan")) for c in day.iterfind("e:Cube", ECB_NS)}
            if not all(k in rates for k in ("USD", "JPY", "GBP", "CAD", "SEK", "CHF")):
                continue
            eur = rates["USD"]  # USD per EUR
            pairs = {
                "EURUSD": eur,
                "USDJPY": rates["JPY"] / eur,
                "GBPUSD": eur / rates["GBP"],
                "USDCAD": rates["CAD"] / eur,
                "USDSEK": rates["SEK"] / eur,
                "USDCHF": rates["CHF"] / eur,
            }
            value = DXY_CONSTANT
            for pair, weight in DXY_WEIGHTS.items():
                value *= pairs[pair] ** weight
            by_date[str(day.get("time"))] = value
    if not by_date:
        raise DataError("ECB XML had no complete rate sets")
    return _clean(pd.Series(list(by_date.values()), index=pd.to_datetime(list(by_date))))


def parse_alpaca_bars(pages: list[dict[str, Any]], symbol: str) -> pd.Series:
    bars = [bar for page in pages for bar in ((page.get("bars") or {}).get(symbol) or [])]
    if not bars:
        raise DataError(f"Alpaca returned no bars for {symbol}")
    stamps = pd.to_datetime([b["t"] for b in bars], utc=True).tz_convert("America/New_York").tz_localize(None)
    return _clean(pd.Series([float(b["c"]) for b in bars], index=stamps))


def _krx_number(text: Any) -> float | None:
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def parse_krx_kospi(payload: dict[str, Any]) -> float | None:
    for row in payload.get("OutBlock_1") or []:
        if str(row.get("IDX_NM", "")).strip() == "코스피":
            return _krx_number(row.get("CLSPRC_IDX"))
    return None


def parse_krx_samsung(payload: dict[str, Any]) -> float | None:
    for row in payload.get("OutBlock_1") or []:
        code = str(row.get("ISU_CD", "")).strip()
        if code == "005930" or (code.startswith("KR7") and code[3:9] == "005930"):
            return _krx_number(row.get("TDD_CLSPRC"))
    return None


# --- Fetchers -------------------------------------------------------------------------


async def fetch_yahoo(client: httpx.AsyncClient, key: str, symbol: str) -> FetchResult:
    errors: list[str] = []
    for number, host in enumerate(YAHOO_HOSTS, start=1):
        url = f"https://{host}/v8/finance/chart/{quote(symbol, safe='')}"
        try:
            response = await get_with_retry(client, url, params={"range": f"{HISTORY_YEARS}y", "interval": "1d"})
            series, appended = parse_yahoo_chart(response.json())
            if key == "y10@yahoo" and float(series.median()) > 25.0:
                series = series / 10.0  # older ^TNX convention: yield x10
            details = ["live quote appended"] if appended else []
            if number > 1:
                details.append(f"backup host {host}")
            return FetchResult(key, symbol, "yahoo", series, details=details)
        except Exception as exc:
            errors.append(f"{host}: {_error(exc)}")
    return FetchResult(key, symbol, "yahoo", None, "; ".join(errors))


async def fetch_treasury(client: httpx.AsyncClient) -> list[FetchResult]:
    year = datetime.now(timezone.utc).year
    texts = []
    try:
        for y in range(year - HISTORY_YEARS + 1, year + 1):
            params = {"type": "daily_treasury_yield_curve", "field_tdr_date_value": str(y), "_format": "csv"}
            response = await get_with_retry(client, TREASURY_CSV_URL.format(year=y), params=params)
            texts.append(response.text)
        parsed = parse_treasury_csv(texts)
        return [FetchResult(k, k.split("@")[0].upper(), "treasury", s) for k, s in parsed.items()]
    except Exception as exc:
        return [
            FetchResult(k, k.split("@")[0].upper(), "treasury", None, _error(exc))
            for k in ("y10@treasury", "t3m@treasury")
        ]


async def fetch_cboe_index(client: httpx.AsyncClient, key: str) -> FetchResult:
    symbol, scale, cutoff = CBOE_INDEXES[key]
    details: list[str] = []
    try:
        try:
            history = parse_cboe_history(
                (await get_with_retry(client, CBOE_HISTORY_URL.format(symbol=symbol))).json(), scale
            )
        except Exception:
            if symbol != "VIX":
                raise
            history = parse_cboe_vix((await get_with_retry(client, CBOE_VIX_CSV_URL)).text)
            details.append("history from VIX_History.csv")
        try:
            close = parse_cboe_quote(
                (await get_with_retry(client, CBOE_QUOTE_URL.format(symbol=symbol))).json(), scale, cutoff
            )
        except Exception as exc:
            close = None
            details.append(f"quote unavailable ({_error(exc)})")
        if close is not None:
            day, value = close
            history = history[history.index != day]
            history = pd.concat([history, pd.Series([value], index=[day])]).sort_index()
            details.append(f"official close for {day.date()} from the delayed quote")
        if key == "t3m@cboe":
            history = history.map(bill_bond_equivalent)
        return FetchResult(key, f"^{symbol}", "cboe", history, details=details)
    except Exception as exc:
        return FetchResult(key, f"^{symbol}", "cboe", None, _error(exc), details)


async def fetch_cboe_tally(client: httpx.AsyncClient) -> list[FetchResult]:
    """Closes and intraday highs for the tally (10Y high, 30Y close and high). Classification only."""
    out = []
    for name, (symbol, scale) in CBOE_TALLY.items():
        keys = [f"{name}_high@cboe"] if name == "y10" else [f"{name}@cboe", f"{name}_high@cboe"]
        try:
            payload = (await get_with_retry(client, CBOE_HISTORY_URL.format(symbol=symbol))).json()
            series = {
                f"{name}@cboe": parse_cboe_history(payload, scale),
                f"{name}_high@cboe": parse_cboe_highs(payload, scale),
            }
            out += [FetchResult(k, f"^{symbol}", "cboe", series[k]) for k in keys]
        except Exception as exc:
            out += [FetchResult(k, f"^{symbol}", "cboe", None, _error(exc)) for k in keys]
    return out


async def fetch_eodhd(client: httpx.AsyncClient) -> list[FetchResult]:
    token = os.environ.get("EODHD_API_KEY")
    keys = [*EODHD_EOD_SYMBOLS, "dxy@eodhd"]
    if not token:
        return [FetchResult(k, k, "eodhd", None, "not configured (EODHD_API_KEY)") for k in keys]
    out: list[FetchResult] = []
    start = (datetime.now(timezone.utc) - timedelta(days=EODHD_HISTORY_DAYS)).strftime("%Y-%m-%d")
    for key, symbol in EODHD_EOD_SYMBOLS.items():
        try:
            response = await get_with_retry(
                client, EODHD_EOD_URL.format(symbol=symbol), params={"api_token": token, "fmt": "json", "from": start}
            )
            out.append(FetchResult(key, symbol, "eodhd", parse_eodhd_eod(response.json())))
        except Exception as exc:
            out.append(FetchResult(key, symbol, "eodhd", None, _error(exc)))
    codes = list(EODHD_FX.values())
    try:
        response = await get_with_retry(
            client,
            EODHD_LIVE_URL.format(symbol=codes[0]),
            params={"api_token": token, "fmt": "json", "s": ",".join(codes[1:])},
        )
        day, value = parse_eodhd_dxy(response.json())
        out.append(FetchResult("dxy@eodhd", "DXY (ICE formula, live FX)", "eodhd", pd.Series([value], index=[day])))
    except Exception as exc:
        out.append(FetchResult("dxy@eodhd", "DXY (ICE formula, live FX)", "eodhd", None, _error(exc)))
    return out


async def fetch_nyfed(client: httpx.AsyncClient) -> FetchResult:
    try:
        response = await get_with_retry(client, NYFED_EFFR_URL)
        return FetchResult("effr@nyfed", "EFFR", "nyfed", parse_nyfed_effr(response.json()))
    except Exception as exc:
        return FetchResult("effr@nyfed", "EFFR", "nyfed", None, _error(exc))


async def fetch_ecb(client: httpx.AsyncClient) -> FetchResult:
    try:
        texts = [(await get_with_retry(client, url)).text for url in (ECB_90D_URL, ECB_DAILY_URL)]
        return FetchResult("dxy@ecb", "DXY (ECB-computed)", "ecb", parse_ecb_dxy(texts))
    except Exception as exc:
        return FetchResult("dxy@ecb", "DXY (ECB-computed)", "ecb", None, _error(exc))


async def fetch_alpaca(client: httpx.AsyncClient) -> list[FetchResult]:
    targets = {"soxx@alpaca": "SOXX", "igv@alpaca": "IGV"}
    targets.update({f"{t.lower()}@alpaca": t for t in (*CYCLE_PRICE_TICKERS, *TALLY_TICKERS.values())})
    key_id, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key_id or not secret:
        return [
            FetchResult(k, s, "alpaca", None, "not configured (APCA_API_KEY_ID / APCA_API_SECRET_KEY)")
            for k, s in targets.items()
        ]
    now = datetime.now(timezone.utc)
    params: dict[str, str] = {
        "symbols": ",".join(targets.values()),
        "timeframe": "1Day",
        "start": (now - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y-%m-%d"),
        "end": (now - timedelta(minutes=ALPACA_SIP_MIN_AGE_MINUTES)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feed": "sip",
        "adjustment": "split",
        "limit": "10000",
    }
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
    try:
        pages: list[dict[str, Any]] = []
        for _ in range(10):
            page = (await get_with_retry(client, ALPACA_BARS_URL, params=params, headers=headers)).json()
            pages.append(page)
            if not page.get("next_page_token"):
                break
            params["page_token"] = page["next_page_token"]
    except Exception as exc:
        return [FetchResult(k, s, "alpaca", None, _error(exc)) for k, s in targets.items()]
    out = []
    for key, symbol in targets.items():
        try:
            bars = drop_unfinished_us_session(parse_alpaca_bars(pages, symbol), now)
            out.append(FetchResult(key, symbol, "alpaca", bars))
        except Exception as exc:
            out.append(FetchResult(key, symbol, "alpaca", None, _error(exc)))
    return out


def drop_unfinished_us_session(series: pd.Series, now: datetime) -> pd.Series:
    """Drop today's daily bar until the US session is over.

    A run during market hours (a manual dispatch at 09:50) otherwise gets a partial bar for
    today, which Yahoo's equally partial bar then "confirms" as a finished close.
    """
    now_ny = now.astimezone(NY)
    if now_ny.strftime("%H:%M") >= US_SESSION_DONE_NY or series.empty:
        return series
    today = pd.Timestamp(now_ny.date())
    return series[series.index < today]


def krx_candidate_dates(today_kst: pd.Timestamp, sessions: int) -> list[str]:
    """Weekdays from yesterday (KST) backwards, newest first. Holidays resolve as empty."""
    days = pd.bdate_range(end=today_kst - pd.Timedelta(days=1), periods=int(sessions * 1.1) + 10)
    return [d.strftime("%Y%m%d") for d in reversed(days)]


async def fetch_krx(client: httpx.AsyncClient, cache: KrxCache) -> list[FetchResult]:
    """KOSPI and Samsung official closes. One call per date, so history is cached.

    KRX posts a session's data at 08:00 KST the next business day; dates within the last
    three days that come back empty are not cached as holidays (they may just be early).
    """
    endpoints = {
        "kospi@krx": ("idx/kospi_dd_trd", parse_krx_kospi, "KOSPI"),
        "samsung@krx": ("sto/stk_bydd_trd", parse_krx_samsung, "005930"),
    }
    api_key = os.environ.get("KRX_API_KEY")
    if not api_key:
        return [FetchResult(k, v[2], "krx", None, "not configured (KRX_API_KEY)") for k, v in endpoints.items()]

    today_kst = pd.Timestamp(datetime.now(timezone(timedelta(hours=9))).date())
    recent_cutoff = (today_kst - pd.Timedelta(days=3)).strftime("%Y%m%d")
    candidates = krx_candidate_dates(today_kst, KRX_HISTORY_SESSIONS)
    semaphore = asyncio.Semaphore(4)
    out = []
    for key, (path, parser, symbol) in endpoints.items():
        known = cache.setdefault(key, {})
        todo = [d for d in candidates if d not in known or (known[d] is None and d >= recent_cutoff)]
        todo = todo[:KRX_MAX_CALLS_PER_RUN]
        errors: list[str] = []

        async def one(
            day: str,
            path: str = path,
            parser: Any = parser,
            known: dict[str, float | None] = known,
            errors: list[str] = errors,
        ) -> None:
            async with semaphore:
                try:
                    response = await get_with_retry(
                        client, f"{KRX_BASE_URL}/{path}", params={"basDd": day}, headers={"AUTH_KEY": api_key}
                    )
                    known[day] = parser(response.json())
                except Exception as exc:
                    errors.append(f"{day}: {_error(exc)}")

        await asyncio.gather(*(one(d) for d in todo))
        values = {d: v for d, v in known.items() if v is not None}
        details = [f"{len(todo)} dates requested"] if todo else []
        if errors:
            details.append(f"{len(errors)} date requests failed")
        if not values:
            out.append(FetchResult(key, symbol, "krx", None, errors[0] if errors else "no KRX data yet", details))
            continue
        series = pd.Series(list(values.values()), index=pd.to_datetime(list(values), format="%Y%m%d"))
        out.append(FetchResult(key, symbol, "krx", _clean(series), None, details))
    return out


async def fetch_all(
    client: httpx.AsyncClient | None = None, krx_cache: KrxCache | None = None
) -> dict[str, FetchResult]:
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )
    cache: KrxCache = krx_cache if krx_cache is not None else {}
    try:
        singles = await asyncio.gather(
            fetch_nyfed(client),
            fetch_ecb(client),
            *(fetch_cboe_index(client, k) for k in CBOE_INDEXES),
            *(fetch_yahoo(client, k, s) for k, s in YAHOO_SYMBOLS.items()),
        )
        groups = await asyncio.gather(
            fetch_treasury(client),
            fetch_alpaca(client),
            fetch_eodhd(client),
            fetch_krx(client, cache),
            fetch_cboe_tally(client),
        )
    finally:
        if owns_client:
            await client.aclose()
    results = [*singles, *(r for group in groups for r in group)]
    return {r.key: r for r in results}
