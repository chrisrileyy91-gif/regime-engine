from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx

import app.data as data
from app.data import (
    DataError,
    bill_bond_equivalent,
    fetch_all,
    fetch_krx,
    parse_alpaca_bars,
    parse_cboe_history,
    parse_cboe_quote,
    parse_cboe_vix,
    parse_ecb_dxy,
    parse_eodhd_dxy,
    parse_eodhd_eod,
    parse_krx_kospi,
    parse_krx_samsung,
    parse_nyfed_effr,
    parse_treasury_csv,
    parse_yahoo_chart,
    redact,
)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "HTTP_BACKOFF_SECONDS", (0.0, 0.0))


# Three sessions at 13:30 UTC (one null close), plus a duplicate intraday bar on the last date.
YAHOO_PAYLOAD = {
    "chart": {
        "result": [
            {
                "meta": {"gmtoffset": -14400},
                "timestamp": [1760016600, 1760103000, 1760275800, 1760290000],
                "indicators": {"quote": [{"close": [4.11, None, 4.05, 4.07]}]},
            }
        ],
        "error": None,
    }
}

TREASURY_CSV = (
    'Date,"1 Mo","3 Mo","2 Yr","10 Yr","30 Yr"\n'
    "10/09/2026,4.13,4.25,4.80,5.24,5.60\n"
    "10/08/2026,4.14,4.23,4.75,5.22,5.60\n"
    "10/07/2026,4.07,4.22,4.77,5.28,5.67\n"
)
ECB_XML = """<?xml version="1.0" encoding="UTF-8"?>
<gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/2002-08-01" xmlns="http://www.ecb.int/vocabulary/2002-08-01/eurofxref">
 <Cube>
  <Cube time="2026-10-13">
   <Cube currency="USD" rate="1.1188"/><Cube currency="JPY" rate="176.91"/><Cube currency="GBP" rate="0.84460"/>
   <Cube currency="CAD" rate="1.5933"/><Cube currency="SEK" rate="11.1698"/><Cube currency="CHF" rate="0.9312"/>
  </Cube>
 </Cube>
</gesmes:Envelope>"""


def test_parse_yahoo_drops_nulls_and_duplicate_live_bar() -> None:
    s, appended = parse_yahoo_chart(YAHOO_PAYLOAD)
    assert list(s.round(2)) == [4.11, 4.07] and not appended
    assert s.index.is_monotonic_increasing and not s.index.has_duplicates


def test_parse_yahoo_appends_newer_live_quote() -> None:
    payload = {"chart": {"result": [dict(YAHOO_PAYLOAD["chart"]["result"][0])]}}  # type: ignore[index]
    payload["chart"]["result"][0]["meta"] = {
        "gmtoffset": -14400,
        "regularMarketPrice": 4.2,
        "regularMarketTime": 1760385600,
    }
    s, appended = parse_yahoo_chart(payload)
    assert appended and s.iloc[-1] == 4.2 and s.index[-1] > s.index[-2]


def test_parse_yahoo_error_raises() -> None:
    with pytest.raises(DataError):
        parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_parse_treasury_two_years_sorted() -> None:
    older = 'Date,"3 Mo","10 Yr"\n12/31/2025,4.10,4.55\n'
    out = parse_treasury_csv([TREASURY_CSV, older])
    assert list(out["y10@treasury"]) == [4.55, 5.28, 5.22, 5.24]
    assert out["t3m@treasury"].iloc[-1] == 4.25


def test_parse_treasury_missing_column_raises() -> None:
    with pytest.raises(DataError):
        parse_treasury_csv(['Date,"5 Yr"\n10/09/2026,5.02\n'])


def test_parse_cboe_vix() -> None:
    s = parse_cboe_vix("DATE,OPEN,HIGH,LOW,CLOSE\n10/12/2026,15.1,16.0,14.9,15.21\n10/13/2026,15.2,15.9,15.0,15.27\n")
    assert s.iloc[-1] == 15.27 and str(s.index[-1].date()) == "2026-10-13"


# Shapes copied from the live Cboe responses fetched 2026-10-13 (20:37 ET).
TNX_QUOTE = {
    "timestamp": "2026-10-14 00:37:22",
    "data": {
        "symbol": "^TNX",
        "current_price": 52.17,
        "close": 52.17,
        "last_trade_time": "2026-10-13T13:59:55.419000-05:00",
    },
}
VIX_QUOTE = {"data": {"close": 15.27, "last_trade_time": "2026-10-13T16:15:01"}}


def test_parse_cboe_history_scales_tnx() -> None:
    payload = {"data": [{"date": "2026-10-08", "close": "52.200000"}, {"date": "2026-10-09", "close": "52.400000"}]}
    s = parse_cboe_history(payload, 10.0)
    assert list(s) == pytest.approx([5.22, 5.24])


def test_cboe_quote_after_session_is_the_close() -> None:
    day, value = parse_cboe_quote(TNX_QUOTE, 10.0, "14:55") or (None, None)
    assert str(day.date()) == "2026-10-13" and value == pytest.approx(5.217)  # type: ignore[union-attr]
    day, value = parse_cboe_quote(VIX_QUOTE, 1.0, "16:14") or (None, None)
    assert str(day.date()) == "2026-10-13" and value == 15.27  # type: ignore[union-attr]


def test_cboe_quote_intraday_or_overnight_is_ignored() -> None:
    overnight = {"data": {"close": 16.0, "last_trade_time": "2026-10-14T05:59:00"}}
    intraday = {"data": {"close": 52.0, "last_trade_time": "2026-10-13T11:00:00-05:00"}}  # 12:00 ET
    assert parse_cboe_quote(overnight, 1.0, "16:14") is None
    assert parse_cboe_quote(intraday, 10.0, "14:55") is None


def test_bill_discount_to_bond_equivalent() -> None:
    # 4.15% discount on a 91-day bill = 4.252% bond-equivalent (365*d / (360 - 91*d)).
    assert bill_bond_equivalent(4.15) == pytest.approx(4.2523, abs=1e-4)


def test_parse_eodhd_eod() -> None:
    s = parse_eodhd_eod([{"date": "2026-10-13", "close": 6384.24}, {"date": "2026-10-14", "close": "NA"}])
    assert list(s) == [6384.24]


def test_eodhd_dxy_matches_ecb_hand_computation() -> None:
    # Same six rates as the ECB 2026-10-13 fix, expressed as USD pairs: 102.268.
    eur = 1.1188
    quotes = {
        "EURUSD": eur,
        "USDJPY": 176.91 / eur,
        "GBPUSD": eur / 0.84460,
        "USDCAD": 1.5933 / eur,
        "USDSEK": 11.1698 / eur,
        "USDCHF": 0.9312 / eur,
    }
    payload = [{"code": f"{k}.FOREX", "close": v, "timestamp": 1792000000} for k, v in quotes.items()]
    _day, value = parse_eodhd_dxy(payload)
    assert value == pytest.approx(102.268, abs=0.001)


def test_eodhd_dxy_missing_pair_raises() -> None:
    with pytest.raises(DataError):
        parse_eodhd_dxy([{"code": "EURUSD.FOREX", "close": 1.1, "timestamp": 1792000000}])


def test_redact_strips_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EODHD_API_KEY", "supersecret123")
    assert redact("GET https://eodhd.com/api/eod/X?api_token=supersecret123 failed") == (
        "GET https://eodhd.com/api/eod/X?api_token=*** failed"
    )


def test_parse_nyfed_effr_filters_type() -> None:
    payload = {
        "refRates": [
            {
                "effectiveDate": "2026-10-09",
                "type": "EFFR",
                "percentRate": 3.88,
                "targetRateFrom": 3.75,
                "targetRateTo": 4.0,
            },
            {
                "effectiveDate": "2026-10-08",
                "type": "EFFR",
                "percentRate": 3.88,
                "targetRateFrom": 3.75,
                "targetRateTo": 4.0,
            },
            {"effectiveDate": "2026-10-09", "type": "OBFR", "percentRate": 3.87},
        ]
    }
    s = parse_nyfed_effr(payload)
    assert list(s) == [3.88, 3.88] and str(s.index[-1].date()) == "2026-10-09"


def test_nyfed_effr_outside_target_range_raises() -> None:
    payload = {
        "refRates": [
            {
                "effectiveDate": "2026-10-09",
                "type": "EFFR",
                "percentRate": 38.8,
                "targetRateFrom": 3.75,
                "targetRateTo": 4.0,
            }
        ]
    }
    with pytest.raises(DataError):
        parse_nyfed_effr(payload)


def test_ecb_dxy_matches_hand_computation_from_tonight() -> None:
    # Hand-computed from the ECB 2026-10-13 fix: 102.268 (Yahoo DX-Y.NYB showed 102.189).
    s = parse_ecb_dxy([ECB_XML])
    assert s.iloc[-1] == pytest.approx(102.268, abs=0.001)


def test_ecb_incomplete_day_skipped() -> None:
    partial = ECB_XML.replace('<Cube currency="SEK" rate="11.1698"/>', "")
    with pytest.raises(DataError):
        parse_ecb_dxy([partial])


def test_parse_alpaca_bars_uses_new_york_date() -> None:
    pages = [{"bars": {"SOXX": [{"t": "2026-10-13T04:00:00Z", "c": 251.3}, {"t": "2026-10-12T04:00:00Z", "c": 258.0}]}}]
    s = parse_alpaca_bars(pages, "SOXX")
    assert list(s) == [258.0, 251.3] and str(s.index[-1].date()) == "2026-10-13"


def test_parse_krx_rows() -> None:
    idx = {
        "OutBlock_1": [{"IDX_NM": "코스피 200", "CLSPRC_IDX": "850.10"}, {"IDX_NM": "코스피", "CLSPRC_IDX": "6,384.24"}]
    }
    sto = {"OutBlock_1": [{"ISU_CD": "000660", "TDD_CLSPRC": "190000"}, {"ISU_CD": "005930", "TDD_CLSPRC": "61,200"}]}
    assert parse_krx_kospi(idx) == 6384.24
    assert parse_krx_samsung(sto) == 61200.0
    assert parse_krx_kospi({"OutBlock_1": []}) is None


@respx.mock
async def test_retry_then_success_on_429() -> None:
    route = respx.get("https://markets.newyorkfed.org/api/rates/unsecured/effr/last/60.json").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(
                200, json={"refRates": [{"effectiveDate": "2026-10-09", "type": "EFFR", "percentRate": 3.88}]}
            ),
        ]
    )
    async with httpx.AsyncClient() as client:
        result = await data.fetch_nyfed(client)
    assert result.ok and route.call_count == 2


@respx.mock
async def test_yahoo_falls_back_to_second_host() -> None:
    respx.get(url__regex=r"https://query1\.finance\.yahoo\.com/.*").mock(return_value=httpx.Response(503))
    respx.get(url__regex=r"https://query2\.finance\.yahoo\.com/.*").mock(
        return_value=httpx.Response(200, json=YAHOO_PAYLOAD)
    )
    async with httpx.AsyncClient() as client:
        result = await data.fetch_yahoo(client, "vix@yahoo", "^VIX")
    assert result.ok and any("backup host" in d for d in result.details)


@respx.mock
async def test_fetch_all_isolates_failures_and_reports_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "KRX_API_KEY", "EODHD_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    history = {"data": [{"date": "2026-10-09", "close": "52.4"}]}
    respx.get(url__regex=r"https://query[12]\.finance\.yahoo\.com/.*").mock(
        return_value=httpx.Response(200, json=YAHOO_PAYLOAD)
    )
    respx.get(url__startswith="https://home.treasury.gov/").mock(return_value=httpx.Response(200, text=TREASURY_CSV))
    respx.get(url__regex=r"https://cdn\.cboe\.com/.*/historical/_TNX\.json").mock(
        return_value=httpx.Response(200, json=history)
    )
    respx.get(url__regex=r"https://cdn\.cboe\.com/.*/quotes/_TNX\.json").mock(
        return_value=httpx.Response(200, json=TNX_QUOTE)
    )
    respx.get(url__regex=r"https://cdn\.cboe\.com/.*/historical/_IRX\.json").mock(return_value=httpx.Response(500))
    respx.get(url__regex=r"https://cdn\.cboe\.com/.*/historical/_VIX\.json").mock(return_value=httpx.Response(404))
    respx.get(url__regex=r"https://cdn\.cboe\.com/.*VIX_History\.csv").mock(
        return_value=httpx.Response(200, text="DATE,CLOSE\n10/12/2026,15.21\n")
    )
    respx.get(url__regex=r"https://cdn\.cboe\.com/.*/quotes/_VIX\.json").mock(
        return_value=httpx.Response(200, json=VIX_QUOTE)
    )
    respx.get(url__startswith="https://markets.newyorkfed.org/").mock(return_value=httpx.Response(404))
    respx.get(url__startswith="https://www.ecb.europa.eu/").mock(return_value=httpx.Response(200, text=ECB_XML))
    results = await fetch_all()
    tnx = results["y10@cboe"]
    assert tnx.ok and tnx.series is not None and tnx.series.iloc[-1] == pytest.approx(5.217)
    assert str(tnx.series.index[-1].date()) == "2026-10-13"  # same-day official close from the quote
    vix = results["vix@cboe"]
    assert vix.ok and vix.series is not None and list(vix.series) == [15.21, 15.27]
    assert not results["t3m@cboe"].ok and not results["effr@nyfed"].ok
    assert results["y10@treasury"].ok and results["dxy@ecb"].ok
    for key in ("soxx@alpaca", "kospi@krx", "kospi@eodhd", "dxy@eodhd"):
        assert "not configured" in (results[key].error or ""), key


@respx.mock
async def test_krx_uses_and_extends_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KRX_API_KEY", "test-key")
    monkeypatch.setattr(data, "KRX_HISTORY_SESSIONS", 3)
    calls: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.headers["AUTH_KEY"] == "test-key"  # header, never the URL
        day = request.url.params["basDd"]
        calls.append(day)
        if "kospi" in request.url.path:
            return httpx.Response(200, json={"OutBlock_1": [{"IDX_NM": "코스피", "CLSPRC_IDX": "6,000.00"}]})
        return httpx.Response(200, json={"OutBlock_1": [{"ISU_CD": "005930", "TDD_CLSPRC": "60,000"}]})

    respx.get(url__startswith="https://data-dbg.krx.co.kr/").mock(side_effect=answer)
    cache: dict = {}
    async with httpx.AsyncClient() as client:
        first = await fetch_krx(client, cache)
        n_first = len(calls)
        second = await fetch_krx(client, cache)
    assert all(r.ok for r in first + second)
    assert n_first > 0 and len(calls) == n_first  # second run served from cache
    assert "AUTH_KEY" not in str(calls)


def test_krx_candidate_dates_skip_weekends() -> None:
    days = data.krx_candidate_dates(pd.Timestamp("2026-10-14"), 5)
    assert days[0] == "20261013" and all(pd.Timestamp(d).weekday() < 5 for d in days)


@respx.mock
async def test_treasury_failure_reports_both_keys() -> None:
    respx.get(url__startswith="https://home.treasury.gov/").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        results = await data.fetch_treasury(client)
    assert sorted(r.key for r in results) == ["t3m@treasury", "y10@treasury", "y2@treasury"]
    assert not any(r.ok for r in results)


def test_unfinished_us_session_bar_is_dropped() -> None:
    """Manual run at 09:50 ET: Alpaca's bar for today is partial and must not count as a close."""
    from datetime import datetime

    from app.data import NY, drop_unfinished_us_session

    idx = pd.to_datetime(["2026-10-13", "2026-10-14"])
    bars = pd.Series([100.0, 98.0], index=idx)
    morning = datetime(2026, 10, 14, 9, 50, tzinfo=NY)
    after = datetime(2026, 10, 14, 16, 25, tzinfo=NY)
    assert list(drop_unfinished_us_session(bars, morning).index) == [idx[0]]
    assert len(drop_unfinished_us_session(bars, after)) == 2


def test_cboe_history_highs() -> None:
    from app.data import parse_cboe_highs

    payload = {
        "data": [
            {"date": "2026-10-13", "high": "52.36", "close": "52.17"},
            {"date": "2026-10-14", "high": "53.29", "close": "51.90"},
        ]
    }
    highs = parse_cboe_highs(payload, 10.0)
    assert highs.iloc[-1] == pytest.approx(5.329)


def test_treasury_csv_reads_the_two_year() -> None:
    text = 'Date,"3 Mo","2 Yr","10 Yr"\n08/28/2026,4.20,4.34,4.80\n08/27/2026,4.20,4.20,4.78\n'
    parsed = data.parse_treasury_csv([text])
    assert parsed["y2@treasury"].diff().iloc[-1] * 100 == pytest.approx(14.0)


def test_treasury_csv_without_two_year_still_reads_ten_year() -> None:
    parsed = data.parse_treasury_csv(['Date,"3 Mo","10 Yr"\n08/28/2026,4.20,4.80\n'])
    assert "y10@treasury" in parsed and "y2@treasury" not in parsed
