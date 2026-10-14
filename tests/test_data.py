from __future__ import annotations

import httpx
import pytest
import respx

from app.data import (
    FRED_CSV_URL,
    DataError,
    fetch_all,
    parse_fred_csv,
    parse_yahoo_chart,
)
from app.features import build_features
from tests.conftest import failed, ok, series

# Three sessions at 13:30 UTC, plus a duplicate intraday bar on the last date.
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


def test_parse_yahoo_drops_nulls_and_duplicate_live_bar() -> None:
    s = parse_yahoo_chart(YAHOO_PAYLOAD)
    assert list(s.round(2)) == [4.11, 4.07]
    assert s.index.is_monotonic_increasing and not s.index.has_duplicates


def test_parse_yahoo_error_raises() -> None:
    with pytest.raises(DataError):
        parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_parse_fred_handles_missing_dots() -> None:
    s = parse_fred_csv("observation_date,DGS2\n2026-10-08,3.61\n2026-10-09,.\n2026-10-10,3.58\n")
    assert list(s) == [3.61, 3.58]


@respx.mock
async def test_fetch_all_isolates_failures() -> None:
    respx.get(url__regex=r"https://query1\.finance\.yahoo\.com/.*").mock(
        return_value=httpx.Response(200, json=YAHOO_PAYLOAD)
    )
    respx.get(url__startswith=FRED_CSV_URL).mock(return_value=httpx.Response(500))
    results = await fetch_all()
    assert results["vix"].ok and results["tnx"].ok
    assert not results["dgs2"].ok and "HTTPStatusError" in (results["dgs2"].error or "")


def test_tnx_x10_feed_is_rescaled() -> None:
    f = build_features({"tnx": ok("tnx", series([41.0, 41.5, 42.0]))})
    assert f.tnx_pct == pytest.approx(4.2)
    assert any("10x" in n for n in f.notes)


def test_tnx_falls_back_to_fred() -> None:
    f = build_features({"tnx": failed("tnx"), "dgs10": ok("dgs10", series([4.1, 4.2]), "fred")})
    assert f.tnx_source == "fred:DGS10" and f.tnx_pct == pytest.approx(4.2)


def test_tnx_crosscheck_flags_disagreement() -> None:
    f = build_features(
        {
            "tnx": ok("tnx", series([4.0, 4.0])),
            "dgs10": ok("dgs10", series([4.5, 4.5]), "fred"),
        }
    )
    assert any("disagree" in n for n in f.notes)


def test_ratio_aligns_mismatched_calendars() -> None:
    igv = series([100.0] * 150, end="2026-10-12")
    soxx = series([200.0] * 150, end="2026-10-09")  # ends a session earlier
    f = build_features({"igv": ok("igv", igv), "soxx": ok("soxx", soxx)})
    assert f.igv_soxx_ratio == pytest.approx(0.5)
