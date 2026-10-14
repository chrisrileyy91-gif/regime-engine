"""The cycle from company filings. Anchored on real SEC data fetched 2026-10-14."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest
import respx

from app.config import CYCLE_COMPANIES, SEC_CONCEPT_URL
from app.fundamentals import (
    build_cycle,
    by_calendar_quarter,
    calendar_quarter,
    company_growth,
    earnings_vs_price,
    fetch_filings,
    group_growth,
    median_acceleration,
    quarterize,
)
from tests.sec_fixture import sec_cache

Q = pd.Period


def facts(ticker: str, metric: str) -> list[dict]:
    return sec_cache()["companies"][ticker][metric]["facts"]


def cq(ticker: str, metric: str) -> pd.Series:
    return by_calendar_quarter(quarterize(facts(ticker, metric)))


# --- quarters from filings ----------------------------------------------------------------


def test_q4_is_full_year_minus_nine_months() -> None:
    """Microsoft files no Q4 10-Q: FY2026 capex $115.948B minus 9M $80.146B = $35.802B."""
    assert cq("MSFT", "capex")[Q("2026Q2")] == pytest.approx(35_802e6)


def test_ytd_chain_without_reported_quarter() -> None:
    """Alphabet reports capex only year-to-date: Q2 2026 = H1 $80.598B minus Q1 $35.674B."""
    assert cq("GOOGL", "capex")[Q("2026Q2")] == pytest.approx(44_924e6)


def test_amazon_trailing_twelve_month_facts_are_ignored() -> None:
    s = cq("AMZN", "capex")
    assert s[Q("2026Q2")] == pytest.approx(54_208e6) and s.max() < 60e9  # a TTM fact would be ~$170B


def test_fiscal_quarters_map_to_nearest_calendar_quarter() -> None:
    assert calendar_quarter(pd.Timestamp("2026-07-26")) == Q("2026Q2")  # Nvidia
    assert calendar_quarter(pd.Timestamp("2026-01-25")) == Q("2025Q4")  # Nvidia fiscal year end
    assert calendar_quarter(pd.Timestamp("2026-05-31")) == Q("2026Q2")  # Oracle fiscal year end
    assert calendar_quarter(pd.Timestamp("2026-02-01")) == Q("2025Q4")  # Broadcom
    assert calendar_quarter(pd.Timestamp("2026-09-03")) == Q("2026Q3")  # Micron 53-week year
    assert cq("NVDA", "revenue")[Q("2026Q2")] == pytest.approx(96_221e6)


def test_restatement_keeps_the_latest_filing() -> None:
    old = {"start": "2025-01-01", "end": "2025-03-31", "val": 100.0, "form": "10-Q", "filed": "2025-04-30"}
    new = {**old, "val": 90.0, "filed": "2026-04-30"}
    assert quarterize([new, old]).iloc[0] == 90.0


def test_non_periodic_forms_are_ignored() -> None:
    fact = {"start": "2025-01-01", "end": "2025-03-31", "val": 100.0, "form": "8-K", "filed": "2025-04-30"}
    assert quarterize([fact]).empty


# --- group readings ------------------------------------------------------------------------


def test_group_growth_needs_every_company() -> None:
    complete = {"NVDA": quarterize(facts("NVDA", "revenue")), "AMD": quarterize(facts("AMD", "revenue"))}
    partial = {**complete, "X": pd.Series({pd.Timestamp("2026-06-30"): 1.0})}
    assert not group_growth(complete).empty and group_growth(partial).empty


def test_median_is_not_moved_by_one_fast_grower() -> None:
    """Palantir grew ~79% in Q2 2026; the median software company grew ~14%."""
    software = {t: quarterize(facts(t, "revenue")) for t, _, _ in CYCLE_COMPANIES["software"]}
    assert company_growth(software["PLTR"])[Q("2026Q2")] > 70
    assert group_growth(software, "median")[Q("2026Q2")] == pytest.approx(13.62, abs=0.01)


def test_median_acceleration_on_real_filings() -> None:
    software = {t: quarterize(facts(t, "revenue")) for t, _, _ in CYCLE_COMPANIES["software"]}
    accel = median_acceleration(software)
    assert accel[Q("2026Q1")] == pytest.approx(0.64, abs=0.01)
    assert accel[Q("2026Q2")] == pytest.approx(0.36, abs=0.01)


# --- the cycle on real filings ---------------------------------------------------------------


def test_real_filings_read_buildout_since_q1_2026() -> None:
    cycle = build_cycle(sec_cache(), {}, effr_minus_bill=-0.38)
    assert (cycle.phase, cycle.since, cycle.confirmed, cycle.through) == ("Buildout", "Q1 2026", True, "Q2 2026")
    metrics = {m.key: m for m in cycle.metrics}
    assert metrics["hyperscalers"].latest == pytest.approx(81.2, abs=0.1)
    assert metrics["chips"].latest == pytest.approx(81.47, abs=0.1)
    assert all(f.ok for f in cycle.filings) and cycle.notes == []


def test_real_filings_history_shows_2025_digestion() -> None:
    """Chip revenue growth rolled over from 81% (Q3 2024) to ~50% through 2025."""
    phases = {h["quarter"]: h["phase_id"] for h in build_cycle(sec_cache(), {}, None).history}
    assert phases["Q3 2024"] == 1
    assert [phases[f"Q{i} 2025"] for i in range(1, 5)] == [2, 2, 2, 2]
    assert phases["Q1 2026"] == 1


def test_software_acceleration_just_short_of_the_line() -> None:
    cycle = build_cycle(sec_cache(), {}, None)
    monet = next(c for c in cycle.checks if c.phase_id == 3)
    assert monet.met is False
    accel = next(c for c in monet.conditions if "two quarters" in c.name)
    assert accel.met is False and "total +0.992" in accel.threshold
    software = next(c for c in cycle.catalysts if c.key == "software")
    assert software.status == "starting"


def test_missing_company_makes_the_cycle_unknown_not_wrong() -> None:
    cache = sec_cache()
    cache["companies"]["META"]["capex"] = {"error": "HTTPStatusError: 503"}
    cycle = build_cycle(cache, {}, None)
    assert not cycle.known and cycle.phase == "Unknown"
    assert any("META" in n for n in cycle.notes)


def test_no_cache_at_all() -> None:
    cycle = build_cycle(None, {}, None)
    assert not cycle.known and all(not f.ok for f in cycle.filings)


# --- catalysts ---------------------------------------------------------------------------------


def _prices(start: float, end: float) -> pd.Series:
    idx = pd.bdate_range("2025-01-01", "2026-07-31")
    return pd.Series([start + (end - start) * i / (len(idx) - 1) for i in range(len(idx))], index=idx)


def test_earnings_vs_price() -> None:
    eps = {"NVDA": quarterize(facts("NVDA", "eps"))}
    # NVDA TTM EPS Q2 2025 -> Q2 2026: 3.52 -> 7.91 (+125%)
    slow = earnings_vs_price(eps, {"NVDA": _prices(100, 150)}, Q("2026Q2"))
    assert slow[0]["eps_growth"] == pytest.approx(124.7, abs=0.5) and slow[0]["outgrowing"] is True
    fast = earnings_vs_price(eps, {"NVDA": _prices(100, 900)}, Q("2026Q2"))
    assert fast[0]["outgrowing"] is False


def test_rate_cut_catalyst() -> None:
    cuts = {c.key: c for c in build_cycle(sec_cache(), {}, effr_minus_bill=0.40).catalysts}["cuts"]
    hikes = {c.key: c for c in build_cycle(sec_cache(), {}, effr_minus_bill=-0.38).catalysts}["cuts"]
    assert cuts.status == "yes" and hikes.status == "not yet" and "hikes are priced" in hikes.detail


# --- fetching ----------------------------------------------------------------------------------


def _concept(name: str, end: str = "2026-06-30") -> dict:
    fact = {"start": "2026-04-01", "end": end, "val": 1e9, "form": "10-Q", "filed": "2026-07-30"}
    return {"entityName": name, "units": {"USD": [fact], "USD/shares": [{**fact, "val": 1.0}]}}


@respx.mock
async def test_fetch_checks_the_company_name_and_keeps_cache_on_failure() -> None:
    names = {cik: name for group in CYCLE_COMPANIES.values() for _, cik, name in group}
    for cik, name in names.items():
        respx.get(url__startswith=SEC_CONCEPT_URL.split("CIK")[0] + f"CIK{cik}/").mock(
            return_value=httpx.Response(200, json=_concept(name if cik != "0000789019" else "SOMEONE ELSE"))
        )
    cache = sec_cache()
    now = datetime(2026, 10, 15, 10, tzinfo=timezone.utc)
    async with httpx.AsyncClient() as client:
        notes = await fetch_filings(client, cache, now)
    assert "facts" in cache["companies"]["MSFT"]["capex"]  # wrong entity: kept the cached filings
    assert any("MSFT" in n and "expected 'MICROSOFT'" in n for n in notes)
    assert cache["companies"]["NVDA"]["revenue"]["facts"][0]["val"] == 1e9  # refreshed
    assert cache["fetched_at"] == now.isoformat(timespec="seconds")


async def test_fetch_skipped_when_fresh() -> None:
    cache = sec_cache(fetched_at="2026-10-15T00:00:00+00:00")
    assert await fetch_filings(None, cache, datetime(2026, 10, 15, 10, tzinfo=timezone.utc)) == []


@respx.mock
async def test_fetch_prefers_the_longer_history_on_a_tie() -> None:
    """Intuit files two revenue tags with the same latest period; only one has the history."""
    short = _concept("INTUIT INC.")
    long = _concept("INTUIT INC.")
    long["units"]["USD"] = long["units"]["USD"] * 1 + [
        {"start": "2025-04-01", "end": "2025-06-30", "val": 5e8, "form": "10-Q", "filed": "2025-07-30"}
    ]
    base = "https://data.sec.gov/api/xbrl/companyconcept/CIK0000896878/us-gaap/"
    respx.get(base + "Revenues.json").mock(return_value=httpx.Response(200, json=short))
    respx.get(base + "RevenueFromContractWithCustomerExcludingAssessedTax.json").mock(
        return_value=httpx.Response(200, json=long)
    )
    respx.get(base + "SalesRevenueNet.json").mock(return_value=httpx.Response(404))
    for cik, name in {cik: name for g in CYCLE_COMPANIES.values() for _, cik, name in g}.items():
        respx.get(url__startswith=f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/").mock(
            return_value=httpx.Response(200, json=_concept(name))
        )
    cache: dict = {}
    async with httpx.AsyncClient() as client:
        await fetch_filings(client, cache, datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert cache["companies"]["INTU"]["revenue"]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


@respx.mock
async def test_sec_403_explains_the_fix() -> None:
    respx.get(url__startswith="https://data.sec.gov/").mock(return_value=httpx.Response(403))
    cache: dict = {}
    async with httpx.AsyncClient() as client:
        notes = await fetch_filings(client, cache, datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert any("SEC_USER_AGENT" in n for n in notes)
    cycle = build_cycle({**cache, "notes": notes}, {}, None)
    assert not cycle.known and any("SEC_USER_AGENT" in n for n in cycle.notes)
