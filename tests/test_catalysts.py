"""'How we go up': semiconductor earnings vs. valuation, software, monetary easing expectations.

Anchored on the real filings fixture (SEC data fetched 2026-10-14). Chip prices are synthetic
quarter-end closes; every displayed number is checked against the rows it came from.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from app.config import THRESHOLDS, VALUATION
from app.features import gap_sessions_ago
from app.fundamentals import (
    RateInputs,
    build_cycle,
    earnings_vs_price,
    eps_reconciliation,
    implied_multiple_change,
    needed_for,
    quarterize,
    valuation_band,
    valuation_grade,
)
from app.models import Catalyst
from tests.sec_fixture import sec_cache

Q = pd.Period
IDX = pd.bdate_range("2023-01-02", "2026-10-30")


def step_prices(growth_pct: float, before: str = "2025-06-30") -> pd.Series:
    """100 through `before`, then 100 * (1 + growth)."""
    s = pd.Series(100.0, index=IDX)
    s[s.index > before] = 100.0 * (1 + growth_pct / 100.0)
    return s


# Price growth on the live page for the year to Q2 2026. In the fixture (filings to Oct 14, 2026)
# Micron's latest filing is already Q3 2026 (quarter ended Sep 3), so its step sits at Sep 30, 2025,
# sized to keep its implied multiple at +17.9% on its Q3 trailing EPS (7.60 -> 74.27).
LIVE_PRICES = {
    "NVDA": step_prices(27),
    "AVGO": step_prices(37),
    "AMD": step_prices(309),
    "MU": step_prices((74.27 / 7.60) * 1.179 * 100 - 100, "2025-09-30"),
}


# The same, all pinned to the year to Q2 2026 (for rows read at a fixed quarter).
Q2_PRICES = {"NVDA": step_prices(27), "AVGO": step_prices(37), "AMD": step_prices(309), "MU": step_prices(837)}


def chip_eps() -> dict[str, pd.Series]:
    return {t: quarterize(sec_cache()["companies"][t]["eps"]["facts"]) for t in ("NVDA", "AVGO", "AMD", "MU")}


def cat(cycle_catalysts: list[Catalyst], key: str) -> Catalyst:
    return {c.key: c for c in cycle_catalysts}[key]


def quarterly_eps(values: list[float], first_end: str = "2024-03-31") -> pd.Series:
    """Calendar-quarter EPS series (quarter ends), for synthetic cases."""
    ends = pd.date_range(first_end, periods=len(values), freq="QE")
    return pd.Series(values, index=ends, dtype=float)


# --- the formula ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("eps", "price", "expected"),
    [
        (124, 309, 82.6),  # AMD (spec example)
        (695, 837, 17.9),  # MU
        (125, 27, -43.6),  # NVDA
        (100, 37, -31.5),  # AVGO
        (50, 50, 0.0),  # stable
        (-20, -50, -37.5),  # both fell; price fell harder
        (50, -10, -40.0),  # earnings up, price down
        (-50, 10, 120.0),  # earnings down, price up
    ],
)
def test_implied_multiple_change(eps: float, price: float, expected: float) -> None:
    assert implied_multiple_change(eps, price) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize(("eps", "price"), [(-100, 10), (-150, 10), (10, -100), (None, 10), (10, None)])
def test_implied_multiple_change_undefined(eps: float | None, price: float | None) -> None:
    """EPS at or below zero (or missing data) has no P/E to compare."""
    assert implied_multiple_change(eps, price) is None


def test_bands_and_their_edges() -> None:
    assert valuation_band(-43.6) == "earnings" and valuation_band(VALUATION.contraction_pct) == "earnings"
    assert valuation_band(-9.9) == "stable" and valuation_band(VALUATION.expansion_pct) == "stable"
    assert valuation_band(10.1) == "expansion" and valuation_band(17.9) == "expansion"
    assert valuation_band(VALUATION.significant_pct) == "significant" and valuation_band(82.6) == "significant"
    assert valuation_band(None) == "unknown"


def test_group_score_grades() -> None:
    grades = [valuation_grade(k, 0, 4) for k in range(5)]
    assert grades == ["weak", "weak", "mixed", "confirming", "strong"]
    assert needed_for("confirming", 4) == 3


def test_unreadable_company_is_unknown_not_false() -> None:
    assert valuation_grade(2, 1, 4) == "unknown"  # 2 (mixed) or 3 (confirming): can't tell
    assert valuation_grade(0, 1, 4) == "weak"  # 0 or 1: weak either way
    assert valuation_grade(3, 1, 4) == "unknown"  # confirming or strong
    assert valuation_grade(0, 0, 0) == "unknown"


# --- rows ------------------------------------------------------------------------------------------


def test_rows_reconcile_to_eps_and_prices() -> None:
    rows = earnings_vs_price(chip_eps(), Q2_PRICES, Q("2026Q2"), ["NVDA", "AVGO", "AMD", "MU"])
    by = {r["ticker"]: r for r in rows}
    assert by["NVDA"]["eps_then"] == pytest.approx(3.52, abs=0.005) and by["NVDA"]["eps_now"] == pytest.approx(7.91)
    for r in rows:
        assert r["eps_growth"] == pytest.approx((r["eps_now"] / r["eps_then"] - 1) * 100, abs=1e-3)
        assert r["price_growth"] == pytest.approx((r["price_now"] / r["price_then"] - 1) * 100, abs=1e-3)
        assert r["multiple_change"] == pytest.approx(
            implied_multiple_change(r["eps_growth"], r["price_growth"]), abs=1e-3
        )
        assert r["outgrowing"] is (r["eps_growth"] > r["price_growth"])
        assert r["price_date"] == "2026-06-30" and r["price_date_then"] == "2025-06-30"
    assert [by[t]["band"] for t in ("NVDA", "AVGO", "AMD", "MU")] == [
        "earnings",
        "earnings",
        "significant",
        "expansion",
    ]
    assert by["MU"]["low_base"] is True and not any(by[t]["low_base"] for t in ("NVDA", "AVGO", "AMD"))
    assert by["NVDA"]["fiscal_end"] == "2026-07-26" and by["MU"]["fiscal_end"] == "2026-05-28"


def test_negative_base_eps_is_excluded_with_its_reason() -> None:
    """AMD's trailing EPS a year before Q2 2024 was negative (-0.03): growth isn't defined."""
    rows = earnings_vs_price(chip_eps(), {"AMD": step_prices(20, "2023-06-30")}, Q("2024Q2"), ["AMD"])
    assert rows[0]["outgrowing"] is None and "zero or negative" in rows[0]["reason"]
    assert rows[0]["multiple_change"] is None and rows[0]["band"] == "unknown"


def test_eps_turning_negative_counts_against_but_has_no_multiple() -> None:
    eps = {"X": quarterly_eps([1, 1, 1, 1, -1, -1, -1, -1])}  # TTM 4 -> -4 (Q4 2024 -> Q4 2025)
    prices = {"X": step_prices(10, "2024-12-31")}
    row = earnings_vs_price(eps, prices, Q("2025Q4"))[0]
    assert row["eps_growth"] == pytest.approx(-200.0) and row["outgrowing"] is False
    assert row["multiple_change"] is None and "turned negative" in row["reason"]


def test_zero_base_eps_is_excluded() -> None:
    eps = {"X": quarterly_eps([0, 0, 0, 0, 1, 1, 1, 1])}
    row = earnings_vs_price(eps, {"X": step_prices(10, "2024-12-31")}, Q("2025Q4"))[0]
    assert row["outgrowing"] is None and "zero or negative" in row["reason"]


def test_missing_company_stays_in_the_group() -> None:
    eps = chip_eps()
    del eps["AVGO"]
    rows = earnings_vs_price(eps, LIVE_PRICES, Q("2026Q2"), ["NVDA", "AVGO", "AMD", "MU"])
    assert len(rows) == 4 and rows[1]["outgrowing"] is None and rows[1]["reason"] == "no EPS filings read"


def test_missing_or_stale_price_is_not_compared() -> None:
    stale = LIVE_PRICES["NVDA"][LIVE_PRICES["NVDA"].index <= "2026-06-15"]  # stops 15 days before quarter end
    rows = earnings_vs_price(chip_eps(), {"NVDA": stale}, Q("2026Q2"), ["NVDA", "AVGO"])
    assert rows[0]["outgrowing"] is None and "within 7 days of Jun 30, 2026" in rows[0]["reason"]
    assert rows[1]["outgrowing"] is None and "close" in rows[1]["reason"]


def test_inconsistent_fiscal_periods_are_flagged() -> None:
    """A company that moved its fiscal year end: its quarters mapped to Q4 2024 and Q4 2025 end
    Nov 30, 2024 and Dec 31, 2025, 396 days apart, so the two trailing years don't compare."""
    ends = pd.to_datetime(
        ["2024-02-29", "2024-05-31", "2024-08-31", "2024-11-30", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    )
    eps = {"X": pd.Series(1.0, index=ends)}
    row = earnings_vs_price(eps, {"X": step_prices(10, "2024-12-31")}, Q("2025Q4"))[0]
    assert row["outgrowing"] is None and "aren't a year apart" in row["reason"]
    assert row["fiscal_end_then"] == "2024-11-30" and row["fiscal_end"] == "2025-12-31"


# --- the catalyst card -------------------------------------------------------------------------------


def test_semiconductor_card_matches_the_live_read() -> None:
    c = cat(build_cycle(sec_cache(), LIVE_PRICES, None).catalysts, "earnings")
    assert c.label == "Semiconductor earnings vs. valuation"
    assert c.status == "mixed" and c.status_text == "Mixed" and c.headline == "2 of 4"
    assert c.observation and c.observation.startswith("2 of 4 semiconductor companies")
    assert c.interpretation and "remains significant in parts of the group (AMD +82.9%)" in c.interpretation
    assert "MU's EPS rose 9.8x from a low base" in c.interpretation
    assert "different quarters" in c.interpretation
    assert c.confirmation and c.confirmation.startswith("At least 3 of 4") and "Closest: MU" in c.confirmation
    assert "files its next 10-Q or 10-K" in c.confirmation
    assert [r["quarter"] for r in c.rows] == ["Q2 2026", "Q2 2026", "Q2 2026", "Q3 2026"]
    labels = [m.label for m in c.methodology]
    assert {"Measured", "Latest filings", "EPS", "Price", "Checks", "Formula", "Limits", "Updated"} <= set(labels)
    eps_note = next(m.value for m in c.methodology if m.label == "EPS")
    assert "GAAP diluted" in eps_note and "No adjusted EPS" in eps_note


def test_every_displayed_number_reconciles_to_its_row() -> None:
    c = cat(build_cycle(sec_cache(), LIVE_PRICES, None).catalysts, "earnings")
    assert [b.label for b in c.bars] == [r["ticker"] for r in c.rows]
    for bar, row in zip(c.bars, c.rows, strict=True):
        assert bar.value == row["multiple_change"]
        assert bar.text == f"{row['multiple_change']:+.1f}%"
        assert bar.sub == f"EPS {row['eps_growth']:+.0f}% · price {row['price_growth']:+.0f}%"
        assert bar.tone == row["band"]
    led = sum(1 for r in c.rows if r["outgrowing"])
    assert c.headline == f"{led} of {len(c.rows)}"
    first = c.rows[0]
    assert f"{first['ticker']} ({first['quarter']}) EPS {first['eps_growth']:+.0f}%" in c.detail


def test_card_turns_unknown_when_a_company_cannot_be_read() -> None:
    prices = {t: s for t, s in LIVE_PRICES.items() if t != "MU"}
    c = cat(build_cycle(sec_cache(), prices, None).catalysts, "earnings")
    assert c.status == "unknown" and c.headline == "2 of 4" and "1 not comparable" in (c.headline_note or "")
    assert c.observation and "MU can't be compared" in c.observation
    assert c.confirmation and "MU must also become comparable" in c.confirmation


def test_strong_group_reads_strong_and_names_the_edge() -> None:
    prices = {t: step_prices(10) for t in ("NVDA", "AVGO", "AMD", "MU")}
    c = cat(build_cycle(sec_cache(), prices, None).catalysts, "earnings")
    assert c.status == "strong" and c.confirmation and c.confirmation.startswith("Met: 4 of 4")
    assert "Closest to flipping" in c.confirmation


def test_no_prices_at_all_is_unknown() -> None:
    c = cat(build_cycle(sec_cache(), {}, None).catalysts, "earnings")
    assert c.status == "unknown" and c.detail == "Not enough earnings or price history."


def test_software_card_separates_company_and_group() -> None:
    c = cat(build_cycle(sec_cache(), {}, None).catalysts, "software")
    up = sum(1 for r in c.rows if r["change"] is not None and r["change"] > 0)
    assert f"{up} of {len(c.rows)} accelerating" in (c.headline_note or "")
    assert c.observation and c.observation.startswith("Group: the median company's revenue growth")
    assert "Companies:" in c.observation
    assert c.confirmation and c.confirmation.startswith("All three")
    for bar, row in zip(c.bars, c.rows, strict=True):
        assert bar.value == pytest.approx(row["change"], abs=1e-3)
        assert bar.tone == ("up" if row["change"] > 0 else "down")


# --- monetary easing ---------------------------------------------------------------------------------

RATES = RateInputs(
    effr=3.88,
    effr_date="2026-11-19",
    points=(
        ("3M", 4.32, "2026-11-20"),
        ("6M", 4.40, "2026-11-20"),
        ("1Y", 4.50, "2026-11-20"),
        ("2Y", 4.63, "2026-11-20"),
    ),
    gap_2y_prev=-0.55,
    prev_date="2026-10-22",
)


def test_rates_card_shows_the_path_and_keeps_policy_separate() -> None:
    c = cat(build_cycle(sec_cache(), {}, -0.44, -0.75, RATES).catalysts, "cuts")
    assert c.status == "not yet" and c.status_text == "Hikes priced" and c.headline == "+0.75 pts"
    assert [b.label for b in c.bars] == ["3M", "6M", "1Y", "2Y"]
    for bar, (_, y, _) in zip(c.bars, RATES.points, strict=True):
        assert bar.value == pytest.approx(y - 3.88) and bar.text == f"{y - 3.88:+.2f}"
    assert c.observation and "Oct 22, 2026" in c.observation and "+0.55" in c.observation
    assert c.interpretation and "less easing" in c.interpretation and "not policy" in c.interpretation
    assert c.confirmation and "Policy side: an FOMC cut" in c.confirmation
    assert any(m.label == "Not read" and "futures" in m.value for m in c.methodology)


def test_rates_card_without_path_still_reads_the_two_year() -> None:
    c = cat(build_cycle(sec_cache(), {}, effr_minus_bill=0.40, effr_minus_2y=0.60).catalysts, "cuts")
    assert c.status == "yes" and c.status_text == "Cuts priced" and c.bars == []


def test_rates_card_falls_back_to_the_bill() -> None:
    c = cat(build_cycle(sec_cache(), {}, effr_minus_bill=-0.38).catalysts, "cuts")
    assert (
        c.status == "not yet" and c.status_text == "Hikes priced" and "2-year yield is missing" in (c.observation or "")
    )


def test_gap_sessions_ago_uses_effr_as_of_that_day() -> None:
    days = pd.bdate_range("2026-09-01", periods=30)
    y2 = pd.Series([4.0 + 0.01 * i for i in range(30)], index=days)
    effr = pd.Series([4.10, 3.88], index=[days[0], days[20]])  # a cut takes effect on day 20
    got = gap_sessions_ago(effr, y2, THRESHOLDS.cuts_change_sessions)
    assert got is not None and got[0] == str(days[9].date()) and got[1] == pytest.approx(4.10 - 4.09)
    assert gap_sessions_ago(effr, y2.iloc[:5], 20) is None


# --- each company through its own latest filing (Nov 2026) ------------------------------------------


def eps_facts(
    quarters: list[tuple[str, str, float]], filed: dict[str, str], form: str = "10-Q"
) -> list[dict[str, Any]]:
    return [
        {"start": start, "end": end, "val": val, "form": form, "filed": filed.get(end, "2099-01-01")}
        for start, end, val in quarters
    ]


QUARTERS = [  # calendar quarters Q3 2024 .. Q3 2026, EPS 1.00 a quarter, then 2.00
    ("2024-07-01", "2024-09-30", 1.0),
    ("2024-10-01", "2024-12-31", 1.0),
    ("2025-01-01", "2025-03-31", 1.0),
    ("2025-04-01", "2025-06-30", 1.0),
    ("2025-07-01", "2025-09-30", 1.0),
    ("2025-10-01", "2025-12-31", 2.0),
    ("2026-01-01", "2026-03-31", 2.0),
    ("2026-04-01", "2026-06-30", 2.0),
    ("2026-07-01", "2026-09-30", 2.0),
]


def test_each_company_reads_its_own_latest_quarter_with_its_filing_date() -> None:
    raw = eps_facts(QUARTERS, {"2026-09-30": "2026-11-04", "2026-06-30": "2026-08-05"})
    newer = quarterize(raw)
    older = quarterize(raw[:-1])  # this company hasn't filed Q3 yet
    rows = earnings_vs_price(
        {"NEW": newer, "OLD": older},
        {"NEW": step_prices(50, "2025-09-30"), "OLD": step_prices(50, "2025-06-30")},
        None,
        ["NEW", "OLD"],
        facts={"NEW": {"eps": raw}, "OLD": {"eps": raw[:-1]}},
    )
    new, old = rows
    assert new["quarter"] == "Q3 2026" and old["quarter"] == "Q2 2026"
    assert new["asof"] == "As of 11/4/2026 earnings (10-Q, quarter ended 9/30/2026)"
    assert old["asof"] == "As of 8/5/2026 earnings (10-Q, quarter ended 6/30/2026)"
    assert new["eps_growth"] == pytest.approx(100.0) and new["price_growth"] == pytest.approx(50.0)
    assert old["eps_growth"] == pytest.approx((7 / 4 - 1) * 100)  # TTM to Q2 2026: 1+2+2+2 vs 4


def test_filing_date_is_the_first_filing_not_a_later_restatement() -> None:
    raw = eps_facts(QUARTERS, {"2026-09-30": "2026-11-04"})
    raw.append({**raw[-1], "filed": "2027-02-20", "form": "10-K"})  # re-reported in the next annual report
    row = earnings_vs_price({"X": quarterize(raw)}, {"X": step_prices(50, "2025-09-30")}, facts={"X": {"eps": raw}})[0]
    assert row["filed"] == "2026-11-04" and "11/4/2026" in row["asof"]


def _consistent(eps: float, shares: float, start: str, end: str) -> tuple[dict[str, Any], dict[str, Any]]:
    base = {"start": start, "end": end, "form": "10-Q", "filed": "2026-11-04"}
    return {**base, "val": eps * shares}, {**base, "val": shares}


def test_eps_reconciles_with_net_income_and_shares() -> None:
    raw = eps_facts(QUARTERS, {})
    income, shares = zip(*(_consistent(v, 1e9, s, e) for s, e, v in QUARTERS), strict=True)
    status, n, note = eps_reconciliation(
        raw, list(income), list(shares), pd.Timestamp("2024-12-31"), pd.Timestamp("2026-09-30")
    )
    assert status == "reconciled" and n == 8 and note is None  # Q4 2024 on: the two trailing years


def test_eps_that_does_not_reconcile_withholds_the_company() -> None:
    raw = eps_facts(QUARTERS, {})
    income, shares = (list(x) for x in zip(*(_consistent(v, 1e9, s, e) for s, e, v in QUARTERS), strict=True))
    income[-1] = {**income[-1], "val": 1.0e9}  # filing says $1B net income, but EPS 2.00 x 1B shares = $2B
    facts = {"X": {"eps": raw, "net_income": income, "diluted_shares": shares}}
    row = earnings_vs_price({"X": quarterize(raw)}, {"X": step_prices(50, "2025-09-30")}, facts=facts)[0]
    assert row["eps_check"] == "failed" and row["outgrowing"] is None
    assert "doesn't reconcile" in row["reason"] and "$2.00B vs net income $1.00B" in row["reason"]


def test_cent_rounding_on_small_eps_is_not_a_failure() -> None:
    """EPS is rounded to the cent: 0.03 x 1.6B shares = $48M against $44M reported is rounding."""
    raw = eps_facts([("2026-07-01", "2026-09-30", 0.03)], {})
    income = [{**raw[0], "val": 44e6}]
    shares = [{**raw[0], "val": 1.6e9}]
    status, n, _ = eps_reconciliation(raw, income, shares, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))
    assert status == "reconciled" and n == 1


def test_missing_net_income_or_shares_is_unchecked_not_failed() -> None:
    raw = eps_facts(QUARTERS, {})
    status, n, _ = eps_reconciliation(raw, [], None, pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31"))
    assert status == "unchecked" and n == 0


def test_price_check_against_yahoo() -> None:
    eps = {"X": quarterize(eps_facts(QUARTERS, {}))}
    alpaca = step_prices(50, "2025-09-30")
    agree = alpaca * 1.001
    row = earnings_vs_price(eps, {"X": alpaca}, price_checks={"X": agree})[0]
    assert row["price_check"] == "verified" and row["outgrowing"] is not None

    differ = alpaca.copy()
    differ[differ.index == "2026-09-30"] = 160.0  # Alpaca 150
    row = earnings_vs_price(eps, {"X": alpaca}, price_checks={"X": differ})[0]
    assert row["price_check"] == "conflict" and row["outgrowing"] is None
    assert "Alpaca 150.00 vs Yahoo 160.00 on Sep 30, 2026" in row["reason"]

    gap = alpaca[alpaca.index != "2026-09-30"]
    row = earnings_vs_price(eps, {"X": alpaca}, price_checks={"X": gap})[0]
    assert row["price_check"] == "official" and row["outgrowing"] is not None  # used, labelled unconfirmed
    assert "Yahoo has no close for Sep 30, 2026" in row["price_check_note"]


def test_software_rows_each_on_their_own_latest_filing() -> None:
    from app.fundamentals import software_latest

    facts = sec_cache()["companies"]
    members = {t: quarterize(facts[t]["revenue"]["facts"]) for t in ("CRM", "ADBE")}
    rows = software_latest(members, ["CRM", "ADBE", "WDAY"])
    by = {r["ticker"]: r for r in rows}
    assert by["CRM"]["quarter"] == "Q2 2026" and by["ADBE"]["quarter"] == "Q3 2026"
    assert by["ADBE"]["change"] == pytest.approx(by["ADBE"]["growth"] - by["ADBE"]["growth_prev"], abs=1e-3)
    assert by["WDAY"]["reason"] == "no revenue filings read"
