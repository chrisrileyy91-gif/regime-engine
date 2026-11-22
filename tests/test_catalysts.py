"""'How we go up': semiconductor earnings vs. valuation, software, monetary easing expectations.

Anchored on the real filings fixture (SEC data fetched 2026-10-14). Chip prices are synthetic
quarter-end closes; every displayed number is checked against the rows it came from.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config import THRESHOLDS, VALUATION
from app.features import gap_sessions_ago
from app.fundamentals import (
    RateInputs,
    build_cycle,
    earnings_vs_price,
    implied_multiple_change,
    needed_for,
    quarterize,
    valuation_band,
    valuation_grade,
)
from app.models import Catalyst
from tests.sec_fixture import sec_cache

Q = pd.Period
IDX = pd.bdate_range("2023-01-02", "2026-07-31")


def step_prices(growth_pct: float, before: str = "2025-06-30") -> pd.Series:
    """100 through `before`, then 100 * (1 + growth)."""
    s = pd.Series(100.0, index=IDX)
    s[s.index > before] = 100.0 * (1 + growth_pct / 100.0)
    return s


# Price growth on the live page, year to Q2 2026.
LIVE_PRICES = {"NVDA": step_prices(27), "AVGO": step_prices(37), "AMD": step_prices(309), "MU": step_prices(837)}


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
    rows = earnings_vs_price(chip_eps(), LIVE_PRICES, Q("2026Q2"), ["NVDA", "AVGO", "AMD", "MU"])
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
    assert "MU's EPS rose 7.9x from a low base" in c.interpretation
    assert c.confirmation and c.confirmation.startswith("At least 3 of 4") and "Closest: MU" in c.confirmation
    assert "Q3 2026" in c.confirmation
    labels = [m.label for m in c.methodology]
    assert {"Measured", "Latest fiscal quarter", "EPS", "Price", "Formula", "Limits", "Updated"} <= set(labels)
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
    assert f"{c.rows[0]['ticker']} EPS {c.rows[0]['eps_growth']:+.0f}%" in c.detail


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
    assert c.confirmation and "Monetization needs all three" in c.confirmation
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
