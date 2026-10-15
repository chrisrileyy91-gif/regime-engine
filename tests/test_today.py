"""Today's stress, the scapegoat tally and the overnight line."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.drivers import build_drivers
from app.today import overnight, stress, tally
from app.verify import verify_all
from tests.conftest import TODAY, market, series
from tests.test_regime import calm


def level(**overrides: object) -> str:
    f = calm(**overrides)
    return stress(f, build_drivers(f, {})).level


def test_calm_market_is_calm() -> None:
    assert level() == "CALM"


def test_high_yield_level_alone_is_not_stress() -> None:
    """The old price rules showed HIGH RISK for months because the 10Y sat at a high level."""
    assert level(tnx_pct=5.22, tnx_pctile=96.0, tnx_chg_1d_bps=-2.0, tnx_delta_z=-0.5) == "CALM"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"tnx_chg_1d_bps": 12.0}, "ELEVATED"),
        ({"vix": 22.0}, "ELEVATED"),
        ({"soxx_rsi": 35.0}, "ELEVATED"),
        ({"vix": 22.0, "soxx_rsi": 35.0}, "STRESSED"),
        ({"vix": 31.0, "soxx_rsi": 50.0}, "STRESSED"),
    ],
)
def test_stress_levels(overrides: dict[str, object], expected: str) -> None:
    assert level(**overrides) == expected


def test_unknown_inputs_are_named() -> None:
    f = calm(vix=None)
    s = stress(f, build_drivers(f, {}))
    assert s.level == "CALM" and any("couldn't check" in r for r in s.reasons)
    f = calm(vix=None, tnx_chg_1d_bps=None, tnx_delta_z=None, soxx_rsi=None)
    assert stress(f, build_drivers(f, {})).level == "UNKNOWN"


N = 300
EVENTS = {  # index from the end -> what happens that day (each day semis fall 3%)
    -80: "rates_close",  # 10Y closes +15 bps
    -70: "rates_intraday",  # CPI-day pattern: 10Y +12 bps at the high, closes -3
    -62: "rates_2y",  # Aug 28 pattern: 2Y +14 bps on hike odds, 10Y/30Y flat
    -60: "rates_30y",  # 30Y closes +12 bps, 10Y flat
    -57: "oil_big",  # Sep 1 pattern: USO +5% in a volatile tape (well under 2 sigma)
    -55: "oil",  # USO +8% on a quiet tape
    -52: "dollar",  # UUP +1.5%
    -50: "fear",  # VIX +27%
    -40: "market",  # equal-weight S&P -2%
    -30: "rotation",  # nothing macro, software +2%
    -20: "chips",  # nothing macro, software -1%
}


def _tally_inputs(newest_drop: bool = False) -> tuple[dict, dict[str, pd.Series]]:
    soxx, igv, rsp, vix = [100.0] * N, [50.0] * N, [150.0] * N, [15.0] * N
    rng = np.random.default_rng(3)
    uso = list(80.0 * np.cumprod(1 + rng.normal(0, 0.01, N)))  # ~1% daily moves
    # War-regime oil over the last 90 sessions: +-3.9% every day (sigma ~3.9%), so +5% is ~1.3 sigma.
    uso[-90:] = list(uso[-91] * np.cumprod([1.039 if k % 2 else 1 / 1.039 for k in range(90)]))
    uup = list(28.0 * np.cumprod(1 + rng.normal(0, 0.002, N)))  # ~0.2% daily moves
    y10 = [4.0] * N
    y30 = [4.5] * N
    y10_hi = [4.01] * N
    y30_hi = [4.51] * N
    y2 = [4.2] * N
    events = dict(EVENTS)
    if newest_drop:
        events[-1] = "chips"
    for i, kind in events.items():
        soxx[i:] = [v * 0.97 for v in soxx[i:]]
        igv[i:] = [v * (1.02 if kind == "rotation" else 0.99) for v in igv[i:]]
        if kind == "rates_close":
            y10[i:] = [v + 0.15 for v in y10[i:]]
            y10_hi[i:] = [v + 0.15 for v in y10_hi[i:]]
        if kind == "rates_intraday":
            y10_hi[i] = y10[i - 1] + 0.12
            y10[i:] = [v - 0.03 for v in y10[i:]]
            y10_hi[i + 1 :] = [v - 0.03 for v in y10_hi[i + 1 :]]
        if kind == "rates_30y":
            y30[i:] = [v + 0.12 for v in y30[i:]]
            y30_hi[i:] = [v + 0.12 for v in y30_hi[i:]]
        if kind == "fear":
            vix[i] = 19.0
        if kind == "oil":
            uso[i:] = [v * 1.08 for v in uso[i:]]
        if kind == "oil_big":
            ratio = uso[i - 1] * 1.05 / uso[i]
            uso[i:] = [v * ratio for v in uso[i:]]
        if kind == "rates_2y":
            y2[i:] = [v + 0.14 for v in y2[i:]]
        if kind == "dollar":
            uup[i:] = [v * 1.015 for v in uup[i:]]
        if kind == "market":
            rsp[i:] = [v * 0.98 for v in rsp[i:]]
    results = market(soxx=series(soxx), igv=series(igv), y10=series(y10), vix=series(vix))
    extras = {
        "market": series(rsp),
        "y10_high": series(y10_hi),
        "y30": series(y30),
        "y30_high": series(y30_hi),
        "oil": series(uso),
        "dollar": series(uup),
        "y2": series(y2),
    }
    return results, extras


def test_tally_classifies_each_drop_by_what_moved_that_day() -> None:
    results, extras = _tally_inputs()
    t = tally(verify_all(results, TODAY.date()), extras)
    assert [d.driver for d in t.days] == [
        "chips",
        "rotation",
        "market",
        "fear",
        "dollar",
        "oil",
        "oil",
        "rates",
        "rates",
        "rates",
        "rates",
    ]
    assert t.counts == {"rates": 4, "oil": 2, "dollar": 1, "fear": 1, "market": 1, "rotation": 1, "chips": 1}


def test_missing_oil_means_nothing_below_it_can_claim_the_day() -> None:
    results, extras = _tally_inputs()
    del extras["oil"]
    drivers = [d.driver for d in tally(verify_all(results, TODAY.date()), extras).days]
    assert drivers.count("rates") == 4 and set(drivers) == {"rates", "unknown"}


def test_headlines_annotate_and_flag_disagreement_without_changing_the_gauge() -> None:
    results, extras = _tally_inputs()
    days = tally(verify_all(results, TODAY.date()), extras).days
    chips_day, rates_day = days[0].date, days[-1].date
    news = {
        chips_day: {"category": "fed", "summary": "Fed speaker hints at another hike.", "evidence": []},
        rates_day: {"category": "ai_demand", "summary": "Hyperscaler cuts capex guide.", "evidence": []},
    }
    t = tally(verify_all(results, TODAY.date()), extras, news)
    assert t.days[0].driver == "chips" and "Headlines point to the Fed" in (t.days[0].news_flag or "")
    assert t.days[-1].driver == "rates" and "headlines point to AI-demand worries" in (t.days[-1].news_flag or "")
    assert t.news_counts == {"fed": 1, "unlabelled": 1}
    assert "disagree" in t.summary


def test_intraday_yield_spike_that_reversed_counts_as_rates() -> None:
    """CPI day, Oct 14: the 10Y rose 11 bps intraday and closed down 3. Close-to-close misses it."""
    results, extras = _tally_inputs()
    day = next(
        d for d in tally(verify_all(results, TODAY.date()), extras).days if d.tnx_bps is not None and d.tnx_bps < 0
    )
    assert day.driver == "rates" and day.tnx_spike_bps == pytest.approx(12.0, abs=0.5)
    assert "at its high" in day.detail


def test_without_intraday_highs_a_close_only_day_cannot_be_called_chip_specific() -> None:
    results, extras = _tally_inputs()
    del extras["y10_high"]
    drivers = [d.driver for d in tally(verify_all(results, TODAY.date()), extras).days]
    assert "chips" not in drivers and "rotation" not in drivers and "unknown" in drivers


def test_newest_day_is_provisional_and_not_counted() -> None:
    results, extras = _tally_inputs(newest_drop=True)
    extras["y30"] = extras["y30"].iloc[:-1]  # the 30Y close hasn't posted yet
    extras["y30_high"] = extras["y30_high"].iloc[:-1]
    t = tally(verify_all(results, TODAY.date()), extras)
    assert t.days[0].provisional and t.days[0].driver == "unknown"
    assert sum(t.counts.values()) == len(t.days) - 1 and "provisional" in t.summary


def test_summary_never_claims_there_was_no_macro_news() -> None:
    results, extras = _tally_inputs()
    text = tally(verify_all(results, TODAY.date()), extras).summary
    assert "no macro trigger" not in text.lower()


def test_without_the_market_series_nothing_is_called_chip_specific() -> None:
    results, extras = _tally_inputs()
    del extras["market"]
    drivers = {d.driver for d in tally(verify_all(results, TODAY.date()), extras).days}
    assert not drivers & {"chips", "rotation", "market"}


def test_overnight_line() -> None:
    results = market(kospi=series([2500.0, 2550.0]), samsung=series([70000.0, 69300.0]))
    items = {i.label: i for i in overnight(verify_all(results, TODAY.date()))}
    assert items["KOSPI"].change_pct == pytest.approx(2.0) and items["Samsung"].change_pct == pytest.approx(-1.0)
    assert items["KOSPI"].date == str(TODAY.date())


def test_overnight_missing_input_is_blank() -> None:
    results = market()
    results = {k: v for k, v in results.items() if not k.startswith("dxy@")}
    items = {i.label: i for i in overnight(verify_all(results, TODAY.date()))}
    assert items["Dollar index"].value is None and items["Dollar index"].date is None


def test_a_failed_headline_read_is_not_shown_as_a_label() -> None:
    results, extras = _tally_inputs()
    first = tally(verify_all(results, TODAY.date()), extras).days[0].date
    t = tally(verify_all(results, TODAY.date()), extras, {first: {"failed": 1, "last_try": "x", "problem": "y"}})
    assert t.days[0].news is None and t.news_counts == {"unlabelled": 2}
