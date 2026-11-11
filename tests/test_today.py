"""Today's stress, the scapegoat tally and the overnight line."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.drivers import build_drivers
from app.today import decompose, overnight, sensitivity, stress, tally
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
        ({"vix_chg_1d_pct": 14.0}, "ELEVATED"),
        ({"soxx_return_z": -2.1}, "ELEVATED"),
        ({"soxx_chg_1d_pct": -4.5}, "ELEVATED"),  # the floor: a big drop even when the z has adapted
        ({"vix_chg_1d_pct": 14.0, "soxx_return_z": -2.1}, "STRESSED"),
        ({"vix": 31.0}, "STRESSED"),
        # Levels are backdrop. VIX above 20 with RSI below 40 used to read Stressed on a flat day
        # (Nov 10, 2026: semis -0.1%); across 39 reads the label was just "RSI < 40".
        ({"vix": 22.0, "soxx_rsi": 35.0}, "CALM"),
    ],
)
def test_stress_levels(overrides: dict[str, object], expected: str) -> None:
    assert level(**overrides) == expected


def test_unknown_inputs_are_named() -> None:
    """Nothing active is only Calm when every driver was read; one unknown makes it Unknown."""
    f = calm(vix_chg_1d_pct=None)
    s = stress(f, build_drivers(f, {}))
    assert s.level == "UNKNOWN" and any("couldn't check" in r for r in s.reasons)
    f = calm(vix_chg_1d_pct=None, tnx_chg_1d_bps=None, tnx_delta_z=None, soxx_chg_1d_pct=None, soxx_return_z=None)
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
    -50: "fear",  # VIX +27% and nothing else: VIX can't claim a day, so this reads as chips
    -40: "market",  # equal-weight S&P -2%: at SOXX's usual beta that explains the whole drop
    -30: "rotation",  # nothing macro, software +2%
    -20: "chips",  # nothing macro, software -1%
}
BETA = 1.5  # SOXX's usual move per 1% in the equal-weight S&P 500


def _tally_inputs(newest_drop: bool = False) -> tuple[dict, dict[str, pd.Series]]:
    igv, vix = [50.0] * N, [15.0] * N
    rng = np.random.default_rng(3)
    daily = rng.normal(0, 0.003, N)  # the broad market's ordinary days; SOXX follows at BETA
    rsp = list(150.0 * np.cumprod(1 + daily))
    soxx = list(100.0 * np.cumprod(1 + BETA * daily))
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
    hyg = [80.0] * N
    hyg[-20:] = [v * 0.99 for v in hyg[-20:]]  # high-yield bonds -1% on the last chips day
    crwv = [100.0] * N
    crwv[-20:] = [v * 0.90 for v in crwv[-20:]]  # and a debt-funded AI name -10%
    extras = {
        "credit": series(hyg),
        "lev_crwv": series(crwv),
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
        "chips",
        "dollar",
        "oil",
        "oil",
        "rates",
        "rates",
        "rates",
        "rates",
    ]
    assert t.counts == {"rates": 4, "oil": 2, "dollar": 1, "market": 1, "rotation": 1, "chips": 2}


def test_explained_share_separates_a_market_day_from_a_chip_day() -> None:
    """Nov 5, 2026 was tagged "VIX spiked"; the question is how much of -12% the market explains."""
    results, extras = _tally_inputs()
    t = tally(verify_all(results, TODAY.date()), extras)
    by = {d.driver: d for d in t.days}
    assert by["market"].macro_share is not None and by["market"].macro_share > 0.8
    assert by["market"].expected_pct == pytest.approx(-3.0, abs=0.6)
    assert by["rotation"].macro_share is not None and by["rotation"].macro_share < 0.4
    assert "expected -" in by["market"].detail and t.avg_macro_share is not None
    assert "On average the broad market and yields explain" in t.summary


def test_a_vix_spike_never_claims_a_day_but_stays_in_the_detail() -> None:
    results, extras = _tally_inputs()
    day = next(d for d in tally(verify_all(results, TODAY.date()), extras).days if (d.vix_pct or 0) > 20)
    assert day.driver == "chips" and "fear" not in day.moved and "VIX +27%" in day.detail


def test_falling_oil_is_not_a_cause() -> None:
    """Nov 4, 2026: oil -8.1% with the equal-weight S&P +1.9% was tagged an "oil shock"."""
    results, extras = _tally_inputs()
    oil = extras["oil"].copy()
    oil.iloc[-20:] = oil.iloc[-20:] * 0.90  # oil -10% on the last chips day
    extras["oil"] = oil
    newest = tally(verify_all(results, TODAY.date()), extras).days[0]
    assert newest.oil_pct is not None and newest.oil_pct < -9 and newest.driver == "chips" and "oil" not in newest.moved


def test_credit_and_the_debt_funded_names_annotate_the_day() -> None:
    results, extras = _tally_inputs()
    newest = tally(verify_all(results, TODAY.date()), extras).days[0]
    assert newest.driver == "chips" and newest.moved == ["credit"]
    assert newest.credit_pct == pytest.approx(-1.0, abs=0.01) and newest.ai_leverage_pct == pytest.approx(
        -10.0, abs=0.01
    )
    assert "high-yield bonds -1.00%" in newest.detail and "debt-funded AI names -10.0%" in newest.detail


def test_sensitivity_recovers_the_betas_and_prints_in_the_summary() -> None:
    rng = np.random.default_rng(5)
    mkt = series(list(rng.normal(0, 0.5, 200)))
    dy = series(list(rng.normal(0, 3.0, 200)))
    soxx = 1.8 * mkt - 0.05 * dy + series(list(rng.normal(0, 0.1, 200)))
    fit = sensitivity(soxx, mkt, dy)
    assert fit is not None
    market, rate, r2, n = fit
    assert market == pytest.approx(1.8, abs=0.05) and rate == pytest.approx(-0.5, abs=0.1)  # per 10 bps
    assert r2 > 0.95 and n == 120
    assert sensitivity(soxx.iloc[:40], mkt.iloc[:40], dy.iloc[:40]) is None
    results, extras = _tally_inputs()
    t = tally(verify_all(results, TODAY.date()), extras)
    assert t.market_beta == pytest.approx(BETA, abs=0.3) and t.rate_beta_10bps is not None
    assert "a 10 bps rise in the 10Y has gone with SOXX" in t.summary


def test_decompose_uses_only_earlier_sessions_and_needs_history() -> None:
    rng = np.random.default_rng(5)
    mkt = series(list(rng.normal(0, 0.5, 200)))
    dy = series(list(rng.normal(0, 3.0, 200)))
    soxx = 1.8 * mkt - 0.05 * dy + series(list(rng.normal(0, 0.1, 200)))
    soxx.iloc[-1], mkt.iloc[-1], dy.iloc[-1] = -6.0, -2.0, 0.0
    expected, share = decompose(soxx, mkt, dy, [soxx.index[-1]])[soxx.index[-1]]
    assert expected == pytest.approx(-3.6, abs=0.15) and share == pytest.approx(0.6, abs=0.03)
    assert decompose(soxx.iloc[:40], mkt.iloc[:40], dy.iloc[:40], [soxx.index[39]]) == {}  # under 60 sessions


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
    assert t.news_counts == {"fed": 1, "unlabelled": 2}
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
    results = {k: v for k, v in results.items() if not k.startswith("kospi@")}
    items = {i.label: i for i in overnight(verify_all(results, TODAY.date()))}
    assert items["KOSPI"].value is None and items["Dollar (UUP)"].value is None


def test_overnight_dollar_comes_from_the_tape() -> None:
    items = {i.label: i for i in overnight(verify_all(market(), TODAY.date()), {"dollar": series([28.0, 28.14])})}
    assert items["Dollar (UUP)"].change_pct == pytest.approx(0.5) and "Alpaca" in (items["Dollar (UUP)"].source or "")


def test_a_failed_headline_read_is_not_shown_as_a_label() -> None:
    results, extras = _tally_inputs()
    first = tally(verify_all(results, TODAY.date()), extras).days[0].date
    t = tally(verify_all(results, TODAY.date()), extras, {first: {"failed": 1, "last_try": "x", "problem": "y"}})
    assert t.days[0].news is None and t.news_counts == {"unlabelled": 3}


def test_a_session_behind_input_is_unknown_not_quiet() -> None:
    """Nov 5 16:25: SOXX withheld and Cboe's VIX settle not yet posted. Yesterday's VIX (16.6)
    read as QUIET and the page said Calm on a -12% semis day with VIX at 28."""
    f = calm(vix=16.61, soxx_rsi=None, soxx_chg_1d_pct=None, soxx_return_z=None)
    f.dates.update({"y10": "2026-11-05", "igv": "2026-11-05", "vix": "2026-11-04"})
    f.dates.pop("soxx", None)
    drivers = build_drivers(f, {})
    assert {d.key: d.state for d in drivers} == {"liquidity": "QUIET", "risk_off": "UNKNOWN", "semis": "UNKNOWN"}
    assert stress(f, drivers).level == "UNKNOWN"


def test_a_session_behind_input_cannot_fire_either() -> None:
    """Yesterday's VIX 31 says nothing about today: unknown, and no VIX>30 override."""
    f = calm(vix=31.0)
    f.dates.update({"y10": "2026-11-05", "igv": "2026-11-05", "soxx": "2026-11-05", "vix": "2026-11-04"})
    drivers = build_drivers(f, {})
    assert next(d for d in drivers if d.key == "risk_off").state == "UNKNOWN"
    assert stress(f, drivers).level == "UNKNOWN"
