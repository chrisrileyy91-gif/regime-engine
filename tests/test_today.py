"""Today's stress, the scapegoat tally and the overnight line."""

from __future__ import annotations

import pandas as pd
import pytest

from app.drivers import build_drivers
from app.today import overnight, stress, tally
from app.verify import verify_all
from tests.conftest import TODAY, flat_then, market, series
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


def _tally_market() -> tuple[dict, pd.Series]:
    """300 sessions; four semis drops of -3%: with a yield jump, a VIX spike, the S&P down 2%, alone."""
    n = 300
    soxx, spy = [100.0] * n, [500.0] * n
    y10 = list(flat_then(4.0, n))
    vix = [15.0] * n
    for i, kind in ((-40, "rates"), (-30, "fear"), (-20, "market"), (-10, "chips")):
        soxx[i:] = [v * 0.97 for v in soxx[i:]]
        if kind == "rates":
            y10[i:] = [v + 0.15 for v in y10[i:]]
        if kind == "fear":
            vix[i] = 19.0
        if kind == "market":
            spy[i:] = [v * 0.98 for v in spy[i:]]
    return market(soxx=series(soxx), y10=series(y10), vix=series(vix)), series(spy)


def test_tally_classifies_each_drop_by_what_moved_that_day() -> None:
    results, spy = _tally_market()
    t = tally(verify_all(results, TODAY.date()), spy)
    assert [d.driver for d in t.days] == ["chips", "market", "fear", "rates"]  # newest first
    assert t.counts == {"rates": 1, "fear": 1, "market": 1, "chips": 1}
    assert "Of the last 4 semis drops of 1.5% or more" in t.summary


def test_without_the_market_series_nothing_is_called_chip_specific() -> None:
    """Chip-specific is a residual: it can't be claimed if the S&P 500 wasn't checked."""
    results, _ = _tally_market()
    drivers = {d.driver for d in tally(verify_all(results, TODAY.date())).days}
    assert "chips" not in drivers and "unknown" in drivers


def test_tally_without_rates_data_says_unknown() -> None:
    results, spy = _tally_market()
    for key in [k for k in results if k.startswith("y10@")]:
        results[key] = results[key].__class__(key, "TNX", key.split("@")[1], None, "down")
    days = tally(verify_all(results, TODAY.date()), spy).days
    assert {d.driver for d in days} <= {"fear", "market", "unknown"} and "chips" not in {d.driver for d in days}


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
